from llm_building_blocks.prompt_evaluator import PromptEvaluator, Prompt
from config.config import get_default_configs, Config
from utils.metrics import evaluate, save_validation_metrics
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from llm_building_blocks.prompt_catalogue import PromptCatalogue, Prompt

class PromptOptimizer:
    def __init__(self, evaluator: PromptEvaluator, config: Config):
        self.evaluator = evaluator
        self.evaluator_config = evaluator.config
        self.llm = evaluator.llm
        self.config = config

    def evaluate_multiple_prompts(self, prompt_catalogue, set, iteration, train=False):
        """
        Evaluate a set of prompts either on the training or validation set.

        Args:
            prompt_catalogue (PromptCatalogue): Catalogue containing all prompts.
            set (pd.DataFrame): Dataset with 'sentence' and 'label' columns.
            iteration (int): Current iteration index.
            train (bool): Flag indicating whether it's a training evaluation.
        """
        if train == True:
            # Evaluate all prompts in prompt catalogue if not already computed
            print(f"Evaluation prompt catalogue in iteration {iteration} for train_set")
            for prompt_id, prompt in prompt_catalogue.get_prompts():
                if prompt.current_score == None and prompt.current_predictions == None:
                    pred = self.evaluator.predict(prompt, list(set["sentence"]))
                    prompt.current_predictions = [label.lower() for label in pred]
                    prompt.current_score = evaluate(prompt.current_predictions, list(set["label"]))["nmae"]

        else:
            print(f"Evaluation prompt catalogue in iteration {iteration} for val set")

            # Compute and save metrics for validation set
            nmaes = []
            accuracys = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            for prompt_id, prompt in prompt_catalogue.get_prompts():
                predictions = self.evaluator.predict(prompt, list(set["sentence"]))
                predictions = [label.lower() for label in predictions]
                metrics = evaluate(predictions, list(set["label"]))
                nmaes.append(metrics["nmae"])
                accuracys.append(metrics["accuracy"])
                precision_scores.append(metrics["precision_macro"])
                recall_scores.append(metrics["recall_macro"])
                f1_scores.append(metrics["f1_macro"])

            mean_nmae = np.mean(nmaes)
            mean_accuracy = np.mean(accuracys)
            mean_precision = np.mean(precision_scores)
            mean_recall = np.mean(recall_scores)
            mean_f1 = np.mean(f1_scores)

            metrics = {"mean_nmae": float(mean_nmae), "mean_accuracy": float(mean_accuracy), "mean_f1": float(mean_f1), "mean_precision": float(mean_precision),"mean_recall": float(mean_recall)}
            save_validation_metrics(self.config, metrics, suffix="_prompts_" + str(iteration))

    def run_optimization_loop(self, prompt_catalogue: PromptCatalogue, train, val, iterations=3) -> Dict[str, Tuple[List[str], float]]:
        """
        Run the full prompt optimization loop for a number of iterations.

        Args:
            prompt_catalogue (PromptCatalogue): Initial catalogue of prompts.
            train (pd.DataFrame): Training dataset with 'sentence' and 'label' columns.
            val (pd.DataFrame): Validation dataset with 'sentence' and 'label' columns.
            iterations (int): Number of optimization iterations.

        Returns:
            Dict[str, Tuple[List[str], float]]: Updated prompt catalogue after optimization.
        """
        target_size = len(prompt_catalogue.get_prompts())
        for i in range(iterations):
            # Evaluate Currente Catalogue on validation data, save
            self.evaluate_multiple_prompts(prompt_catalogue, val, i)

            # Evaluate Current Catalogue on train
            self.evaluate_multiple_prompts(prompt_catalogue, train, i, train=True)
            # Reorder Catalogue by Scores
            prompt_catalogue.reorder_catalogue_by_score()

            # Save Prompt Catalogue
            prompt_catalogue.save(iteration=i, config=self.config)

            # Optimize Prompts returns new prompts as strings and respective scores
            new_prompts_str = self.optimize_prompts(prompt_catalogue, self.config.prompt.k_best_worst)

            # We store scores of old + new catalogue and update catalogue
            prompt_catalogue.append_prompts(new_prompts_str)

            # Evaluate new prompts
            self.evaluate_multiple_prompts(prompt_catalogue, train, i, train=True)

            # We select prompts to have same amount of prompts as in beginning
            self.greedy_select_prompts(prompt_catalogue, list(train["label"]), size=target_size)

        # Final Evaluation
        prompt_catalogue.save(iteration=iterations, config=self.config)
        self.evaluate_multiple_prompts(prompt_catalogue, val, iterations)
        prompt_catalogue.reorder_catalogue_by_score()

        return prompt_catalogue

    def greedy_select_prompts(self, catalogue: PromptCatalogue, labels, size):
        """
        Select a subset of prompts using greedy selection to maximize accuracy.

        Args:
            catalogue (PromptCatalogue): Prompt catalogue with current predictions.
            labels (List[str]): Ground truth labels for training data.
            size (int): Number of prompts to select.
        """
        S = set()

        def evaluate_set(prompt_ids, catalogue: PromptCatalogue):
            if not prompt_ids:
                return 0.0, []  # Return empty list for best_prompt_per_sample when set is empty
                
            # For each sample, find the best prompt in the set
            correct_predictions = 0
            best_prompt_per_sample = []
            
            for i in range(len(labels)):
                best_prompt_for_sample = None
                best_prediction = None
                
                # Find the prompt that gives the correct prediction for this sample
                for prompt_id in prompt_ids:
                    # Select Prompt that matches prompt_id
                    prompt = next((v for k, v in catalogue.get_prompts() if k == prompt_id), None)
                    predictions = prompt.current_predictions
                    if predictions[i].lower() == labels[i]:
                        best_prompt_for_sample = prompt_id
                        best_prediction = predictions[i]
                        break
                
                # If no prompt got it right, find the one that's most confident
                if best_prompt_for_sample is None:
                    for prompt_id in prompt_ids:
                        prompt = next((v for k, v in catalogue.get_prompts() if k == prompt_id), None)
                        predictions = prompt.current_predictions
                        if best_prediction is None or predictions[i] != 'neutral':  # Prefer non-neutral predictions
                            best_prompt_for_sample = prompt_id
                            best_prediction = predictions[i]
                
                best_prompt_per_sample.append(best_prompt_for_sample)
                if best_prediction.lower() == labels[i]:
                    correct_predictions += 1
            
            # Compute accuracy score
            accuracy = correct_predictions / len(labels)
            return accuracy, best_prompt_per_sample
        

        for j in range(size):
            best_improvement = -float('inf')
            best_prompt_id = None
            best_prompt_per_sample = None
            
            # Try each remaining prompt
            for prompt_id, prompt in catalogue.get_prompts():
                if prompt_id not in S:
                    # Calculate improvement when adding this prompt
                    current_score, _ = evaluate_set(S, catalogue)
                    new_score, new_best_prompts = evaluate_set(S | {prompt_id}, catalogue)
                    improvement = new_score - current_score
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_prompt_id = prompt_id
                        best_prompt_per_sample = new_best_prompts
            
            # Add best prompt to set
            if best_prompt_id is not None:
                S.add(best_prompt_id)

        catalogue.filter_prompts(S)

    def optimize_prompts(self, prompt_catalogue: PromptCatalogue, k):
        """
        Generate new prompts by prompting the LLM using top and bottom k prompts.

        Args:
            prompt_catalogue (PromptCatalogue): Catalogue containing current prompts.
            k (int): Number of top and bottom prompts to use for optimization.

        Returns:
            List[str]: List of newly generated prompt strings.
        """
        base_prompt_good = "<start_of_turn>user\nHere are some really good prompts:"
        base_prompt_bad = "<start_of_turn>user\nHere are some prompts that don't work well:"

        top_k = prompt_catalogue.get_prompt_list_str()[:k]
        flop_k = prompt_catalogue.get_prompt_list_str()[-k:]

        for i in range(k):
            base_prompt_good += f"Prompt {i+1}:\n{top_k[i]}\n\n"
            base_prompt_bad += f"Prompt {i+1}:\n{flop_k[i]}\n\n"

        base_prompt_good += f"Based on these, suggest exactly 1 new prompt templates in a similar style. The goal is to always generate a prompt that can distinguish sentiments of a review as either postive, negative, or neutral. You will want to generate long prompts, that are very specific. Always include as a first sentence in your prompt that the output should be negative, positive or neutral, nothing more. Do not put brakets in your output or any special characters in there, just natural language. Your prompt should not work on all inputs, but very well on a certain type of inputs. So try to produce expert prompts for certain reviews. For example this sentence: 'I highly recommend any location but his.' should be classified as negative. This sentence:'They are just as good at 'soft skills' as translating.' should be classified as positive. Only output the new prompt, nothing else. Avoid at all cost to output anything else than the plain prompt. That means no prefix or suffix. \n\n<end_of_turn>\n<start_of_turn>model\n"
        base_prompt_bad += f"Based on these, suggest exactly 1 improved prompt templates, that could work better. The goal is to always generate a prompt that can distinguish sentiments of a review as either postive, negative, or neutral. You will want to generate long prompts, that are very specific. Always include as a first sentence in your prompt that the output should be negative, positive or neutral, nothing more. Do not put brakets in your output or any special characters in there, just natural language. Your prompt should not work on all inputs, but very well on a certain type of inputs. So try to produce expert prompts for certain reviews. For example this sentence: 'I highly recommend any location but his.' should be classified as negative. This sentence: They are just as good at 'soft skills' as translating.' should be classified as positive. Only output the new prompt, nothing else. Avoid at all cost to output anything else than the plain prompt. That means no prefix or suffix. \n\n<end_of_turn>\n<start_of_turn>model\n"

        generated = []

        for label, prompt_text in [("good", base_prompt_good), ("bad", base_prompt_bad)]:
            for i in range(k): 
                res = self.evaluator.llm.create_completion(
                    prompt=prompt_text, max_tokens=170, temperature=0.9, stop=["\n"]
                )
                generated.append(res["choices"][0]["text"].strip())

        return generated