import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from os.path import dirname
from enum import IntEnum
from typing import Dict, List, Tuple
from numpy.typing import ArrayLike
from llama_cpp import Llama as LLM, llama_kv_self_clear
from re import split
import re
import time
from tqdm import tqdm
import csv


@dataclass
class PromptEvaluatorConfig:
    llm_gguf_path: str

    n_gpu_layers: int
    n_ctx: int

    temperature: float
    top_k: int
    top_p: float
    min_p: float

    stops: list[str]

    max_n_thinking_tokens: int
    verbose: bool
    debug: bool = False

    @staticmethod
    def for_gemma_3_4b_it(max_n_thinking_tokens=1000, verbose=False, debug=False):
        return PromptEvaluatorConfig(
            llm_gguf_path=f"{dirname(__file__)}/../../llms/gemma-3-4b-it-q4_0.gguf",
            n_gpu_layers=-1,
            # n_ctx=1024,
            n_ctx=8192,
            # CHANGED THE ABOVE TO MAKE IT FASTER
            # Parameters from https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune
            temperature=1.0,
            top_k=64,
            top_p=0.95,
            min_p=0.0,
            stops=["<end_of_turn>"],
            max_n_thinking_tokens=max_n_thinking_tokens,
            verbose=verbose,
            debug=debug,
        )


class Sentiment(IntEnum):
    POSITIVE = 0
    NEUTRAL = 1
    NEGATIVE = 2


@dataclass
class Prompt:
    template: str
    """
    A string template that guides the text generation and logit evaluation.
    
    All instances of
    - `<INPUT>` will be replaced with the classifiable text.
    - `<THINK>` and `<COUNT>` will be replaced with thinking until just before a stop token is reached.

    The template MUST specify 1+ evaluators after the first occurence of `<INPUT>` in this order:
    
    1. 0+ `<COUNT>` instances that will return one normalized distribution of sentiment-mapped phrase occurences in their generated text snippets.
    2. 0 or 1 `<PROBE>` instance that will return a constrained next-phrase probability distribution.

    Each evaluator has the same weight in the final probability distribution.
    """

    sentiment_map: Dict[str, Sentiment]
    """
    A mapping from detectable substrings to sentiments.
    """

    @staticmethod
    def prompt_catalogue() -> Dict[str, "Prompt"]:
        return {
            "direct_v1": Prompt.direct_example(),
            "direct_v2": Prompt.second_direct_example(),
            "emoji_example": Prompt.emoji_example(),
            "direct_v3": Prompt(
                template="<start_of_turn>user\nCan you analyze the sentiment in this review? Reply only with one word: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4": Prompt(
                template="<start_of_turn>user\nSentiment classification task. Choose one of the following: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4longV1": Prompt(
                template="<start_of_turn>user\nYou are an expert in recognizing people's sentiments. Most of the text you see will be reviews. If there is a negative word in the sentence, that is not enough to say negative. Same for positive. If you are unsure if positive or neutral, rather take neutral. If you are unsure if negative or neutral, rather take neutral. Be careful of sarcasm. Reply only with one word: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4longV2": Prompt(
                template="<start_of_turn>user\nYou are an expert in recognizing people's sentiments. Most of the text you see will be from the internet. If there is a positive word in the sentence, that is not enough to say positive. Same for negative. If you are unsure if positive or neutral, rather take neutral. If you are unsure if negative or neutral, rather take neutral. Be careful of sarcasm. Reply only with one word: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4longV3": Prompt(
                template="<start_of_turn>user\nSentiment classification task. Most of the text you see will be reviews. If there is a positive word in the sentence, that is not enough to say positive. If there is a negative word in the sentence, that is not enough to say negative. If you are unsure if positive or neutral, rather take neutral. If you are unsure if negative or neutral, rather take neutral. Be careful of sarcasm. Reply only with one word: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4longV4": Prompt(
                template="<start_of_turn>user\nSentiment classification task. Most of the text you see will be reviews. If there is a positive word in the sentence, that is not enough to say positive. If there is a negative word in the sentence, that is not enough to say negative. If you are unure rather take neutral. Be careful of sarcasm. Reply only with one word: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4internet": Prompt(
                template="<start_of_turn>user\nSentiment classification task. These are sentences from the internet. Choose one of the following: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4crowdaware": Prompt(
                template="<start_of_turn>user\nSentiment classification task. These are sentecnes from the internet. It is common for people to use different writingstyles on the internet. Choose one of the following: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4extrasarcasm": Prompt(
                template="<start_of_turn>user\nSentiment classification task. There can be some sarcasam pay attention to this. Choose one of the following: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4carefully": Prompt(
                template="<start_of_turn>user\nSentiment classification task. Choose carefully one of the following: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4nosinglewordpred": Prompt(
                template="<start_of_turn>user\nSentiment classification task. Dont let yourself be influenced by single words too much. Aanalyze the sentence as a whole. Choose carefully one of the following: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4_karltest": Prompt(
                template="<start_of_turn>user\nClassify the sentiment! Choose one of the following sentiments: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4_karltest_v2": Prompt(
                template="<start_of_turn>user\nDetermine the sentiment of the following sentence. Respond with only one word: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4_emphatic": Prompt(
                template="<start_of_turn>user\nThis is a sentiment classification task. Your job is to carefully determine whether the review is positive, negative, or neutral. Respond with only one word.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4_shorter": Prompt(
                template="<start_of_turn>user\nClassify the sentiment: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>\\",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v4_assertive": Prompt(
                template="<start_of_turn>user\nClassify the sentiment of this review. Choose only from: positive, negative, or neutral. Do not explain your answer.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>\\",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "direct_v5": Prompt(
                template="<start_of_turn>user\nWhat is the sentiment of this review? Choose oen of the following: positive, negative, or neutral. Pay attention to irony and sarcasm!\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "thoughtful_analyst": Prompt(
                template="<start_of_turn>user\nPlease act as a sentiment analysis expert. Assess the following text and reply with one word: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "professional_opinion": Prompt(
                template="<start_of_turn>user\nGive a professional sentiment judgment for this review: positive, negative, or neutral?\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "sarcasm_aware": Prompt(
                template="<start_of_turn>user\nBe cautious of irony and sarcasm. Is the sentiment of the following review positive, negative, or neutral?\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
            "market_review_style": Prompt(
                template="<start_of_turn>user\nAnalyze the tone of the customer review below as if preparing for a product sentiment report. Reply with one of: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            ),
        }
    
    @staticmethod
    def export_base_prompts_to_csv():
        """Export the base prompts to base_prompts.csv format"""
        prompts = Prompt.prompt_catalogue()
        
        # Create DataFrame with required columns
        df = pd.DataFrame(columns=['idx', 'prompt', 'accuracy'])
        
        # Convert each prompt to the required format
        for idx, prompt in prompts.items():
            # Format the prompt string to match the CSV format
            prompt_str = f"<start_of_turn>user\n{prompt.template}<end_of_turn>\n<start_of_turn>model\n<PROBE>"
            # Replace all newlines with \Line
            prompt_str = prompt_str.replace("\n", "\Line")
            # Add row to DataFrame
            df = pd.concat([df, pd.DataFrame({
                'idx': [idx],
                'prompt': [prompt_str],
                'accuracy': [None]  # Accuracy will be calculated during evaluation
            })], ignore_index=True)
        
        # Save to CSV
        df.to_csv("../../data/base_prompts.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\n")
        print(f"Exported {len(df)} base prompts to base_prompts.csv")


    @staticmethod
    def direct_example():
        """
        Directly asks the model to classify the sentiment of a text.
        Uses the <PROBE> evaluator to convert logits to probabilities.
        """
        return Prompt(
            template="<start_of_turn>user\nYou are an expert in recognizing people's sentiments. Reply only with one word: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
            sentiment_map={
                "positive": Sentiment.POSITIVE,
                "negative": Sentiment.NEGATIVE,
                "neutral": Sentiment.NEUTRAL,
            },
        )

    @staticmethod
    def second_direct_example():
        """
        Directly asks the model to classify the sentiment of a text.
        Uses the <PROBE> evaluator to convert logits to probabilities.
        """
        return Prompt(
            template="<start_of_turn>user\nWhat is the sentiment of this review?\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
            sentiment_map={
                "positive": Sentiment.POSITIVE,
                "negative": Sentiment.NEGATIVE,
                "neutral": Sentiment.NEUTRAL,
            },
        )

    @staticmethod
    def emoji_example():
        """
        Provides the model with a set of emojis and asks it to use them to summarize the sentiment of a text.
        Uses the <THINK> marker to generate an internal thought process.
        Uses the <COUNT> evaluator to convert emoji occurrences in the final answer to probabilities.
        """
        return Prompt(
            template="<start_of_turn>user\nI would like to use a subset of the following emojis to summarize my sentiment: 🥰, 😘, 🤗, 😎, 👍, 🧐, ✍, 👀, 🤐, 😶, 🙄, 😪, 😢, 😡, 💩\nThis is a sentence without emojis where I express my sentiment:\n\n<INPUT>\n\nThink carefully about which emojis best summarize my sentiment. Then, let me know and I'll ask for your final verdict.<end_of_turn>\n<start_of_turn>model\n<THINK><end_of_turn>\n<start_of_turn>user\nOkay, so without explanations, which of the provided emojis best summarize my sentiment?<end_of_turn>\n<start_of_turn>model\n<COUNT>",
            sentiment_map={
                "🥰": Sentiment.POSITIVE,
                "😘": Sentiment.POSITIVE,
                "🤗": Sentiment.POSITIVE,
                "😎": Sentiment.POSITIVE,
                "👍": Sentiment.POSITIVE,
                "🧐": Sentiment.NEUTRAL,
                "✍": Sentiment.NEUTRAL,
                "👀": Sentiment.NEUTRAL,
                "🤐": Sentiment.NEUTRAL,
                "😶": Sentiment.NEUTRAL,
                "🙄": Sentiment.NEGATIVE,
                "😪": Sentiment.NEGATIVE,
                "😢": Sentiment.NEGATIVE,
                "😡": Sentiment.NEGATIVE,
                "💩": Sentiment.NEGATIVE,
            },
        )


class PromptEvaluator:
    config: PromptEvaluatorConfig
    llm: LLM

    def __init__(
        self,
        config: PromptEvaluatorConfig,
    ):
        self.config = config
        self.llm = LLM(
            model_path=config.llm_gguf_path,
            n_gpu_layers=config.n_gpu_layers,
            n_ctx=config.n_ctx,
            logits_all=True,
            verbose=config.verbose,
        )

    def predict_proba(self, prompt: Prompt, X: ArrayLike):
        X = np.asarray(X, dtype=np.str_)

        # Plan how to fulfill the template.
        plan = split(
            r"(<(?:THINK|PROBE|COUNT)>)",
            prompt.template,
        )
        try:
            # If <PROBE> is present, ignore everything after its first occurence.
            cutoff_idx = plan.index("<PROBE>") + 1
        except ValueError:
            try:
                # If <COUNT> is present, ignore everything after its last occurence.
                cutoff_idx = len(plan) - plan[::-1].index("<COUNT>")
            except ValueError:
                raise ValueError(
                    "Prompt template must contain at least one <PROBE> or <COUNT> after <INPUT>.",
                )
        plan = plan[:cutoff_idx]

        if self.config.verbose:
            print(f"Evaluation plan: {plan}")

        # Use the specified stop tokens to stop generation early.
        stop_tokens = [
            self.llm.tokenizer_.encode(stop, add_bos=False, special=True)[0]
            for stop in self.config.stops
        ]
        phrases = prompt.sentiment_map.keys()

        # Reset the LLM and evaluate the beginning of sequence token.
        self.llm.reset()
        self.llm.input_ids.fill(-1)
        self.llm.scores.fill(0)
        llama_kv_self_clear(self.llm.ctx)
        self.llm.eval([self.llm.token_bos()])

        def score_single_input(x: str) -> np.ndarray:
            # Soft-reset the LLM until right after the beginning of sequence token.
            self.llm.n_tokens = 1

            # Count occurrences of phrases in <COUNT> blocks.
            phrase_counts = np.zeros(len(phrases), dtype=np.int_)
            # Evaluate the probabilities of phrases occuring in the <PROBE> block.
            phrase_probe_probabilities = np.zeros(len(phrases), dtype=np.float64)

            for action in plan:
                match action:
                    case "<THINK>" | "<COUNT>":
                        n_tokens_before_thinking = self.llm.n_tokens

                        # Generate until a stop token is reached or the maximum number of tokens is reached.
                        for token in self.llm.generate(
                            self.llm.input_ids[:n_tokens_before_thinking],
                            temp=self.config.temperature,
                            top_k=self.config.top_k,
                            top_p=self.config.top_p,
                            min_p=self.config.min_p,
                        ):
                            if (
                                token in stop_tokens
                                or self.llm.n_tokens - n_tokens_before_thinking
                                >= self.config.max_n_thinking_tokens
                            ):
                                break

                        if self.config.verbose:
                            print(
                                f"Thought for {self.llm.n_tokens - n_tokens_before_thinking} tokens and produced",
                                repr(
                                    self.llm.tokenizer_.decode(
                                        self.llm.input_ids[
                                            n_tokens_before_thinking : self.llm.n_tokens
                                        ]
                                    )
                                ),
                            )

                        # Only count phrases in <COUNT> blocks.
                        if action == "<COUNT>":
                            generated_text = self.llm.tokenizer_.decode(
                                self.llm.input_ids[
                                    n_tokens_before_thinking : self.llm.n_tokens
                                ]
                            )
                            phrase_counts += [generated_text.count(p) for p in phrases]

                    case "<PROBE>":
                        n_tokens_before_probing = self.llm.n_tokens

                        for i, phrase in enumerate(phrases):
                            # Determine the probability of insertion for this phrase.
                            tokenized_phrase = self.llm.tokenizer_.encode(
                                phrase, add_bos=False, special=True
                            )
                            self.llm.eval(tokenized_phrase)

                            logprobs = LLM.logits_to_logprobs(
                                self.llm.scores[
                                    (self.llm.n_tokens - len(tokenized_phrase) - 1) : (
                                        self.llm.n_tokens - 1
                                    )
                                ],
                            )[:, tokenized_phrase]
                            phrase_probe_probabilities[i] = np.exp(logprobs.sum())

                            # Soft-reset the LLM to the state before the <PROBE> block.
                            self.llm.n_tokens = n_tokens_before_probing

                    case _:  # Fully constrained blocks.
                        # Replace <INPUT> markers with x.
                        tokenized_text = self.llm.tokenizer_.encode(
                            action.replace("<INPUT>", x), add_bos=False, special=True
                        )

                        # Try to reuse precomputed logprobs by checking the soft-reset area.
                        n_precomputed_tokens = LLM.longest_token_prefix(
                            tokenized_text, self.llm.input_ids[self.llm.n_tokens :]
                        )
                        self.llm.n_tokens += n_precomputed_tokens
                        self.llm.eval(tokenized_text[n_precomputed_tokens:])

                        if self.config.verbose and n_precomputed_tokens > 0:
                            print(
                                f"Reusing {n_precomputed_tokens} precomputed tokens that spell",
                                repr(
                                    self.llm.tokenizer_.decode(
                                        tokenized_text[:n_precomputed_tokens]
                                    )
                                ),
                            )

            # Normalize the phrase counts to probabilities and weigh them 1:1 with the probe probabilities.
            phrase_probabilitites = phrase_probe_probabilities + (
                phrase_counts / max(1, phrase_counts.sum())
            )

            # Realize the phrase-sentiment mapping.
            sentiment_probabilities = np.zeros(len(Sentiment), dtype=np.float64)
            for i, phrase in enumerate(phrases):
                sentiment_probabilities[
                    prompt.sentiment_map[phrase]
                ] += phrase_probabilitites[i]

            # Return normalized sentiment probabilities.
            return sentiment_probabilities / sentiment_probabilities.sum()

        results = []
        iterator = tqdm(X, desc="Evaluating") if self.config.debug else X

        for x in iterator:
            if self.config.debug:
                print(f"\nInput: {x}")
                start_time = time.time()
            result = score_single_input(x)
            if self.config.debug:
                print(f"Done in {time.time() - start_time:.2f} sec")
            results.append(result)

        return pd.DataFrame(
            results,
            columns=[Sentiment(i).name for i in range(len(Sentiment))],
            index=X,
        )
        # return pd.DataFrame(
        #     [score_single_input(x) for x in X],
        #     columns=[Sentiment(i).name for i in range(len(Sentiment))],
        #     index=X,
        # )

    def predict(self, prompt: Prompt, X: ArrayLike):
        return self.predict_proba(prompt, X).idxmax(axis=1)


class PromptOptimizer:
    def __init__(self, evaluator: PromptEvaluator):
        self.evaluator = evaluator
        self.config = evaluator.config
        self.llm = evaluator.llm

    def fill_prompt_template(self, prompt: Prompt, input_text: str) -> str:
        return prompt.template.replace("<INPUT>", input_text)

    mutable_catalogue = Prompt.prompt_catalogue().copy()

    def add_prompt(self, label: str, text: str):
        self.mutable_catalogue[label] = Prompt(
            template=text,
            sentiment_map={
                "positive": Sentiment.POSITIVE,
                "negative": Sentiment.NEGATIVE,
                "neutral": Sentiment.NEUTRAL,
            },
        )

    def evaluate_prompts(
        self, X: List[str], y_true: List[str]
    ) -> Dict[str, Tuple[List[str], float]]:
        # Add caching logic at the beginning
        import os
        cache_path = "../../data/prompt_eval_cache.csv"
        cache_df = pd.read_csv(cache_path) if os.path.exists(cache_path) else pd.DataFrame(columns=["idx", "accuracy", "predictions"])

        results = {}
        prompts = pd.read_csv("../../data/current_prompt_catalogue.csv")

        # Convert cache to dictionary for fast lookup
        cached_results = {row["idx"]: (eval(row["predictions"]), row["accuracy"]) for _, row in cache_df.iterrows()}

        for idx, row in prompts.iterrows():
            print(f"\n--- Evaluating with prompt: {row['idx']} ---")
            # Use cache if available
            if row["idx"] in cached_results:
                print(f"Using cached result for {row['idx']}")
                results[idx] = cached_results[row["idx"]]
                prompts.loc[idx, "accuracy"] = cached_results[row["idx"]][1]
                continue
            # creating prompts out of the string prompts
            prompt = Prompt(
                template=row["prompt"]
                .replace("Prompt(template='", "")
                .replace(
                    "sentiment_map={'positive': <Sentiment.POSITIVE: 0>, 'negative': <Sentiment.NEGATIVE: 2>, 'neutral': <Sentiment.NEUTRAL: 1>}), \"",
                    "",
                ),
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
            )
            # Run model
            y_proba = evaluator.predict_proba(prompt, X)

            with pd.option_context("display.float_format", "{:0.4f}".format):
                print(
                    "--------------------------------------------------------\n",
                    y_proba,
                    "\n--------------------------------------------------------",
                )

            # Get predicted labels
            y_pred = y_proba.idxmax(axis=1).tolist()

            # Compute accuracy
            accuracy = accuracy_score(
                [label.lower() for label in y_true], [pred.lower() for pred in y_pred]
            )
            results[idx] = (y_pred, accuracy)
            prompts.loc[idx, "accuracy"] = accuracy

            print(f"Accuracy: {accuracy:.2f}")

        # Update the cache file with all current results
        new_cache_df = pd.DataFrame([
            {
                "idx": prompts.iloc[int(prompt_id)]["idx"],
                "accuracy": accuracy,
                "predictions": str(predictions)
            }
            for prompt_id, (predictions, accuracy) in results.items()
        ])
        new_cache_df.to_csv(cache_path, index=False)

        # Sort prompts by accuracy descending
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1][1], reverse=True))
        best_prompts = self.greedy_select_prompts(sorted_results, X, y_true, k=24)
        
        # Print detailed information about selected prompts
        print("\nSelected prompts and their individual accuracies:")
        for prompt_id, (predictions, accuracy) in best_prompts.items():
            # Find the correct row using index-based lookup
            try:
                prompt_text = prompts.iloc[int(prompt_id)]['prompt']
                prompt_name = prompts.iloc[int(prompt_id)]['idx']
            except (IndexError, ValueError):
                print(f"Could not find prompt with ID {prompt_id}")
                continue

            print(f"\nPrompt {prompt_name}:")
            print(f"  Text: {prompt_text}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Predictions: {predictions}")
            print(f"  True labels: {y_true}")
        
        # Create new DataFrame with best prompts
        best_prompts_df = pd.DataFrame([
            {
                'idx': prompts.iloc[int(prompt_id)]['idx'],
                'prompt': prompts.iloc[int(prompt_id)]['prompt'],
                'accuracy': accuracy
            }
            for prompt_id, (_, accuracy) in best_prompts.items()
            if int(prompt_id) < len(prompts)
        ])
                
        # Save to current_prompt_catalogue.csv
        best_prompts_df.to_csv("../../data/current_prompt_catalogue.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\n")
        best_prompts_df.to_csv("../../data/best_promts.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\n")
        print(f"\nSaved {len(best_prompts_df)} best prompts to current_prompt_catalogue.csv")
        
        return best_prompts

    def greedy_select_prompts(
        self, 
        results: Dict[str, Tuple[List[str], float]], 
        X: List[str], 
        y_true: List[str], 
        k: int = 24
    ) -> Dict[str, Tuple[List[str], float]]:
        """
        Greedy algorithm to select the best combination of k prompts.
        For each sample, we select the best performing prompt from the set.
        
        Args:
            results: Dictionary mapping prompt IDs to (predictions, accuracy) tuples
            X: Input texts
            y_true: True labels
            k: Number of prompts to select
            
        Returns:
            Dictionary of selected prompts with their predictions and accuracies
        """
        # Initialize empty set S
        S = set()
        selected_results = {}
        
        # Convert y_true to lowercase for consistent comparison
        y_true_lower = [label.lower() for label in y_true]
        
        # Function to evaluate accuracy of a set of prompts
        def evaluate_set(prompt_set):
            if not prompt_set:
                return 0.0, []  # Return empty list for best_prompt_per_sample when set is empty
                
            # For each sample, find the best prompt in the set
            correct_predictions = 0
            best_prompt_per_sample = []
            
            for i in range(len(X)):
                best_prompt_for_sample = None
                best_prediction = None
                
                # Find the prompt that gives the correct prediction for this sample
                for prompt_id in prompt_set:
                    predictions = results[prompt_id][0]
                    if predictions[i].lower() == y_true_lower[i]:
                        best_prompt_for_sample = prompt_id
                        best_prediction = predictions[i]
                        break
                
                # If no prompt got it right, find the one that's most confident
                if best_prompt_for_sample is None:
                    for prompt_id in prompt_set:
                        predictions = results[prompt_id][0]
                        if best_prediction is None or predictions[i] != 'neutral':  # Prefer non-neutral predictions
                            best_prompt_for_sample = prompt_id
                            best_prediction = predictions[i]
                
                best_prompt_per_sample.append(best_prompt_for_sample)
                if best_prediction.lower() == y_true_lower[i]:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(X)
            return accuracy, best_prompt_per_sample
        
        # Main greedy selection loop
        for j in range(k):
            best_improvement = -float('inf')
            best_prompt = None
            best_prompt_per_sample = None
            
            # Try each remaining prompt
            for prompt_id in results:
                if prompt_id not in S:
                    # Calculate improvement when adding this prompt
                    current_score, _ = evaluate_set(S)
                    new_score, new_best_prompts = evaluate_set(S | {prompt_id})
                    improvement = new_score - current_score
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_prompt = prompt_id
                        best_prompt_per_sample = new_best_prompts
            
            # Add best prompt to set
            if best_prompt is not None:
                S.add(best_prompt)
                selected_results[best_prompt] = results[best_prompt]
                print(f"\nSelected prompt {best_prompt} with improvement {best_improvement:.4f}")
                print("Best prompt per sample distribution:")
                prompt_counts = {}
                for prompt in best_prompt_per_sample:
                    prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
                for prompt, count in prompt_counts.items():
                    print(f"  Prompt {prompt}: {count} samples")
        
        # Print final evaluation
        final_score, final_best_prompts = evaluate_set(S)
        print(f"\nFinal ensemble accuracy: {final_score:.4f}")
        print("Final best prompt per sample distribution:")
        prompt_counts = {}
        for prompt in final_best_prompts:
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
        for prompt, count in prompt_counts.items():
            print(f"  Prompt {prompt}: {count} samples")
        
        return selected_results

    def optimize_prompts(
        self, prompts: Dict[str, Tuple[List[str], float]], k: int
    ) -> Dict[str, "Prompt"]:
        # nimmt die sortierten Resultate, gibt die ersten und letzten k an das completion modell
        base_prompt_good = "<start_of_turn>user\nHere are some really good prompts:"
        base_prompt_bad = "<start_of_turn>user\nHere are some prompts that don't work well:"
        count_base_prompt = "<start_of_turn>user\nHere is an example of a prompt I would give an expert to generate a sentiment mapping from a sentence to emojis: I would like to use a subset of the following emojis to summarize my sentiment: 🥰, 😘, 🤗, 😎, 👍, 🧐, ✍, 👀, 🤐, 😶, 🙄, 😪, 😢, 😡, 💩\nThis is a sentence without emojis where I express my sentiment<INPUT>. I want you to produce 15 similar prompts, that also use these emojis to summarize the emotions found in the text. Make the prompt very long and precise. You can include examples, such as : for this negative word, use this emoji. Make the prompts, such that each prompt works well for certain types of reviews. Not all prompts have to classify all reviews well. Only output the new prompts in the following format: Prompt: <Prompt>. At the end of the prompt there should be this: \n\n<INPUT>\n<end_of_turn>\n<start_of_turn>model\n"

        prompts = pd.read_csv('../../data/current_prompt_catalogue.csv')
        
        # Weil das im csv zu Problemen führt, ersetzen wir beim Lesen und schreiben \n durch \Line
        top_k = prompts["prompt"].str.replace("\Line", "\n")[:k]
        flop_k = prompts["prompt"].str.replace("\Line", "\n")[-k:]

        for i, name in enumerate(top_k):
            template = name.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
            print(template)
            base_prompt_good += f"Prompt {i+1}:\n{template}\n\n"

        for i, name in enumerate(flop_k):
            template = name.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
            print(template)
            base_prompt_bad += f"Prompt {i+1}:\n{template}\n\n"

        
        base_prompt_good += f"Based on these, suggest exactly 1 new prompt templates in a similar style. The goal is to always generate the sentiment of a review as either postive, negative, or neutral. You will want to generate long prompts, that are very specific. Always include in your prompt that the output should be negative, positive or neutral, nothing more. Do not put brakets in your output. Your prompt should not work on all inputs, but very well on a certain type of inputs. So try to produce expert prompts for certain reviews. For example this sentence: 'I highly recommend any location but his.' should be classified as negative. This sentence:'They are just as good at 'soft skills' as translating.' should be classified as positive. Only output the new prompts in the following format: Prompt: <Prompt>. At the end of the prompt there should be this: \n\n<INPUT>\n<end_of_turn>\n<start_of_turn>model\n"
        base_prompt_bad += f"Based on these, suggest exactly 1 improved prompt templates, that could work better. The goal is to always be to generate the sentiment of a review as either postive, negative, or neutral. You will want to generate long prompts, that are very specific. Always include in your prompt that the output should be negative, positive or neutral, nothing more. Do not put brakets in your output. Your prompt should not work on all inputs, but very well on a certain type of inputs. So try to produce expert prompts for certain reviews. For example this sentence: 'I highly recommend any location but his.' should be classified as negative. This sentence: They are just as good at 'soft skills' as translating.' should be classified as positive. Only output the new prompts in the following format: Prompt: <Prompt>.  Followed by this literal string: '\n\n<INPUT>\n\n'<end_of_turn>\n<start_of_turn>model\n"
        count_base_prompt += f"Based on these, suggest exactly 1 improved prompt templates, that could work better."

        completions = {}

        for label, prompt_text in [("good", base_prompt_good), ("bad", base_prompt_bad), ("count", count_base_prompt)]:
            for i in range(k): 
                res = self.evaluator.llm.create_completion(
                    prompt=prompt_text, max_tokens=170, temperature=0.9, stop=[]
                )
                generated = res["choices"][0]["text"].strip()
                completions[f"{label}_generated_{i+1}"] = generated

        formatted_completions = self.format_prompts(completions)

        return formatted_completions

    def format_prompts(self, completions: Dict[str, "Prompt"]) -> Dict[str, "Prompt"]:
        generated_prompts = {}
        counter = 1
        for key, text in completions.items():
            prompts = re.findall(r"Prompt:\s*(.*?)(?=Prompt:|$)", text, flags=re.DOTALL)
            prompts = [p.strip() for p in prompts if p.strip()]

            for i, p in enumerate(prompts):
                generated_prompts[f"{key}_{i+1}"] = (
                    f"<start_of_turn>user\Line{p}<end_of_turn>\Line<start_of_turn>model\Line<PROBE>"
                )

                with open("../../data/prompt_catalogue.csv", "a") as pc:
                    f = csv.writer(pc)
                    unique_id = int(time.time() * 100)
                    f.writerow(
                        [
                            f"{key}_{unique_id}",
                            generated_prompts[f"{key}_{i+1}"].replace("\n", "\Line"),
                            "",
                        ]
                    )
                # Write to current_prompt_catalogue.csv as well
                with open("../../data/current_prompt_catalogue.csv", "a") as cpc:
                    f_current = csv.writer(cpc)
                    f_current.writerow(
                        [
                            f"{key}_{unique_id}",
                            generated_prompts[f"{key}_{i+1}"].replace("\n", "\Line"),
                            "",
                        ]
                    )

        return generated_prompts

    def run_optimization_loop(self, sample_size=None, n_iterations = 10, new_prompts_per_class=8)-> Dict[str, Tuple[List[str], float]]:

        # Export base prompts first
        Prompt.export_base_prompts_to_csv()

        df = pd.read_csv("../../data/training.csv")
        X = df["sentence"].tolist()
        y_true = [label.upper() for label in df["label"]]
        if sample_size:
            X = X[:sample_size]
            y_true = y_true[:sample_size]

        # Initialize catalogues with base prompts
        base_prompts = pd.read_csv("../../data/base_prompts.csv")
        base_prompts.to_csv("../../data/prompt_catalogue.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\n")
        base_prompts.to_csv("../../data/current_prompt_catalogue.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\n")
        print("Initialized catalogues with base prompts")

        # Optimization loop
        results = optimizer.evaluate_prompts(X, y_true)
        for iteration in range(n_iterations):
            print(f"\n=== Starting Optimization Iteration {iteration + 1}/{n_iterations} ===")

            # Generate new prompts based on best ones
            completions = optimizer.optimize_prompts(results, new_prompts_per_class)  # Will add 3k entries k per class
            print(f"\nGenerated new prompts for iteration {iteration + 1}")
            
            # Evaluate current prompts and get best ones
            results = optimizer.evaluate_prompts(X, y_true)
            print(f"\nCompleted evaluation for iteration {iteration + 1}")
            
            print(f"\n=== Completed Optimization Iteration {iteration + 1}/{n_iterations} ===")

        print("\nOptimization complete!")
        print("Final best prompts are in current_prompt_catalogue.csv")
        print("All generated prompts are in prompt_catalogue.csv")
        return results


if __name__ == "__main__":

    config = PromptEvaluatorConfig.for_gemma_3_4b_it(verbose=True, debug=True)
    evaluator = PromptEvaluator(config)
    optimizer = PromptOptimizer(evaluator)

    optimizer.run_optimization_loop(sample_size=20, n_iterations=3, new_prompts_per_class=8) #Sample size: the number of sampes forom X that we evaluate against for final run let out or set None #N_iterations: optimization iterations #new_prompts_per_class in total we get 3*this new entries, for bad,good,count so 8 is reccomended to get 24