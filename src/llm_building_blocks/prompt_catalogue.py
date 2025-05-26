from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

class Sentiment(IntEnum):
    POSITIVE = 0
    NEUTRAL = 1
    NEGATIVE = 2


@dataclass
class Prompt:
    # A prompt contains a string, a sentiment map that maps from string to Sentiment (as defined above)
    # Furthermore we store the current predictions of our train / test set in order to have it close to the object, same for score
    template: str
    sentiment_map: Dict[str, Sentiment]
    current_predictions: List[str]
    current_score: float

    # Standard Prompt that can be instantiated if needed
    @staticmethod
    def direct_example():
        return Prompt(
            template="<start_of_turn>user\nYou are an expert in recognizing people's sentiments. Reply only with one word: positive, negative, or neutral.\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
            sentiment_map={
                "positive": Sentiment.POSITIVE,
                "negative": Sentiment.NEGATIVE,
                "neutral": Sentiment.NEUTRAL,
            },
        )
    
    # "Random" Prompt that can be instantiated if needed
    @staticmethod
    def random_example():
        """
        Directly asks the model to classify the sentiment of a text.
        Uses the <PROBE> evaluator to convert logits to probabilities.
        """
        return Prompt(
            template="<start_of_turn>user\nRandomly reply only with one word: positive, negative, or neutral.  \n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
            sentiment_map={
                "positive": Sentiment.POSITIVE,
                "negative": Sentiment.NEGATIVE,
                "neutral": Sentiment.NEUTRAL,
            },
        )


class PromptCatalogue:
    def __init__(self, prompt_list_str):
        # We can create a Prompt Catalogue from strings
        self.prompt_list_str = prompt_list_str
        self.id_counter = 0
        self.prompts = self._create_prompts_from_str_list(self.prompt_list_str)

    def _create_prompts_from_str_list(self, list):
        # Converts a list of prompt strings into a dictionary of Prompt objects.
        # Each prompt is assigned a unique key using the id_counter.
        prompts = {}
        for i, prompt_str in enumerate(list):
            prompt = Prompt(
                template=f"<start_of_turn>user\n{prompt_str}\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
                sentiment_map={
                    "positive": Sentiment.POSITIVE,
                    "negative": Sentiment.NEGATIVE,
                    "neutral": Sentiment.NEUTRAL,
                },
                current_predictions=None,
                current_score=None
            )
            prompts[f"prompt_{self.id_counter}"] = prompt
            self.id_counter += 1
        return prompts

    def save(self, iteration, config):
        # Saves the current prompts and their scores to a CSV file for the given iteration.
        # The file is stored in the experiment's "prompts" directory.
        prompts_strs = []
        scores = []

        for i, (prompt_id, prompt) in enumerate(self.get_prompts()):
            prompts_strs.append(self.get_prompt_list_str()[i])
            scores.append(prompt.current_score)

        directory = config.data.experiment_output_dir / f"experiment_{config.experiment.experiment_id}"
        prompt_dir = directory / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_df = pd.DataFrame({"prompt": prompts_strs, "score": scores})
        csv_path = prompt_dir / f"prompts_iteration_{iteration}.csv"
        prompt_df.to_csv(csv_path)

    def get_prompt_at_pos(self, pos):
        return list(self.prompts.values())[pos]

    def get_prompts(self):
        return list(self.prompts.items())
    
    def get_prompt_list_str(self):
        return self.prompt_list_str

    def append_prompts(self, prompts_str):
        # Appends new prompts from a list of strings to the catalogue.
        # Each new prompt is assigned a unique key and added to both the prompts dictionary and the prompt string list.
        for prompt in prompts_str:
            self.prompts[f"prompt_{self.id_counter}"] = self._generate_fresh_prompt(prompt)
            self.id_counter += 1
            self.prompt_list_str.append(prompt)

    def filter_prompts(self, ids):
        # Removes prompts whose IDs are not in the provided list of IDs.
        # Also updates the prompt string list to reflect the removed prompts.
        print(f"Removing ids {ids}")
        remove_indices = []
        for i, (prompt_id, prompt) in enumerate(self.get_prompts()):
            if prompt_id not in ids:
                self.prompts.pop(prompt_id)
                remove_indices.append(i)

        self.prompt_list_str = [item for i, item in enumerate(self.prompt_list_str) if i not in remove_indices]

            

    def _generate_fresh_prompt(self, prompt_str):
        # Generates a fresh prompt from a string
        prompt = Prompt(
            template=f"<start_of_turn>user\n{prompt_str}\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<PROBE>",
            sentiment_map={
                "positive": Sentiment.POSITIVE,
                "negative": Sentiment.NEGATIVE,
                "neutral": Sentiment.NEUTRAL,
            },
            current_predictions=None,
            current_score=None
        )
        return prompt
    
    def reorder_catalogue_by_score(self):
        # Reorder the catalogue by scores (descending), as well as the string list of prompts.
        zipped = list(zip(self.prompts.items(), self.prompt_list_str))
        sorted_zipped = sorted(zipped, key=lambda x: x[0][1].current_score or 0, reverse=True)
        sorted_items = [item for item, _ in sorted_zipped]
        self.prompt_list_str = [prompt_str for _, prompt_str in sorted_zipped]
        self.prompts = dict(sorted_items)

    def __str__(self):
        # For better printing
        output = []
        for i, (prompt_id, prompt) in enumerate(self.prompts.items()):
            output.append(f"{i+1}. Prompt: {self.prompt_list_str[i]}")
            output.append(f"   ID: {prompt_id}")
            output.append(f"   Template:\n{prompt.template}")
            output.append(f"   Current Prediction: {prompt.current_predictions}")
            output.append(f"   Current Score: {prompt.current_score}")
            output.append("")  # Blank line for separation
        return "\n".join(output)