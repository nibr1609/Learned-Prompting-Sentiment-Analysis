import numpy as np
from dataclasses import dataclass
from itertools import islice
from os.path import dirname
from enum import StrEnum
from typing import Dict
from numpy.typing import ArrayLike
from llama_cpp import Llama as LLM
from re import split, RegexFlag


@dataclass
class PromptEvaluatorConfig:
    llm_gguf_path: str
    verbose: bool = True

    @staticmethod
    def for_gemma_3_4b_it():
        return PromptEvaluatorConfig(
            llm_gguf_path=f"{dirname(__file__)}/../../llms/gemma-3-4b-it-q4_0.gguf",
        )


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Prompt:
    template: str
    """
    A string template that guides the text generation and logit evaluation.
    
    All instances of
    - `<INPUT>` will be replaced with the classifiable text.
    - `<THINK>` and `<COUNT>` will be replaced with thinking until just before a stop token is reached.

    The template MUST specify 1+ evaluators after the first occurence of `<INPUT>` in this order:
    
    1. 0+ `<COUNT>` instances that will return the normalized distribution of sentiment-mapped phrase occurences in their generated text snippets.
    2. 0 or 1 `<PROBE>` instance that will return a constrained next-phrase probability distribution.

    Each evaluator has the same weight in the final probability distribution.
    """

    sentiment_map: Dict[str, Sentiment]
    """
    A mapping from detectable substrings to sentiments. 
    """

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

    @staticmethod
    def emoji_example():
        return Prompt(
            template="<start_of_turn>user\nI would like to use some of these emojis: 🥰, 😘, 🤗, 😎, 👍, 🧐, ✍, 👀, 🤐, 😶, 🙄, 😪, 😢, 😡, 💩. Please explain for each its meaning.<end_of_turn><start_of_turn>model\n<THINK><end_of_turn>\n<start_of_turn>user\nPlease choose a subset of the provided emojis to decorate my message:\n\n<INPUT><end_of_turn>\n<start_of_turn>model\n<COUNT>",
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
            n_gpu_layers=-1,
            n_ctx=8192,
            logits_all=True,
            verbose=True,
        )

    def evaluate(self, prompt: Prompt, X: ArrayLike):
        X = np.asarray(X, dtype=np.str_)

        plan_before_input, plan_with_input = map(
            # Isolate actions in their own steps.
            lambda x: split(
                r"(<(?:THINK|PROBE|COUNT)>)",
                x,
            ),
            # Split the prompt into two parts: before and after the first <INPUT>.
            split(
                r"(<INPUT>.*)",
                prompt.template,
                maxsplit=1,
                flags=RegexFlag.DOTALL,
            )[:2],
        )

        # Validate plan_with_input.
        try:
            # If <PROBE> is present, ignore everything after its first occurence.
            cutoff_idx = plan_with_input.index("<PROBE>") + 1
        except ValueError:
            try:
                # If <COUNT> is present, ignore everything after its last occurence.
                cutoff_idx = len(plan_with_input) - plan_with_input[::-1].index(
                    "<COUNT>"
                )
            except ValueError:
                raise ValueError(
                    "Prompt template must contain at least one <PROBE> or <COUNT> after <INPUT>.",
                )

        plan_with_input = plan_with_input[:cutoff_idx]

        if self.config.verbose:
            print(
                f"Evaluation plan:\n\tonce: {plan_before_input}\n\tfor x∈X: {plan_with_input}"
            )

        # Reset the LLM.
        self.llm.reset()
        self.llm.eval(self.llm.tokenize(bytes(), add_bos=True))

        for action in plan_before_input:
            match action:
                case "<THINK>":
                    self.llm.create_completion(x)

        def evaluate_input(x: str):
            pass

        self.llm.reset()

        return

        self.llm.reset()
        prompt_before_input_tokens = self.llm.tokenize(
            bytes(prompt_before_input, "utf-8"), add_bos=True, special=True
        )
        self.llm.eval(prompt_before_input_tokens)
        n_prompt_before_input_tokens = self.llm.n_tokens

        positive_tokens = self.llm.tokenize(bytes("Positive", "utf-8"), add_bos=False)
        negative_tokens = self.llm.tokenize(bytes("Negative", "utf-8"), add_bos=False)
        neutral_tokens = self.llm.tokenize(bytes("Neutral", "utf-8"), add_bos=False)

        def process(x: str):
            print("Evaluating x = ", x)

            self.llm.n_tokens = n_prompt_prefix_tokens

            self.llm.eval(
                self.llm.tokenize(
                    bytes(
                        f"""{x}<end_of_turn>
<start_of_turn>model
The sentiment of the text is: """,
                        "utf-8",
                    ),
                    add_bos=False,
                    special=True,
                )
            )

            n_tokens_before_answer = self.llm.n_tokens

            def get_probability(predicate_tokens):
                self.llm.n_tokens = n_tokens_before_answer

                self.llm.eval(predicate_tokens)

                logprobs = LLM.logits_to_logprobs(
                    self.llm.scores[
                        (self.llm.n_tokens - len(predicate_tokens) - 1) : (
                            self.llm.n_tokens - 1
                        )
                    ],
                )[:, predicate_tokens]

                # print(
                #     "Seen tokens = ",
                #     self.llm.detokenize(
                #         self.llm.input_ids[: self.llm.n_tokens], special=True
                #     ),
                # )

                return np.exp(logprobs.sum())

            probs = np.asarray(
                [
                    get_probability(predicate_tokens)
                    for predicate_tokens in [
                        positive_tokens,
                        negative_tokens,
                        neutral_tokens,
                    ]
                ]
            )

            probs = probs / probs.sum()

            print("probs = ", np.round(probs, 3))
            print("predicate = ", probs.argmax())

            return 0

        np.vectorize(process)(X)


if __name__ == "__main__":
    config = PromptEvaluatorConfig.for_gemma_3_4b_it()
    evaluator = PromptEvaluator(config)

    evaluator.evaluate(
        prompt=Prompt.direct_example(),
        X=[
            "FUCKING HATE THIS MOVIE! it sucks so bad... is what I would say if I was a loser. But actually, I love it!",
            "I love this movie! It's so good!",
            "This movie is okay, not great but not bad either.",
            "I don't like this movie at all. It's terrible!",
        ],
    )
