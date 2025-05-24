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
from config.config import get_default_configs, Config
from utils.metrics import evaluate, save_validation_metrics
from models.prompt_catalogue import PromptCatalogue, Prompt, Sentiment

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
    verbose: bool = False
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

    def predict(self, prompt: Prompt, X: ArrayLike):
        return self.predict_proba(prompt, X).idxmax(axis=1)