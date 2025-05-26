from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import time
import torch

@dataclass
class ModelConfig:
    model_name: str
    model_type: str
    pretrained: bool = False
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 3
    max_length: int = 128
    dropout: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    label_mapping: Dict[int, str] = field(default_factory=lambda: {
        0: "negative",
        1: "neutral",
        2: "positive"
    })

@dataclass
class ExperimentConfig:
    experiment_name: str
    mode: str
    experiment_id: str = field(default_factory=lambda: str(int(time.time())))
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 4
    save_best_model: bool = True
    early_stopping_patience: int = 3
    log_interval: int = 100
    max_test_samples: Optional[int] = None
    max_train_samples: Optional[int] = None
    max_train_samples_prompt_optimizer: Optional[int] = None
    validation_set_split: Optional[float] = 0.2
    save_per_epoch: int = 4
    two_stage_setting: Optional[str] = ""
    selector_eval_setting: Optional[str] = ""

@dataclass
class DataConfig:
    train_path: Path
    test_path: Path
    submission_dir: Path
    experiment_output_dir: Path
    model_output_dir: Path
    llm_path: Path
    random_state: int = 42



@dataclass
class PromptConfig:
    prompt_list: List[str] = None
    k_best_worst: Optional[int] = 3
    optimizer_iterations: int = 3

@dataclass
class Config:
    model: ModelConfig
    experiment: ExperimentConfig
    data: DataConfig
    prompt: PromptConfig

def get_default_configs() -> Config:
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data"
    
    model_config = ModelConfig(
        model_name="default_model",
        pretrained=True,
        model_type="HuggingFaceModel"
    )

    experiment_config = ExperimentConfig(
        experiment_name="default_experiment",
        mode="inference"
    )

    data_config = DataConfig(
        train_path=data_path / "training.csv",
        test_path=data_path / "test.csv",
        submission_dir=base_path / "submissions",
        experiment_output_dir=base_path / "results/experiments",
        model_output_dir="/work/scratch/nbritz/models",
        llm_path=base_path / "llms/gemma-3-4b-it-q4_0.gguf",
    )

    prompt_config = PromptConfig(
        [
            "You are a highly skilled sentiment analysis expert specializing in analyzing online customer reviews for e-commerce businesses. Categorize it as definitively either positive, negative, or neutral. ",
            "You are a highly experienced sentiment analysis expert specializing in analyzing online reviews of restaurants. Provide a concise sentiment classification – “positive”, “negative”, or “neutral”. Do not include any explanations or justifications; simply state the sentiment. ",
            "Sentiment classification task. There can be some sarcasam pay attention to this. Choose one of the following: positive, negative, or neutral.",
            "You are a professional brand monitor tasked with assessing customer feedback. Your role is to categorize the sentiment of each review as either positive, negative, or neutral. Your response MUST be limited to a single word: positive, negative, or neutral. Prioritize accuracy above all else.",
            "Sentiment classification task. Dont let yourself be influenced by single words too much. Analyze the sentence as a whole. Choose carefully one of the following: positive, negative, or neutral.",
            "You are a highly experienced sentiment analysis expert specializing in analyzing whether people find prices appropriate. Provide a concise sentiment classification – “positive”, “negative”, or “neutral”. Do not include any explanations or justifications; simply state the sentiment. ",
            "Do a sentiment classification task. Specialize on people's emotion such as anger or joy. Provide a concise sentiment classification – “positive”, “negative”, or “neutral”. Do not include any explanations or justifications; simply state the sentiment. ",
            "You are a highly skilled sentiment analysis expert. You will receive reviews about movies. Answer with one word: 'positive', 'negative', or 'neutral'. ",
            "You are a highly skilled sentiment analysis expert. Focus on double negations in sentences. Answer with one word: 'positive', 'negative', or 'neutral'. ",
            "You are a highly skilled sentiment analysis expert. Your task is to read sentences and determine the sentiment expressed. The sentiment should be classified as either 'positive', 'negative', or 'neutral'. Provide only the single word sentiment classification."
        ],
        k_best_worst=3
    )

    return Config(
        model=model_config,
        experiment=experiment_config,
        data=data_config,
        prompt=prompt_config
    ) 