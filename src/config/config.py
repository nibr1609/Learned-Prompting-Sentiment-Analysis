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
    #prompt_list: List[str] = []

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
    validation_set_split: Optional[float] = 0.2
    save_per_epoch: int = 4

@dataclass
class DataConfig:
    train_path: Path
    test_path: Path
    submission_dir: Path
    experiment_output_dir: Path
    model_output_dir: Path
    val_split: float = 0.1
    random_state: int = 42

@dataclass
class Config:
    model: ModelConfig
    experiment: ExperimentConfig
    data: DataConfig

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
        experiment_output_dir=base_path / "experiments",
        model_output_dir="/work/scratch/nbritz/models"
    )

    return Config(
        model=model_config,
        experiment=experiment_config,
        data=data_config
    ) 