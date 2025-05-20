import json
from pathlib import Path
from typing import Dict, Any
import argparse
from config.config import Config, ModelConfig, ExperimentConfig, DataConfig, PromptConfig
from models.huggingface_model import BERTHuggingFaceModel
from models.two_stage_model import TwoStageModel
from experiments.experiment_runner import ExperimentRunner
from utils.submission_creation import create_submission

def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_config_from_json(config_dict: Dict[str, Any]) -> Config:
    """Create Config object from JSON configuration."""
    # Convert string paths to Path objects
    for key in ['train_path', 'test_path', 'submission_dir', 'experiment_output_dir', 'save_dir', 'model_output_dir']:
        if key in config_dict['data']:
            config_dict['data'][key] = Path(config_dict['data'][key])
        elif key in config_dict['experiment']:
            config_dict['experiment'][key] = Path(config_dict['experiment'][key])
    
    # Create individual configs
    model_config = ModelConfig(**config_dict['model'])
    experiment_config = ExperimentConfig(**config_dict['experiment'])
    data_config = DataConfig(**config_dict['data'])
    prompt_config = PromptConfig(**config_dict['prompt'])
    
    # Combine into single config
    return Config(
        model=model_config,
        experiment=experiment_config,
        data=data_config,
        prompt=prompt_config
    )

def get_model_class(model_type: str):
    """Get the appropriate model class based on the model name."""
    # TODO: Load class directly
    model_classes = {
        "HuggingFaceModel": BERTHuggingFaceModel,
        "TwoStageModel": TwoStageModel
    }
    return model_classes.get(model_type)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run sentiment analysis experiment')
    parser.add_argument('-c', '--config', 
                      type=str, 
                      required=True,
                      help='Path to the experiment configuration JSON file')
    
    # Parse arguments
    args = parser.parse_args()
    config_path = Path("../experiments/" + args.config + ".json").resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load and parse experiment configuration
    config_dict = load_experiment_config(config_path)
    config = create_config_from_json(config_dict)
    
    # Get appropriate model class
    model_class = get_model_class(config.model.model_type)
    if model_class is None:
        raise ValueError(f"Unknown model: {config.model.model_name}")
    
    # Initialize model
    model = model_class(config)
    
    # Initialize experiment runner
    runner = ExperimentRunner(config, config_path, model)
    
    assert config.experiment.mode in ["train_inference", "inference"], "mode in config must be set to train_inference or inference"
    runner.run()

if __name__ == "__main__":
    main() 