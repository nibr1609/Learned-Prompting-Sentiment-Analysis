import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config.config import Config
from models.base_model import BaseSentimentModel
from utils.submission_creation import create_submission

import json

class ExperimentRunner:
    def __init__(self, config: Config, config_path, model: BaseSentimentModel):
        self.config = config
        self.model = model
        
        # Create experiment-specific directory
        self.experiment_dir = config.data.experiment_output_dir / f"experiment_{config.experiment.experiment_id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment directory: {self.experiment_dir}")
        self.save_config(config_path)

        # Every model expects differently formatted datasets. That's why we require the model to implement this method
        self.train_data, self.val_data, self.test_data, self.val_exists = self.model.preprocess_dataset(config)

    # Makes predictions on test set and validation set if set
    def predict(self) -> np.ndarray:
        if self.val_exists:
            return self.model.predict(self.test_data, self.val_data)
        else:
            return self.model.predict(self.test_data, None)
    
    # Makes predictions on train set optionally evaluates on validation set in parallel
    def train(self) -> None:
        if self.val_exists:
            self.model.train(self.train_data, self.val_data)
        else:
            self.model.train(self.train_data)
        

    # Runs experiment, trains / finetunes and makes inference or just inference
    def run(self):
        if self.config.experiment.mode == "train_inference":
            self.train()
            test_pred, val_pred = self.predict()
            if val_pred is not None:
                metrics = self.model.evaluate(val_pred, self.val_data)
                self.save_validation_metrics(metrics)

        if self.config.experiment.mode == "inference":
            test_pred, val_pred = self.predict()
            if val_pred is not None:
                metrics = self.model.evaluate(val_pred, self.val_data)
                self.save_validation_metrics(metrics)

        # Creates a submission on test set
        create_submission(test_pred, self.config)

    
    # Saves calculated metrics on validation set
    def save_validation_metrics(self, metrics):
        validation_set_dir = self.experiment_dir / "val_set_metrics"
        validation_set_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = validation_set_dir / "metrics.json"

        # Write the metrics dictionary to the file
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    # Writes config to results to inform what config produced which results
    def save_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        new_config_path = self.experiment_dir / "config.json"

        with open(new_config_path, "w") as f:
            json.dump(config, f, indent=4)

