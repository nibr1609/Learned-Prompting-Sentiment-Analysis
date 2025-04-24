import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config.config import Config
from metrics.metrics_tracker import MetricsTracker
from models.base_model import BaseSentimentModel
from data.SentimentAnalysisDataset import get_sentiment_dataset, create_collate_fn

class ExperimentRunner:
    def __init__(self, config: Config, model: BaseSentimentModel):
        self.config = config
        self.model = model
        self.device = torch.device(config.experiment.device)
        self.model.to(self.device)
        
        # Create experiment-specific directory
        self.experiment_dir = config.data.experiment_output_dir / f"experiment_{config.experiment.experiment_id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment directory: {self.experiment_dir}")
        
        # Initialize metrics tracker with experiment directory
        self.metrics_tracker = MetricsTracker(config.experiment.experiment_name, self.experiment_dir)
        
        # Load datasets
        self.train_dataset = get_sentiment_dataset("train", self.model.tokenizer, config.data.train_path)
        self.test_dataset = get_sentiment_dataset("test", self.model.tokenizer, config.data.test_path)
        
        # Create subset of test dataset if max_test_samples is set
        if config.experiment.max_test_samples is not None:
            from torch.utils.data import Subset
            self.test_dataset = Subset(
                self.test_dataset,
                range(min(config.experiment.max_test_samples, len(self.test_dataset)))
            )
            print(f"Using first {len(self.test_dataset)} samples for testing")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.model.batch_size,
            shuffle=True,
            num_workers=config.experiment.num_workers,
            collate_fn=create_collate_fn(self.model.tokenizer)
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.model.batch_size,
            shuffle=False,
            num_workers=config.experiment.num_workers,
            collate_fn=create_collate_fn(self.model.tokenizer)
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.model.learning_rate,
            weight_decay=config.model.weight_decay
        )
        
        total_steps = len(self.train_loader) * config.model.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    
    def evaluate_pretrained(self) -> np.ndarray:
        """Evaluate a pretrained model and save metrics.
        
        Returns:
            np.ndarray: Array of predictions
        """
        print("Evaluating pretrained model...")
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # For BERTweet, batch is already a dictionary of tensors
                inputs = batch
                
                # Move all tensors to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=-1).cpu().numpy()
                all_predictions.extend(predictions)
        
        return np.array(all_predictions)  # Return predictions for submission
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            inputs, labels = batch
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        self.metrics_tracker.update_train_metrics(
            avg_loss,
            np.array(all_predictions),
            np.array(all_labels)
        )
        
        return avg_loss
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                # For BERTweet, batch is already a dictionary of tensors
                inputs = batch
                labels = inputs.pop('labels') if 'labels' in inputs else None
                
                # Move all tensors to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if labels is not None:
                    labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                if labels is not None:
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=-1).cpu().numpy()
                    all_predictions.extend(predictions)
                    all_labels.extend(labels.cpu().numpy())
                else:
                    predictions = torch.argmax(outputs, dim=-1).cpu().numpy()
                    all_predictions.extend(predictions)
        
        if all_labels:  # Only calculate metrics if we have labels
            avg_loss = total_loss / len(loader)
            self.metrics_tracker.update_val_metrics(
                avg_loss,
                np.array(all_predictions),
                np.array(all_labels)
            )
            return self.metrics_tracker.get_best_metrics()
        return {}  # Return empty dict if no labels (test set)
    
    def run(self):
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.config.model.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.model.num_epochs}")
            
            # Training
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")
            
            # Evaluation
            val_metrics = self.evaluate(self.test_loader)
            print(f"Validation Metrics: {val_metrics}")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0
                if self.config.experiment.save_best_model:
                    model_path = self.experiment_dir / f"model_best_{self.config.experiment.experiment_id}.pt"
                    print(f"Saving best model to {model_path}")
                    self.model.save(model_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.experiment.early_stopping_patience:
                print("Early stopping triggered")
                break
        
        # Save final metrics
        metrics_path = self.experiment_dir / f"metrics_{self.config.experiment.experiment_id}.json"
        print(f"Saving metrics to {metrics_path}")
        self.metrics_tracker.save_metrics()
        
        # Save final model
        if self.config.experiment.save_best_model:
            model_path = self.experiment_dir / f"model_final_{self.config.experiment.experiment_id}.pt"
            print(f"Saving final model to {model_path}")
            self.model.save(model_path)
    
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_predictions = []
        
        total_samples = len(loader.dataset)
        processed_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting", total=len(loader)):
                inputs = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=-1).cpu().numpy()
                all_predictions.extend(predictions)
                
                # Update progress
                processed_samples += len(predictions)
                print(f"Processed {processed_samples}/{total_samples} samples ({processed_samples/total_samples*100:.1f}%)")
        
        return np.array(all_predictions) 