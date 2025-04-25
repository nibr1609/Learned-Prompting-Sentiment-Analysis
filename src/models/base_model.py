import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
from config.config import ModelConfig
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

class BaseSentimentModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")
    
    def preprocess_dataset(self):
        raise NotImplementedError("Subclasses must implement forward method")
    
    def evaluate(self):
        raise NotImplementedError("Subclasses must implement forward method")
    
    def evaluate(self, pred, labels):
        cm = confusion_matrix(labels, pred)
        
        # Avoid division by zero
        FP = cm.sum(axis=0) - np.diag(cm)
        TN = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)

        # False positive rate: FP / (FP + TN)
        fp_rate_per_class = FP / (FP + TN + 1e-10)  # add small value to avoid div by 0
        macro_fp_rate = fp_rate_per_class.mean()

        metrics = {
            "accuracy": accuracy_score(labels, pred),
            "precision_macro": precision_score(labels, pred, average='macro', zero_division=0),
            "recall_macro": recall_score(labels, pred, average='macro', zero_division=0),
            "f1_macro": f1_score(labels, pred, average='macro', zero_division=0),
            "precision_weighted": precision_score(labels, pred, average='weighted', zero_division=0),
            "recall_weighted": recall_score(labels, pred, average='weighted', zero_division=0),
            "f1_weighted": f1_score(labels, pred, average='weighted', zero_division=0),
            "false_positive_rate_macro": macro_fp_rate,
            "false_positive_rate_per_class": fp_rate_per_class.tolist(),
            "confusion_matrix": cm.tolist()
        }

        return metrics