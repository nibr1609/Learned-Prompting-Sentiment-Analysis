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
    
    def evaluate(self, pred, labels):
        raise NotImplementedError("Subclass must implement evaluate")