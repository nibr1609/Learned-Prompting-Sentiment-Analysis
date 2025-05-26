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

class BaseSentimentModel():
    # Base Sentiment Model, this includes all required methods but none are implemented
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def train(self, train, val=None):
        raise NotImplementedError("Subclasses must implement train method")

    def predict(self, test, val=None):
        raise NotImplementedError("Subclasses must implement predict method")
    
    def preprocess_dataset(self, config):
        raise NotImplementedError("Subclasses must implement preprocess method")
    
    def evaluate(self, pred, labels):
        raise NotImplementedError("Subclass must implement evaluate method")