import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
from config.config import ModelConfig

class BaseSentimentModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_labels = 3  # positive, neutral, negative
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")
    
    def save(self, path: Path):
        """Save model weights and configuration"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: Path) -> 'BaseSentimentModel':
        """Load model from checkpoint"""
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions for a batch of inputs"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = torch.argmax(outputs, dim=-1)
        return predictions 