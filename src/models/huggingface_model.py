import torch
import torch.nn as nn
from typing import Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base_model import BaseSentimentModel
from config.config import Config

class HuggingFaceModel(BaseSentimentModel):
    def __init__(self, config: Config):
        super().__init__(config.model)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model.model_name)
        
        # Freeze all parameters since we're only doing inference
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Get BERTweet outputs
        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        return outputs.logits 