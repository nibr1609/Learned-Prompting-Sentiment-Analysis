from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
import torch
from transformers import AutoTokenizer

class BaseTokenizer(ABC):
    @abstractmethod
    def __call__(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize text and return a dictionary of tensors."""
        pass

class SimpleTokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_token_id = 0  # PAD token is always 0
        self.unk_token_id = 1  # UNK token is always 1
        
        # Add special tokens
        self.word_to_idx[self.pad_token] = self.pad_token_id
        self.word_to_idx[self.unk_token] = self.unk_token_id
        self.idx_to_word[self.pad_token_id] = self.pad_token
        self.idx_to_word[self.unk_token_id] = self.unk_token
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        # Count word frequencies
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size-2 words (minus special tokens)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size-2]:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def __call__(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize text and return a dictionary of tensors."""
        max_length = kwargs.get('max_length', 128)
        truncation = kwargs.get('truncation', True)
        return_tensors = kwargs.get('return_tensors', 'pt')
        
        # Tokenize
        words = text.lower().split()
        if truncation:
            words = words[:max_length]
        words = words + [self.pad_token] * (max_length - len(words))
        
        # Convert to ids
        input_ids = [self.word_to_idx.get(word, self.unk_token_id) for word in words]
        
        # Create attention mask
        attention_mask = [1 if word != self.pad_token else 0 for word in words]
        
        # Convert to tensors
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)
            }
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

class TransformersTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def __call__(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Use the transformers tokenizer."""
        return self.tokenizer(text, **kwargs) 