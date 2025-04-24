import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from typing import Dict, Any
from .tokenizers import BaseTokenizer
         
def one_hot_encode_labels(df: DataFrame):
    assert "label" in df.columns, "column 'label' must be in dataframe columns"
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["label"].map(label_mapping)
    return df

class SentimentAnalysisDataset(Dataset):
    def __init__(self, split: str, tokenizer: BaseTokenizer, data_path: Path):
        """
        Initialize the dataset.
        
        Args:
            split: 'train' or 'test'
            tokenizer: Any tokenizer that follows the BaseTokenizer interface
            data_path: Path to the data file
        """
        self.split = split
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data_path)
        
        # Convert labels to integers for training data
        if split == 'train':
            self.data = one_hot_encode_labels(self.data)
        
        # If it's a simple tokenizer, fit it on the training data
        if hasattr(tokenizer, 'fit') and split == 'train':
            tokenizer.fit(self.data['sentence'].tolist())
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        
        # Tokenize the text
        inputs = self.tokenizer(row["sentence"], truncation=True, return_tensors="pt")
        
        # For training data, include labels
        if self.split == 'train':
            return {
                **inputs,
                'labels': torch.tensor(row["label"], dtype=torch.long)
            }
        
        return inputs
    

def get_sentiment_dataset(split: str, tokenizer, data_path: Path) -> SentimentAnalysisDataset:
    assert split == "train" or split == "test", "split must be set to 'train' or 'test'"
    return SentimentAnalysisDataset(split, tokenizer, data_path)

class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Separate inputs and labels (if not test set)
        if isinstance(batch[0], tuple):
            texts = [item[0]["input_ids"].squeeze(0) for item in batch]
            attention_masks = [item[0]["attention_mask"].squeeze(0) for item in batch]
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        else:
            texts = [item["input_ids"].squeeze(0) for item in batch]
            attention_masks = [item["attention_mask"].squeeze(0) for item in batch]
            labels = None

        # Pad sequences
        padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

        # Create the final input dictionary
        inputs = {
            "input_ids": padded_texts,
            "attention_mask": padded_masks
        }

        return (inputs, labels) if labels is not None else inputs

def create_collate_fn(tokenizer):
    return CollateFn(tokenizer) 