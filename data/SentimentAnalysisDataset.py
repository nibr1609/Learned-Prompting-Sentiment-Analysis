import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
         
MODULE_DIR = Path(__file__).parent

class SentimentAnalysisDataset(Dataset):
    def __init__(self, df, tokenizer, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        inputs = self.tokenizer(row["sentence"], truncation=True, return_tensors="pt")

        inputs = {key: val.squeeze(0) for key, val in inputs.items()}

        if self.is_test:
            return inputs

        label = torch.tensor(row["label"], dtype=torch.long)
        return inputs, label
    

def one_hot_encode_labels(df: DataFrame):
    assert "label" in df.columns, "column 'label' must be in dataframe columns"
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["label"].map(label_mapping)

    return df

def get_sentiment_dataset(type_dataset: bool, tokenizer) -> SentimentAnalysisDataset:
    assert type_dataset == "train" or type_dataset == "test", "type_dataset must be set to 'train' or 'test'"
    
    if type_dataset == "train":
        dataset = pd.read_csv(MODULE_DIR / "training.csv")
        dataset = one_hot_encode_labels(dataset)
        return SentimentAnalysisDataset(dataset, tokenizer)
    else:
        dataset = pd.read_csv(MODULE_DIR / "test.csv")
        return SentimentAnalysisDataset(dataset, tokenizer, is_test=True)
    
def create_collate_fn(tokenizer):
    def collate_fn(batch):
        # Separate inputs and labels (if not test set)
        if isinstance(batch[0], tuple):
            inputs = [item[0] for item in batch]
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        else:
            inputs = [item for item in batch]
            labels = None

        # Convert list of dictionaries to dictionary of lists
        batch_inputs = {key: [d[key] for d in inputs] for key in inputs[0]}

        # Pad only (no truncation here)
        encodings = tokenizer.pad(batch_inputs, padding=True, return_tensors="pt")

        return (encodings, labels) if labels is not None else encodings   

    return collate_fn