import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
from llm_building_blocks.prompt_evaluator import PromptEvaluator, PromptEvaluatorConfig, Prompt, Sentiment
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, List
from config.config import Config, get_default_configs
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy  as np
import datetime
from llm_building_blocks.prompt_catalogue import Prompt, PromptCatalogue
from utils.metrics import save_validation_metrics, evaluate

def get_labels(data, prompt_catalogue: PromptCatalogue, evaluator, config):
    """Compute soft prompt selection probabilities based on evaluator predictions.

    Args:
        data (pd.DataFrame): DataFrame containing 'sentence' and 'label' columns.
        prompt_catalogue (PromptCatalogue): Catalogue of prompts to evaluate.
        evaluator (PromptEvaluator): Evaluator used to get probabilities for each prompt.
        config (Config): Configuration object for experiment settings.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, n_prompts) with soft selection probabilities.
    """
    sentences = list(data["sentence"])

    # Make predictions for all prompts
    predictions_list = []
    print(prompt_catalogue)
    for name, prompt in prompt_catalogue.get_prompts():
        prediction = np.array(evaluator.predict_proba(prompt, sentences))
        predictions_list.append(prediction)

    preds = torch.Tensor(np.stack(predictions_list, axis=0))

    true_sentiments_text = list(data["label"])

    def convert_sentiments_to_index(label):
        mapping = {"positive": Sentiment.POSITIVE.value, "neutral": Sentiment.NEUTRAL.value, "negative": Sentiment.NEGATIVE.value}
        return mapping[label]

    true_sentiments = torch.Tensor(list(map(convert_sentiments_to_index,true_sentiments_text))).long()

    # Compute Loss as described in methods section of Paper
    losses = []
    for i in range(preds.shape[0]):
        logits_i = preds[i]
        loss_i = - F.cross_entropy(logits_i, true_sentiments, reduction="none")
        losses.append(loss_i)

    loss_matrix = torch.stack(losses, dim=0)
    softmax_output = F.softmax(loss_matrix, dim=0).T

    def convert_index_to_sentiment(label):
        mapping = {Sentiment.POSITIVE.value: "positive", Sentiment.NEUTRAL.value: "neutral",  Sentiment.NEGATIVE.value: "negative"}
        return mapping[label]

    # save metrics on prompt level
    for i in range(preds.shape[0]):
        logits_i = preds[i]
        metrics = evaluate([convert_index_to_sentiment(label.item()) for label in logits_i.argmax(dim=1)], true_sentiments_text)
        save_validation_metrics(config, metrics, suffix="_selection_individual_" + str(i))

    return softmax_output

class SentenceToPromptDataset(Dataset):
    """A PyTorch dataset that maps sentences to soft prompt labels."""
    def __init__(self, config: Config, prompt_catalogue: PromptCatalogue, evaluator, data):
        self.config = config
        self.data = data
        self.labels = get_labels(self.data, prompt_catalogue, evaluator, self.config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data.iloc[idx]["sentence"], self.labels[idx]

class SentenceToPromptModule(pl.LightningDataModule):
    """A Lightning data module for loading sentence-to-prompt datasets."""
    def __init__(
        self,
        config:Config,
        prompt_catalogue: PromptCatalogue,
        evaluator,
        train,
        val,
        *args,
        **kwargs,
    ):
        """Initialize the data module.

        Args:
            config (Config): Experiment configuration.
            prompt_catalogue (PromptCatalogue): Catalogue of prompts.
            evaluator (PromptEvaluator): Evaluator for prompt probabilities.
            train (pd.DataFrame): Training data.
            val (pd.DataFrame): Validation data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super(SentenceToPromptModule, self).__init__(*args, **kwargs)

        self.prompt_catalogue = prompt_catalogue
        self.config = config
        self.batch_size = self.config.model.batch_size
        self.num_workers = self.config.experiment.num_workers
        self.evaluator = evaluator
        self.train = train
        self.val = val

    def setup(self, stage=None):
        """Prepare datasets for training or validation.

        Args:
            stage (str, optional): Stage in training ('fit', etc.).
        """
        if stage in ("fit", None):
            self.train_dataset = SentenceToPromptDataset(config=self.config, prompt_catalogue=self.prompt_catalogue, evaluator=self.evaluator, data=self.train)
            self.val_dataset = SentenceToPromptDataset(config=self.config, prompt_catalogue=self.prompt_catalogue, evaluator=self.evaluator, data=self.val)

    def train_dataloader(self):
        # Return a DataLoader for the training dataset.
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        # Return a DataLoader for the validation dataset.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

class PromptSelector(pl.LightningModule):
    """A Lightning module to train a model that selects prompts based on input sentences."""

    def __init__(self, config: Config, prompt_catalogue: PromptCatalogue, *args, **kwargs):
        super(PromptSelector, self).__init__(*args, **kwargs)

        self.output_layers = len(prompt_catalogue.get_prompts())

        print("number_hidden_layers")
        print(self.output_layers)

        self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilroberta-base",
            num_labels=self.output_layers,
        )
        self.config = config

        self.save_hyperparameters()

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding="max_length", truncation=True).to(self.config.experiment.device)
        return self.model(**inputs).logits

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y_true)


        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y_true)

        y_pred = torch.argmax(logits, dim=1)

        unique_preds, counts = torch.unique(y_pred, return_counts=True)
        print(f"Batch {batch_idx} prediction distribution:", dict(zip(unique_preds.tolist(), counts.tolist())))

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.model.learning_rate)

        return {
            "optimizer": optimizer,
        }



