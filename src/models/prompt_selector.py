import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
from models.prompt_evaluator import PromptEvaluator, PromptEvaluatorConfig, Prompt, Sentiment
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, List
from config.config import Config, get_default_configs
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy  as np
import datetime
from models.prompt_catalogue import Prompt, PromptCatalogue

def get_labels(data, prompt_catalogue: PromptCatalogue, evaluator):
    # Dummy
    cached = False
    if (cached):
        pass

    sentences = list(data["sentence"])

    predictions_list = []
    print(prompt_catalogue)
    for name, prompt in prompt_catalogue.get_prompts():
        prediction = np.array(evaluator.predict_proba(prompt, sentences))
        predictions_list.append(prediction)

    preds = torch.Tensor(np.stack(predictions_list, axis=0))

    true_sentiments = list(data["label"])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    np.save(f"preds_{timestamp}.npy", np.stack(predictions_list, axis=0))
    np.save(f"true_{timestamp}.npy", np.array(list(data["label"])))

    def convert_sentiments_to_index(label):
        mapping = {"positive": Sentiment.POSITIVE.value, "neutral": Sentiment.NEUTRAL.value, "negative": Sentiment.NEGATIVE.value}
        return mapping[label]

    true_sentiments = torch.Tensor(list(map(convert_sentiments_to_index,true_sentiments))).long()

    losses = []
    for i in range(preds.shape[0]):             # for each model i=0..k-1
        logits_i = preds[i]         # shape (n, m)
        loss_i = - F.cross_entropy(logits_i, true_sentiments, reduction="none")  # (n,)
        losses.append(loss_i)

    loss_matrix = torch.stack(losses, dim=0)    # shape (k, n)
    softmax_output = F.softmax(loss_matrix, dim=0).T

    np.save(f"labels_{timestamp}.npy", softmax_output)

    return softmax_output

class SentenceToPromptDataset(Dataset):
    def __init__(self, config: Config, prompt_catalogue: PromptCatalogue, evaluator, data):
        self.config = config
        self.data = data
        self.labels = get_labels(self.data, prompt_catalogue, evaluator)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data.iloc[idx]["sentence"], self.labels[idx]

class SentenceToPromptModule(pl.LightningDataModule):
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
        super(SentenceToPromptModule, self).__init__(*args, **kwargs)

        self.prompt_catalogue = prompt_catalogue
        self.config = config
        self.batch_size = self.config.model.batch_size
        self.num_workers = self.config.experiment.num_workers
        self.evaluator = evaluator
        self.train = train
        self.val = val

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = SentenceToPromptDataset(config=self.config, prompt_catalogue=self.prompt_catalogue, evaluator=self.evaluator, data=self.train)
            self.val_dataset = SentenceToPromptDataset(config=self.config, prompt_catalogue=self.prompt_catalogue, evaluator=self.evaluator, data=self.val)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

class PromptSelector(pl.LightningModule):
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

        # Calculate accuracy
        #y_pred = torch.argmax(logits, dim=-1)
        #acc = (y_pred == y_true).float().mean()

        # Log metrics
        #self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        #self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y_true)

        y_pred = torch.argmax(logits, dim=1)
        # acc = (y_pred == y_true).float().mean()
        # precisions, recalls, f1scores, supports = precision_recall_fscore_support(
        #     y_true.cpu(), y_pred.cpu(), zero_division=0
        # )

        unique_preds, counts = torch.unique(y_pred, return_counts=True)
        print(f"Batch {batch_idx} prediction distribution:", dict(zip(unique_preds.tolist(), counts.tolist())))

        #self.log("val_loss", loss, on_epoch=True, on_step=False)
        # self.log("val_acc", acc, on_epoch=True, on_step=False)
        # self.log(
        #     "val_precision",
        #     precisions.mean(),
        #     on_epoch=True,
        #     on_step=False,
        #     prog_bar=False,
        # )
        # self.log("val_recall", recalls.mean(), on_epoch=True, on_step=False)
        # self.log("val_f1", f1scores.mean(), on_epoch=True, on_step=False, prog_bar=True)

        # for i, (p, r, f1, _) in enumerate(zip(precisions, recalls, f1scores, supports)):
        #     self.log(f"val_precision_{i}", p, on_epoch=True, on_step=False)
        #     self.log(f"val_recall_{i}", r, on_epoch=True, on_step=False)
        #     self.log(f"val_f1_{i}", f1, on_epoch=True, on_step=False)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.model.learning_rate)

        return {
            "optimizer": optimizer,
        }
    
    # def predict_sentences(self, sentences: List[str]):
    #     self.eval()
    #     print(self.model.device)
    #     with torch.no_grad():
    #         logits = self(sentences)
    #         probs = torch.softmax(logits, dim=-1)
    #         preds = torch.argmax(probs, dim=-1)
        
    #     return preds.cpu().numpy(), probs.cpu().numpy()


