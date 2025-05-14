import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
from models.prompt_evaluator import PromptEvaluator, PromptEvaluatorConfig, Prompt
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, List
from config.config import Config, get_default_configs
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def get_labels(data, prompt_catalogue):
    # Dummy
    return torch.tensor([0, 1, 2, 3, 4, 5, 4, 3, 2, 1])

class SentenceToPromptDataset(Dataset):
    def __init__(self, prompt_catalogue: List["Prompt"], data_path):
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)

        # TODO limit dataset, do validation etc.

        self.data = self.data[:10]

        self.labels = get_labels(self.data, prompt_catalogue)

        # self.labels = {
        #     cls_name: cls_label
        #     for cls_label, cls_name in enumerate(
        #         ["giraffe", "hippo", "lion", "warthog", "zebra"]
        #     )
        # }
        # self.items = [
        #     (filename, self.labels[class_name])
        #     for filename in os.listdir(data_dir)
        #     if filename.endswith(".jpg")
        #     and (class_name := filename.split("-")[0]) in self.labels.keys()
        # ]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data.iloc[idx]["sentence"], self.labels[idx]

class SentenceToPromptModule(pl.LightningDataModule):
    def __init__(
        self,
        config:Config,
        prompt_catalogue: List["Prompt"],
        *args,
        **kwargs,
    ):
        super(SentenceToPromptModule, self).__init__(*args, **kwargs)

        self.prompt_catalogue = prompt_catalogue
        self.config = config
        self.batch_size = self.config.model.batch_size
        self.num_workers = self.config.experiment.num_workers

    def setup(self, stage=None):
        dataset = SentenceToPromptDataset(prompt_catalogue=self.prompt_catalogue, data_path=self.config.data.train_path)

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [1-self.config.experiment.validation_set_split, self.config.experiment.validation_set_split], generator=generator
        )

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
    def __init__(self, config: Config, prompt_catalogue: List["Prompt"], *args, **kwargs):
        super(PromptSelector, self).__init__(*args, **kwargs)

        self.output_layers = len(prompt_catalogue)

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
        y_pred = torch.argmax(logits, dim=-1)
        acc = (y_pred == y_true).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y_true)

        y_pred = torch.argmax(logits, dim=1)
        acc = (y_pred == y_true).float().mean()
        precisions, recalls, f1scores, supports = precision_recall_fscore_support(
            y_true.cpu(), y_pred.cpu(), zero_division=0
        )

        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_acc", acc, on_epoch=True, on_step=False)
        self.log(
            "val_precision",
            precisions.mean(),
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )
        self.log("val_recall", recalls.mean(), on_epoch=True, on_step=False)
        self.log("val_f1", f1scores.mean(), on_epoch=True, on_step=False, prog_bar=True)

        for i, (p, r, f1, _) in enumerate(zip(precisions, recalls, f1scores, supports)):
            self.log(f"val_precision_{i}", p, on_epoch=True, on_step=False)
            self.log(f"val_recall_{i}", r, on_epoch=True, on_step=False)
            self.log(f"val_f1_{i}", f1, on_epoch=True, on_step=False)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.model.learning_rate)

        return {
            "optimizer": optimizer,
        }
    

# Main training function
def main(
):
    config = get_default_configs()
    prompt_catalogue = [Prompt(None, None), Prompt(None, None), Prompt(None, None), Prompt(None, None), Prompt(None, None), Prompt(None, None)]

    data_module = SentenceToPromptModule(
        config, prompt_catalogue
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-{val_f1:.2f}",
        monitor="val_f1",
        mode="max",
        save_top_k=10,
        save_last=True,
    )
    logger = TensorBoardLogger("logs/", name="sentence_to_prompt")

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
    )

    with trainer.init_module():
        model = PromptSelector(config, prompt_catalogue)

    trainer.fit(model, data_module)

    return model, trainer


if __name__ == "__main__":
    main()


