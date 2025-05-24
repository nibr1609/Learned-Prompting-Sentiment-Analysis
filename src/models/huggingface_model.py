from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import numpy as np
import torch
from models.base_model import BaseSentimentModel
from config.config import Config
import evaluate
import pandas as pd
from math import ceil
from transformers import DataCollatorWithPadding
from transformers import DebertaV2Tokenizer
from utils.metrics import evaluate

class BERTHuggingFaceModel(BaseSentimentModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self.tokenizer = self.load_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model.model_name,
            num_labels=3,
        )
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU available, using CPU.")
    
    def preprocess_dataset(self, config: Config):
        # 1) load
        ds = load_dataset("csv", data_files={
            "train": str(config.data.train_path),
            "test":  str(config.data.test_path)
        })

        # 2) optional subsampling
        if config.experiment.max_train_samples:
            ds["train"] = (
                ds["train"]
                .shuffle(seed=42)
                .select(range(config.experiment.max_train_samples))
            )
        if config.experiment.max_test_samples:
            ds["test"] = (
                ds["test"].select(range(config.experiment.max_test_samples))
            )

        # 3) split off a val fold if requested
        validation_exists = False
        split = config.experiment.validation_set_split or 0.0
        if split > 0:
            validation_exists = True
            split_ds = ds["train"].train_test_split(
                test_size=split, seed=42
            )
            ds["train"] = split_ds["train"]
            ds["val"]   = split_ds["test"]

        if "label" in ds["test"].column_names:
            ds["test"] = ds["test"].remove_columns("label")

        label2id = {"negative": 0, "neutral": 1, "positive": 2}

        def tokenize(batch):
            toks = self.tokenizer(batch["sentence"],
                                padding=True, truncation=True)
            # only add labels if they exist in this batch
            if "label" in batch and batch["label"] is not None:
                toks["labels"] = [
                    label2id[lab] if isinstance(lab, str) else int(lab)
                    for lab in batch["label"]
                ]
            return toks

        ds = ds.map(tokenize, batched=True)

        # now set torch format per-split
        # train always has labels
        ds["train"].set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        # if you have a validation split
        if "val" in ds:
            ds["val"].set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"]
            )
        # test has no labels, so leave them off
        ds["test"].set_format(
            type="torch",
            columns=["input_ids", "attention_mask"]
        )

        return ds["train"], ds["val"] if validation_exists else None, ds["test"], validation_exists

    def train(self, train_ds, val_ds=None):
        model_root_directory = self.config.data.model_output_dir / (self.config.experiment.experiment_name + "_" + self.config.experiment.experiment_id)

        model_root_directory.mkdir(parents=True, exist_ok=True)

        self.model.to(self.config.experiment.device)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer,
            pad_to_multiple_of=8)

        # save save_per_epoch times per epoch
        steps_per_epoch = ceil(len(train_ds) / self.config.model.batch_size)
        eval_save_steps = max(1, steps_per_epoch // self.config.experiment.save_per_epoch)

        # save memory:
        #self.model.gradient_checkpointing_enable()

        args = TrainingArguments(
            output_dir=model_root_directory / "output",
            logging_dir=model_root_directory / "logs",
            learning_rate=self.config.model.learning_rate,
            per_device_train_batch_size=self.config.model.batch_size,
            per_device_eval_batch_size=self.config.model.batch_size,
            num_train_epochs=self.config.model.num_epochs,
            weight_decay=self.config.model.weight_decay,
            logging_strategy="epoch",
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=eval_save_steps,
            save_steps=eval_save_steps,
            save_total_limit = 3,
            label_names=["labels"],
            load_best_model_at_end = True,
            metric_for_best_model = "f1",
            lr_scheduler_type = "linear"
        )

        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=-1)
            true_labels = p.label_ids
            
            accuracy = accuracy_metric.compute(predictions=preds, references=true_labels)
            f1 = f1_metric.compute(predictions=preds, references=true_labels, average="macro")
            precision = precision_metric.compute(predictions=preds, references=true_labels, average="macro")
            recall = recall_metric.compute(predictions=preds, references=true_labels, average="macro")

            return {
                "accuracy": accuracy["accuracy"],
                "f1": f1["f1"],
                "precision": precision["precision"],
                "recall": recall["recall"],
            }
        
        if val_ds is not None:
            self.trainer = Trainer(model=self.model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics, data_collator=self.data_collator, callbacks = [EarlyStoppingCallback(early_stopping_patience=3)])
        else:
            self.trainer = Trainer(model=self.model, args=args, train_dataset=train_ds, compute_metrics=compute_metrics, data_collator=self.data_collator, callbacks = [EarlyStoppingCallback(early_stopping_patience=self.config.experiment.early_stopping_patience)])
        self.trainer.can_return_loss = True
        self.trainer.train()

    def predict_logits(self, test_ds, val_ds):
        # if nobody trained, still create a Trainer for inference
        if not hasattr(self, "trainer"):
            args = TrainingArguments(
                output_dir=self.config.data.model_output_dir / "hf_tmp",
                per_device_eval_batch_size=self.config.model.batch_size or 32,
                logging_strategy="no",
                save_strategy="no",
            )
            if not hasattr(self, "data_collator"):
                self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=8)

            self.trainer = Trainer(model=self.model, args=args, data_collator=self.data_collator)

            

        val_logits = None
        if val_ds is not None:
            val_logits = self.trainer.predict(val_ds).predictions

        print(test_ds)
        test_logits  = self.trainer.predict(test_ds).predictions
        return test_logits, val_logits
    
    def predict(self, test_ds, val_ds):
        test_logits, val_logits = self.predict_logits(test_ds, val_ds)

        val_labels = None
        if val_logits is not None:
            val_labels = np.argmax(val_logits, axis=-1)
            val_labels = np.array([ {0:"negative",1:"neutral",2:"positive", None:None}[lab]
                                for lab in val_labels ])
        
        test_labels = np.argmax(test_logits, axis=-1)
        test_labels = np.array([ {0:"negative",1:"neutral",2:"positive", None:None}[lab]
                               for lab in test_labels ])

        return test_labels, val_labels


    def evaluate(self, pred, true_data):
        true_labels = true_data["labels"]
        true_labels = np.array([ {0:"negative",1:"neutral",2:"positive", None:None}[lab]
                                for lab in np.array(true_labels) ])
        metrics = evaluate(pred, true_labels)
        print(metrics)
        return metrics
    
    def load_tokenizer(self):
        # No fast tokenizer available for deberta
        if self.config.model.model_name == "microsoft/deberta-v3-large":
            tokenizer =DebertaV2Tokenizer.from_pretrained(self.config.model.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_name)

        return tokenizer

