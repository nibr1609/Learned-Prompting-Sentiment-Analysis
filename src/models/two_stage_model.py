from models.base_model import BaseSentimentModel
from llm_building_blocks.prompt_evaluator import PromptEvaluator, PromptEvaluatorConfig, Prompt
from llm_building_blocks.prompt_selector import PromptSelector, SentenceToPromptModule
from config.config import Config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
import torch
from llm_building_blocks.prompt_catalogue import PromptCatalogue, Prompt
from llm_building_blocks.prompt_optimizer import PromptOptimizer
from utils.metrics import evaluate


class TestDataset(Dataset):
    """Dataset wrapping a list of sentences for prediction."""

    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        return self.sentences[idx]


class TwoStageModel(BaseSentimentModel):
    """A two-stage sentiment analysis model supporting prompt selection and optimization."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.selector_model = None
        self.prompt_catalogue = None
        self.selector_trainer = None
        self.prompt_config = PromptEvaluatorConfig.for_gemma_3_4b_it(config, verbose=True, debug=True)
        self.evaluator = PromptEvaluator(self.prompt_config)

    def preprocess_dataset(self, config: Config):
        """Load and preprocess datasets for training, validation, and testing. Specific to this kind of Model

        Args:
            config (Config): Configuration containing dataset paths and options.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, bool]:
                train dataset, validation dataset or None, test dataset, flag if validation exists.
        """
        train = pd.read_csv(str(config.data.train_path))
        test = pd.read_csv(str(config.data.test_path))

        if config.experiment.max_train_samples:
            train = train.sample(n=config.experiment.max_train_samples, random_state=43).reset_index(drop=True)

        if config.experiment.max_test_samples:
            test = test.iloc[:config.experiment.max_test_samples]

        val_exists = True
        split = config.experiment.validation_set_split or 0.0
        if split > 0:
            train, val = train_test_split(train, test_size=split, random_state=42)
        else:
            val = None
            val_exists = False

        return train, val, test, val_exists

    def train(self, train, val):
        """Train the model according to the configured two-stage setting.

        Args:
            train (pd.DataFrame): Training dataset.
            val (Optional[pd.DataFrame]): Validation dataset or None.
        """
        if self.config.experiment.two_stage_setting == "selection":
            self.prompt_catalogue = self.load_prompt_catalogue()
            model, trainer = self.train_prompt_selector(train, val)
            print("finished training")
            self.selector_model = model
            self.selector_trainer = trainer

        elif self.config.experiment.two_stage_setting == "optimize_then_select":
            self.prompt_catalogue = self.load_prompt_catalogue()
            model, trainer = self.train_prompt_selector(train, val)
            print("finished training")

        elif self.config.experiment.two_stage_setting == "only_optimize":
            self.prompt_catalogue = self.load_prompt_catalogue()
            optimizer = PromptOptimizer(self.evaluator, self.config)
            train_rest = train.iloc[:self.config.experiment.max_train_samples_prompt_optimizer]
            self.prompt_catalogue = optimizer.run_optimization_loop(
                prompt_catalogue=self.prompt_catalogue,
                train=train_rest,
                val=val,
                iterations=self.config.prompt.optimizer_iterations,
            )

        elif self.config.experiment.two_stage_setting == "optimize_and_select":
            self.prompt_catalogue = self.load_prompt_catalogue()
            optimizer = PromptOptimizer(self.evaluator, self.config)
            train_rest = train.iloc[:self.config.experiment.max_train_samples_prompt_optimizer]
            self.prompt_catalogue = optimizer.run_optimization_loop(
                prompt_catalogue=self.prompt_catalogue,
                train=train_rest,
                val=val,
                iterations=self.config.prompt.optimizer_iterations,
            )
            model, trainer = self.train_prompt_selector(train, val)
            self.selector_model = model
            self.selector_trainer = trainer

        else:
            raise NotImplementedError("Training not defined for inference model")

    def predict(self, test, val):
        """Make predictions on test (and optionally validation) data.

        Args:
            test (pd.DataFrame): Test dataset.
            val (Optional[pd.DataFrame]): Validation dataset or None.

        Returns:
            Tuple[List[str], Optional[List[str]]]:
                Predictions on test data, and optionally predictions on validation data.
        """
        print("start predicting")
        if self.config.experiment.two_stage_setting == "direct_inference":
            prompt = Prompt.direct_example()
            return self.predict_single_prompt(prompt, test, val)

        elif self.config.experiment.two_stage_setting in ["selection", "optimize_and_select"]:
            assert self.selector_model is not None, "Prompt Selector has to be trained first"
            if val is not None:
                prompts_val = self.predict_sentences(list(val["sentence"]))
                prompts_val = torch.cat(prompts_val, dim=0)
            prompts_test = self.predict_sentences(list(test["sentence"]))
            prompts_test = torch.cat(prompts_test, dim=0)

            if self.config.experiment.selector_eval_setting == "single":
                prediction_val = None
                if val is not None:
                    prediction_val = self.predict_from_prompt_list(prompts_val, val)

                prediction_test = self.predict_from_prompt_list(prompts_test, test)

                return prediction_test, prediction_val if val is not None else None

        elif self.config.experiment.two_stage_setting == "only_optimize":
            prompt = self.prompt_catalogue.get_prompts()[0][1]
            print("top prompt")
            print(prompt)
            return self.predict_single_prompt(prompt, test, val)

    def evaluate(self, pred, val_data):
        """Evaluate predictions against true labels.

        Args:
            pred (List[str]): Predicted labels.
            val_data (pd.DataFrame): Validation data containing true labels.

        Returns:
            dict: Evaluation metrics.
        """
        true_labels = val_data["label"]
        metrics = evaluate(pred, true_labels)
        print(metrics)
        return metrics

    def predict_sentences(self, sentences):
        """Predict soft prompt probabilities for a list of sentences.

        Args:
            sentences (List[str]): Sentences to predict.

        Returns:
            List[torch.Tensor]: List of prediction tensors for batches.
        """
        dataset = TestDataset(sentences)
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=self.config.experiment.num_workers,
            persistent_workers=True,
        )
        predictions = self.selector_trainer.predict(model=self.selector_model, dataloaders=data_loader)
        return predictions

    def predict_from_prompt_list(self, prompts, data):
        """Predict sentiment labels from prompt selection probabilities.

        Args:
            prompts (torch.Tensor): Soft prompt selection tensor.
            data (pd.DataFrame): Dataset containing sentences.

        Returns:
            List[str]: List of predicted labels.
        """
        prompts = prompts.argmax(axis=1)
        predictions = []
        for i, prompt_idx in enumerate(prompts):
            print(f"predicting {i}/{len(prompts)}")
            prompt = self.prompt_catalogue.get_prompt_at_pos(prompt_idx.item())
            pred = self.evaluator.predict(prompt, [data.iloc[i]["sentence"]])
            predictions.append(pred[0])

        predictions = [pred.lower() for pred in predictions]
        return predictions

    def predict_single_prompt(self, prompt, test, val=None):
        """Predict labels using a single fixed prompt.

        Args:
            prompt (Prompt): Prompt to use for prediction.
            test (pd.DataFrame): Test dataset.
            val (Optional[pd.DataFrame]): Validation dataset or None.

        Returns:
            Tuple[List[str], Optional[List[str]]]: Predictions on test and optionally validation data.
        """
        if val is not None:
            X_val = list(val["sentence"])
            prediction_val = np.array(self.evaluator.predict(prompt, X_val))
            prediction_val = [pred.lower() for pred in prediction_val]

        X_test = list(test["sentence"])
        prediction_test = np.array(self.evaluator.predict(prompt, X_test))
        prediction_test = [pred.lower() for pred in prediction_test]

        return prediction_test, prediction_val if val is not None else None

    def load_prompt_catalogue(self):
        """Load prompt catalogue from configuration.

        Returns:
            PromptCatalogue: Loaded prompt catalogue.
        """
        return PromptCatalogue(self.config.prompt.prompt_list)

    def train_prompt_selector(self, train, val):
        """Train the prompt selector model.

        Args:
            train (pd.DataFrame): Training data.
            val (Optional[pd.DataFrame]): Validation data or None.

        Returns:
            Tuple[PromptSelector, Trainer]: Trained model and its trainer.
        """
        assert self.prompt_catalogue is not None, "Prompt catalogue must be set before training Prompt Selector"
        data_module = SentenceToPromptModule(
            self.config, self.prompt_catalogue, self.evaluator, train, val
        )

        model_root_directory = self.config.data.model_output_dir / (self.config.experiment.experiment_name + "_" + self.config.experiment.experiment_id)
        model_root_directory.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(
            max_epochs=self.config.model.num_epochs,
            accelerator="auto",
            devices="auto",
            logger=False,
            enable_checkpointing=False,
        )

        with trainer.init_module():
            model = PromptSelector(self.config, self.prompt_catalogue)

        trainer.fit(model, data_module)

        return model, trainer
