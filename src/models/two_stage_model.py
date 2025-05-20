from models.base_model import BaseSentimentModel
from models.prompt_evaluator import PromptEvaluator, PromptEvaluatorConfig, Prompt
from models.prompt_selector import PromptSelector, SentenceToPromptModule
from config.config import Config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
import torch

class TestDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        return self.sentences[idx]

class TwoStageModel(BaseSentimentModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.selector_model = None
        self.prompt_catalogue = None
        self.selector_trainer = None
        self.prompt_config = PromptEvaluatorConfig.for_gemma_3_4b_it(verbose=True, debug=True)
        self.evaluator = PromptEvaluator(self.prompt_config)

    def preprocess_dataset(self, config: Config):
        train = pd.read_csv(str(config.data.train_path))
        test = pd.read_csv(str(config.data.test_path))

        if config.experiment.max_train_samples:
            train = train.sample(n=config.experiment.max_train_samples, random_state=43).reset_index(drop=True)

        if config.experiment.max_test_samples:
            test = test.iloc[:config.experiment.max_test_samples]

        # Split off a validation fold if requested
        val_exists = True
        split = config.experiment.validation_set_split or 0.0
        if split > 0:
            train, val = train_test_split(train, test_size=split, random_state=42)
        else:
            val = None
            val_exists = False

        return train, val, test, val_exists
    
    def train(self, train, val):
        if self.config.experiment.two_stage_setting == "selection":
            self.prompt_catalogue = self.load_prompt_catalogue()
            #self.prompt_catalogue = {"b": Prompt.random_example(), "a": Prompt.direct_example()}
            model, trainer = self.train_prompt_selector(train, val)
            print("finished training")
            self.selector_model = model
            self.selector_trainer = trainer
            pass
        elif self.config.experiment.two_stage_setting == "optimize_then_select":
            self.prompt_catalogue = self.load_prompt_catalogue()
            #self.improved_prompt_catalogue = self.optimizer.improve_catalogue(self.config, self.prompt_catalogue, train, val, rounds=2)

            model, trainer = self.train_prompt_selector(train, val)
            print("finished training")
        else:
            raise NotImplementedError("Training not defined for inference model")
    

    def predict(self, test, val):
        print("start predicting")
        if self.config.experiment.two_stage_setting == "direct_inference":
            prompt = Prompt.direct_example()
            return self.predict_single_prompt(prompt, test, val)

        elif self.config.experiment.two_stage_setting == "selection":
            assert self.selector_model is not None, "Prompt Selector has to be trained first"
            if val is not None:
                prompts_val = self.predict_sentences(list(val["sentence"]))
                prompts_val = torch.cat(prompts_val, dim=0)
            prompts_test = self.predict_sentences(list(test["sentence"]))
            prompts_test = torch.cat(prompts_test, dim=0)



            print(prompts_val)
            print("##")
            print(prompts_test)
            if self.config.experiment.selector_eval_setting == "single":
                prediction_val = None
                if val is not None:
                    prediction_val = self.predict_from_prompt_list(prompts_val, val)

                prediction_test = self.predict_from_prompt_list(prompts_test, test)


                return prediction_test, prediction_val if val is not None else None

    

    def evaluate(self, pred, val_data):
        true_labels = val_data["label"]
        metrics = super().evaluate(pred, true_labels)
        print(metrics)
        return metrics
    
    def predict_sentences(self, sentences):
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
        prompts= prompts.argmax(axis=1) #TODO: check axis
        predictions = []
        for i, prompt_idx in enumerate(prompts): # TODO maybe it's faster to group by prompt and then predict sentence batches per promp
            print(f"predicting {i}/{len(prompts)}")
            prompt = list(self.prompt_catalogue.values())[prompt_idx.item()]
            pred = self.evaluator.predict(prompt, [data.iloc[i]["sentence"]])
            predictions.append(pred[0])

        predictions = [pred.lower() for pred in predictions]

        return predictions

    def predict_single_prompt(self, prompt, test, val=None):
        if val is not None:

            X_val = list(val["sentence"])
            prediction_val = np.array(self.evaluator.predict(prompt, X_val))
            prediction_val = [pred.lower() for pred in prediction_val]



        X_test = list(test["sentence"])
        # TODO: Make Prompt Selectable
        prediction_test = np.array(self.evaluator.predict(prompt, X_test))
        prediction_test = [pred.lower() for pred in prediction_test]

        return prediction_test, prediction_val if val is not None else None

    def load_prompt_catalogue(self):
        return Prompt.load_prompts(self.config.prompt.prompt_list)
    
    def train_prompt_selector(self, train, val):
        assert self.prompt_catalogue is not None, "Prompt catalogue must be set before training Prompt Selector"
        data_module = SentenceToPromptModule(
            self.config, self.prompt_catalogue, self.evaluator, train, val
        )


        model_root_directory = self.config.data.model_output_dir / (self.config.experiment.experiment_name + "_" + self.config.experiment.experiment_id)
        model_root_directory.mkdir(parents=True, exist_ok=True)

        # checkpoint_callback = ModelCheckpoint(
        #     dirpath=model_root_directory / "output/",
        #     filename="{epoch}",
        #     #monitor="val_f1",
        #     mode="max",
        #     save_top_k=5,
        #     save_last=True,
        # )

        # logger = TensorBoardLogger(model_root_directory / "logs/", name="sentence_to_prompt")

        trainer = Trainer(
            max_epochs=self.config.model.num_epochs,
            accelerator="auto",
            devices="auto",
            #callbacks=[checkpoint_callback],
            logger=False,
            enable_checkpointing=False,
            #log_every_n_steps=10,
        )

        with trainer.init_module():
            model = PromptSelector(self.config, self.prompt_catalogue)

        trainer.fit(model, data_module)
    
        return model, trainer