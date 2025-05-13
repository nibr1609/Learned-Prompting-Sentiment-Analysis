from models.base_model import BaseSentimentModel
from models.prompt_evaluator import PromptEvaluator, PromptEvaluatorConfig, Prompt
from config.config import Config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PromptSimpleInferenceModel(BaseSentimentModel):
    def __init__(self, config: Config):
        super().__init__(config)
        

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
        raise NotImplementedError("Training not defined for inference model")
    

    def predict(self, test, val):

        if val is not None:
            config = PromptEvaluatorConfig.for_gemma_3_4b_it(verbose=True, debug=True)
            evaluator = PromptEvaluator(config)
            X_val = list(val["sentence"])
            prediction_val = np.array(evaluator.predict(Prompt.direct_example(), X_val))
            prediction_val = [pred.lower() for pred in prediction_val]


        config = PromptEvaluatorConfig.for_gemma_3_4b_it(verbose=True, debug=True)
        evaluator = PromptEvaluator(config)

        X_test = list(test["sentence"])
        # TODO: Make Prompt Selectable
        prediction_test = np.array(evaluator.predict(Prompt.direct_example(), X_test))
        prediction_test = [pred.lower() for pred in prediction_test]

        print(X_val)
        print()
        print(prediction_val[:50])
        print()
        print(val["label"][:10])

        return prediction_test, prediction_val if val is not None else None

    def evaluate(self, pred, val_data):
        true_labels = val_data["label"]
        metrics = super().evaluate(pred, true_labels)
        print(metrics)
        return metrics
    