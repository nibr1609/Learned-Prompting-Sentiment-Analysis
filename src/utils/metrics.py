import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import json

def evaluate(pred, labels):
    cm = confusion_matrix(labels, pred, labels=['positive', 'neutral', 'negative'])
    
    # Avoid division by zero
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)

    # False positive rate: FP / (FP + TN)
    fp_rate_per_class = FP / (FP + TN + 1e-10)  # add small value to avoid div by 0
    macro_fp_rate = fp_rate_per_class.mean()

    def nmae(y_true, y_pred):
        sentiment_to_int = {"negative": 0, "neutral": 1, "positive": 2}
        y_true = np.array([sentiment_to_int[s] for s in y_true])
        y_pred = np.array([sentiment_to_int[s] for s in y_pred])
        mae = np.mean(np.abs(y_true - y_pred))
        return float(0.5 * (2 - mae))

    metrics = {
        "accuracy": accuracy_score(labels, pred),
        "nmae": nmae(labels, pred),
        "precision_macro": precision_score(labels, pred, average='macro', zero_division=0),
        "recall_macro": recall_score(labels, pred, average='macro', zero_division=0),
        "f1_macro": f1_score(labels, pred, average='macro', zero_division=0),
        "precision_weighted": precision_score(labels, pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(labels, pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(labels, pred, average='weighted', zero_division=0),
        "false_positive_rate_macro": macro_fp_rate,
        "false_positive_rate_per_class": fp_rate_per_class.tolist(),
        "confusion_matrix": cm.tolist()
    }

    return metrics

def save_validation_metrics(config, metrics, suffix=None):
    directory = config.data.experiment_output_dir / f"experiment_{config.experiment.experiment_id}"
    validation_set_dir = directory / "val_set_metrics"
    validation_set_dir.mkdir(parents=True, exist_ok=True)

    if suffix is None:
        metrics_path = validation_set_dir / "metrics.json"
    else:
        metrics_path = validation_set_dir / f"metrics{suffix}.json"

    # Write the metrics dictionary to the file
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)