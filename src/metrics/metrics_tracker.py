import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class MetricsTracker:
    def __init__(self, experiment_name: str, save_dir: Path):
        self.experiment_name = experiment_name
        self.save_dir = save_dir / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_metrics: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': []
        }
        
        self.val_metrics: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': []
        }
        
        self.best_val_metrics: Dict[str, float] = {
            'loss': float('inf'),
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    def update_train_metrics(self, loss: float, predictions: np.ndarray, labels: np.ndarray):
        self.train_metrics['loss'].append(loss)
        self._update_metrics(self.train_metrics, predictions, labels)
    
    def update_val_metrics(self, loss: float, predictions: np.ndarray, labels: np.ndarray):
        self.val_metrics['loss'].append(loss)
        self._update_metrics(self.val_metrics, predictions, labels)
        
        # Update best metrics
        if self.val_metrics['f1'][-1] > self.best_val_metrics['f1']:
            self.best_val_metrics = {
                'loss': self.val_metrics['loss'][-1],
                'accuracy': self.val_metrics['accuracy'][-1],
                'f1': self.val_metrics['f1'][-1],
                'precision': self.val_metrics['precision'][-1],
                'recall': self.val_metrics['recall'][-1]
            }
    
    def _update_metrics(self, metrics: Dict[str, List[float]], predictions: np.ndarray, labels: np.ndarray):
        metrics['accuracy'].append(accuracy_score(labels, predictions))
        metrics['f1'].append(f1_score(labels, predictions, average='weighted'))
        metrics['precision'].append(precision_score(labels, predictions, average='weighted'))
        metrics['recall'].append(recall_score(labels, predictions, average='weighted'))
    
    def save_metrics(self):
        metrics = {
            'train': self.train_metrics,
            'val': self.val_metrics,
            'best_val': self.best_val_metrics
        }
        
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def get_best_metrics(self) -> Dict[str, float]:
        return self.best_val_metrics 