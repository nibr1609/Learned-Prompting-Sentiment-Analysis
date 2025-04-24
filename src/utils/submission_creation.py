import time
import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Dict
from config.config import Config

def create_submission(predictions: np.ndarray, config: Config) -> None:
    """
    Create a submission file from the predictions.
    
    Args:
        predictions: numpy array of predictions
        config: unified configuration object containing all necessary parameters
    """
    # Ensure predictions is a numpy array
    predictions = np.array(predictions)
    
    # Map numeric predictions to string labels
    string_predictions = np.array([config.model.label_mapping[str(pred)] for pred in predictions])
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'label': string_predictions
    })
    
    # Create output directory if it doesn't exist
    config.data.submission_dir.mkdir(parents=True, exist_ok=True)
    
    # Save submission with experiment name and ID
    submission_path = config.data.submission_dir / f"{config.experiment.experiment_name}_submission_{config.experiment.experiment_id}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}") 