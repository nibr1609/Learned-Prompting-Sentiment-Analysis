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
        predictions: numpy array of predictions (negative, neutral, positive as elements)
        config: unified configuration object containing all necessary parameters
    """
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'label': predictions
    })
    
    # Create output directory if it doesn't exist
    config.data.submission_dir.mkdir(parents=True, exist_ok=True)
    
    # Save submission in submission folder
    submission_path = config.data.submission_dir / f"{config.experiment.experiment_name}_submission_{config.experiment.experiment_id}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}") 