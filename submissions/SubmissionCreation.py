import time
import pandas as pd
import numpy as np
from pathlib import Path
import os

MODULE_DIR = Path(__file__).parent


def create_submission(predictions):

    try:
        length = len(pd.read_csv(MODULE_DIR / "../data/test.csv"))
    except FileNotFoundError:
        print("Test file has been moved!")
        return None

    submission = pd.DataFrame({
        "id": np.arange(length),
        "label": predictions
    })

    (MODULE_DIR / "output").mkdir(parents=True, exist_ok=True)
    submission.to_csv(MODULE_DIR / ("output/submission" + str(int(time.time())) + ".csv"), index=False)