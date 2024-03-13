import pickle
import re
from pathlib import Path

__date__ = "130324"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/finalized_model_{__date__}.pkl", "rb") as f:
    model = pickle.load(f)


def predict_pipeline(pred_X):
    pred = model.predict([pred_X])
    return pred[0]
