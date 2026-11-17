import numpy as np
import pandas as pd


def evaluate_predictions(path, labels, classes):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError("Missing or unsafe prediction file")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_id", "prediction"] or len(frame) != len(labels):
        raise ValueError("Wrong columns or prediction count")
    if not np.array_equal(frame.row_id.to_numpy(), np.arange(len(labels))):
        raise ValueError("Prediction IDs missing, reordered or duplicated")
    if frame.prediction.isna().any() or not set(frame.prediction.tolist()).issubset(set(classes)):
        raise ValueError("Predictions outside original label domain")
    return float(np.mean(frame.prediction.to_numpy() == labels))
