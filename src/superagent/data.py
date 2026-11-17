"""Format-based adapters. No task-identity-to-algorithm dispatch."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import write_json, sha256


def load_split(spec, root: Path, split):
    path = (root / spec[f"{split}_features"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Data paths must stay inside the supplied data root")
    fmt = spec["format"]
    if fmt == "categorical_npy":
        array = np.load(path, allow_pickle=bool(spec.get("trusted_pickle", False)))
        if array.ndim != 2:
            raise ValueError("Expected feature matrix")
        columns = spec.get("feature_columns") or [f"feature_{i}" for i in range(array.shape[1])]
        features = pd.DataFrame(array, columns=columns)
        label_path = (root / spec[f"{split}_labels"]).resolve()
        if not label_path.is_relative_to(root.resolve()):
            raise ValueError("Labels must stay inside data root")
        labels = np.load(label_path, allow_pickle=False).reshape(-1)
    elif fmt in {"csv", "image_csv"}:
        table = pd.read_csv(path)
        label = spec["label_column"]
        columns = spec["feature_columns"]
        if label in columns or len(set(columns)) != len(columns):
            raise ValueError("Feature columns must be unique and exclude the target")
        features = table.loc[:, columns].copy()
        labels = table[label].to_numpy()
    else:
        raise ValueError(f"Unsupported explicit format: {fmt}")
    if len(features) != len(labels) or len(labels) == 0 or pd.isna(labels).any():
        raise ValueError("Empty, misaligned or missing labels")
    return features, np.asarray(labels)


def export_features(features, destination, schema, spec, source_root):
    destination.mkdir(parents=True, exist_ok=True)
    features = features.copy()
    if spec["format"] == "image_csv":
        for column in spec["image_columns"]:
            for index, value in features[column].items():
                source = (source_root / str(value)).resolve()
                if not source.is_relative_to(source_root.resolve()) or not source.is_file():
                    raise ValueError("Invalid image path")
                target = Path("images") / f"{column}_{index}{source.suffix.lower()}"
                (destination / target).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination / target)
                features.loc[index, column] = str(target)
    features.to_csv(destination / "features.csv", index=False)
    write_json(destination / "schema.json", schema)


def prepare_development(spec, root: Path, run: Path):
    # Deliberately never opens test paths here.
    train_x, train_y = load_split(spec, root, "train")
    val_x, val_y = load_split(spec, root, "val")
    if list(train_x.columns) != list(val_x.columns):
        raise ValueError("Training/validation schema mismatch")
    classes = np.unique(train_y).tolist()
    if len(classes) < 2 or not set(val_y.tolist()).issubset(set(classes)):
        raise ValueError("Invalid classification label domain")
    schema = {"modality": spec["modality"], "feature_columns": list(train_x.columns),
              "classes": classes, "metric": "accuracy", "image_columns": spec.get("image_columns", [])}
    export_features(train_x, run / "data/train", schema, spec, root)
    pd.DataFrame({"label": train_y}).to_csv(run / "data/train/labels.csv", index=False)
    export_features(val_x, run / "data/val", schema, spec, root)
    (run / "private").mkdir(exist_ok=True)
    np.save(run / "private/validation_labels.npy", val_y, allow_pickle=False)
    profile = {**schema, "train_rows": len(train_x), "validation_rows": len(val_x),
               "missing_counts": train_x.isna().sum().to_dict(),
               "distinct_counts": train_x.nunique().to_dict(),
               "dtypes": train_x.dtypes.astype(str).to_dict(),
               "training_sample": json.loads(train_x.head(3).to_json(orient="records")),
               "label_counts": {str(k): int(v) for k, v in pd.Series(train_y).value_counts().items()}}
    write_json(run / "profile.json", profile)
    write_json(run / "data_hashes.json", {str(p.relative_to(run)): sha256(p) for p in (run / "data").rglob("*") if p.is_file()})
    return profile
