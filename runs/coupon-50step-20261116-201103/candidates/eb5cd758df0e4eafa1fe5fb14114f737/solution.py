import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def get_feature_columns(schema, df):
    cols = schema.get("feature_columns")
    if cols:
        cols = [c for c in cols if c in df.columns]
    else:
        cols = [c for c in df.columns if c != "row_id"]
    return cols


def prepare_features(df, feature_columns):
    X = df[feature_columns].copy()
    cat_cols = []
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype).startswith("str") or X[c].dtype.name == "category":
            X[c] = X[c].astype(str).fillna("__NA__")
            cat_cols.append(c)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, cat_cols


def train(args):
    data_dir = args.data
    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)

    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    label_col = "label" if "label" in labels.columns else labels.columns[0]
    y_raw = labels[label_col].values

    feature_columns = get_feature_columns(schema, features)
    X, cat_cols = prepare_features(features, feature_columns)

    classes = np.unique(y_raw)
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y = np.array([class_to_idx[str(v)] for v in y_raw], dtype=int)

    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass" if len(classes) > 2 else "Logloss",
        verbose=False,
        thread_count=2,
        random_seed=42,
        allow_writing_files=False,
    )

    train_pool = Pool(X, y, cat_features=cat_indices)
    model.fit(train_pool)

    model.save_model(os.path.join(model_dir, "model.cbm"))

    meta = {
        "feature_columns": feature_columns,
        "cat_cols": cat_cols,
        "idx_to_class": {str(k): (v.item() if hasattr(v, "item") else v) for k, v in idx_to_class.items()},
        "n_classes": int(len(classes)),
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def predict(args):
    data_dir = args.data
    model_dir = args.model
    output_path = args.output

    with open(os.path.join(model_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    feature_columns = meta["feature_columns"]
    cat_cols = meta["cat_cols"]
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X, _ = prepare_features(features, feature_columns)
    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    model = CatBoostClassifier()
    model.load_model(os.path.join(model_dir, "model.cbm"))

    preds = model.predict(X)
    preds = np.asarray(preds).ravel().astype(int)

    original = [idx_to_class[int(p)] for p in preds]

    out_df = pd.DataFrame({"row_id": np.arange(len(features)), "prediction": original})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)
    p_pred.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()