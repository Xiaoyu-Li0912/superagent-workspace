import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def get_feature_columns(schema, features_df):
    cols = schema.get("feature_columns")
    if cols:
        return [c for c in cols if c in features_df.columns]
    return [c for c in features_df.columns if c != "row_id"]


def train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    feature_cols = get_feature_columns(schema, features)
    X = features[feature_cols].copy()
    y_raw = labels["label"].astype(str).values

    classes = np.unique(y_raw)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[v] for v in y_raw], dtype=np.int64)

    cat_cols = [c for c in feature_cols if X[c].dtype == object or str(X[c].dtype) == "str"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    encoder = None
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = encoder.fit_transform(X[cat_cols].astype(str))
    else:
        X_cat = np.empty((len(X), 0))

    if num_cols:
        X_num = X[num_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    else:
        X_num = np.empty((len(X), 0))

    X_all = np.hstack([X_num, X_cat]) if X_num.shape[1] and X_cat.shape[1] else (X_num if X_num.shape[1] else X_cat)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=None,
        random_state=42,
    )
    model.fit(X_all, y)

    with open(os.path.join(model_dir, "label_mapping.json"), "w") as f:
        json.dump({"classes": [str(c) for c in classes]}, f)
    with open(os.path.join(model_dir, "columns.json"), "w") as f:
        json.dump({"feature_cols": feature_cols, "cat_cols": cat_cols, "num_cols": num_cols}, f)

    import pickle
    with open(os.path.join(model_dir, "pipeline.pkl"), "wb") as f:
        pickle.dump({"encoder": encoder, "model": model}, f)


def predict(data_dir, model_dir, output_path):
    import pickle

    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))

    with open(os.path.join(model_dir, "columns.json"), "r") as f:
        col_info = json.load(f)
    with open(os.path.join(model_dir, "label_mapping.json"), "r") as f:
        label_info = json.load(f)
    with open(os.path.join(model_dir, "pipeline.pkl"), "rb") as f:
        pipeline = pickle.load(f)

    feature_cols = col_info["feature_cols"]
    cat_cols = col_info["cat_cols"]
    num_cols = col_info["num_cols"]
    encoder = pipeline["encoder"]
    model = pipeline["model"]
    classes = label_info["classes"]

    X = features.reindex(columns=feature_cols)

    if cat_cols:
        X_cat = encoder.transform(X[cat_cols].astype(str))
    else:
        X_cat = np.empty((len(X), 0))

    if num_cols:
        X_num = X[num_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    else:
        X_num = np.empty((len(X), 0))

    X_all = np.hstack([X_num, X_cat]) if X_num.shape[1] and X_cat.shape[1] else (X_num if X_num.shape[1] else X_cat)

    preds_idx = model.predict(X_all)
    preds = [classes[int(i)] for i in preds_idx]

    out = pd.DataFrame({"row_id": np.arange(len(features)), "prediction": preds})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.model)
    elif args.command == "predict":
        predict(args.data, args.model, args.output)


if __name__ == "__main__":
    main()