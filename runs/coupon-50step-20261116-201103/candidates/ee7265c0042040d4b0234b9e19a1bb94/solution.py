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


def get_feature_columns(schema):
    cols = schema.get("feature_columns")
    if cols:
        return list(cols)
    return None


def build_encoder():
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def prepare_features(df, feature_cols, encoder=None, fit=False):
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = X[c].astype(object)
        X[c] = X[c].where(X[c].notna(), "__MISSING__")
        X[c] = X[c].astype(str)
    if fit:
        encoder = build_encoder()
        arr = encoder.fit_transform(X)
    else:
        arr = encoder.transform(X)
    return arr.astype(np.float64), encoder


def cmd_train(args):
    data_dir = args.data
    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)

    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    feature_cols = get_feature_columns(schema)
    if feature_cols is None:
        feature_cols = [c for c in features.columns]

    label_col = "label" if "label" in labels.columns else labels.columns[0]
    y_raw = labels[label_col].values

    classes = np.unique(y_raw)
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[str(v)] for v in y_raw], dtype=np.int64)

    X, encoder = prepare_features(features, feature_cols, fit=True)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=None,
        random_state=42,
    )
    model.fit(X, y)

    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(model_dir, "label_mapping.json"), "w") as f:
        json.dump([str(c) for c in classes], f)

    import pickle
    with open(os.path.join(model_dir, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


def cmd_predict(args):
    data_dir = args.data
    model_dir = args.model
    output_path = args.output

    import pickle

    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))

    with open(os.path.join(model_dir, "feature_columns.json"), "r") as f:
        feature_cols = json.load(f)
    with open(os.path.join(model_dir, "label_mapping.json"), "r") as f:
        classes = json.load(f)
    with open(os.path.join(model_dir, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    X, _ = prepare_features(features, feature_cols, encoder=encoder, fit=False)
    preds_idx = model.predict(X)
    preds = [classes[int(i)] for i in preds_idx]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df = pd.DataFrame({"row_id": np.arange(len(preds)), "prediction": preds})
    out_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()