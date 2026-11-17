import argparse
import json
import os

import numpy as np
import pandas as pd


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


def build_encoders(train_df, feature_cols):
    encoders = {}
    for col in feature_cols:
        series = train_df[col].astype(object)
        series = series.where(pd.notna(series), "__MISSING__")
        series = series.astype(str)
        uniques = sorted(series.unique().tolist())
        mapping = {v: i for i, v in enumerate(uniques)}
        encoders[col] = mapping
    return encoders


def apply_encoders(df, feature_cols, encoders):
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        series = df[col].astype(object)
        series = series.where(pd.notna(series), "__MISSING__")
        series = series.astype(str)
        mapping = encoders[col]
        out[col] = series.map(mapping).fillna(-1).astype(np.int64)
    return out


def train(args):
    data_dir = args.data
    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)

    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    label_col = "label" if "label" in labels.columns else labels.columns[0]
    y_raw = labels[label_col].astype(str).values

    feature_cols = get_feature_columns(schema, features)

    encoders = build_encoders(features, feature_cols)
    X = apply_encoders(features, feature_cols, encoders)

    classes = sorted(pd.unique(y_raw).tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[v] for v in y_raw], dtype=np.int64)

    model = None
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass" if len(classes) > 2 else "Logloss",
            verbose=False,
            thread_count=2,
            random_seed=42,
        )
        model.fit(X.values, y)
        model_type = "catboost"
    except Exception:
        model = None

    if model is None:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        )
        model.fit(X.values, y)
        model_type = "sklearn_hgb"

    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "encoders": encoders,
                "classes": classes,
                "model_type": model_type,
            },
            f,
        )

    if model_type == "catboost":
        model.save_model(os.path.join(model_dir, "model.cbm"))
    else:
        import pickle
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)


def predict(args):
    data_dir = args.data
    model_dir = args.model
    output_path = args.output

    with open(os.path.join(model_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    encoders = meta["encoders"]
    classes = meta["classes"]
    model_type = meta["model_type"]

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X = apply_encoders(features, feature_cols, encoders)

    if model_type == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(os.path.join(model_dir, "model.cbm"))
        preds = model.predict(X.values)
        preds = np.array(preds).reshape(-1).astype(np.int64)
    else:
        import pickle
        with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        preds = model.predict(X.values).astype(np.int64)

    pred_labels = [classes[int(p)] for p in preds]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    result = pd.DataFrame(
        {
            "row_id": np.arange(len(features), dtype=np.int64),
            "prediction": pred_labels,
        }
    )
    result.to_csv(output_path, index=False)


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