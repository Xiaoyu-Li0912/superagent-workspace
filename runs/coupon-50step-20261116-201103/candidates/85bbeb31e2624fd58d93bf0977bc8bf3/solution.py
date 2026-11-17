import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def get_feature_columns(schema, df):
    cols = schema.get("feature_columns")
    if cols:
        return [c for c in cols if c in df.columns]
    return [c for c in df.columns if c != "row_id"]


def build_frame(df, feature_cols, cat_cols, num_cols):
    out = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c in cat_cols:
            out[c] = df[c].astype("object").where(df[c].notna(), "__NA__").astype(str)
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def train(args):
    data_dir = args.data
    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)

    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    label_col = "label" if "label" in labels.columns else labels.columns[0]
    y_raw = labels[label_col].values

    feature_cols = get_feature_columns(schema, features)

    # Determine categorical vs numeric from schema dtypes and data
    dtypes = schema.get("dtypes", {})
    cat_cols = []
    num_cols = []
    for c in feature_cols:
        dt = str(dtypes.get(c, "")).lower()
        if dt in ("object", "str", "string", "category"):
            cat_cols.append(c)
        else:
            # fallback: check pandas dtype
            if features[c].dtype == object:
                cat_cols.append(c)
            else:
                num_cols.append(c)

    X = build_frame(features, feature_cols, cat_cols, num_cols)

    # Label mapping
    classes = np.unique(y_raw)
    label_to_idx = {str(v): i for i, v in enumerate(classes)}
    idx_to_label = {i: v for v, i in label_to_idx.items()}
    y = np.array([label_to_idx[str(v)] for v in y_raw], dtype=np.int64)

    # Try CatBoost first, fallback to sklearn
    model = None
    model_type = None
    try:
        from catboost import CatBoostClassifier, Pool
        cat_idx = [X.columns.get_loc(c) for c in cat_cols]
        X_cb = X.copy()
        for c in cat_cols:
            X_cb[c] = X_cb[c].astype(str)
        model = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.08,
            loss_function="MultiClass" if len(classes) > 2 else "Logloss",
            verbose=False,
            thread_count=2,
            random_seed=42,
            allow_writing_files=False,
        )
        model.fit(X_cb, y, cat_features=cat_idx)
        model_type = "catboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer

        if cat_cols:
            pre = ColumnTransformer(
                transformers=[
                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
                    ("num", "passthrough", num_cols),
                ],
                remainder="drop",
            )
        else:
            pre = "passthrough"
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
        model = Pipeline([("pre", pre), ("clf", clf)])
        model.fit(X, y)
        model_type = "sklearn"

    # Persist everything
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "cat_cols": cat_cols,
                "num_cols": num_cols,
                "idx_to_label": {str(k): (v.item() if hasattr(v, "item") else v) for k, v in idx_to_label.items()},
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
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    idx_to_label = {int(k): v for k, v in meta["idx_to_label"].items()}
    model_type = meta["model_type"]

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X = build_frame(features, feature_cols, cat_cols, num_cols)

    if model_type == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(os.path.join(model_dir, "model.cbm"))
        for c in cat_cols:
            X[c] = X[c].astype(str)
        preds = model.predict(X)
        preds = np.array(preds).reshape(-1).astype(int)
    else:
        import pickle
        with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        preds = model.predict(X)
        preds = np.array(preds).reshape(-1).astype(int)

    labels_out = [idx_to_label[int(p)] for p in preds]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df = pd.DataFrame({"row_id": np.arange(len(features)), "prediction": labels_out})
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