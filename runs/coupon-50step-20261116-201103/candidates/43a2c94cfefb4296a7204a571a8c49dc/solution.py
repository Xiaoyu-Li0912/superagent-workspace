import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False


# ---------- Schema / feature helpers ----------


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


ORDINAL_MAPS = {
    "age": {
        "below21": 0,
        "21": 1,
        "26": 2,
        "31": 3,
        "36": 4,
        "41": 5,
        "46": 6,
        "50plus": 7,
    },
    "income": {
        "Less than $12500": 0,
        "$12500 - $24999": 1,
        "$25000 - $37499": 2,
        "$37500 - $49999": 3,
        "$50000 - $62499": 4,
        "$62500 - $74999": 5,
        "$75000 - $87499": 6,
        "$87500 - $99999": 7,
        "$100000 or More": 8,
    },
    "education": {
        "Some High School": 0,
        "High School Graduate": 1,
        "Some college - no degree": 2,
        "Associates degree": 3,
        "Bachelors degree": 4,
        "Graduate degree (Masters or Doctorate)": 5,
    },
    "Bar": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "CoffeeHouse": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "CarryAway": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "RestaurantLessThan20": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "Restaurant20To50": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "time": {"7AM": 0, "10AM": 1, "2PM": 2, "6PM": 3, "10PM": 4},
    "expiration": {"2h": 0, "1d": 1},
}

CATEGORICAL_COLS = [
    "coupon",
    "gender",
    "maritalStatus",
    "occupation",
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
    "time",
    "expiration",
]


def _to_str(s):
    return s.astype(str).fillna("__NA__")


def build_features(df, feature_columns):
    """Return (X, cat_cols). All transforms are row-local (no fit needed)."""
    X = df[feature_columns].copy()

    for col, mapping in ORDINAL_MAPS.items():
        if col in X.columns:
            s = _to_str(X[col])
            X[col + "_ord"] = s.map(mapping).astype(float)

    for c in [
        "temperature",
        "has_children",
        "toCoupon_GEQ5min",
        "toCoupon_GEQ15min",
        "toCoupon_GEQ25min",
        "direction_same",
        "direction_opp",
    ]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    if "toCoupon_GEQ15min" in X.columns and "toCoupon_GEQ25min" in X.columns:
        X["distance_bucket"] = X["toCoupon_GEQ25min"].fillna(0).astype(int) * 2 + X[
            "toCoupon_GEQ15min"
        ].fillna(0).astype(int)
    if "direction_same" in X.columns and "direction_opp" in X.columns:
        X["direction"] = X["direction_same"].fillna(0).astype(int) * 2 + X[
            "direction_opp"
        ].fillna(0).astype(int)

    if "time" in X.columns:
        t = _to_str(X["time"])
        X["time_ord"] = t.map(ORDINAL_MAPS["time"]).astype(float)

    def _interact(a, b, name):
        if a in X.columns and b in X.columns:
            X[name] = _to_str(X[a]) + "|" + _to_str(X[b])

    _interact("coupon", "time", "coupon_x_time")
    _interact("coupon", "expiration", "coupon_x_expiration")
    _interact("maritalStatus", "has_children", "marital_x_children")
    _interact("coupon", "direction_opp", "coupon_x_direction")

    cat_cols = []
    for c in X.columns:
        if (
            X[c].dtype == object
            or str(X[c].dtype).startswith("str")
            or X[c].dtype.name == "category"
        ):
            X[c] = _to_str(X[c])
            cat_cols.append(c)

    return X, cat_cols


# ---------- Training ----------


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
    X, cat_cols = build_features(features, feature_columns)

    classes = np.unique(y_raw)
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y = np.array([class_to_idx[str(v)] for v in y_raw], dtype=int)
    n_classes = len(classes)

    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cb_models = []
    lgb_models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        cb = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            eval_metric="Accuracy",
            verbose=False,
            thread_count=2,
            random_seed=42 + fold,
            allow_writing_files=False,
            od_type="Iter",
            od_wait=100,
        )
        train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        val_pool = Pool(X_va, y_va, cat_features=cat_indices)
        cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
        cb_models.append(cb)

        if HAS_LGB:
            X_tr_l = X_tr.copy()
            X_va_l = X_va.copy()
            for c in cat_cols:
                X_tr_l[c] = X_tr_l[c].astype("category")
                X_va_l[c] = X_va_l[c].astype("category")
            for c in cat_cols:
                cats = X_tr_l[c].cat.categories
                X_va_l[c] = pd.Categorical(X_va_l[c], categories=cats)

            lgb_train = lgb.Dataset(X_tr_l, label=y_tr, categorical_feature=cat_cols)
            lgb_val = lgb.Dataset(
                X_va_l, label=y_va, categorical_feature=cat_cols, reference=lgb_train
            )
            params = {
                "objective": "binary" if n_classes == 2 else "multiclass",
                "num_class": n_classes if n_classes > 2 else 1,
                "metric": "binary_error" if n_classes == 2 else "multi_error",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 5,
                "lambda_l2": 1.0,
                "verbose": -1,
                "num_threads": 2,
                "seed": 42 + fold,
            }
            if n_classes > 2:
                params.pop("num_class")
            lgb_model = lgb.train(
                params,
                lgb_train,
                num_boost_round=1500,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            lgb_models.append(lgb_model)

    for i, m in enumerate(cb_models):
        m.save_model(os.path.join(model_dir, f"cb_{i}.cbm"))
    for i, m in enumerate(lgb_models):
        m.save_model(os.path.join(model_dir, f"lgb_{i}.txt"))

    cat_vocab = {c: sorted(X[c].astype(str).unique().tolist()) for c in cat_cols}

    meta = {
        "feature_columns": feature_columns,
        "cat_cols": cat_cols,
        "cat_vocab": cat_vocab,
        "idx_to_class": {
            str(k): (v.item() if hasattr(v, "item") else v)
            for k, v in idx_to_class.items()
        },
        "n_classes": int(n_classes),
        "n_cb": len(cb_models),
        "n_lgb": len(lgb_models),
        "has_lgb": HAS_LGB,
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


# ---------- Prediction ----------


def predict(args):
    data_dir = args.data
    model_dir = args.model
    output_path = args.output

    with open(os.path.join(model_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    feature_columns = meta["feature_columns"]
    cat_cols = meta["cat_cols"]
    cat_vocab = meta.get("cat_vocab", {})
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    n_classes = meta["n_classes"]
    n_cb = meta["n_cb"]
    n_lgb = meta.get("n_lgb", 0)

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X, _ = build_features(features, feature_columns)

    proba_sum = np.zeros((len(X), n_classes), dtype=float)
    count = 0
    for i in range(n_cb):
        cb = CatBoostClassifier()
        cb.load_model(os.path.join(model_dir, f"cb_{i}.cbm"))
        p = cb.predict_proba(X)
        if p.ndim == 1:
            p = np.column_stack([1 - p, p])
        proba_sum += p
        count += 1

    if n_lgb > 0:
        if not HAS_LGB:
            raise RuntimeError(
                "Model was trained with LightGBM but lightgbm is not importable at predict time."
            )
        X_l = X.copy()
        for c in cat_cols:
            cats = cat_vocab.get(c, None)
            if cats is not None:
                X_l[c] = pd.Categorical(X_l[c].astype(str), categories=cats)
            else:
                X_l[c] = X_l[c].astype("category")
        for i in range(n_lgb):
            booster = lgb.Booster(model_file=os.path.join(model_dir, f"lgb_{i}.txt"))
            p = booster.predict(X_l)
            if n_classes == 2:
                p = np.asarray(p).ravel()
                p = np.column_stack([1 - p, p])
            proba_sum += p
            count += 1

    proba = proba_sum / max(count, 1)
    preds = np.argmax(proba, axis=1).astype(int)
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