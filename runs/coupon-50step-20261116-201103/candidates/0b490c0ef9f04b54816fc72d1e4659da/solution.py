import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

# Frequency ordinal maps
FREQ_MAP = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
INCOME_MAP = {
    "Less than $12500": 0,
    "$12500 - $24999": 1,
    "$25000 - $37499": 2,
    "$37500 - $49999": 3,
    "$50000 - $62499": 4,
    "$62500 - $74999": 5,
    "$75000 - $87499": 6,
    "$87500 - $99999": 7,
    "$100000 or More": 8,
}
AGE_MAP = {
    "below21": 0,
    "21": 1,
    "26": 2,
    "31": 3,
    "36": 4,
    "41": 5,
    "46": 6,
    "50plus": 7,
}
TIME_MAP = {"7AM": 7, "10AM": 10, "2PM": 14, "6PM": 18, "10PM": 22}
FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def get_feature_columns(schema, features_df):
    cols = schema.get("feature_columns")
    if cols:
        return [c for c in cols if c in features_df.columns]
    return [c for c in features_df.columns if c != "row_id"]


def engineer_features(df):
    out = df.copy()
    # Ordinal encodings
    for c in FREQ_COLS:
        if c in out.columns:
            out[c + "_ord"] = out[c].map(FREQ_MAP).fillna(-1).astype(int)
    if "income" in out.columns:
        out["income_ord"] = out["income"].map(INCOME_MAP).fillna(-1).astype(int)
    if "age" in out.columns:
        out["age_ord"] = out["age"].map(AGE_MAP).fillna(-1).astype(int)
    if "time" in out.columns:
        out["time_ord"] = out["time"].map(TIME_MAP).fillna(-1).astype(int)
    if "temperature" in out.columns:
        out["temperature_num"] = (
            pd.to_numeric(out["temperature"], errors="coerce").fillna(-1).astype(int)
        )

    # Activity counts
    freq_ord_cols = [c + "_ord" for c in FREQ_COLS if c + "_ord" in out.columns]
    if freq_ord_cols:
        out["activity_sum"] = out[freq_ord_cols].clip(lower=0).sum(axis=1)
        out["activity_any"] = (out[freq_ord_cols].clip(lower=0) > 0).sum(axis=1)

    # Coupon x time interaction
    if "coupon" in out.columns and "time_ord" in out.columns:
        out["coupon_time"] = (
            out["coupon"].astype(str) + "_" + out["time_ord"].astype(str)
        )
    if "coupon" in out.columns and "expiration" in out.columns:
        out["coupon_exp"] = (
            out["coupon"].astype(str) + "_" + out["expiration"].astype(str)
        )

    # Direction consistency
    if "direction_same" in out.columns and "direction_opp" in out.columns:
        ds = pd.to_numeric(out["direction_same"], errors="coerce").fillna(0)
        do = pd.to_numeric(out["direction_opp"], errors="coerce").fillna(0)
        out["direction_same_num"] = ds.astype(int)
        out["direction_opp_num"] = do.astype(int)

    # Distance bucket
    if "toCoupon_GEQ15min" in out.columns and "toCoupon_GEQ25min" in out.columns:
        g15 = (
            pd.to_numeric(out["toCoupon_GEQ15min"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        g25 = (
            pd.to_numeric(out["toCoupon_GEQ25min"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        out["dist_bucket"] = g15 + g25

    return out


def build_matrices(train_df, other_dfs, feature_cols):
    """Return list of (X, cat_cols, num_cols) for each df, using consistent columns."""
    # Determine cat/num on engineered train
    cat_cols = []
    num_cols = []
    for c in feature_cols:
        if c not in train_df.columns:
            continue
        s = train_df[c]
        if (
            s.dtype == object
            or str(s.dtype) == "str"
            or str(s.dtype).startswith("string")
        ):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return cat_cols, num_cols


def prepare_X(df, feature_cols, cat_cols, num_cols):
    X = pd.DataFrame(index=df.index)
    for c in cat_cols:
        X[c] = df[c].astype(str).fillna("__NA__") if c in df.columns else "__NA__"
    for c in num_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(float)
        else:
            X[c] = -1.0
    return X


def train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    base_cols = get_feature_columns(schema, features)
    X_raw = features[base_cols].copy()
    X_eng = engineer_features(X_raw)

    # All engineered columns become features
    feature_cols = list(X_eng.columns)

    y_raw = labels["label"].astype(str).values
    classes = np.unique(y_raw)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[v] for v in y_raw], dtype=np.int64)

    cat_cols, num_cols = build_matrices(X_eng, [X_eng], feature_cols)
    X = prepare_X(X_eng, feature_cols, cat_cols, num_cols)

    # Class weights
    counts = np.bincount(y, minlength=len(classes)).astype(float)
    class_weight = {
        i: (counts.sum() / (len(classes) * counts[i])) for i in range(len(classes))
    }
    sample_weight = np.array([class_weight[int(v)] for v in y], dtype=float)

    models = {}

    # CatBoost
    try:
        from catboost import CatBoostClassifier, Pool

        cat_idx = [X.columns.get_loc(c) for c in cat_cols]
        X_cb = X.copy()
        for c in cat_cols:
            X_cb[c] = X_cb[c].astype(str)
        cb = CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            thread_count=2,
            verbose=False,
            loss_function="Logloss",
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
        )
        cb.fit(X_cb, y, cat_features=cat_idx, sample_weight=sample_weight)
        models["catboost"] = cb
    except Exception as e:
        print("CatBoost failed:", e)

    # LightGBM
    try:
        import lightgbm as lgb

        X_lgb = X.copy()
        for c in cat_cols:
            X_lgb[c] = X_lgb[c].astype("category")
        lg = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2,
            verbose=-1,
        )
        lg.fit(X_lgb, y, categorical_feature=cat_cols, sample_weight=sample_weight)
        models["lightgbm"] = lg
    except Exception as e:
        print("LightGBM failed:", e)

    # HistGradientBoosting fallback / third member
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat_enc = (
            enc.fit_transform(X[cat_cols].astype(str))
            if cat_cols
            else np.empty((len(X), 0))
        )
        X_num_arr = (
            X[num_cols].values.astype(float) if num_cols else np.empty((len(X), 0))
        )
        X_hgb = (
            np.hstack([X_num_arr, X_cat_enc])
            if X_num_arr.shape[1] and X_cat_enc.shape[1]
            else (X_num_arr if X_num_arr.shape[1] else X_cat_enc)
        )
        hgb = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=None,
            random_state=42,
        )
        hgb.fit(X_hgb, y, sample_weight=sample_weight)
        models["hgb"] = {"model": hgb, "encoder": enc}
    except Exception as e:
        print("HGB failed:", e)

    with open(os.path.join(model_dir, "label_mapping.json"), "w") as f:
        json.dump({"classes": [str(c) for c in classes]}, f)
    with open(os.path.join(model_dir, "columns.json"), "w") as f:
        json.dump(
            {"feature_cols": feature_cols, "cat_cols": cat_cols, "num_cols": num_cols},
            f,
        )

    with open(os.path.join(model_dir, "pipeline.pkl"), "wb") as f:
        pickle.dump({"models": models}, f)


def predict(data_dir, model_dir, output_path):
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
    classes = label_info["classes"]
    models = pipeline["models"]

    base_cols = [c for c in feature_cols if c in features.columns]
    X_raw = features.reindex(
        columns=[c for c in feature_cols if c in features.columns]
    ).copy()
    # Rebuild engineered features from raw schema columns
    schema_cols = get_feature_columns(schema, features)
    X_eng = engineer_features(features[schema_cols].copy())
    X = prepare_X(X_eng, feature_cols, cat_cols, num_cols)

    probas = []
    weights = []

    if "catboost" in models:
        cb = models["catboost"]
        X_cb = X.copy()
        for c in cat_cols:
            X_cb[c] = X_cb[c].astype(str)
        p = cb.predict_proba(X_cb)
        probas.append(p)
        weights.append(1.0)

    if "lightgbm" in models:
        lg = models["lightgbm"]
        X_lgb = X.copy()
        for c in cat_cols:
            X_lgb[c] = X_lgb[c].astype("category")
        p = lg.predict_proba(X_lgb)
        probas.append(p)
        weights.append(1.0)

    if "hgb" in models:
        hgb_info = models["hgb"]
        hgb = hgb_info["model"]
        enc = hgb_info["encoder"]
        X_cat_enc = (
            enc.transform(X[cat_cols].astype(str))
            if cat_cols
            else np.empty((len(X), 0))
        )
        X_num_arr = (
            X[num_cols].values.astype(float) if num_cols else np.empty((len(X), 0))
        )
        X_hgb = (
            np.hstack([X_num_arr, X_cat_enc])
            if X_num_arr.shape[1] and X_cat_enc.shape[1]
            else (X_num_arr if X_num_arr.shape[1] else X_cat_enc)
        )
        p = hgb.predict_proba(X_hgb)
        probas.append(p)
        weights.append(1.0)

    if not probas:
        raise RuntimeError("No models available")

    w = np.array(weights) / np.sum(weights)
    avg = np.zeros_like(probas[0])
    for wi, p in zip(w, probas):
        avg += wi * p
    preds_idx = np.argmax(avg, axis=1)
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
