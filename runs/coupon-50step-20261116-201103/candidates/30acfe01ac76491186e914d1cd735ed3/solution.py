import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

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


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def get_feature_columns(schema, features_df):
    cols = schema.get("feature_columns")
    if cols:
        return [c for c in cols if c in features_df.columns]
    return [c for c in features_df.columns if c != "row_id"]


def engineer(df, feature_cols):
    """Return (cat_df, num_df) with engineered features. Stateless."""
    out = pd.DataFrame(index=df.index)
    cat_cols = []
    num_cols = []

    freq_cols = [
        "Bar",
        "CoffeeHouse",
        "CarryAway",
        "RestaurantLessThan20",
        "Restaurant20To50",
    ]
    for c in freq_cols:
        if c in df.columns:
            out[c + "_ord"] = df[c].map(FREQ_MAP).astype(float)
            num_cols.append(c + "_ord")

    if "income" in df.columns:
        out["income_ord"] = df["income"].map(INCOME_MAP).astype(float)
        num_cols.append("income_ord")

    if "age" in df.columns:
        out["age_ord"] = df["age"].map(AGE_MAP).astype(float)
        num_cols.append("age_ord")

    if "time" in df.columns:
        out["time_ord"] = df["time"].map(TIME_MAP).astype(float)
        num_cols.append("time_ord")

    if "temperature" in df.columns:
        out["temperature_num"] = pd.to_numeric(
            df["temperature"], errors="coerce"
        ).astype(float)
        num_cols.append("temperature_num")

    freq_present = [c for c in freq_cols if c in df.columns]
    if freq_present:
        ord_matrix = np.column_stack(
            [df[c].map(FREQ_MAP).fillna(0).values for c in freq_present]
        )
        out["activity_count"] = (ord_matrix >= 2).sum(axis=1).astype(float)
        out["activity_sum"] = ord_matrix.sum(axis=1).astype(float)
        num_cols.extend(["activity_count", "activity_sum"])

    if "coupon" in df.columns and "time" in df.columns:
        out["coupon_time"] = df["coupon"].astype(str) + "|" + df["time"].astype(str)
        cat_cols.append("coupon_time")

    if "coupon" in df.columns and "expiration" in df.columns:
        out["coupon_exp"] = (
            df["coupon"].astype(str) + "|" + df["expiration"].astype(str)
        )
        cat_cols.append("coupon_exp")

    geq_cols = ["toCoupon_GEQ5min", "toCoupon_GEQ15min", "toCoupon_GEQ25min"]
    if all(c in df.columns for c in geq_cols):
        g = df[geq_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        out["dist_bucket"] = g.sum(axis=1).astype(float)
        num_cols.append("dist_bucket")

    passthrough_cats = [
        "coupon",
        "expiration",
        "gender",
        "maritalStatus",
        "education",
        "occupation",
        "has_children",
    ]
    for c in passthrough_cats:
        if c in df.columns:
            out[c] = df[c].astype(str)
            cat_cols.append(c)

    for c in [
        "direction_same",
        "direction_opp",
        "toCoupon_GEQ5min",
        "toCoupon_GEQ15min",
        "toCoupon_GEQ25min",
    ]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
            num_cols.append(c)

    return out, cat_cols, num_cols


def train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    feature_cols = get_feature_columns(schema, features)
    X_raw = features[feature_cols].copy()
    y_raw = labels["label"].astype(str).values

    classes = sorted(np.unique(y_raw).tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[v] for v in y_raw], dtype=np.int64)

    X_eng, cat_cols, num_cols = engineer(X_raw, feature_cols)

    for c in cat_cols:
        X_eng[c] = X_eng[c].fillna("__NA__").astype(str)
    for c in num_cols:
        X_eng[c] = X_eng[c].fillna(-999.0)

    # Persist category levels learned on training data for LightGBM consistency
    cat_categories = {c: sorted(X_eng[c].astype(str).unique().tolist()) for c in cat_cols}

    from sklearn.utils.class_weight import compute_sample_weight

    sample_w = compute_sample_weight("balanced", y)

    models = {}

    try:
        from catboost import CatBoostClassifier

        cat_idx = [X_eng.columns.get_loc(c) for c in cat_cols]
        X_cb = X_eng.copy()
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
            cat_features=cat_idx,
        )
        cb.fit(X_cb, y, sample_weight=sample_w)
        models["catboost"] = cb
    except Exception as e:
        print("CatBoost failed:", e)

    try:
        import lightgbm as lgb

        X_lgb = X_eng.copy()
        for c in cat_cols:
            X_lgb[c] = pd.Categorical(X_lgb[c].astype(str), categories=cat_categories[c])
        lgbm = lgb.LGBMClassifier(
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
        lgbm.fit(X_lgb, y, sample_weight=sample_w, categorical_feature=cat_cols)
        models["lightgbm"] = lgbm
        models["lgb_cat_cols"] = cat_cols
    except Exception as e:
        print("LightGBM failed:", e)

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_hgb = X_eng.copy()
        if cat_cols:
            X_hgb[cat_cols] = enc.fit_transform(X_hgb[cat_cols].astype(str))
        X_hgb = X_hgb[num_cols + cat_cols].values.astype(float)
        hgb = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=None,
            random_state=42,
        )
        hgb.fit(X_hgb, y, sample_weight=sample_w)
        models["hgb"] = hgb
        models["hgb_enc"] = enc
        models["hgb_cols"] = num_cols + cat_cols
    except Exception as e:
        print("HGB failed:", e)

    with open(os.path.join(model_dir, "label_mapping.json"), "w") as f:
        json.dump({"classes": [str(c) for c in classes]}, f)
    with open(os.path.join(model_dir, "columns.json"), "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "cat_cols": cat_cols,
                "num_cols": num_cols,
                "eng_cols": list(X_eng.columns),
                "cat_categories": cat_categories,
            },
            f,
        )

    with open(os.path.join(model_dir, "pipeline.pkl"), "wb") as f:
        pickle.dump(models, f)


def predict(data_dir, model_dir, output_path):
    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))

    with open(os.path.join(model_dir, "columns.json"), "r") as f:
        col_info = json.load(f)
    with open(os.path.join(model_dir, "label_mapping.json"), "r") as f:
        label_info = json.load(f)
    with open(os.path.join(model_dir, "pipeline.pkl"), "rb") as f:
        models = pickle.load(f)

    feature_cols = col_info["feature_cols"]
    cat_cols = col_info["cat_cols"]
    num_cols = col_info["num_cols"]
    eng_cols = col_info["eng_cols"]
    cat_categories = col_info.get("cat_categories", {})
    classes = label_info["classes"]

    X_raw = features.reindex(columns=feature_cols)
    X_eng, _, _ = engineer(X_raw, feature_cols)
    X_eng = X_eng.reindex(columns=eng_cols)

    for c in cat_cols:
        X_eng[c] = X_eng[c].fillna("__NA__").astype(str)
    for c in num_cols:
        X_eng[c] = X_eng[c].fillna(-999.0)

    proba_list = []
    weights = []

    if "catboost" in models:
        X_cb = X_eng.copy()
        for c in cat_cols:
            X_cb[c] = X_cb[c].astype(str)
        p = models["catboost"].predict_proba(X_cb)
        proba_list.append(p)
        weights.append(1.0)

    if "lightgbm" in models:
        X_lgb = X_eng.copy()
        for c in cat_cols:
            cats = cat_categories.get(c)
            if cats is not None:
                X_lgb[c] = pd.Categorical(X_lgb[c].astype(str), categories=cats)
            else:
                X_lgb[c] = X_lgb[c].astype("category")
        p = models["lightgbm"].predict_proba(X_lgb)
        proba_list.append(p)
        weights.append(1.0)

    if "hgb" in models:
        enc = models["hgb_enc"]
        hgb_cols = models["hgb_cols"]
        X_hgb = X_eng.copy()
        if cat_cols:
            X_hgb[cat_cols] = enc.transform(X_hgb[cat_cols].astype(str))
        X_hgb = X_hgb[hgb_cols].values.astype(float)
        p = models["hgb"].predict_proba(X_hgb)
        proba_list.append(p)
        weights.append(0.8)

    if not proba_list:
        raise RuntimeError("No models available for prediction")

    w = np.array(weights)
    w = w / w.sum()
    proba = np.zeros_like(proba_list[0])
    for p, wi in zip(proba_list, w):
        proba += wi * p

    preds_idx = np.argmax(proba, axis=1)
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