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
        out["activity_mean"] = ord_matrix.mean(axis=1).astype(float)
        out["activity_std"] = ord_matrix.std(axis=1).astype(float)
        out["activity_max"] = ord_matrix.max(axis=1).astype(float)
        out["activity_never"] = (ord_matrix == 0).sum(axis=1).astype(float)
        num_cols.extend(
            [
                "activity_count",
                "activity_sum",
                "activity_mean",
                "activity_std",
                "activity_max",
                "activity_never",
            ]
        )

    if "coupon" in df.columns and "time" in df.columns:
        out["coupon_time"] = df["coupon"].astype(str) + "|" + df["time"].astype(str)
        cat_cols.append("coupon_time")

    if "coupon" in df.columns and "expiration" in df.columns:
        out["coupon_exp"] = (
            df["coupon"].astype(str) + "|" + df["expiration"].astype(str)
        )
        cat_cols.append("coupon_exp")

    if "coupon" in df.columns and "has_children" in df.columns:
        out["coupon_children"] = (
            df["coupon"].astype(str) + "|" + df["has_children"].astype(str)
        )
        cat_cols.append("coupon_children")

    if "coupon" in df.columns and "age" in df.columns:
        out["coupon_age"] = df["coupon"].astype(str) + "|" + df["age"].astype(str)
        cat_cols.append("coupon_age")

    if "coupon" in df.columns and "income" in df.columns:
        out["coupon_income"] = df["coupon"].astype(str) + "|" + df["income"].astype(str)
        cat_cols.append("coupon_income")

    if "coupon" in df.columns and "maritalStatus" in df.columns:
        out["coupon_marital"] = (
            df["coupon"].astype(str) + "|" + df["maritalStatus"].astype(str)
        )
        cat_cols.append("coupon_marital")

    if "coupon" in df.columns and "occupation" in df.columns:
        out["coupon_occ"] = (
            df["coupon"].astype(str) + "|" + df["occupation"].astype(str)
        )
        cat_cols.append("coupon_occ")

    if "coupon" in df.columns and "toCoupon_GEQ15min" in df.columns:
        out["coupon_dist15"] = (
            df["coupon"].astype(str) + "|" + df["toCoupon_GEQ15min"].astype(str)
        )
        cat_cols.append("coupon_dist15")

    if "coupon" in df.columns and "direction_same" in df.columns:
        out["coupon_dir"] = (
            df["coupon"].astype(str) + "|" + df["direction_same"].astype(str)
        )
        cat_cols.append("coupon_dir")

    if "time" in df.columns and "expiration" in df.columns:
        out["time_exp"] = df["time"].astype(str) + "|" + df["expiration"].astype(str)
        cat_cols.append("time_exp")

    if "temperature" in df.columns:
        out["temperature_cat"] = df["temperature"].astype(str)
        cat_cols.append("temperature_cat")

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

    cat_categories = {
        c: sorted(X_eng[c].astype(str).unique().tolist()) for c in cat_cols
    }

    from sklearn.model_selection import StratifiedKFold

    # Milder class weighting aligned with accuracy metric (not balanced accuracy)
    class_counts = np.bincount(y)
    n_total = len(y)
    # natural prior weights ~ 1.0; slight boost to minority to help recall
    w0 = 1.0
    w1 = 1.15
    sample_w = np.where(y == 0, w0, w1)

    n_classes = len(classes)
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_cb = np.zeros((len(y), n_classes))
    oof_cb2 = np.zeros((len(y), n_classes))
    oof_lgb = np.zeros((len(y), n_classes))
    oof_lgb2 = np.zeros((len(y), n_classes))
    oof_hgb = np.zeros((len(y), n_classes))
    has_cb = True
    has_lgb = True
    has_hgb = True

    try:
        from catboost import CatBoostClassifier
    except Exception:
        has_cb = False
    try:
        import lightgbm as lgb
    except Exception:
        has_lgb = False
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder
    except Exception:
        has_hgb = False

    cat_idx = [X_eng.columns.get_loc(c) for c in cat_cols]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_eng, y)):
        X_tr = X_eng.iloc[tr_idx]
        X_va = X_eng.iloc[va_idx]
        y_tr = y[tr_idx]
        w_tr = sample_w[tr_idx]

        if has_cb:
            X_cb_tr = X_tr.copy()
            X_cb_va = X_va.copy()
            for c in cat_cols:
                X_cb_tr[c] = X_cb_tr[c].astype(str)
                X_cb_va[c] = X_cb_va[c].astype(str)
            cb = CatBoostClassifier(
                iterations=1500,
                learning_rate=0.04,
                depth=7,
                l2_leaf_reg=4.0,
                random_seed=42,
                thread_count=2,
                verbose=False,
                loss_function="Logloss",
                cat_features=cat_idx,
                early_stopping_rounds=100,
            )
            cb.fit(X_cb_tr, y_tr, sample_weight=w_tr, eval_set=(X_cb_va, y[va_idx]))
            oof_cb[va_idx] = cb.predict_proba(X_cb_va)

            # Diverse CatBoost: Lossguide + CrossEntropy
            cb2 = CatBoostClassifier(
                iterations=1500,
                learning_rate=0.04,
                depth=8,
                grow_policy="Lossguide",
                max_leaves=48,
                l2_leaf_reg=6.0,
                random_seed=2024,
                thread_count=2,
                verbose=False,
                loss_function="CrossEntropy",
                cat_features=cat_idx,
                bootstrap_type="Bayesian",
                bagging_temperature=1.0,
                early_stopping_rounds=100,
            )
            cb2.fit(X_cb_tr, y_tr, sample_weight=w_tr, eval_set=(X_cb_va, y[va_idx]))
            oof_cb2[va_idx] = cb2.predict_proba(X_cb_va)

        if has_lgb:
            X_lgb_tr = X_tr.copy()
            X_lgb_va = X_va.copy()
            for c in cat_cols:
                X_lgb_tr[c] = pd.Categorical(
                    X_lgb_tr[c].astype(str), categories=cat_categories[c]
                )
                X_lgb_va[c] = pd.Categorical(
                    X_lgb_va[c].astype(str), categories=cat_categories[c]
                )
            lgbm = lgb.LGBMClassifier(
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=25,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                cat_smooth=20.0,
                cat_l2=10.0,
                random_state=42,
                n_jobs=2,
                verbose=-1,
            )
            lgbm.fit(
                X_lgb_tr,
                y_tr,
                sample_weight=w_tr,
                categorical_feature=cat_cols,
                eval_set=[(X_lgb_va, y[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            oof_lgb[va_idx] = lgbm.predict_proba(X_lgb_va)

            # Diverse LightGBM: DART boosting
            lgbm2 = lgb.LGBMClassifier(
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=48,
                min_child_samples=20,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_lambda=3.0,
                cat_smooth=20.0,
                cat_l2=10.0,
                boosting_type="dart",
                drop_rate=0.1,
                skip_drop=0.5,
                random_state=2024,
                n_jobs=2,
                verbose=-1,
            )
            lgbm2.fit(
                X_lgb_tr,
                y_tr,
                sample_weight=w_tr,
                categorical_feature=cat_cols,
            )
            oof_lgb2[va_idx] = lgbm2.predict_proba(X_lgb_va)

        if has_hgb:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_hgb_tr = X_tr.copy()
            X_hgb_va = X_va.copy()
            if cat_cols:
                X_hgb_tr[cat_cols] = enc.fit_transform(X_hgb_tr[cat_cols].astype(str))
                X_hgb_va[cat_cols] = enc.transform(X_hgb_va[cat_cols].astype(str))
            cols_order = num_cols + cat_cols
            X_hgb_tr = X_hgb_tr[cols_order].values.astype(float)
            X_hgb_va = X_hgb_va[cols_order].values.astype(float)
            hgb = HistGradientBoostingClassifier(
                max_iter=600,
                learning_rate=0.04,
                max_depth=None,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42,
            )
            hgb.fit(X_hgb_tr, y_tr, sample_weight=w_tr)
            oof_hgb[va_idx] = hgb.predict_proba(X_hgb_va)

    stack_parts = []
    stack_names = []
    if has_cb:
        stack_parts.append(oof_cb)
        stack_names.append("cb")
        stack_parts.append(oof_cb2)
        stack_names.append("cb2")
    if has_lgb:
        stack_parts.append(oof_lgb)
        stack_names.append("lgb")
        stack_parts.append(oof_lgb2)
        stack_names.append("lgb2")
    if has_hgb:
        stack_parts.append(oof_hgb)
        stack_names.append("hgb")

    if not stack_parts:
        raise RuntimeError("No base models trained")

    stack_X = np.hstack(stack_parts)

    # Feature-augmented meta-learner: append a compact set of raw numeric features
    meta_num_cols = [c for c in num_cols if c in X_eng.columns]
    meta_feat = X_eng[meta_num_cols].values.astype(float)
    # standardize for LR stability
    meta_mean = meta_feat.mean(axis=0)
    meta_std = meta_feat.std(axis=0) + 1e-8
    meta_feat_norm = (meta_feat - meta_mean) / meta_std
    stack_X_aug = np.hstack([stack_X, meta_feat_norm])

    from sklearn.linear_model import LogisticRegression

    meta_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    meta_lr.fit(stack_X_aug, y)

    # Second meta-learner: small CatBoost on stacked probs
    meta_cb = None
    if has_cb:
        try:
            meta_cb = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=3,
                l2_leaf_reg=3.0,
                random_seed=42,
                thread_count=2,
                verbose=False,
                loss_function="Logloss",
            )
            meta_cb.fit(stack_X, y)
        except Exception:
            meta_cb = None

    final_models = {}

    if has_cb:
        X_cb_full = X_eng.copy()
        for c in cat_cols:
            X_cb_full[c] = X_cb_full[c].astype(str)
        cb_full = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=4.0,
            random_seed=42,
            thread_count=2,
            verbose=False,
            loss_function="Logloss",
            cat_features=cat_idx,
        )
        cb_full.fit(X_cb_full, y, sample_weight=sample_w)
        final_models["catboost"] = cb_full

        cb2_full = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.04,
            depth=8,
            grow_policy="Lossguide",
            max_leaves=48,
            l2_leaf_reg=6.0,
            random_seed=2024,
            thread_count=2,
            verbose=False,
            loss_function="CrossEntropy",
            cat_features=cat_idx,
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
        )
        cb2_full.fit(X_cb_full, y, sample_weight=sample_w)
        final_models["catboost2"] = cb2_full

    if has_lgb:
        X_lgb_full = X_eng.copy()
        for c in cat_cols:
            X_lgb_full[c] = pd.Categorical(
                X_lgb_full[c].astype(str), categories=cat_categories[c]
            )
        lgbm_full = lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=25,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=2.0,
            cat_smooth=20.0,
            cat_l2=10.0,
            random_state=42,
            n_jobs=2,
            verbose=-1,
        )
        lgbm_full.fit(
            X_lgb_full, y, sample_weight=sample_w, categorical_feature=cat_cols
        )
        final_models["lightgbm"] = lgbm_full

        lgbm2_full = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=48,
            min_child_samples=20,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=3.0,
            cat_smooth=20.0,
            cat_l2=10.0,
            boosting_type="dart",
            drop_rate=0.1,
            skip_drop=0.5,
            random_state=2024,
            n_jobs=2,
            verbose=-1,
        )
        lgbm2_full.fit(
            X_lgb_full, y, sample_weight=sample_w, categorical_feature=cat_cols
        )
        final_models["lightgbm2"] = lgbm2_full

    if has_hgb:
        enc_full = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_hgb_full = X_eng.copy()
        if cat_cols:
            X_hgb_full[cat_cols] = enc_full.fit_transform(
                X_hgb_full[cat_cols].astype(str)
            )
        cols_order = num_cols + cat_cols
        X_hgb_full_arr = X_hgb_full[cols_order].values.astype(float)
        hgb_full = HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.04,
            max_depth=None,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        hgb_full.fit(X_hgb_full_arr, y, sample_weight=sample_w)
        final_models["hgb"] = hgb_full
        final_models["hgb_enc"] = enc_full
        final_models["hgb_cols"] = cols_order

    final_models["meta"] = meta_lr
    final_models["meta_cb"] = meta_cb
    final_models["meta_mean"] = meta_mean
    final_models["meta_std"] = meta_std
    final_models["meta_num_cols"] = meta_num_cols
    final_models["stack_names"] = stack_names

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
        pickle.dump(final_models, f)


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
    stack_names = models.get("stack_names", [])

    X_raw = features.reindex(columns=feature_cols)
    X_eng, _, _ = engineer(X_raw, feature_cols)
    X_eng = X_eng.reindex(columns=eng_cols)

    for c in cat_cols:
        X_eng[c] = X_eng[c].fillna("__NA__").astype(str)
    for c in num_cols:
        X_eng[c] = X_eng[c].fillna(-999.0)

    stack_parts = []

    if "catboost" in models and "cb" in stack_names:
        X_cb = X_eng.copy()
        for c in cat_cols:
            X_cb[c] = X_cb[c].astype(str)
        stack_parts.append(models["catboost"].predict_proba(X_cb))

    if "catboost2" in models and "cb2" in stack_names:
        X_cb = X_eng.copy()
        for c in cat_cols:
            X_cb[c] = X_cb[c].astype(str)
        stack_parts.append(models["catboost2"].predict_proba(X_cb))

    if "lightgbm" in models and "lgb" in stack_names:
        X_lgb = X_eng.copy()
        for c in cat_cols:
            cats = cat_categories.get(c)
            if cats is not None:
                X_lgb[c] = pd.Categorical(X_lgb[c].astype(str), categories=cats)
            else:
                X_lgb[c] = X_lgb[c].astype("category")
        stack_parts.append(models["lightgbm"].predict_proba(X_lgb))

    if "lightgbm2" in models and "lgb2" in stack_names:
        X_lgb = X_eng.copy()
        for c in cat_cols:
            cats = cat_categories.get(c)
            if cats is not None:
                X_lgb[c] = pd.Categorical(X_lgb[c].astype(str), categories=cats)
            else:
                X_lgb[c] = X_lgb[c].astype("category")
        stack_parts.append(models["lightgbm2"].predict_proba(X_lgb))

    if "hgb" in models and "hgb" in stack_names:
        enc = models["hgb_enc"]
        hgb_cols = models["hgb_cols"]
        X_hgb = X_eng.copy()
        if cat_cols:
            X_hgb[cat_cols] = enc.transform(X_hgb[cat_cols].astype(str))
        X_hgb = X_hgb[hgb_cols].values.astype(float)
        stack_parts.append(models["hgb"].predict_proba(X_hgb))

    if not stack_parts:
        raise RuntimeError("No models available for prediction")

    stack_X = np.hstack(stack_parts)

    meta_num_cols = models.get("meta_num_cols", [])
    if meta_num_cols:
        meta_feat = X_eng[meta_num_cols].values.astype(float)
        meta_feat_norm = (meta_feat - models["meta_mean"]) / models["meta_std"]
        stack_X_aug = np.hstack([stack_X, meta_feat_norm])
    else:
        stack_X_aug = stack_X

    proba_lr = models["meta"].predict_proba(stack_X_aug)

    meta_cb = models.get("meta_cb")
    if meta_cb is not None:
        proba_cb = meta_cb.predict_proba(stack_X)
        proba = 0.5 * proba_lr + 0.5 * proba_cb
    else:
        proba = proba_lr

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
