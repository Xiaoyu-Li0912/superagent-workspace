import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

try:
    from catboost import CatBoostClassifier

    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False

SEED = 42
N_FOLDS = 5

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
EDU_MAP = {
    "Some High School": 0,
    "High School Graduate": 1,
    "Some college - no degree": 2,
    "Associates degree": 3,
    "Bachelors degree": 4,
    "Graduate degree (Masters, Doctorate)": 5,
}
FREQ_MAP = {
    "never": 0,
    "less1": 1,
    "1~3": 2,
    "4~8": 3,
    "gt8": 4,
}
TIME_MAP = {
    "7AM": 7,
    "10AM": 10,
    "2PM": 14,
    "6PM": 18,
    "10PM": 22,
}
EXP_MAP = {"2h": 0, "1d": 1}

FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]
COUPON_AFFINITY = {
    "Bar": "Bar",
    "Coffee House": "CoffeeHouse",
    "Carry out & Take away": "CarryAway",
    "Restaurant(<20)": "RestaurantLessThan20",
    "Restaurant(20-50)": "Restaurant20To50",
}


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def get_feature_columns(schema, df):
    cols = schema.get("feature_columns")
    if cols:
        return [c for c in cols if c in df.columns]
    return list(df.columns)


def engineer(df):
    out = pd.DataFrame(index=df.index)

    for c in df.columns:
        out[c] = df[c].astype(object).where(df[c].notna(), "__MISSING__").astype(str)

    if "temperature" in df.columns:
        out["temp_num"] = pd.to_numeric(df["temperature"], errors="coerce")
    else:
        out["temp_num"] = np.nan

    out["income_ord"] = df["income"].map(INCOME_MAP).astype(float) if "income" in df.columns else np.nan
    out["age_ord"] = df["age"].map(AGE_MAP).astype(float) if "age" in df.columns else np.nan
    out["edu_ord"] = df["education"].map(EDU_MAP).astype(float) if "education" in df.columns else np.nan
    out["time_hour"] = df["time"].map(TIME_MAP).astype(float) if "time" in df.columns else np.nan
    out["exp_ord"] = df["expiration"].map(EXP_MAP).astype(float) if "expiration" in df.columns else np.nan

    for c in FREQ_COLS:
        if c in df.columns:
            out[c + "_ord"] = df[c].map(FREQ_MAP).astype(float)
        else:
            out[c + "_ord"] = np.nan

    geq5 = pd.to_numeric(df["toCoupon_GEQ5min"], errors="coerce").fillna(0) if "toCoupon_GEQ5min" in df.columns else pd.Series(0, index=df.index)
    geq15 = pd.to_numeric(df["toCoupon_GEQ15min"], errors="coerce").fillna(0) if "toCoupon_GEQ15min" in df.columns else pd.Series(0, index=df.index)
    geq25 = pd.to_numeric(df["toCoupon_GEQ25min"], errors="coerce").fillna(0) if "toCoupon_GEQ25min" in df.columns else pd.Series(0, index=df.index)
    out["dist_bucket"] = geq5 + geq15 + geq25

    coupon = df["coupon"].astype(str) if "coupon" in df.columns else pd.Series("", index=df.index)
    for coup_val, freq_col in COUPON_AFFINITY.items():
        is_coup = (coupon == coup_val).astype(int)
        if freq_col in df.columns:
            freq_val = df[freq_col].map(FREQ_MAP).fillna(-1)
        else:
            freq_val = pd.Series(-1, index=df.index)
        out[f"aff_{freq_col}"] = is_coup * freq_val
        out[f"is_{freq_col}"] = is_coup

    habit_sum = np.zeros(len(df))
    for c in FREQ_COLS:
        if c in df.columns:
            habit_sum = habit_sum + df[c].map(FREQ_MAP).fillna(0).values
    out["habit_sum"] = habit_sum

    if "has_children" in df.columns and "maritalStatus" in df.columns:
        out["children_marital"] = out["has_children"] + "_" + out["maritalStatus"]
    else:
        out["children_marital"] = "__MISSING__"

    return out


def get_cat_cols(df_eng):
    cat_cols = []
    for c in df_eng.columns:
        if df_eng[c].dtype == object:
            cat_cols.append(c)
    return cat_cols


def build_models():
    models = []
    if HAS_CAT:
        models.append(
            (
                "cat",
                CatBoostClassifier(
                    iterations=600,
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=3.0,
                    random_seed=SEED,
                    verbose=False,
                    thread_count=2,
                    loss_function="Logloss",
                    auto_class_weights="Balanced",
                ),
            )
        )
    if HAS_LGB:
        models.append(
            (
                "lgb",
                lgb.LGBMClassifier(
                    n_estimators=600,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=SEED,
                    n_jobs=2,
                    verbose=-1,
                    class_weight="balanced",
                ),
            )
        )
    models.append(
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.06,
                max_depth=None,
                l2_regularization=1.0,
                random_state=SEED,
                class_weight="balanced",
            ),
        )
    )
    return models


def make_hgb_matrix(eng, cat_cols, num_cols, enc, fit=False):
    if fit:
        X_ord = enc.fit_transform(eng[cat_cols]).astype(np.float64)
    else:
        X_ord = enc.transform(eng[cat_cols]).astype(np.float64)
    X_num = eng[num_cols].values.astype(np.float64)
    return np.hstack([X_num, X_ord])


def cmd_train(args):
    data_dir = args.data
    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)

    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    feature_cols = get_feature_columns(schema, features)
    label_col = "label" if "label" in labels.columns else labels.columns[0]
    y_raw = labels[label_col].values

    classes = np.unique(y_raw)
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[str(v)] for v in y_raw], dtype=np.int64)

    eng = engineer(features[feature_cols])
    cat_cols = get_cat_cols(eng)
    cat_idx = [eng.columns.get_loc(c) for c in cat_cols]
    num_cols = [c for c in eng.columns if c not in cat_cols]

    X_tree = eng.copy()
    for c in cat_cols:
        X_tree[c] = X_tree[c].astype(str)

    models = build_models()
    model_names = [n for n, _ in models]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), len(models)))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tree, y)):
        fold_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_hgb_tr = make_hgb_matrix(eng.iloc[tr_idx], cat_cols, num_cols, fold_enc, fit=True)
        X_hgb_va = make_hgb_matrix(eng.iloc[va_idx], cat_cols, num_cols, fold_enc, fit=False)
        for mi, (name, _) in enumerate(models):
            m = build_models()[mi][1]
            if name == "cat":
                m.fit(X_tree.iloc[tr_idx], y[tr_idx], cat_features=cat_idx)
                oof[va_idx, mi] = m.predict_proba(X_tree.iloc[va_idx])[:, 1]
            elif name == "lgb":
                m.fit(X_tree.iloc[tr_idx], y[tr_idx], categorical_feature=cat_idx)
                oof[va_idx, mi] = m.predict_proba(X_tree.iloc[va_idx])[:, 1]
            else:
                m.fit(X_hgb_tr, y[tr_idx])
                oof[va_idx, mi] = m.predict_proba(X_hgb_va)[:, 1]

    stacker = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    stacker.fit(oof, y)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_hgb_full = make_hgb_matrix(eng, cat_cols, num_cols, enc, fit=True)

    final_models = {}
    for mi, (name, _) in enumerate(models):
        m = build_models()[mi][1]
        if name == "cat":
            m.fit(X_tree, y, cat_features=cat_idx)
        elif name == "lgb":
            m.fit(X_tree, y, categorical_feature=cat_idx)
        else:
            m.fit(X_hgb_full, y)
        final_models[name] = m

    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(model_dir, "label_mapping.json"), "w") as f:
        json.dump([c.item() if hasattr(c, "item") else c for c in classes], f)
    with open(os.path.join(model_dir, "cat_cols.json"), "w") as f:
        json.dump(cat_cols, f)
    with open(os.path.join(model_dir, "num_cols.json"), "w") as f:
        json.dump(num_cols, f)
    with open(os.path.join(model_dir, "model_names.json"), "w") as f:
        json.dump(model_names, f)
    with open(os.path.join(model_dir, "encoder.pkl"), "wb") as f:
        pickle.dump(enc, f)
    with open(os.path.join(model_dir, "stacker.pkl"), "wb") as f:
        pickle.dump(stacker, f)
    with open(os.path.join(model_dir, "models.pkl"), "wb") as f:
        pickle.dump(final_models, f)


def cmd_predict(args):
    data_dir = args.data
    model_dir = args.model
    output_path = args.output

    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))

    with open(os.path.join(model_dir, "feature_columns.json"), "r") as f:
        feature_cols = json.load(f)
    with open(os.path.join(model_dir, "label_mapping.json"), "r") as f:
        classes = json.load(f)
    with open(os.path.join(model_dir, "cat_cols.json"), "r") as f:
        cat_cols = json.load(f)
    with open(os.path.join(model_dir, "num_cols.json"), "r") as f:
        num_cols = json.load(f)
    with open(os.path.join(model_dir, "model_names.json"), "r") as f:
        model_names = json.load(f)
    with open(os.path.join(model_dir, "encoder.pkl"), "rb") as f:
        enc = pickle.load(f)
    with open(os.path.join(model_dir, "stacker.pkl"), "rb") as f:
        stacker = pickle.load(f)
    with open(os.path.join(model_dir, "models.pkl"), "rb") as f:
        final_models = pickle.load(f)

    eng = engineer(features[feature_cols])

    X_hgb = make_hgb_matrix(eng, cat_cols, num_cols, enc, fit=False)

    X_tree = eng.copy()
    for c in cat_cols:
        X_tree[c] = X_tree[c].astype(str)

    proba = np.zeros((len(eng), len(model_names)))
    for mi, name in enumerate(model_names):
        m = final_models[name]
        if name in ("cat", "lgb"):
            proba[:, mi] = m.predict_proba(X_tree)[:, 1]
        else:
            proba[:, mi] = m.predict_proba(X_hgb)[:, 1]

    final_proba = stacker.predict_proba(proba)[:, 1]
    preds_idx = (final_proba >= 0.5).astype(int)
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