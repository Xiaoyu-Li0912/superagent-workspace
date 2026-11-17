import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import joblib

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# Ordinal encodings (domain knowledge, fixed, not learned from labels)
# ---------------------------------------------------------------------------
FREQ_MAP = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
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
TIME_MAP = {"7AM": 7, "10AM": 10, "2PM": 14, "6PM": 18, "10PM": 22}
EXP_MAP = {"2h": 0, "1d": 1}

HABIT_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]

# Categorical columns that will be passed as native categoricals to models.
CAT_FEATURES = [
    "time",
    "coupon",
    "expiration",
    "gender",
    "age",
    "maritalStatus",
    "education",
    "occupation",
    "income",
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]


def coupon_to_habit(coupon):
    c = str(coupon)
    if "Bar" in c:
        return "Bar"
    if "Coffee" in c:
        return "CoffeeHouse"
    if "Carry" in c or "Take" in c:
        return "CarryAway"
    if "20-50" in c or "20To50" in c or "(20-50)" in c:
        return "Restaurant20To50"
    return "RestaurantLessThan20"


def engineer(df, freq_maps=None, fit=False):
    """Feature engineering. If fit=True, learn frequency maps from df."""
    df = df.copy()

    for c in HABIT_COLS:
        if c in df.columns:
            df[c + "_ord"] = df[c].map(FREQ_MAP).astype(float)
        else:
            df[c + "_ord"] = np.nan

    df["age_ord"] = (
        df["age"].astype(str).map(AGE_MAP).astype(float)
        if "age" in df.columns
        else np.nan
    )
    df["income_ord"] = (
        df["income"].astype(str).map(INCOME_MAP).astype(float)
        if "income" in df.columns
        else np.nan
    )
    df["time_ord"] = (
        df["time"].astype(str).map(TIME_MAP).astype(float)
        if "time" in df.columns
        else np.nan
    )
    df["exp_ord"] = (
        df["expiration"].astype(str).map(EXP_MAP).astype(float)
        if "expiration" in df.columns
        else np.nan
    )

    df["temp_num"] = pd.to_numeric(df.get("temperature"), errors="coerce").astype(float)
    df["has_children_num"] = pd.to_numeric(
        df.get("has_children"), errors="coerce"
    ).astype(float)

    for c in [
        "toCoupon_GEQ5min",
        "toCoupon_GEQ15min",
        "toCoupon_GEQ25min",
        "direction_same",
        "direction_opp",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        else:
            df[c] = np.nan

    # Coupon-habit interaction
    habit_for_coupon = (
        df["coupon"].map(coupon_to_habit)
        if "coupon" in df.columns
        else pd.Series(["RestaurantLessThan20"] * len(df))
    )
    match_vals = []
    for i, h in enumerate(habit_for_coupon.values):
        col = h + "_ord"
        match_vals.append(df.iloc[i][col] if col in df.columns else np.nan)
    df["coupon_habit_ord"] = match_vals
    df["coupon_habit_match"] = (df["coupon_habit_ord"].fillna(-1) >= 2).astype(float)

    # Distance
    df["dist_min"] = (
        df["toCoupon_GEQ5min"].fillna(0)
        + df["toCoupon_GEQ15min"].fillna(0)
        + df["toCoupon_GEQ25min"].fillna(0)
    )
    df["dist_far"] = (df["toCoupon_GEQ25min"].fillna(0) > 0).astype(float)
    df["same_dir"] = df["direction_same"].fillna(0)
    df["dist_decay"] = np.exp(-df["dist_min"].fillna(0))

    # Coupon type flags
    coupon_str = (
        df["coupon"].astype(str)
        if "coupon" in df.columns
        else pd.Series([""] * len(df))
    )
    df["is_bar"] = coupon_str.str.contains("Bar").astype(float)
    df["is_coffee"] = coupon_str.str.contains("Coffee").astype(float)
    df["is_cheap_rest"] = coupon_str.str.contains("Less|<20", regex=True).astype(float)
    df["is_exp_rest"] = coupon_str.str.contains("20-50|20To50", regex=True).astype(
        float
    )
    df["is_carry"] = coupon_str.str.contains("Carry|Take", regex=True).astype(float)

    df["evening"] = (df["time_ord"] >= 17).astype(float)
    df["bar_evening"] = df["is_bar"] * df["evening"]
    df["coffee_morning"] = df["is_coffee"] * (df["time_ord"] <= 10).astype(float)
    df["exp_short"] = (df["exp_ord"] == 0).astype(float)

    df["single"] = (
        (df["maritalStatus"].astype(str) == "Single").astype(float)
        if "maritalStatus" in df.columns
        else 0.0
    )
    df["has_kids"] = df["has_children_num"].fillna(0)
    df["kid_friendly"] = df["has_kids"] * (df["is_cheap_rest"] + df["is_carry"])

    habit_ords = df[[c + "_ord" for c in HABIT_COLS]]
    df["habit_sum"] = habit_ords.fillna(0).sum(axis=1)
    df["habit_mean"] = habit_ords.mean(axis=1)
    df["coupon_habit_ratio"] = df["coupon_habit_ord"] / (df["habit_mean"] + 0.5)
    df["coupon_habit_gap"] = df["coupon_habit_ord"] - df["habit_mean"]

    # Frequency encodings (learned only on training data)
    if freq_maps is None:
        freq_maps = {}
    for c in ["occupation", "coupon", "income", "age"]:
        key = "freq_" + c
        if c in df.columns:
            if fit:
                vc = df[c].astype(str).value_counts(normalize=True)
                freq_maps[key] = vc.to_dict()
            mp = freq_maps.get(key, {})
            df[key] = df[c].astype(str).map(mp).fillna(0.0).astype(float)
        else:
            df[key] = 0.0

    return df, freq_maps


def build_matrix(df, feature_cols):
    """Build X with correct dtypes; missing categoricals -> 'missing'."""
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c in df.columns:
            X[c] = df[c]
        else:
            X[c] = np.nan
    for c in CAT_FEATURES:
        if c in X.columns:
            X[c] = (
                X[c]
                .fillna("missing")
                .astype(str)
                .replace({"nan": "missing", "None": "missing", "": "missing"})
            )
    for c in X.columns:
        if c not in CAT_FEATURES:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def get_cat_idx(feature_cols):
    return [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]


def _fit_catboost(X, y, cat_idx, seed=42, iterations=800, depth=6, lr=0.05):
    m = CatBoostClassifier(
        iterations=iterations,
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        thread_count=2,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
    )
    m.fit(X, y, cat_features=cat_idx)
    return m


def _fit_lgbm(X, y, cat_idx):
    Xl = X.copy()
    cat_categories = {}
    for i in cat_idx:
        c = Xl.columns[i]
        cats = sorted(Xl[c].astype(str).unique().tolist())
        cat_categories[c] = cats
        Xl[c] = pd.Categorical(Xl[c].astype(str), categories=cats)
    m = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
    m.fit(Xl, y, categorical_feature=[Xl.columns[i] for i in cat_idx])
    return m, cat_categories


def _lgbm_predict_proba(m, X, cat_idx, cat_categories):
    Xl = X.copy()
    for i in cat_idx:
        c = Xl.columns[i]
        cats = cat_categories[c]
        Xl[c] = pd.Categorical(Xl[c].astype(str), categories=cats)
    return m.predict_proba(Xl)[:, 1]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    feats = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    y = labels["label"].values

    schema = {}
    schema_path = os.path.join(data_dir, "schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema = json.load(f)

    base_cols = schema.get("feature_columns", None)
    if not base_cols:
        base_cols = [c for c in feats.columns if c != "row_id"]

    feats_eng, freq_maps = engineer(feats, fit=True)
    eng_numeric = [c for c in feats_eng.columns if c not in feats.columns]
    feature_cols = list(base_cols) + eng_numeric
    # keep only columns that actually exist in engineered frame
    feature_cols = [c for c in feature_cols if c in feats_eng.columns]

    X = build_matrix(feats_eng, feature_cols)
    cat_idx = get_cat_idx(feature_cols)

    # Simple holdout for blend weight selection (train-only, no leakage)
    rng = np.random.RandomState(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = max(1, int(0.15 * len(X)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr, Xva = X.iloc[tr_idx], X.iloc[val_idx]
    ytr, yva = y[tr_idx], y[val_idx]

    m_cb = _fit_catboost(Xtr, ytr, cat_idx, seed=42, iterations=600, depth=6, lr=0.05)
    p_cb = m_cb.predict_proba(Xva)[:, 1]
    m_lgb, cat_categories = _fit_lgbm(Xtr, ytr, cat_idx)
    p_lgb = _lgbm_predict_proba(m_lgb, Xva, cat_idx, cat_categories)

    best_w, best_acc = 0.5, -1.0
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * p_cb + (1.0 - w) * p_lgb
        acc = ((blend >= 0.5).astype(int) == yva).mean()
        if acc > best_acc:
            best_acc = acc
            best_w = float(w)

    # Refit on full training data
    final_cb = _fit_catboost(X, y, cat_idx, seed=42, iterations=900, depth=6, lr=0.04)
    final_lgb, cat_categories = _fit_lgbm(X, y, cat_idx)

    classes = np.unique(y).tolist()
    joblib.dump(
        {
            "model_cb": final_cb,
            "model_lgb": final_lgb,
            "weight_cb": best_w,
            "feature_cols": feature_cols,
            "cat_features": CAT_FEATURES,
            "cat_idx": cat_idx,
            "cat_categories": cat_categories,
            "freq_maps": freq_maps,
            "classes": classes,
        },
        os.path.join(model_dir, "pipeline.joblib"),
    )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict(data_dir, model_dir, output_path):
    bundle = joblib.load(os.path.join(model_dir, "pipeline.joblib"))
    feature_cols = bundle["feature_cols"]
    cat_idx = bundle["cat_idx"]
    cat_categories = bundle["cat_categories"]
    freq_maps = bundle["freq_maps"]
    classes = bundle["classes"]
    w_cb = bundle["weight_cb"]

    feats = pd.read_csv(os.path.join(data_dir, "features.csv"))
    feats_eng, _ = engineer(feats, freq_maps=freq_maps, fit=False)
    X = build_matrix(feats_eng, feature_cols)

    p_cb = bundle["model_cb"].predict_proba(X)[:, 1]
    p_lgb = _lgbm_predict_proba(bundle["model_lgb"], X, cat_idx, cat_categories)
    blend = w_cb * p_cb + (1.0 - w_cb) * p_lgb
    pred_idx = (blend >= 0.5).astype(int)
    preds = [classes[int(p)] for p in pred_idx]

    out = pd.DataFrame({"row_id": np.arange(len(preds)), "prediction": preds})
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--data", required=True)
    p_tr.add_argument("--model", required=True)

    p_pr = sub.add_parser("predict")
    p_pr.add_argument("--data", required=True)
    p_pr.add_argument("--model", required=True)
    p_pr.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        train(args.data, args.model)
    elif args.cmd == "predict":
        predict(args.data, args.model, args.output)
    else:
        sys.exit("Unknown command")
