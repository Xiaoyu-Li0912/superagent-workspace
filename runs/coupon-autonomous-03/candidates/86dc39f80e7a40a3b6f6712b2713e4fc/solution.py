import argparse, json, os, sys
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

# ---------- Feature engineering ----------
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
    if "Less" in c or "<20" in c:
        return "RestaurantLessThan20"
    return "RestaurantLessThan20"


def engineer(df):
    df = df.copy()
    # ordinal encodings
    for c in HABIT_COLS:
        df[c + "_ord"] = df[c].map(FREQ_MAP).astype(float)
    df["age_ord"] = df["age"].astype(str).map(AGE_MAP).astype(float)
    df["income_ord"] = df["income"].astype(str).map(INCOME_MAP).astype(float)
    df["time_ord"] = df["time"].astype(str).map(TIME_MAP).astype(float)
    df["exp_ord"] = df["expiration"].astype(str).map(EXP_MAP).astype(float)
    df["temp_num"] = pd.to_numeric(df["temperature"], errors="coerce").astype(float)
    df["has_children"] = pd.to_numeric(df["has_children"], errors="coerce").astype(
        float
    )
    for c in [
        "toCoupon_GEQ5min",
        "toCoupon_GEQ15min",
        "toCoupon_GEQ25min",
        "direction_same",
        "direction_opp",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # coupon-habit match: does the customer already frequent the coupon's venue?
    habit_for_coupon = df["coupon"].map(coupon_to_habit)
    match_vals = []
    for i, h in enumerate(habit_for_coupon.values):
        match_vals.append(df.iloc[i][h + "_ord"] if h in HABIT_COLS else np.nan)
    df["coupon_habit_ord"] = match_vals
    df["coupon_habit_match"] = (df["coupon_habit_ord"].fillna(-1) >= 2).astype(float)

    # distance feasibility: coupon requires travel
    df["dist_min"] = (
        df["toCoupon_GEQ5min"].fillna(0)
        + df["toCoupon_GEQ15min"].fillna(0)
        + df["toCoupon_GEQ25min"].fillna(0)
    )
    df["dist_far"] = (df["toCoupon_GEQ25min"].fillna(0) > 0).astype(float)
    df["same_dir"] = df["direction_same"].fillna(0)

    # coupon x time interaction (bar/restaurant more likely in evening)
    df["is_bar"] = df["coupon"].astype(str).str.contains("Bar").astype(float)
    df["is_coffee"] = df["coupon"].astype(str).str.contains("Coffee").astype(float)
    df["is_cheap_rest"] = (
        df["coupon"].astype(str).str.contains("Less|<20", regex=True).astype(float)
    )
    df["is_exp_rest"] = (
        df["coupon"].astype(str).str.contains("20-50|20To50", regex=True).astype(float)
    )
    df["is_carry"] = (
        df["coupon"].astype(str).str.contains("Carry|Take", regex=True).astype(float)
    )
    df["evening"] = (df["time_ord"] >= 17).astype(float)
    df["bar_evening"] = df["is_bar"] * df["evening"]
    df["coffee_morning"] = df["is_coffee"] * (df["time_ord"] <= 10).astype(float)
    df["exp_short"] = (df["exp_ord"] == 0).astype(float)

    # social: children + marital
    df["single"] = (df["maritalStatus"].astype(str) == "Single").astype(float)
    df["has_kids"] = df["has_children"].fillna(0)
    df["kid_friendly"] = df["has_kids"] * (df["is_cheap_rest"] + df["is_carry"])

    # overall habit engagement
    df["habit_sum"] = df[[c + "_ord" for c in HABIT_COLS]].fillna(0).sum(axis=1)
    df["habit_mean"] = df[[c + "_ord" for c in HABIT_COLS]].mean(axis=1)

    return df


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


def build_matrix(df, feature_cols):
    X = df[feature_cols].copy()
    # ensure categoricals are strings and NaN -> "missing"
    for c in CAT_FEATURES:
        if c in X.columns:
            X[c] = (
                X[c]
                .fillna("missing")
                .astype(str)
                .replace({"nan": "missing", "None": "missing"})
            )
    # numeric fill
    for c in X.columns:
        if c not in CAT_FEATURES:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def get_cat_idx(feature_cols):
    return [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]


# ---------- Train ----------
def train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    feats = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    y = labels["label"].values

    feats_eng = engineer(feats)
    # feature set: raw categoricals + engineered numerics
    eng_numeric = [c for c in feats_eng.columns if c not in feats.columns]
    feature_cols = list(feats.columns) + eng_numeric
    X = build_matrix(feats_eng, feature_cols)
    cat_idx = get_cat_idx(feature_cols)

    # CV to sanity-check (no metric printed per contract; just for internal model selection)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for tr, va in skf.split(X, y):
        m = CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
            thread_count=2,
            cat_features=cat_idx,
            bootstrap_type="Bayesian",
            bagging_temperature=0.5,
        )
        m.fit(X.iloc[tr], y[tr])
        pred = m.predict(X.iloc[va]).ravel()
        cv_scores.append((pred == y[va]).mean())
    cv_mean = float(np.mean(cv_scores))
    print(f"CV accuracy (internal): {cv_mean:.4f}")

    # final model on all training data
    final = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        thread_count=2,
        cat_features=cat_idx,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
    )
    final.fit(X, y)

    # label mapping (original labels)
    classes = np.unique(y).tolist()
    joblib.dump(
        {
            "model": final,
            "feature_cols": feature_cols,
            "cat_features": CAT_FEATURES,
            "classes": classes,
        },
        os.path.join(model_dir, "pipeline.joblib"),
    )
    print("Model saved.")


# ---------- Predict ----------
def predict(data_dir, model_dir, output_path):
    bundle = joblib.load(os.path.join(model_dir, "pipeline.joblib"))
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    cat_features = bundle["cat_features"]
    classes = bundle["classes"]

    feats = pd.read_csv(os.path.join(data_dir, "features.csv"))
    feats_eng = engineer(feats)
    X = build_matrix(feats_eng, feature_cols)
    # align columns
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols]

    raw_preds = model.predict(X).ravel()
    # map class indices back to original labels
    preds = [classes[int(p)] for p in raw_preds]
    out = pd.DataFrame({"row_id": np.arange(len(preds)), "prediction": preds})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} predictions to {output_path}")


# ---------- Main ----------
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
    else:
        predict(args.data, args.model, args.output)