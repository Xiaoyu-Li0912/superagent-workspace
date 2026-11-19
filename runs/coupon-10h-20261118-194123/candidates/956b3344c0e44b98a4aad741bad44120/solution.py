"""
solution.py

In-vehicle coupon acceptance prediction (binary classification, metric = accuracy).

Key improvements over prior version:
  1. Full-data retraining: CV is used only for OOF threshold tuning and to estimate
     the number of boosting iterations. Final models are fit on 100% of train rows
     (previous version kept 80%-trained fold models -> undertrained ensemble).
  2. Heterogeneous blend: CatBoost (native categoricals / ordered target stats)
     + LightGBM (ordinal-encoded categoricals) averaged in logit space.
  3. Extra interaction categoricals: coupon x occupation, coupon x income.

Usage:
  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier, Pool

try:
    import lightgbm as lgb

    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


# =========================================================================== #
# STEP 1: DATA PROCESSING & FEATURE ENGINEERING
# =========================================================================== #
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
EDU_MAP = {
    "Some High School": 0,
    "High School Graduate": 1,
    "Some college - no degree": 2,
    "Associates degree": 3,
    "Bachelors degree": 4,
    "Graduate degree (Masters or Doctorate)": 5,
}
FREQ_MAP = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
# Real hour ordinals: keeps the wall-clock signal monotone for the trees.
TIME_MAP = {"7AM": 7, "10AM": 10, "2PM": 14, "6PM": 18, "10PM": 22}
EXPIRATION_HOURS = {"2h": 2.0, "1d": 24.0}
# Persisted constant code for the coupon venue (deterministic, never fitted).
COUPON_CODE = {
    "Bar": 1,
    "Coffee House": 2,
    "Carry out & Take away": 3,
    "Restaurant(<20)": 4,
    "Restaurant(20-50)": 5,
}
COUPON_VENUE = {
    "Bar": "Bar",
    "Coffee House": "CoffeeHouse",
    "Carry out & Take away": "CarryAway",
    "Restaurant(<20)": "RestaurantLessThan20",
    "Restaurant(20-50)": "Restaurant20To50",
}
FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]
FREQ_ORD = [c + "_freq" for c in FREQ_COLS]
TEXT_COLS = [
    "time",
    "coupon",
    "expiration",
    "gender",
    "age",
    "maritalStatus",
    "education",
    "occupation",
    "income",
] + FREQ_COLS
# toCoupon_GEQ5min is constant (==1, 1 distinct value) -> dropped to avoid a
# dummy categorical in LightGBM and a wasted split in CatBoost.
BINARY_COLS = [
    "has_children",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
    "direction_opp",
]

BASE_NUM = [
    "temperature",
    "has_children",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
]
ORD_NUM = ["age_ord", "income_ord", "edu_ord", "time_ord", "expiration_hours"]
ENG_NUM = FREQ_ORD + [
    "social_index",
    "matched_venue_freq",
    "distance_level",
    "affinity_ratio",
    "matched_minus_mean",
    "dining_engagement",
    "coupon_matched_x",
]
NUM_FEATS = BASE_NUM + ORD_NUM + ENG_NUM

CAT_FEATS = [
    "time",
    "coupon",
    "expiration",
    "gender",
    "maritalStatus",
    "education",
    "occupation",
    "income",
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
    "coupon_time",
    "coupon_exp",
    "matched_bucket",
    "coupon_occupation",
    "coupon_income",
    "coupon_matched_freq",
    "matched_venue_x_has_children",
]

FEATURES = NUM_FEATS + CAT_FEATS


def load_schema(data_dir):
    path = os.path.join(data_dir, "schema.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _restrict_to_schema(df, schema):
    cols = schema.get("feature_columns")
    if cols:
        df = df[[c for c in cols if c in df.columns]].copy()
    return df


def clean_features(df):
    df = df.copy()
    for c in ["temperature"] + BINARY_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    for c in FREQ_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("never")
    for c in ["gender", "maritalStatus", "education", "occupation", "income"]:
        if c in df.columns:
            df[c] = df[c].fillna("missing")
    for c in BINARY_COLS:
        if c in df.columns:
            df[c] = df[c].astype(float)
    return df


def _matched_venue_freq(df):
    out = pd.Series(np.full(len(df), np.nan), index=df.index)
    if "coupon" not in df.columns:
        return out
    venue_col = df["coupon"].map(COUPON_VENUE)
    for c in FREQ_COLS:
        if c not in df.columns:
            continue
        mask = (venue_col == c).values
        if mask.any():
            out.loc[df.index[mask]] = df.loc[mask, c].map(FREQ_MAP).astype(float).values
    return out


def engineer_features(df):
    df = df.copy()
    if "age" in df.columns:
        df["age_ord"] = df["age"].map(AGE_MAP).astype(float)
    if "income" in df.columns:
        df["income_ord"] = df["income"].map(INCOME_MAP).astype(float)
    if "education" in df.columns:
        df["edu_ord"] = df["education"].map(EDU_MAP).astype(float)
    if "time" in df.columns:
        df["time_ord"] = df["time"].map(TIME_MAP).astype(float)
    if "expiration" in df.columns:
        df["expiration_hours"] = df["expiration"].map(EXPIRATION_HOURS).astype(float)

    for c in FREQ_COLS:
        if c in df.columns:
            df[c + "_freq"] = df[c].map(FREQ_MAP).astype(float)

    existing_freq_ord = [c for c in FREQ_ORD if c in df.columns]
    df["social_index"] = (
        df[existing_freq_ord].fillna(0.0).sum(axis=1) if existing_freq_ord else 0.0
    )

    df["matched_venue_freq"] = _matched_venue_freq(df)
    mvf = df["matched_venue_freq"]
    df["matched_bucket"] = np.select(
        [mvf.isna(), mvf == 0, mvf <= 2, mvf >= 3],
        ["unknown", "never", "rare", "regular"],
        default="unknown",
    )

    g15 = df["toCoupon_GEQ15min"].fillna(0.0) if "toCoupon_GEQ15min" in df else 0.0
    g25 = df["toCoupon_GEQ25min"].fillna(0.0) if "toCoupon_GEQ25min" in df else 0.0
    df["distance_level"] = g15 + g25

    if {"coupon", "time"}.issubset(df.columns):
        df["coupon_time"] = df["coupon"].astype(str) + "|" + df["time"].astype(str)
    if {"coupon", "expiration"}.issubset(df.columns):
        df["coupon_exp"] = df["coupon"].astype(str) + "|" + df["expiration"].astype(str)
    if {"coupon", "occupation"}.issubset(df.columns):
        df["coupon_occupation"] = (
            df["coupon"].astype(str) + "|" + df["occupation"].astype(str)
        )
    if {"coupon", "income"}.issubset(df.columns):
        df["coupon_income"] = df["coupon"].astype(str) + "|" + df["income"].astype(str)

    # ---- target-free affinity / conditional-propensity features ------- #
    # All deterministic per-row functions of constant maps + already
    # engineered columns -> nothing is fitted here (predict path identical).
    mvf_num = pd.to_numeric(df["matched_venue_freq"], errors="coerce")
    social_num = pd.to_numeric(df["social_index"], errors="coerce")
    df["affinity_ratio"] = (
        (mvf_num / (social_num + 1.0)).fillna(0.0).clip(0.0, 5.0).astype(float)
    )
    if existing_freq_ord:
        freq_mean = df[existing_freq_ord].fillna(0.0).mean(axis=1)
        freq_sum = df[existing_freq_ord].fillna(0.0).sum(axis=1)
    else:
        freq_mean = pd.Series(0.0, index=df.index)
        freq_sum = pd.Series(0.0, index=df.index)
    df["matched_minus_mean"] = (mvf_num - freq_mean).fillna(0.0).astype(float)
    df["dining_engagement"] = freq_sum.fillna(0.0).astype(float)
    if "coupon" in df.columns:
        coupon_code = df["coupon"].map(COUPON_CODE).astype(float)
    else:
        coupon_code = pd.Series(np.nan, index=df.index)
    df["coupon_matched_x"] = (coupon_code * mvf_num).fillna(0.0).astype(float)

    if "coupon" in df.columns:
        df["coupon_matched_freq"] = (
            df["coupon"].astype(str) + "|" + df["matched_bucket"].astype(str)
        )
    if "has_children" in df.columns:
        df["matched_venue_x_has_children"] = (
            df["matched_bucket"].astype(str)
            + "|"
            + df["has_children"].fillna(-1).astype(int).astype(str)
        )

    for c in NUM_FEATS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_FEATS:
        if c not in df.columns:
            df[c] = "missing"
        df[c] = df[c].astype(str)
    return df


def load_train(data_dir):
    features_path = os.path.join(data_dir, "features.csv")
    labels_path = os.path.join(data_dir, "labels.csv")
    schema = load_schema(data_dir)
    feats = _restrict_to_schema(pd.read_csv(features_path), schema)
    labels = pd.read_csv(labels_path)
    label_col = "label" if "label" in labels.columns else labels.columns[-1]
    y = labels[label_col].values
    assert len(feats) == len(y), "features/labels row mismatch"
    return feats, y, schema


def load_predict_features(data_dir):
    schema = load_schema(data_dir)
    feats = _restrict_to_schema(
        pd.read_csv(os.path.join(data_dir, "features.csv")), schema
    )
    return feats, schema


def prepare_train(data_dir):
    raw, y, schema = load_train(data_dir)
    return engineer_features(clean_features(raw)), y, schema


def prepare_predict(data_dir):
    raw, schema = load_predict_features(data_dir)
    return engineer_features(clean_features(raw)), schema


# =========================================================================== #
# STEP 2: MODEL DESIGN
# =========================================================================== #
CATBOOST_PARAMS = dict(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=4.0,
    random_strength=1.0,
    bagging_temperature=1.0,
    loss_function="Logloss",
    eval_metric="Logloss",
    od_type="Iter",
    od_wait=150,
    thread_count=2,
    verbose=False,
    random_seed=42,
)


def build_catboost(**ov):
    p = dict(CATBOOST_PARAMS)
    p.update(ov)
    return CatBoostClassifier(**p)


LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=30,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=2.0,
    num_threads=2,
    seed=42,
    verbose=-1,
)


def _make_X_cat(eng, feature_order=None, cat_features=None):
    feature_order = feature_order or FEATURES
    cat_features = cat_features or CAT_FEATS
    X = eng[feature_order].copy()
    for c in cat_features:
        X[c] = X[c].astype(str)
    return X.reset_index(drop=True)


def build_cat_category_maps(eng):
    """Persisted vocabularies -> integer codes for LightGBM categorical features."""
    maps = {}
    for c in CAT_FEATS:
        vals = pd.Series(eng[c].astype(str))
        uniq = sorted(vals.unique().tolist())
        maps[c] = {v: i for i, v in enumerate(uniq)}
    return maps


def _make_X_lgb(eng, cat_maps):
    X = eng[FEATURES].copy().reset_index(drop=True)
    for c in CAT_FEATS:
        m = cat_maps[c]
        unseen = len(m)
        X[c] = X[c].astype(str).map(lambda v: m.get(v, unseen)).astype(np.int64)
    for c in NUM_FEATS:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
    return X


# =========================================================================== #
# STEP 3: TRAINING / PREDICTION
# =========================================================================== #
BUNDLE_NAME = "coupon_model_bundle.joblib"
META_NAME = "model_metadata.json"


def _fit_cv_catboost(X, y, n_splits, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    best_iters = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = build_catboost(random_seed=seed + fold)
        m.fit(
            Pool(X.iloc[tr], y[tr], cat_features=CAT_FEATS),
            eval_set=Pool(X.iloc[va], y[va], cat_features=CAT_FEATS),
            use_best_model=True,
            verbose=False,
        )
        oof[va] = m.predict_proba(Pool(X.iloc[va], cat_features=CAT_FEATS))[:, 1]
        it = m.get_best_iteration()
        best_iters.append(int(it) if it is not None and it > 0 else 300)
        print(
            f"[train][cat] fold {fold}: best_iter={best_iters[-1]} "
            f"acc@0.5={float(((oof[va]>=0.5).astype(int)==y[va]).mean()):.4f}"
        )
    return oof, best_iters


def _fit_cv_lgb(X_l, y, n_splits, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    best_iters = []
    for fold, (tr, va) in enumerate(skf.split(X_l, y)):
        dtr = lgb.Dataset(
            X_l.iloc[tr],
            label=y[tr],
            categorical_feature=CAT_FEATS,
            free_raw_data=False,
        )
        dva = lgb.Dataset(
            X_l.iloc[va],
            label=y[va],
            categorical_feature=CAT_FEATS,
            reference=dtr,
            free_raw_data=False,
        )
        params = dict(LGB_PARAMS)
        params["seed"] = seed + fold
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=3000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        bi = int(booster.best_iteration) if booster.best_iteration else 300
        best_iters.append(max(50, bi))
        oof[va] = booster.predict(X_l.iloc[va], num_iteration=bi)
        print(
            f"[train][lgb] fold {fold}: best_iter={best_iters[-1]} "
            f"acc@0.5={float(((oof[va]>=0.5).astype(int)==y[va]).mean()):.4f}"
        )
    return oof, best_iters


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _joint_fusion_search(oof_cat, oof_lgb, y):
    """Joint (blend weight, threshold) grid search in logit space on OOF probs.

    Returns (w_cat, threshold, best_accuracy).  Degenerate / flat-accuracy
    regions are broken by preferring the w nearest the CV-best single model.
    """
    thresholds = np.linspace(0.30, 0.70, 81)
    logit_cat = _logit(oof_cat)
    if oof_lgb is None:
        cand_w = [1.0]
        logit_lgb = None
        preferred_w = 1.0
    else:
        cand_w = [float(w) for w in np.linspace(0.0, 1.0, 11)]
        logit_lgb = _logit(oof_lgb)
        acc_cat = float(((oof_cat >= 0.5).astype(int) == y).mean())
        acc_lgb = float(((oof_lgb >= 0.5).astype(int) == y).mean())
        preferred_w = 1.0 if acc_cat >= acc_lgb else 0.0

    results = []
    for w in cand_w:
        if logit_lgb is None:
            proba = _sigmoid(logit_cat)
        else:
            proba = _sigmoid(w * logit_cat + (1.0 - w) * logit_lgb)
        accs = np.asarray(
            [((proba >= t).astype(int) == y).mean() for t in thresholds], dtype=float
        )
        bi = int(np.argmax(accs))
        results.append((float(accs[bi]), float(w), float(thresholds[bi])))

    best_acc = max(r[0] for r in results)
    tied = [r for r in results if abs(r[0] - best_acc) < 1e-12]
    chosen = min(tied, key=lambda r: abs(r[1] - preferred_w))
    return chosen[1], chosen[2], chosen[0]


def train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    eng, y_raw, _ = prepare_train(data_dir)
    y_raw = np.asarray(y_raw).ravel()
    classes = np.unique(y_raw).tolist()
    label_to_idx = {l: i for i, l in enumerate(classes)}
    y = np.asarray([label_to_idx[v] for v in y_raw], dtype=int)

    X_cat = _make_X_cat(eng)
    cat_maps = build_cat_category_maps(eng)
    X_lgb = _make_X_lgb(eng, cat_maps)

    n_splits = 5 if len(y) >= 50 else max(2, min(5, max(2, len(y) // 2)))

    # ---- CV for OOF + iteration estimation (train rows only) ---------- #
    # Two StratifiedKFold(5) repetitions shrink the selection variance of the
    # OOF vector used to pick the fusion operating point.  With a single model
    # (no LightGBM) a single repetition is used and w is forced to (1.0, 0.0).
    rep_seeds = (42, 1042) if _HAS_LGB else (42,)
    oof_cat_sum = np.zeros(len(y), dtype=float)
    oof_lgb_sum = np.zeros(len(y), dtype=float) if _HAS_LGB else None
    it_cat, it_lgb = [], []
    for rep_seed in rep_seeds:
        oof_cat_r, it_cat_r = _fit_cv_catboost(X_cat, y, n_splits, seed=rep_seed)
        oof_cat_sum += oof_cat_r
        it_cat += it_cat_r
        if _HAS_LGB:
            oof_lgb_r, it_lgb_r = _fit_cv_lgb(X_lgb, y, n_splits, seed=rep_seed)
            oof_lgb_sum += oof_lgb_r
            it_lgb += it_lgb_r
    n_rep = float(len(rep_seeds))
    oof_cat = oof_cat_sum / n_rep
    oof_lgb = (oof_lgb_sum / n_rep) if _HAS_LGB else None

    # ---- joint (w, threshold) search on the averaged OOF -------------- #
    best_w, best_thr, best_acc = _joint_fusion_search(oof_cat, oof_lgb, y)
    if _HAS_LGB:
        oof_blend = _sigmoid(
            best_w * _logit(oof_cat) + (1.0 - best_w) * _logit(oof_lgb)
        )
    else:
        oof_blend = oof_cat
    acc_05 = float(((oof_blend >= 0.5).astype(int) == y).mean())

    print(
        f"[train] OOF blend acc@0.5={acc_05:.4f}  @{best_thr:.3f}={best_acc:.4f} "
        f"w_cat={best_w:.2f} (cat={float(((oof_cat>=0.5).astype(int)==y).mean()):.4f}"
        + (
            f", lgb={float(((oof_lgb>=0.5).astype(int)==y).mean()):.4f}"
            if _HAS_LGB
            else ""
        )
        + ")"
    )

    # ---- FINAL RETRAIN ON 100% OF TRAIN ROWS -------------------------- #
    final_iters_cat = max(100, int(np.median(it_cat)))
    cat_model = build_catboost(iterations=final_iters_cat, random_seed=42)
    cat_model.fit(Pool(X_cat, y, cat_features=CAT_FEATS), verbose=False)
    print(f"[train] final CatBoost retrained on {len(y)} rows, iters={final_iters_cat}")

    lgb_model = None
    final_iters_lgb = None
    if _HAS_LGB:
        final_iters_lgb = max(50, int(np.median(it_lgb)))
        dtr = lgb.Dataset(
            X_lgb, label=y, categorical_feature=CAT_FEATS, free_raw_data=False
        )
        lgb_model = lgb.train(
            dict(LGB_PARAMS), dtr, num_boost_round=int(final_iters_lgb * 1.10)
        )
        print(
            f"[train] final LightGBM retrained on {len(y)} rows, "
            f"iters={int(final_iters_lgb*1.10)}"
        )

    bundle = {
        "cat_model": cat_model,
        "lgb_model": lgb_model,
        "has_lgb": _HAS_LGB,
        "cat_maps": cat_maps,
        "threshold": best_thr,
        "classes": classes,
        "cat_features": list(CAT_FEATS),
        "num_features": list(NUM_FEATS),
        "feature_order": list(FEATURES),
        "blend_weights": (float(best_w), float(1.0 - best_w)),
        "threshold": best_thr,
        "oof_accuracy_05": acc_05,
        "oof_accuracy_best": best_acc,
    }
    joblib.dump(bundle, os.path.join(model_dir, BUNDLE_NAME))

    meta = {
        "model_type": "catboost_plus_lightgbm_full_retrain_blend",
        "n_folds": n_splits,
        "cat_final_iters": final_iters_cat,
        "lgb_final_iters": (int(final_iters_lgb * 1.10) if final_iters_lgb else None),
        "threshold": best_thr,
        "blend_weights": [float(best_w), float(1.0 - best_w)],
        "classes": classes,
        "cat_features": list(CAT_FEATS),
        "feature_order": list(FEATURES),
        "n_train_rows": int(len(y)),
        "oof_accuracy_05": acc_05,
        "oof_accuracy_best": best_acc,
    }
    with open(os.path.join(model_dir, META_NAME), "w") as f:
        json.dump(meta, f, indent=2)

    return best_acc


def predict(data_dir, model_dir, output_path):
    bundle = joblib.load(os.path.join(model_dir, BUNDLE_NAME))
    cat_model = bundle["cat_model"]
    lgb_model = bundle["lgb_model"]
    cat_maps = bundle["cat_maps"]
    thr = float(bundle["threshold"])
    classes = bundle["classes"]
    feature_order = bundle["feature_order"]
    cat_features = bundle["cat_features"]
    w_cat, w_lgb = bundle.get("blend_weights", (1.0, 0.0))

    eng, _ = prepare_predict(data_dir)
    n = len(eng)
    if n == 0:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame({"row_id": [], "prediction": []}).to_csv(output_path, index=False)
        print("[predict] empty input -> wrote empty predictions")
        return

    X_cat = _make_X_cat(eng, feature_order=feature_order, cat_features=cat_features)
    p_cat = cat_model.predict_proba(Pool(X_cat, cat_features=cat_features))[:, 1]

    if lgb_model is not None:
        X_lgb = _make_X_lgb(eng, cat_maps)
        p_lgb = lgb_model.predict(X_lgb)
        proba = _sigmoid(w_cat * _logit(p_cat) + w_lgb * _logit(p_lgb))
    else:
        proba = p_cat

    pred_idx = (proba >= thr).astype(int)
    predictions = np.asarray(classes)[pred_idx]

    out = pd.DataFrame({"row_id": np.arange(n, dtype=int), "prediction": predictions})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(
        f"[predict] wrote {len(out)} predictions to {output_path} (threshold={thr:.3f})"
    )


# =========================================================================== #
# CLI
# =========================================================================== #
if __name__ == "__main__":
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
        score = train(args.data, args.model)
        print(f"Final Validation Score: {score}")
    elif args.command == "predict":
        predict(args.data, args.model, args.output)
