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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

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

# ---- Target-encoding specification (high-cardinality interaction keys) ---- #
# These keys are too sparse for greedy tree splits on ~8k rows; a prior
# smoothed numeric encoding gives the same conditional base-rate signal at
# much lower variance.  CatBoost keeps the RAW categorical contract below.
TE_KEYS = ["coupon_occupation", "coupon_income", "coupon_time", "coupon_exp"]
TE_COL_NAMES = ["te_" + k for k in TE_KEYS]
TE_M = 20.0
# LightGBM / HGB see the numeric TE columns plus ONLY low-cardinality
# categoricals (each <= ~30 levels); high-card interaction keys are excluded.
CAT_FEATS_LGB = [
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
    "matched_bucket",
    "coupon_matched_freq",
    "matched_venue_x_has_children",
]
LGB_FEATURES = NUM_FEATS + TE_COL_NAMES + CAT_FEATS_LGB


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


# --------------------------------------------------------------------------- #
# Target encoding (prior-smoothed).  State = {"maps", "prior", "m"} and is
# persisted in the bundle; prediction only ever calls transform_*.
# --------------------------------------------------------------------------- #
def fit_target_encoding(df, y, m=TE_M):
    """Fit smoothed level->rate maps on the ROWS PROVIDED (fold-train only)."""
    df = df.reset_index(drop=True)
    y = np.asarray(y).ravel().astype(float)
    prior = float(y.mean()) if len(y) else 0.5
    maps = {}
    for key in TE_KEYS:
        if key not in df.columns or len(y) == 0:
            maps[key] = {}
            continue
        frame = pd.DataFrame({"k": df[key].astype(str).values, "y": y})
        agg = frame.groupby("k")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + prior * m) / (agg["count"] + m)
        maps[key] = enc.to_dict()
    return {"maps": maps, "prior": prior, "m": float(m)}


def transform_target_encoding(df, te_state):
    """Apply persisted maps; unseen/NaN levels fall back to the stored prior."""
    df = df.reset_index(drop=True).copy()
    maps = te_state.get("maps", {})
    prior = float(te_state.get("prior", 0.5))
    for key, col in zip(TE_KEYS, TE_COL_NAMES):
        m = maps.get(key, {})
        if key in df.columns:
            levels = df[key].astype(str)
        else:
            levels = pd.Series(["missing"] * len(df))
        df[col] = levels.map(lambda v: m.get(v, prior)).astype(np.float32)
    return df


def oof_target_encoding(df, y, m=TE_M, seed=42, n_splits=5):
    """Inner-OOF encoding for TRAIN rows: a row's own label never contributes."""
    df = df.reset_index(drop=True)
    y = np.asarray(y).ravel()
    n = len(df)
    prior = float(y.mean()) if n else 0.5
    out = {col: np.full(n, np.nan, dtype=np.float32) for col in TE_COL_NAMES}
    if n == 0:
        return pd.DataFrame(out)
    counts = np.bincount(y.astype(int))
    nz = counts[counts > 0]
    k = int(max(2, min(n_splits, int(nz.min())))) if nz.size >= 2 else 0
    if k < 2 or n < k:
        for col in out:
            out[col][:] = prior
        return pd.DataFrame(out)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(n), y):
        st = fit_target_encoding(df.iloc[tr].reset_index(drop=True), y[tr], m=m)
        enc = transform_target_encoding(df.iloc[va].reset_index(drop=True), st)
        for col in TE_COL_NAMES:
            out[col][va] = enc[col].values
    for col in TE_COL_NAMES:
        out[col] = np.where(np.isnan(out[col]), prior, out[col]).astype(np.float32)
    return pd.DataFrame(out)


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

HGB_PARAMS = dict(
    max_iter=600,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=True,
    n_iter_no_change=40,
    validation_fraction=0.15,
    random_state=42,
)


def build_hgb(**ov):
    p = dict(HGB_PARAMS)
    p.update(ov)
    return HistGradientBoostingClassifier(**p)


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


def _make_X_lgb(eng, cat_maps, te_df=None):
    """Numeric TE columns + low-cardinality categorical codes only."""
    eng = eng.reset_index(drop=True)
    if te_df is not None:
        te_df = te_df.reset_index(drop=True)
    X = pd.DataFrame(index=np.arange(len(eng)))
    for c in NUM_FEATS:
        if c in eng.columns:
            X[c] = pd.to_numeric(eng[c], errors="coerce").astype(np.float32)
        else:
            X[c] = np.float32(np.nan)
    for col in TE_COL_NAMES:
        if te_df is not None and col in te_df.columns:
            X[col] = pd.to_numeric(te_df[col], errors="coerce").values.astype(
                np.float32
            )
        else:
            X[col] = np.float32(np.nan)
    for c in CAT_FEATS_LGB:
        m = cat_maps.get(c, {})
        unseen = len(m)
        if c in eng.columns:
            levels = eng[c].astype(str)
        else:
            levels = pd.Series(["missing"] * len(eng))
        X[c] = levels.map(lambda v, _m=m, _u=unseen: _m.get(v, _u)).astype(np.int64)
    return X[LGB_FEATURES]


# =========================================================================== #
# STEP 3: TRAINING / PREDICTION
# =========================================================================== #
BUNDLE_NAME = "coupon_model_bundle.joblib"
META_NAME = "model_metadata.json"


def _fit_cv_models(eng, X_cat, y, cat_maps, n_splits, seed=42):
    """One CV pass producing OOF probabilities for CatBoost / LGB / HGB.

    The target encoder is fitted strictly on fold-train rows and fold-train rows
    get an inner-OOF encoding, so no row's own label ever enters its encoding.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(y)
    oof_cat = np.zeros(n, dtype=float)
    oof_lgb = np.zeros(n, dtype=float)
    oof_hgb = np.zeros(n, dtype=float)
    it_cat, it_lgb, it_hgb = [], [], []
    for fold, (tr, va) in enumerate(skf.split(X_cat, y)):
        tr = np.asarray(tr, dtype=int)
        va = np.asarray(va, dtype=int)
        y_tr, y_va = y[tr], y[va]
        eng_tr = eng.iloc[tr].reset_index(drop=True)
        eng_va = eng.iloc[va].reset_index(drop=True)

        # ---- fold-internal target encoding ---------------------------- #
        te_tr = oof_target_encoding(eng_tr, y_tr, m=TE_M, seed=seed + fold)
        te_fold_state = fit_target_encoding(eng_tr, y_tr, m=TE_M)
        te_va = transform_target_encoding(eng_va, te_fold_state)
        Xl_tr = _make_X_lgb(eng_tr, cat_maps, te_tr)
        Xl_va = _make_X_lgb(eng_va, cat_maps, te_va)

        # ---- CatBoost (raw categorical contract preserved) ------------- #
        m_cat = build_catboost(random_seed=seed + fold)
        m_cat.fit(
            Pool(X_cat.iloc[tr], y_tr, cat_features=CAT_FEATS),
            eval_set=Pool(X_cat.iloc[va], y_va, cat_features=CAT_FEATS),
            use_best_model=True,
            verbose=False,
        )
        oof_cat[va] = m_cat.predict_proba(
            Pool(X_cat.iloc[va], cat_features=CAT_FEATS)
        )[:, 1]
        bi = m_cat.get_best_iteration()
        it_cat.append(int(bi) if bi is not None and bi > 0 else 300)

        # ---- LightGBM (numeric TE + low-card categoricals) ------------- #
        if _HAS_LGB:
            params = dict(LGB_PARAMS)
            params["seed"] = seed + fold
            dtr = lgb.Dataset(
                Xl_tr,
                label=y_tr,
                categorical_feature=CAT_FEATS_LGB,
                free_raw_data=False,
            )
            dva = lgb.Dataset(
                Xl_va,
                label=y_va,
                categorical_feature=CAT_FEATS_LGB,
                reference=dtr,
                free_raw_data=False,
            )
            booster = lgb.train(
                params,
                dtr,
                num_boost_round=3000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(150, verbose=False)],
            )
            bi = int(booster.best_iteration) if booster.best_iteration else 300
            bi = max(50, bi)
            it_lgb.append(bi)
            oof_lgb[va] = booster.predict(Xl_va, num_iteration=bi)

        # ---- HistGradientBoosting on the same encoded matrix ----------- #
        m_hgb = build_hgb(random_state=seed + fold)
        m_hgb.fit(Xl_tr.astype(np.float64).values, y_tr)
        oof_hgb[va] = m_hgb.predict_proba(Xl_va.astype(np.float64).values)[:, 1]
        it_hgb.append(int(getattr(m_hgb, "n_iter_", 300)))

        acc = lambda p: float(((p >= 0.5).astype(int) == y_va).mean())
        msg = (
            f"[train] fold {fold} (seed={seed}): cat_it={it_cat[-1]} "
            f"cat_acc={acc(oof_cat[va]):.4f} hgb_it={it_hgb[-1]} "
            f"hgb_acc={acc(oof_hgb[va]):.4f}"
        )
        if _HAS_LGB:
            msg += f" lgb_it={it_lgb[-1]} lgb_acc={acc(oof_lgb[va]):.4f}"
        print(msg)
    return {
        "oof_cat": oof_cat,
        "oof_lgb": oof_lgb,
        "oof_hgb": oof_hgb,
        "it_cat": it_cat,
        "it_lgb": it_lgb,
        "it_hgb": it_hgb,
    }


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _fit_meta(Z, y, seed=42, n_splits=5):
    """Regularized LR stacker on OOF logits + inner-CV threshold selection.

    The threshold is chosen on an inner CV of the meta-learner (NOT on the same
    rows used to fit its weights), removing the winner's curse of the previous
    891-point joint grid search.
    """
    thresholds = np.linspace(0.30, 0.70, 81)
    counts = np.bincount(np.asarray(y, dtype=int))
    nz = counts[counts > 0]
    k = int(max(2, min(n_splits, int(nz.min())))) if nz.size >= 2 else 0
    if k >= 2:
        inner = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        rows = []
        for tr, va in inner.split(Z, y):
            m = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
            m.fit(Z[tr], y[tr])
            p = m.predict_proba(Z[va])[:, 1]
            rows.append([((p >= t).astype(int) == y[va]).mean() for t in thresholds])
        mean_acc = np.asarray(rows, dtype=float).mean(axis=0)
        bi = int(np.argmax(mean_acc))
        tied = np.where(np.abs(mean_acc - mean_acc[bi]) < 1e-12)[0]
        # tie-break toward 0.5 (less extreme operating point -> less variance)
        best_thr = float(thresholds[tied[np.argmin(np.abs(thresholds[tied] - 0.5))]])
        inner_acc = float(mean_acc[bi])
    else:
        best_thr, inner_acc = 0.5, float("nan")

    final_meta = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
    final_meta.fit(Z, y)
    return final_meta, best_thr, inner_acc


def train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    eng, y_raw, _ = prepare_train(data_dir)
    eng = eng.reset_index(drop=True)
    y_raw = np.asarray(y_raw).ravel()
    classes = np.unique(y_raw).tolist()
    label_to_idx = {l: i for i, l in enumerate(classes)}
    y = np.asarray([label_to_idx[v] for v in y_raw], dtype=int)

    X_cat = _make_X_cat(eng, feature_order=FEATURES, cat_features=CAT_FEATS)
    cat_maps = build_cat_category_maps(eng)

    n_splits = 5 if len(y) >= 50 else max(2, min(5, max(2, len(y) // 2)))

    # ---- CV: OOF logits for the stacker + iteration estimation -------- #
    # Two StratifiedKFold(5) repetitions (seeds 42 / 1042) average out fold
    # noise before the meta-learner sees the logits.
    rep_seeds = (42, 1042) if _HAS_LGB else (42,)
    oof = {"cat": np.zeros(len(y)), "lgb": np.zeros(len(y)), "hgb": np.zeros(len(y))}
    iters = {"cat": [], "lgb": [], "hgb": []}
    for rep_seed in rep_seeds:
        r = _fit_cv_models(eng, X_cat, y, cat_maps, n_splits, seed=rep_seed)
        for k in oof:
            oof[k] = oof[k] + r["oof_" + k]
        for k in iters:
            iters[k] = iters[k] + r["it_" + k]
    n_rep = float(len(rep_seeds))
    for k in oof:
        oof[k] = oof[k] / n_rep

    # ---- nested stacking layer ---------------------------------------- #
    meta_features = ["cat"] + (["lgb"] if _HAS_LGB else []) + ["hgb"]
    if len(classes) < 2:
        meta_model, best_thr, best_acc = None, 0.5, 1.0
    else:
        Z = np.column_stack([_logit(oof[k]) for k in meta_features])
        meta_model, best_thr, best_acc = _fit_meta(Z, y, seed=42)
        oof_meta = meta_model.predict_proba(Z)[:, 1]
        acc_05 = float(((oof_meta >= 0.5).astype(int) == y).mean())
        print(
            "[train] OOF meta acc@0.5={:.4f}  inner-CV thr={:.3f} -> {:.4f} "
            "(cat={:.4f}".format(
                acc_05,
                best_thr,
                best_acc,
                float(((oof["cat"] >= 0.5).astype(int) == y).mean()),
            )
            + (
                ", lgb={:.4f}, hgb={:.4f})".format(
                    float(((oof["lgb"] >= 0.5).astype(int) == y).mean()),
                    float(((oof["hgb"] >= 0.5).astype(int) == y).mean()),
                )
                if _HAS_LGB
                else ", hgb={:.4f})".format(
                    float(((oof["hgb"] >= 0.5).astype(int) == y).mean())
                )
            )
        )

    # ---- FINAL RETRAIN ON 100% OF TRAIN ROWS -------------------------- #
    # Target-encoding maps are fitted on ALL training rows here (legal: only
    # training labels are used) and persisted for the read-only predict path.
    te_state = fit_target_encoding(eng, y, m=TE_M)
    te_all = transform_target_encoding(eng, te_state)
    X_lgb_full = _make_X_lgb(eng, cat_maps, te_all)

    final_iters_cat = max(100, int(np.median(iters["cat"])))
    cat_model = build_catboost(iterations=final_iters_cat, random_seed=42)
    cat_model.fit(Pool(X_cat, y, cat_features=CAT_FEATS), verbose=False)
    print(f"[train] final CatBoost retrained on {len(y)} rows, iters={final_iters_cat}")

    lgb_model = None
    final_iters_lgb = None
    if _HAS_LGB:
        final_iters_lgb = max(50, int(np.median(iters["lgb"])))
        dtr = lgb.Dataset(
            X_lgb_full,
            label=y,
            categorical_feature=CAT_FEATS_LGB,
            free_raw_data=False,
        )
        lgb_model = lgb.train(
            dict(LGB_PARAMS), dtr, num_boost_round=int(final_iters_lgb * 1.10)
        )
        print(
            f"[train] final LightGBM retrained on {len(y)} rows, "
            f"iters={int(final_iters_lgb*1.10)}"
        )

    final_iters_hgb = (
        max(50, int(np.median(iters["hgb"]))) if iters["hgb"] else 300
    )
    hgb_model = build_hgb(
        random_state=42, early_stopping=False, max_iter=final_iters_hgb
    )
    hgb_model.fit(X_lgb_full.astype(np.float64).values, y)
    print(f"[train] final HGB retrained on {len(y)} rows, iters={final_iters_hgb}")

    bundle = {
        "cat_model": cat_model,
        "lgb_model": lgb_model,
        "hgb_model": hgb_model,
        "meta_model": meta_model,
        "meta_features": list(meta_features),
        "has_lgb": _HAS_LGB,
        "cat_maps": cat_maps,
        "te_state": te_state,
        "te_m": float(TE_M),
        "te_col_names": list(TE_COL_NAMES),
        "threshold": best_thr,
        "classes": classes,
        "cat_features": list(CAT_FEATS),
        "lgb_cat_features": list(CAT_FEATS_LGB),
        "num_features": list(NUM_FEATS),
        "feature_order": list(FEATURES),
        "lgb_feature_order": list(LGB_FEATURES),
        "oof_accuracy_meta": float(best_acc),
    }
    joblib.dump(bundle, os.path.join(model_dir, BUNDLE_NAME))

    meta = {
        "model_type": "catboost_lightgbm_hgb_te_nested_stack_full_retrain",
        "n_folds": n_splits,
        "cat_final_iters": final_iters_cat,
        "lgb_final_iters": (int(final_iters_lgb * 1.10) if final_iters_lgb else None),
        "hgb_final_iters": final_iters_hgb,
        "threshold": best_thr,
        "meta_features": list(meta_features),
        "te_keys": list(TE_KEYS),
        "te_col_names": list(TE_COL_NAMES),
        "te_m": float(TE_M),
        "classes": classes,
        "cat_features": list(CAT_FEATS),
        "lgb_cat_features": list(CAT_FEATS_LGB),
        "feature_order": list(FEATURES),
        "lgb_feature_order": list(LGB_FEATURES),
        "n_train_rows": int(len(y)),
        "oof_accuracy_meta": float(best_acc),
    }
    with open(os.path.join(model_dir, META_NAME), "w") as f:
        json.dump(meta, f, indent=2)

    return best_acc


def predict(data_dir, model_dir, output_path):
    bundle = joblib.load(os.path.join(model_dir, BUNDLE_NAME))
    cat_model = bundle["cat_model"]
    lgb_model = bundle.get("lgb_model")
    hgb_model = bundle.get("hgb_model")
    cat_maps = bundle["cat_maps"]
    te_state = bundle["te_state"]
    meta_model = bundle.get("meta_model")
    meta_features = bundle.get("meta_features", ["cat"])
    thr = float(bundle["threshold"])
    classes = bundle["classes"]
    feature_order = bundle.get("feature_order", FEATURES)
    cat_features = bundle.get("cat_features", CAT_FEATS)
    lgb_feature_order = bundle.get("lgb_feature_order", LGB_FEATURES)

    eng, _ = prepare_predict(data_dir)
    n = len(eng)
    if n == 0:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame({"row_id": [], "prediction": []}).to_csv(output_path, index=False)
        print("[predict] empty input -> wrote empty predictions")
        return

    eng = eng.reset_index(drop=True)
    X_cat = _make_X_cat(eng, feature_order=feature_order, cat_features=cat_features)
    probs = {
        "cat": cat_model.predict_proba(Pool(X_cat, cat_features=cat_features))[:, 1]
    }

    if lgb_model is not None or hgb_model is not None:
        # TE maps come straight from the bundle -> transform only, never refit.
        te_all = transform_target_encoding(eng, te_state)
        X_lgb = _make_X_lgb(eng, cat_maps, te_all)[lgb_feature_order]
        if lgb_model is not None:
            probs["lgb"] = np.asarray(lgb_model.predict(X_lgb), dtype=float)
        if hgb_model is not None:
            probs["hgb"] = hgb_model.predict_proba(
                X_lgb.astype(np.float64).values
            )[:, 1]

    if meta_model is not None and all(k in probs for k in meta_features):
        Z = np.column_stack([_logit(probs[k]) for k in meta_features])
        proba = meta_model.predict_proba(Z)[:, 1]
    else:
        proba = probs["cat"]

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