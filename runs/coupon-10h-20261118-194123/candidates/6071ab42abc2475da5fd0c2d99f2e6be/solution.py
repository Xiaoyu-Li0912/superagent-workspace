#!/usr/bin/env python3
"""Coupon acceptance prediction.

Pipeline:
  * robust tabular feature engineering (ordinal maps + one-hot + numeric block),
  * ensemble of HistGradientBoosting / ExtraTrees / (optional) LightGBM,
  * OOF-optimized simplex blend weights,
  * single pickled bundle persisted under --model, loaded at predict time.

Usage:
  python solution.py train   --data /data  --model /model
  python solution.py predict --data /data  --model /model --output /output/predictions.csv
"""

import argparse
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

try:
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:
    HAS_LGB = False


# ---------------------------------------------------------------------------
# Feature-engineering configuration
# ---------------------------------------------------------------------------
ORDINAL_CATS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
    "income",
    "age",
]
PLAIN_CATS = [
    "time",
    "coupon",
    "expiration",
    "gender",
    "maritalStatus",
    "education",
    "occupation",
]
NUMERIC_COLS = [
    "temperature",
    "has_children",
    "toCoupon_GEQ5min",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
    "direction_opp",
]
# The 70-127 row missing pattern in the FREQ columns is not random.
MISSING_FLAG_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
=======
# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======
# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
=======
    "Carry out": "CarryAway_ord",
}

# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
=======
# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
=======
# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======
# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
    "CoffeeHouse": _ORD_LEVELS,
    "CarryAway": _ORD_LEVELS,
    "RestaurantLessThan20": _ORD_LEVELS,
    "Restaurant20To50": _ORD_LEVELS,
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
}
=======
# Coupon -> coarse venue group code (tree friendly).
COUPON_GROUPS = {
    "Restaurant(<20)": 0,
    "Coffee House": 1,
    "Bar": 2,
    "Carry out": 3,
    "Restaurant(20-50)": 4,
}
# Coupon -> the ordinal column that describes the user's affinity for that venue.
COUPON_VENUE = {
    "Coffee House": "CoffeeHouse_ord",
    "Bar": "Bar_ord",
    "Restaurant(<20)": "RestaurantLessThan20_ord",
    "Restaurant(20-50)": "Restaurant20To50_ord",
    "Carry out": "CarryAway_ord",
}

# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
    "CoffeeHouse": _ORD_LEVELS,
    "CarryAway": _ORD_LEVELS,
    "RestaurantLessThan20": _ORD_LEVELS,
    "Restaurant20To50": _ORD_LEVELS,
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
}
=======
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
=======
# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======
# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
=======
    "Carry out": "CarryAway_ord",
}

# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
=======
# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
=======
# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======
# Repeated-CV / seed-bagging configuration.
SEED_BAG = [0, 1, 2]
N_REPEATS = 3
# Smoothing constant for target encoding (defined BEFORE TargetEncoder uses it).
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
=======
TE_SMOOTH_M = 20

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
=======

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
    "CoffeeHouse": _ORD_LEVELS,
    "CarryAway": _ORD_LEVELS,
    "RestaurantLessThan20": _ORD_LEVELS,
    "Restaurant20To50": _ORD_LEVELS,
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
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _load_schema(data_dir):
    p = os.path.join(data_dir, "schema.json")
    if os.path.exists(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _read_features(data_dir, schema=None):
    p = os.path.join(data_dir, "features.csv")
    df = pd.read_csv(p)
    if schema:
        cols = schema.get("feature_columns")
        if cols:
            keep = [c for c in cols if c in df.columns]
            if keep:
                df = df[keep]
    return df.reset_index(drop=True)


def _read_labels(data_dir):
    p = os.path.join(data_dir, "labels.csv")
    lbl = pd.read_csv(p)
    col = "label" if "label" in lbl.columns else lbl.columns[0]
    return lbl[col].tolist()


# ---------------------------------------------------------------------------
# String-safe column extraction (NaN/None -> "nan")
# ---------------------------------------------------------------------------
def _safe_str(v):
    if v is None:
        return "nan"
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        if v == int(v):
            return str(int(v))
        return str(v)
    return str(v)


def _col_str(df, c):
    n = len(df)
    if c not in df.columns:
        return ["nan"] * n
    return [_safe_str(v) for v in df[c].tolist()]


# ---------------------------------------------------------------------------
# Leakage-safe smoothed target encoding
# ---------------------------------------------------------------------------
def _te_keys(df):
    """Return dict {te_col_name: list-of-category-keys} for the current frame."""
    occ = _col_str(df, "occupation")
    coupon = _col_str(df, "coupon")
    return {
        "te_occ": occ,
        "te_coupon_occ": [c + "||" + o for c, o in zip(coupon, occ)],
    }


class TargetEncoder(object):
    """Smoothed target statistics: encoded = (sum_y + prior*m) / (count + m)."""

    def __init__(self, m=TE_SMOOTH_M):
        self.m = int(m)
        self.prior = 0.0
        self.maps = {}
        self.te_col_names = list(_te_keys(pd.DataFrame({"occupation": [], "coupon": []})).keys())
        self.fitted = False

    def fit(self, df, y):
        y = np.asarray(y, dtype=np.float64)
        self.m = int(TE_SMOOTH_M)
        self.prior = float(np.mean(y)) if len(y) else 0.0
        keys = _te_keys(df)
        self.maps = {}
        for name, vals in keys.items():
            acc = {}
            for v, t in zip(vals, y):
                a = acc.get(v)
                if a is None:
                    a = [0.0, 0.0]
                    acc[v] = a
                a[0] += t
                a[1] += 1.0
            m, prior = self.m, self.prior
            self.maps[name] = {
                k: (a[0] + prior * m) / (a[1] + m) for k, a in acc.items()
            }
        self.te_col_names = list(keys.keys())
        self.fitted = True
        return self

    def transform(self, df):
        n = len(df)
        out = np.zeros((n, len(self.te_col_names)), dtype=np.float32)
        keys = _te_keys(df)
        for j, name in enumerate(self.te_col_names):
            mp = self.maps.get(name, {})
            prior = self.prior
            vals = keys.get(name, ["nan"] * n)
            out[:, j] = np.array(
                [mp.get(v, prior) for v in vals], dtype=np.float32
            )
        return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_features(df, ohe=None, fit=False):
    n = len(df)

    # 1) plain categorical block
    cat_data = {c: _col_str(df, c) for c in PLAIN_CATS}
    cat_df = pd.DataFrame(cat_data)

    # 2) ordinal-encoded block -> numeric
    num_data = {}
    for c in ORDINAL_CATS:
        mp = ORDINAL_MAPS.get(c, {})
        vals = _col_str(df, c)
        num_data[c + "_ord"] = np.array([mp.get(v, -1) for v in vals], dtype=np.float32)
    # 3) direct numeric block
    for c in NUMERIC_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            num_data[c + "_num"] = s.fillna(-1).astype(np.float32).values
        else:
            num_data[c + "_num"] = np.full(n, -1, dtype=np.float32)

    # 3b) explicit missingness flags for the FREQ columns
    for c in MISSING_FLAG_COLS:
        vals = _col_str(df, c)
        num_data["miss_" + c] = np.array(
            [1.0 if v == "nan" else 0.0 for v in vals], dtype=np.float32
        )

    # 4) cheap interactions
    income = num_data["income_ord"]
    age = num_data["age_ord"]
    num_data["income_x_age"] = (income * age).astype(np.float32)
    dist = (
        num_data["toCoupon_GEQ5min_num"]
        + num_data["toCoupon_GEQ15min_num"]
        + num_data["toCoupon_GEQ25min_num"]
    )
    num_data["dist_sum"] = dist.astype(np.float32)
    num_data["dist_x_same"] = (dist * num_data["direction_same_num"]).astype(np.float32)

    # 4b) distance bucket + coupon-group interaction
    g25 = num_data["toCoupon_GEQ25min_num"]
    g15 = num_data["toCoupon_GEQ15min_num"]
    dist_bucket = np.where(g25 >= 1, 0.0, np.where(g15 >= 1, 1.0, 2.0)).astype(np.float32)
    num_data["dist_bucket"] = dist_bucket

    coupon_str = _col_str(df, "coupon")
    cg = np.array([COUPON_GROUPS.get(v, 0) for v in coupon_str], dtype=np.float32)
    num_data["coupon_group"] = cg
    num_data["coupon_group_x_dist"] = (cg + 1.0) * (dist_bucket + 1.0)

    # 4c) interest-match: user's venue frequency ordinal vs coupon venue type
    interest = np.full(n, -1.0, dtype=np.float32)
    for cname, ven_col in COUPON_VENUE.items():
        mask = np.array([v == cname for v in coupon_str], dtype=bool)
        if mask.any() and ven_col in num_data:
            interest[mask] = num_data[ven_col][mask]
    num_data["interest_match"] = interest

    num_df = pd.DataFrame(num_data)

    # 5) one-hot for plain categoricals
    if fit or ohe is None:
        ohe = _make_ohe()
        X_cat = ohe.fit_transform(cat_df)
    else:
        X_cat = ohe.transform(cat_df)
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()
    X_cat = np.asarray(X_cat, dtype=np.float32)

    X = np.hstack([X_cat, num_df.values.astype(np.float32)])
    return X, ohe


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def _make_models(seed, n_classes):
    """Two strong, decorrelated booster families (ExtraTrees dropped)."""
    models = []
    models.append(
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_iter=800,
                learning_rate=0.03,
                max_leaf_nodes=31,
                min_samples_leaf=25,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=int(seed),
            ),
        )
    )
    if HAS_LGB:
        models.append(
            (
                "lgb",
                lgb.LGBMClassifier(
                    n_estimators=1200,
                    learning_rate=0.03,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=int(seed),
                    n_jobs=2,
                    verbose=-1,
                ),
            )
        )
    return models


def _is_lgb(model):
    return HAS_LGB and type(model).__name__ == "LGBMClassifier"


def _lgb_fit(model, X, y, n_classes):
    """LightGBM with an internal early-stopping split carved from training rows."""
    n = len(X)
    metric = "binary_logloss" if n_classes <= 2 else "multi_logloss"
    if n > 300:
        rng = np.random.RandomState(0)
        perm = rng.permutation(n)
        nv = max(50, int(0.15 * n))
        va = perm[:nv]
        tr = perm[nv:]
        try:
            model.fit(
                X[tr],
                y[tr],
                eval_set=[(X[va], y[va])],
                eval_metric=metric,
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            return model
        except Exception:
            pass
    model.fit(X, y)
    return model


def _fit_model(model, X, y, n_classes):
    if _is_lgb(model):
        return _lgb_fit(model, X, y, n_classes)
    model.fit(X, y)
    return model


def _fit_predict(model, X_tr, y_tr, X_te, n_classes):
    _fit_model(model, X_tr, y_tr, n_classes)
    p = np.asarray(model.predict_proba(X_te), dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != n_classes:
        idx = p.argmax(axis=1) if p.ndim == 2 else np.zeros(len(X_te), dtype=int)
        p = np.eye(n_classes)[idx]
    return model, p


# ---------------------------------------------------------------------------
# Blend-weight search (coordinate ascent on the simplex)
# ---------------------------------------------------------------------------
def _learn_blend_weights(oof, names, y):
    n = len(names)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    w = np.ones(n, dtype=np.float64) / n

    def _acc(weights):
        prob = np.zeros_like(oof[names[0]])
        for i, nm in enumerate(names):
            prob += oof[nm] * weights[i]
        return float((prob.argmax(1) == y).mean())

    best = _acc(w)
    for _ in range(60):
        improved = False
        for i in range(n):
            for delta in (0.20, -0.20, 0.10, -0.10, 0.05, -0.05):
                w2 = w.copy()
                w2[i] = max(0.0, w2[i] + delta)
                s = float(w2.sum())
                if s <= 1e-9:
                    continue
                w2 /= s
                acc = _acc(w2)
                if acc > best + 1e-6:
                    best = acc
                    w = w2
                    improved = True
        if not improved:
            break
    return [float(x) for x in w]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    schema = _load_schema(data_dir)
    df = _read_features(data_dir, schema)
    y_raw = _read_labels(data_dir)
    if len(y_raw) != len(df):
        raise RuntimeError(
            f"[train] features rows ({len(df)}) != labels rows ({len(y_raw)})"
        )

    labels_arr = np.unique(np.asarray(y_raw))
    labels_list = labels_arr.tolist()
    l2i = {l: i for i, l in enumerate(labels_list)}
    y = np.array([l2i[v] for v in y_raw], dtype=np.int64)
    n_classes = len(labels_list)

    # Base (leak-free) design matrix: one-hot + ordinals + numerics + flags +
    # interactions.  The target-encoded block is appended per-fold below.
    X_base, ohe = build_features(df, ohe=None, fit=True)
    n_base = int(X_base.shape[1])

    member_names = [f"hgb_{s}" for s in SEED_BAG]
    if HAS_LGB:
        member_names += [f"lgb_{s}" for s in SEED_BAG]
    print(
        f"[train] rows={len(df)} n_classes={n_classes} n_base={n_base} "
        f"members={member_names} labels={labels_list} lgb={HAS_LGB}"
    )

    # Repeated stratified CV: TE fitted on TRAIN rows only per fold.
    oof_sum = {nm: np.zeros((len(y), n_classes), dtype=np.float64) for nm in member_names}
    oof_cnt = {nm: np.zeros(len(y), dtype=np.float64) for nm in member_names}
    for r in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + r)
        for k, (tr_idx, va_idx) in enumerate(skf.split(X_base, y)):
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            te = TargetEncoder()
            te.fit(df.iloc[tr_idx], y[tr_idx])
            X_tr = np.hstack([X_base[tr_idx], te.transform(df.iloc[tr_idx])])
            X_va = np.hstack([X_base[va_idx], te.transform(df.iloc[va_idx])])
            for s in SEED_BAG:
                for nm, m in _make_models(s, n_classes):
                    mname = f"{nm}_{s}"
                    try:
                        _, p = _fit_predict(m, X_tr, y[tr_idx], X_va, n_classes)
                        oof_sum[mname][va_idx] += p
                        oof_cnt[mname][va_idx] += 1.0
                    except Exception as exc:
                        print(f"[train] rep {r} fold {k} {mname} failed: {exc}")

    oof = {}
    for nm in member_names:
        cnt = np.maximum(oof_cnt[nm], 1.0)[:, None]
        oof[nm] = oof_sum[nm] / cnt

    accs = {nm: float((oof[nm].argmax(1) == y).mean()) for nm in member_names}
    print("[train] OOF " + " | ".join(f"{nm}={accs[nm]:.4f}" for nm in member_names))

    weights = _learn_blend_weights(oof, member_names, y)
    if len(weights) != len(member_names):
        weights = [1.0 / len(member_names)] * len(member_names)
    blend_prob = np.zeros((len(y), n_classes), dtype=np.float64)
    for i, nm in enumerate(member_names):
        blend_prob += oof[nm] * weights[i]
    blend_acc = float((blend_prob.argmax(1) == y).mean())
    print(
        f"[train] blend weights={np.round(weights, 3).tolist()} "
        f"acc={blend_acc:.4f}"
    )

    # Final TE maps fitted on ALL training rows for inference.  Training
    # features are cross-fitted so no row's own label leaks into its features.
    te_final = TargetEncoder()
    te_final.fit(df, y)
    X_full_te = np.zeros(
        (len(df), len(te_final.te_col_names)), dtype=np.float32
    )
    te_cnt = np.zeros(len(df), dtype=np.float64)
    for r in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + r)
        for tr_idx, va_idx in skf.split(X_base, y):
            te_cv = TargetEncoder()
            te_cv.fit(df.iloc[tr_idx], y[tr_idx])
            X_full_te[va_idx] += te_cv.transform(df.iloc[va_idx])
            te_cnt[va_idx] += 1.0
    te_cnt = np.maximum(te_cnt, 1.0)[:, None]
    X_full_te = X_full_te / te_cnt
    X_full = np.hstack([X_base, X_full_te])

    final_models = []
    for s in SEED_BAG:
        for nm, m in _make_models(s, n_classes):
            mname = f"{nm}_{s}"
            try:
                _fit_model(m, X_full, y, n_classes)
                final_models.append((mname, m))
            except Exception as exc:
                print(f"[train] final {mname} failed: {exc}")

    if not final_models:
        raise RuntimeError("[train] no base model trained successfully")

    avail_names = [nm for nm, _ in final_models]
    weight_map = dict(zip(member_names, weights))
    blend_weights = [float(weight_map.get(nm, 0.0)) for nm in avail_names]
    if sum(blend_weights) <= 0:
        blend_weights = [1.0 / len(avail_names)] * len(avail_names)

    counts = np.bincount(y, minlength=n_classes)
    majority_idx = int(counts.argmax())

    bundle = {
        "ohe": ohe,
        "te": te_final,
        "models": final_models,
        "model_names": avail_names,
        "blend_weights": blend_weights,
        "labels": labels_list,
        "n_classes": n_classes,
        "has_lgb": HAS_LGB,
        "n_base": n_base,
        "majority_idx": majority_idx,
    }
    path = os.path.join(model_dir, "bundle.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[train] saved bundle -> {path} " f"(models={[nm for nm,_ in final_models]})"
    )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict(data_dir, model_dir, output_path):
    path = os.path.join(model_dir, "bundle.pkl")
    if not os.path.exists(path):
        raise RuntimeError(f"[predict] missing bundle: {path}")
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    schema = _load_schema(data_dir)
    df = _read_features(data_dir, schema)
    X_base, _ = build_features(df, ohe=bundle["ohe"], fit=False)
    te = bundle.get("te")
    if te is not None:
        X = np.hstack([X_base, te.transform(df)])
    else:
        X = X_base
    n_base = bundle.get("n_base")
    if n_base is not None and int(n_base) != int(X_base.shape[1]):
        print(
            f"[predict] WARNING base width {X_base.shape[1]} != trained {n_base}"
        )
    n_classes = int(bundle["n_classes"])
    labels = bundle["labels"]
    names = bundle["model_names"]
    weights = np.asarray(bundle["blend_weights"], dtype=np.float64)
    models = dict(bundle["models"])

    if len(weights) != len(names):
        weights = np.ones(len(names), dtype=np.float64) / max(len(names), 1)
    if weights.sum() <= 0:
        weights = np.ones(len(names), dtype=np.float64) / max(len(names), 1)
    else:
        weights = weights / weights.sum()

    prob = np.zeros((len(df), n_classes), dtype=np.float64)
    used = []
    for i, nm in enumerate(names):
        m = models.get(nm)
        if m is None:
            continue
        try:
            p = np.asarray(m.predict_proba(X), dtype=np.float64)
            if p.ndim != 2 or p.shape[1] != n_classes:
                idx = p.argmax(axis=1) if p.ndim == 2 else np.zeros(len(X), int)
                p = np.eye(n_classes)[idx]
            prob += p * weights[i]
            used.append(nm)
        except Exception as exc:
            print(f"[predict] model {nm} failed: {exc}")

    if not used:
        print("[predict] no model succeeded; falling back to majority class")
        prob = np.zeros((len(df), n_classes), dtype=np.float64)
        prob[:, int(bundle.get("majority_idx", 0))] = 1.0
    else:
        w_sum = float(weights[[names.index(nm) for nm in used]].sum())
        if w_sum > 0:
            prob = prob / w_sum
        print(f"[predict] used={used} weights={np.round(weights, 3).tolist()}")

    idx = prob.argmax(axis=1)
    pred_labels = [labels[int(i)] for i in idx]

    out_df = pd.DataFrame(
        {
            "row_id": np.arange(len(df), dtype=np.int64),
            "prediction": pred_labels,
        }
    )
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[predict] wrote {output_path} ({len(out_df)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Coupon acceptance prediction")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--model", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.data, args.model)
    elif args.cmd == "predict":
        predict(args.data, args.model, args.output)
    else:
        ap.error("unknown command")


if __name__ == "__main__":
    main()