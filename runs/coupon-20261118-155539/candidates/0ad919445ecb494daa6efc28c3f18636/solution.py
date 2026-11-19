"""
Coupon acceptance prediction — GBDT ensemble with native categorical handling.

Tier 2/3 redesign: replace the DeepFM + one-hot + stacking pipeline with a
strong gradient-boosting ensemble:

  * 2 x CatBoost (native categorical features via cat_features=strings)
  * 2 x LightGBM       (numeric + TE frame)
  * XGBoost, HGB, ExtraTrees (numeric + TE frame)

All members are stacked via a small logistic-regression meta-learner trained
on OOF probabilities, with an OOF-driven threshold search. The leakage-safe
feature engineer (inner-fold target encodings + coupon-conditional centering)
is preserved from the previous solution.
"""

from __future__ import annotations

import os
import json
import argparse
import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

# ==========================================================================
# STEP 1: DATA PROCESSING & FEATURE ENGINEERING
# ==========================================================================
FREQ_MAP = {"never": 0.0, "less1": 1.0, "1~3": 2.0, "4~8": 3.0, "gt8": 4.0}
AGE_MAP = {
    "below21": 0.0,
    "21": 1.0,
    "26": 2.0,
    "31": 3.0,
    "36": 4.0,
    "41": 5.0,
    "46": 6.0,
    "50plus": 7.0,
}
INCOME_MAP = {
    "Less than $12500": 0.0,
    "$12500 - $24999": 1.0,
    "$25000 - $37499": 2.0,
    "$37500 - $49999": 3.0,
    "$50000 - $62499": 4.0,
    "$62500 - $74999": 5.0,
    "$75000 - $87499": 6.0,
    "$87500 - $99999": 7.0,
    "$100000 or More": 8.0,
}
EDUCATION_MAP = {
    "Some High School": 0.0,
    "High School Graduate": 1.0,
    "Some college - no degree": 2.0,
    "Associates degree": 3.0,
    "Bachelors degree": 4.0,
    "Graduate degree (Masters, Doctorate)": 5.0,
}
TIME_MAP = {"7AM": 0.0, "10AM": 1.0, "2PM": 2.0, "6PM": 3.0, "10PM": 4.0}
EXPIRATION_MAP = {"2h": 0.0, "1d": 1.0}

FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]
HABIT_COLUMN = {
    "Bar": "Bar_num",
    "Coffee House": "CoffeeHouse_num",
    "Restaurant(<20)": "RestaurantLessThan20_num",
    "Restaurant(20-50)": "Restaurant20To50_num",
    "Carry out & Take away": "CarryAway_num",
}
NOMINAL_COLS = ["coupon", "gender", "maritalStatus", "occupation"]
ORDINAL_MAPS = {
    "time": TIME_MAP,
    "expiration": EXPIRATION_MAP,
    "age": AGE_MAP,
    "income": INCOME_MAP,
    "education": EDUCATION_MAP,
}
NUMERIC_COLS = [
    "temperature",
    "has_children",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
]
# String columns passed to CatBoost with native cat_features.
CAT_STRING_COLS = [
    "coupon",
    "gender",
    "maritalStatus",
    "occupation",
    "age",
    "income",
    "education",
    "time",
    "expiration",
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]


class FeatureEngineer:
    """Stateful, leakage-safe feature transformer."""

    TE_KEYS = [
        "coupon",
        "occupation",
        "coupon_x_occupation",
        "coupon_x_time",
        "coupon_x_expiration",
        "coupon_x_distance",
        "coupon_x_direction",
        "coupon_x_age",
        "coupon_x_children",
    ]
    TE_COLS = ["te__" + k for k in TE_KEYS]

    CC_COLS = [
        "time_num",
        "expiration_num",
        "distance_bucket",
        "direction_same_num",
        "habit_match",
        "age_num",
        "income_num",
        "has_children_num",
    ]

    def __init__(self, te_smoothing: float = 25.0):
        self.fill_values = {}
        self.nominal_categories = {}
        self.feature_columns = None
        self.te_smoothing = float(te_smoothing)
        self.te_prior = 0.5
        self.te_maps = {}
        self.te_active = False
        self._te_recursion = False
        self.coupon_means = {}
        self.coupon_global_means = {}

    @staticmethod
    def _normalize(s):
        s = s.astype("object")
        out = s.astype(str).str.strip()
        return out.where(~out.isin(["nan", "None", "", "NaN"]), np.nan)

    @staticmethod
    def _col_or_nan(df, name, n):
        if name in df.columns:
            return df[name].reset_index(drop=True)
        return pd.Series([np.nan] * n, dtype="object")

    def fit(self, df, y=None):
        df = df.reset_index(drop=True)
        for c in FREQ_COLS:
            self.fill_values[c] = -1.0
        for c, mapping in ORDINAL_MAPS.items():
            s = self._normalize(self._col_or_nan(df, c, len(df)))
            mapped = s.map(mapping)
            mode = mapped.dropna().mode()
            self.fill_values[c] = float(mode.iloc[0]) if len(mode) else 0.0
        for c in NOMINAL_COLS:
            s = self._normalize(self._col_or_nan(df, c, len(df))).fillna("missing")
            self.nominal_categories[c] = sorted(s.dropna().unique().tolist())
        for c in NUMERIC_COLS:
            v = pd.to_numeric(self._col_or_nan(df, c, len(df)), errors="coerce")
            med = v.dropna().median()
            self.fill_values[c] = float(med) if pd.notna(med) else 0.0

        self.transform(df)  # establish feature_columns

        n_fit = len(df)
        coupon_arr_fit = (
            self._normalize(self._col_or_nan(df, "coupon", n_fit))
            .fillna("missing")
            .to_numpy()
            .astype(str)
        )
        base_df = self.transform(df)
        self.coupon_means = {}
        self.coupon_global_means = {}
        for c in self.CC_COLS:
            if c not in base_df.columns:
                continue
            vals = base_df[c].to_numpy(np.float64)
            finite = np.isfinite(vals)
            gmean = float(vals[finite].mean()) if finite.any() else 0.0
            self.coupon_global_means[c] = gmean
            per = {}
            for cp in np.unique(coupon_arr_fit):
                mask = coupon_arr_fit == cp
                v = vals[mask]
                v = v[np.isfinite(v)]
                per[str(cp)] = float(v.mean()) if len(v) else gmean
            self.coupon_means[c] = per
        self.feature_columns = None
        self.transform(df)

        if y is not None:
            y_arr = np.asarray(y).astype(np.float64).ravel()
            if len(y_arr) == len(df) and len(y_arr) > 0:
                self.te_prior = float(np.clip(y_arr.mean(), 1e-6, 1.0 - 1e-6))
                m = self.te_smoothing
                keys = self._build_te_frame(df)
                self.te_maps = {}
                for k in self.TE_KEYS:
                    kv = keys[k].astype(str)
                    agg = (
                        pd.DataFrame({"k": kv, "y": y_arr})
                        .groupby("k")["y"]
                        .agg(["sum", "count"])
                    )
                    prior = self.te_prior
                    smoothed = (agg["sum"] + m * prior) / (agg["count"] + m)
                    self.te_maps[k] = {
                        str(idx): (float(smoothed.loc[idx]), int(agg["count"].loc[idx]))
                        for idx in agg.index
                    }
                self.te_active = True
                self.feature_columns = None
                self.transform(df)
        return self

    def _raw_te_components(self, df):
        n = len(df)
        out = self.transform(df)
        coupon_arr = (
            self._normalize(self._col_or_nan(df, "coupon", n))
            .fillna("missing")
            .to_numpy()
            .astype(str)
        )
        occ_arr = (
            self._normalize(self._col_or_nan(df, "occupation", n))
            .fillna("missing")
            .to_numpy()
            .astype(str)
        )
        age_num = out["age_num"].to_numpy()
        age_bucket = np.floor(age_num / 2.0).astype(int).astype(str)
        children = out["has_children_num"].to_numpy().astype(int).astype(str)
        time_v = out["time_num"].to_numpy().astype(str)
        exp_v = out["expiration_num"].to_numpy().astype(str)
        dist_v = out["distance_bucket"].astype(str)
        dir_v = out["direction_same_num"].to_numpy().astype(str)
        return {
            "coupon": coupon_arr,
            "occupation": occ_arr,
            "coupon_x_occupation": coupon_arr + "||" + occ_arr,
            "coupon_x_time": coupon_arr + "||" + time_v,
            "coupon_x_expiration": coupon_arr + "||" + exp_v,
            "coupon_x_distance": coupon_arr + "||" + dist_v,
            "coupon_x_direction": coupon_arr + "||" + dir_v,
            "coupon_x_age": coupon_arr + "||" + age_bucket,
            "coupon_x_children": coupon_arr + "||" + children,
        }

    def _build_te_frame(self, df):
        return self._raw_te_components(df)

    def transform(self, df):
        df = df.reset_index(drop=True)
        n = len(df)
        out = {}

        for c, mapping in ORDINAL_MAPS.items():
            s = self._normalize(self._col_or_nan(df, c, n))
            out[c + "_num"] = (
                s.map(mapping).astype(float).fillna(self.fill_values.get(c, 0.0))
            )

        for c in FREQ_COLS:
            s = self._normalize(self._col_or_nan(df, c, n))
            out[c + "_num"] = (
                s.map(FREQ_MAP).astype(float).fillna(self.fill_values.get(c, -1.0))
            )

        for c in NUMERIC_COLS:
            v = pd.to_numeric(self._col_or_nan(df, c, n), errors="coerce").astype(float)
            out[c + "_num"] = v.fillna(self.fill_values.get(c, 0.0))

        g15 = out["toCoupon_GEQ15min_num"].to_numpy()
        g25 = out["toCoupon_GEQ25min_num"].to_numpy()
        bucket = np.where(g15 < 0.5, 0.0, np.where(g25 < 0.5, 1.0, 2.0))
        out["distance_bucket"] = bucket.astype(float)
        out["direction_and_close"] = out["direction_same_num"].to_numpy() * (
            g15 < 0.5
        ).astype(float)

        coupon_s = self._normalize(self._col_or_nan(df, "coupon", n)).fillna("missing")
        coupon_arr = coupon_s.to_numpy()
        habit_val = np.full(n, -1.0, dtype=float)
        for ctype, hcol in HABIT_COLUMN.items():
            mask = coupon_arr == ctype
            if mask.any():
                habit_val[mask] = out[hcol].to_numpy()[mask]
        out["habit_match"] = habit_val
        out["habit_high"] = (habit_val >= 2.0).astype(float)
        out["habit_missing"] = (habit_val < 0.0).astype(float)

        out["expiry_x_distance"] = (
            out["expiration_num"].to_numpy() * out["distance_bucket"]
        )
        out["has_children_x_age"] = (
            out["has_children_num"].to_numpy() * out["age_num"].to_numpy()
        )

        for col in NOMINAL_COLS:
            cats = self.nominal_categories.get(col, ["missing"])
            s = (
                self._normalize(self._col_or_nan(df, col, n))
                .fillna("missing")
                .to_numpy()
            )
            for c in cats:
                out[f"{col}__{c}"] = (s == c).astype(float)
            out[f"{col}__is_missing"] = (
                self._normalize(self._col_or_nan(df, col, n))
                .isna()
                .astype(float)
                .to_numpy()
            )

        if self.coupon_means:
            coupon_arr_s = coupon_arr.astype(str)
            for c in self.CC_COLS:
                if c not in out:
                    continue
                base = np.asarray(out[c], dtype=np.float64)
                gm = self.coupon_global_means.get(c, 0.0)
                per = self.coupon_means.get(c, {})
                means_arr = np.fromiter(
                    (per.get(k, gm) for k in coupon_arr_s),
                    dtype=np.float64,
                    count=len(coupon_arr_s),
                )
                out["cc__" + c] = (base - means_arr).astype(np.float32)

        if (
            self.te_active
            and self.te_maps
            and not getattr(self, "_te_recursion", False)
        ):
            self._te_recursion = True
            try:
                comp = self._raw_te_components(df)
            finally:
                self._te_recursion = False
            default = float(self.te_prior) if self.te_prior is not None else 0.5
            for te_col, k in zip(self.TE_COLS, self.TE_KEYS):
                mp = self.te_maps.get(k, {})
                kv = comp[k]
                vals = np.fromiter(
                    (mp[kk][0] if kk in mp else default for kk in kv),
                    dtype=np.float32,
                    count=len(kv),
                )
                out[te_col] = vals
                p = np.clip(vals, 1e-6, 1.0 - 1e-6)
                out[te_col + "_logit"] = (np.log(p) - np.log1p(-p)).astype(np.float32)

        result = pd.DataFrame(out)
        if self.feature_columns is None and not getattr(self, "_te_recursion", False):
            self.feature_columns = list(result.columns)
        if self.feature_columns is not None:
            result = result.reindex(columns=self.feature_columns, fill_value=0.0)
        return result.astype(np.float32)


def transform_native_cols(df):
    """Stateless: raw categorical strings for CatBoost native cat_features."""
    n = len(df)
    out = {}
    for c in CAT_STRING_COLS:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
        else:
            s = pd.Series(["missing"] * n, dtype=object)
        out[c] = s.where(~s.isin(["nan", "None", "", "NaN"]), "missing").to_numpy()
    return pd.DataFrame(out)


def drop_onehot(df):
    drop = [c for c in df.columns if any(c.startswith(p + "__") for p in NOMINAL_COLS)]
    return df.drop(columns=drop)


# ==========================================================================
# STEP 2: I/O
# ==========================================================================
def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def load_train(data_dir):
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    schema = load_schema(data_dir)
    return features, labels["label"].to_numpy(), schema


def load_predict(data_dir):
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    schema = load_schema(data_dir)
    return features, schema


# ==========================================================================
# STEP 3: MODEL ZOO (GBDT ONLY + NATIVE CAT)
# ==========================================================================
def build_ensemble_spec(seed=42):
    specs = []
    try:
        from catboost import CatBoostClassifier

        specs.append(
            (
                "cat_a",
                "cat",
                CatBoostClassifier(
                    iterations=900,
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=3.0,
                    random_seed=seed,
                    verbose=0,
                    thread_count=2,
                    bootstrap_type="Bernoulli",
                    subsample=0.85,
                ),
            )
        )
        specs.append(
            (
                "cat_b",
                "cat",
                CatBoostClassifier(
                    iterations=1100,
                    learning_rate=0.04,
                    depth=8,
                    l2_leaf_reg=6.0,
                    random_seed=seed + 1,
                    verbose=0,
                    thread_count=2,
                    bootstrap_type="Bernoulli",
                    subsample=0.85,
                ),
            )
        )
    except Exception as e:
        print(f"[zoo] catboost unavailable: {e}")

    try:
        import lightgbm as lgb

        specs.append(
            (
                "lgb_a",
                "num",
                lgb.LGBMClassifier(
                    n_estimators=600,
                    learning_rate=0.04,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    n_jobs=2,
                    random_state=seed,
                    verbose=-1,
                ),
            )
        )
        specs.append(
            (
                "lgb_b",
                "num",
                lgb.LGBMClassifier(
                    n_estimators=800,
                    learning_rate=0.03,
                    num_leaves=63,
                    min_child_samples=10,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=3.0,
                    n_jobs=2,
                    random_state=seed + 1,
                    verbose=-1,
                ),
            )
        )
    except Exception as e:
        print(f"[zoo] lightgbm unavailable: {e}")

    try:
        import xgboost as xgb

        specs.append(
            (
                "xgb",
                "num",
                xgb.XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.04,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    min_child_weight=3,
                    tree_method="hist",
                    n_jobs=2,
                    random_state=seed,
                    verbosity=0,
                    eval_metric="logloss",
                ),
            )
        )
    except Exception as e:
        print(f"[zoo] xgboost unavailable: {e}")

    specs.append(
        (
            "et",
            "num",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=3,
                min_samples_split=4,
                max_features="sqrt",
                n_jobs=2,
                random_state=seed,
            ),
        )
    )
    specs.append(
        (
            "hgb",
            "num",
            HistGradientBoostingClassifier(
                max_iter=600,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=seed,
            ),
        )
    )
    return specs


def _fit_one(spec_item, df_num, df_cat, y_tr):
    name, kind, mdl = spec_item
    if kind == "cat":
        X = pd.concat([drop_onehot(df_num), df_cat], axis=1)
        cat_idx = [X.columns.get_loc(c) for c in CAT_STRING_COLS if c in X.columns]
        mdl.fit(X, y_tr, cat_features=cat_idx)
    else:
        X = df_num.to_numpy(np.float32)
        mdl.fit(X, y_tr)
    return mdl


def _predict_one(spec_item, mdl, df_num, df_cat):
    name, kind, _ = spec_item
    if kind == "cat":
        X = pd.concat([drop_onehot(df_num), df_cat], axis=1)
        return mdl.predict_proba(X)[:, 1].astype(np.float64)
    else:
        X = df_num.to_numpy(np.float32)
        return mdl.predict_proba(X)[:, 1].astype(np.float64)


# ==========================================================================
# STEP 4: OOF, STACKING, FINAL FIT
# ==========================================================================
def build_inner_kfold_te(features_df, y_int, fe_full, n_splits=5, seed=42):
    df_full = fe_full.transform(features_df)
    te_cols = [c for c in df_full.columns if c.startswith("te__")]
    if not te_cols:
        return df_full
    te_vals = np.zeros((len(features_df), len(te_cols)), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(len(y_int)), y_int):
        fe_fold = FeatureEngineer().fit(features_df.iloc[tr], y=y_int[tr])
        part = fe_fold.transform(features_df.iloc[va])
        for j, c in enumerate(te_cols):
            if c in part.columns:
                te_vals[va, j] = part[c].to_numpy(np.float32)
            else:
                te_vals[va, j] = np.float32(fe_fold.te_prior)
    df_oof = df_full.copy()
    for j, c in enumerate(te_cols):
        df_oof[c] = te_vals[:, j]
    return df_oof


def encode_labels(y):
    y = np.asarray(y)
    if y.dtype.kind in "OUS":
        uniq = sorted(set(y.tolist()))
    else:
        uniq = sorted(set(y.astype(np.int64).tolist()))
    mapping = {v: i for i, v in enumerate(uniq)}
    yi = np.array([mapping[v] for v in y.tolist()], dtype=np.int64)
    return yi, uniq


def run_oof(features_df, y_int, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y_int)), y_int))
    oof = {}
    for fi, (tr, va) in enumerate(splits):
        fe = FeatureEngineer().fit(features_df.iloc[tr], y=y_int[tr])
        df_tr = build_inner_kfold_te(
            features_df.iloc[tr], y_int[tr], fe, n_splits=5, seed=seed
        )
        df_va = fe.transform(features_df.iloc[va])
        df_tr_cat = transform_native_cols(features_df.iloc[tr])
        df_va_cat = transform_native_cols(features_df.iloc[va])

        specs = build_ensemble_spec(seed=seed + fi * 11)
        msg = f"fold {fi} |"
        for spec_item in specs:
            name = spec_item[0]
            mdl = _fit_one(spec_item, df_tr, df_tr_cat, y_int[tr])
            p = _predict_one(spec_item, mdl, df_va, df_va_cat)
            if name not in oof:
                oof[name] = np.zeros(len(y_int))
            oof[name][va] = p
            acc = float(((p >= 0.5).astype(int) == y_int[va]).mean())
            msg += f" {name}:{acc:.4f}"
        print(msg)
    return oof


def fit_meta_stacker(stacked_oof, y_int):
    from sklearn.linear_model import LogisticRegression

    n = len(y_int)
    oof_meta = np.zeros(n)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    for tr_i, va_i in skf.split(stacked_oof, y_int):
        m = LogisticRegression(C=0.3, max_iter=2000)
        m.fit(stacked_oof[tr_i], y_int[tr_i])
        oof_meta[va_i] = m.predict_proba(stacked_oof[va_i])[:, 1]

    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.30, 0.701, 0.002):
        acc = float(((oof_meta >= t).astype(int) == y_int).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    print(f"stacking meta-CV acc = {best_acc:.5f} at t = {best_t:.4f}")

    meta = LogisticRegression(C=0.3, max_iter=2000)
    meta.fit(stacked_oof, y_int)
    return meta, best_t, best_acc


def final_fit(features_df, y_int, seed=100):
    fe = FeatureEngineer().fit(features_df, y=y_int)
    df_num = build_inner_kfold_te(features_df, y_int, fe, n_splits=5, seed=42)
    df_cat = transform_native_cols(features_df)

    specs = build_ensemble_spec(seed=seed)
    fitted = []
    for spec_item in specs:
        name = spec_item[0]
        mdl = _fit_one(spec_item, df_num, df_cat, y_int)
        fitted.append((name, spec_item[1], mdl, spec_item[2]))
        print(f"  fitted {name}")
    return fe, fitted


# ==========================================================================
# STEP 5: PERSISTENCE
# ==========================================================================
def save_artifacts(
    model_dir, fe, fitted, meta_learner, model_names, threshold, code_to_label
):
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "bundle.pkl"), "wb") as f:
        pickle.dump(
            {
                "fe": fe,
                "fitted": fitted,  # list of (name, kind, mdl, original_spec_mdl)
                "meta_learner": meta_learner,
                "model_names": list(model_names),
                "threshold": float(threshold),
                "code_to_label": list(code_to_label),
                "cat_columns": list(CAT_STRING_COLS),
            },
            f,
        )


def load_artifacts(model_dir):
    with open(os.path.join(model_dir, "bundle.pkl"), "rb") as f:
        return pickle.load(f)


# ==========================================================================
# STEP 6: PREDICT
# ==========================================================================
def predict_pipeline(data_dir, model_dir, output_path):
    features, _ = load_predict(data_dir)
    bundle = load_artifacts(model_dir)
    fe = bundle["fe"]
    fitted = bundle["fitted"]
    meta_learner = bundle["meta_learner"]
    model_names = bundle["model_names"]
    threshold = float(bundle["threshold"])
    code_to_label = bundle["code_to_label"]

    df_num = fe.transform(features)
    df_cat = transform_native_cols(features)

    preds = {}
    for name, kind, mdl, _orig in fitted:
        spec_stub = (name, kind, mdl)
        preds[name] = _predict_one(spec_stub, mdl, df_num, df_cat)

    n = len(df_num)
    stacked = np.column_stack([preds.get(nm, np.full(n, 0.5)) for nm in model_names])
    p_meta = meta_learner.predict_proba(stacked)[:, 1]
    pred_code = (p_meta >= threshold).astype(np.int64)
    pred_label = np.array([code_to_label[c] for c in pred_code.tolist()])

    out = pd.DataFrame(
        {"row_id": np.arange(len(pred_label), dtype=np.int64), "prediction": pred_label}
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"wrote {len(out)} predictions to {output_path}")
    return out


# ==========================================================================
# STEP 7: MAIN
# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train")
    pt.add_argument("--data", required=True)
    pt.add_argument("--model", required=True)
    pp = sub.add_parser("predict")
    pp.add_argument("--data", required=True)
    pp.add_argument("--model", required=True)
    pp.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.cmd == "train":
        features, y_raw, _ = load_train(args.data)
        y_int, label_values = encode_labels(y_raw)
        print(f"n_train={len(features)} pos_rate={y_int.mean():.4f}")

        oof = run_oof(features, y_int, n_splits=5, seed=42)
        model_names = sorted(oof.keys())
        stacked = np.column_stack([oof[n] for n in model_names])
        print(f"stacked OOF shape={stacked.shape} names={model_names}")

        meta_learner, best_t, best_acc = fit_meta_stacker(stacked, y_int)
        print(f"Final stacked validation accuracy = {best_acc:.5f} (t={best_t:.4f})")

        fe, fitted = final_fit(features, y_int, seed=100)
        save_artifacts(
            args.model,
            fe,
            fitted,
            meta_learner,
            model_names,
            best_t,
            label_values,
        )
        print("artifacts saved")
        print(f"Final Validation Score: {best_acc:.5f}")
    else:
        predict_pipeline(args.data, args.model, args.output)


if __name__ == "__main__":
    main()
