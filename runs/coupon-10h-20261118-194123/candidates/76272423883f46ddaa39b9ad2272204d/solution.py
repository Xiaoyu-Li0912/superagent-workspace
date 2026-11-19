# solution.py
"""
Coupon acceptance prediction — multi-model stacked ensemble.

Pipeline:
  Leakage-safe FE (fold-fit for CV, full-train-fit at refit) with CV-safe
  target-encoding of interaction categoricals. Base learners per fold:
    * LightGBM  (native cats + TE numerics)
    * CatBoost  (native cats)
    * Embedding-MLP (focal loss)
  OOF probs -> LogisticRegression stacker (robust at n~8k) + threshold tune.
  Refit all three on full TRAIN, apply persisted stacker, write predictions.

CLI:
  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

# ===========================================================================
# Constants / FE
# ===========================================================================
VENUE_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]
VENUE_INDEX = {c: i for i, c in enumerate(VENUE_COLS)}
FREQ_MAP = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
TIME_MAP = {"7am": 0, "10am": 1, "2pm": 2, "6pm": 3, "10pm": 4}
TIME_HOUR = {"7am": 7, "10am": 10, "2pm": 14, "6pm": 18, "10pm": 22}

RAW_CAT_COLS = [
    "time",
    "expiration",
    "gender",
    "maritalStatus",
    "education",
    "occupation",
    "income",
    "age",
]

INTERACTION_COLS = [
    "coupon_x_exp",
    "coupon_x_dir",
    "coupon_x_age",
    "coupon_x_time",
    "gender_x_coupon",
    "children_x_coupon",
    "distance_x_coupon",
    "time_x_exp",
]


def _norm(s):
    return str(s).strip().lower()


def _coupon_to_venue(coupon_val):
    s = _norm(coupon_val)
    if "bar" in s:
        return "Bar"
    if "coffee" in s or "cafe" in s:
        return "CoffeeHouse"
    if "carry" in s or "take" in s or "away" in s:
        return "CarryAway"
    if "rest" in s:
        if "20" in s and "50" in s:
            return "Restaurant20To50"
        if "<" in s or "less" in s or "under" in s:
            return "RestaurantLessThan20"
        return "Restaurant20To50" if "20" in s else "RestaurantLessThan20"
    return None


def _age_ord(v):
    s = _norm(v)
    if "below" in s or "under" in s:
        return 0
    if "plus" in s or "+" in s:
        return 7
    m = re.search(r"\d+", s)
    if m:
        n = int(m.group())
        return int(max(1, min(6, (n - 21) // 5 + 1)))
    return -1


def _income_ord(v):
    s = _norm(v).replace(",", "").replace(" ", "")
    if "less" in s:
        return 0
    if "more" in s or "100000" in s:
        return 8
    m = re.search(r"(\d+)", s)
    if m:
        lo = int(m.group(1))
        return int(max(0, min(8, lo // 12500)))
    return -1


def _edu_ord(v):
    s = _norm(v)
    if "some high" in s:
        return 0
    if "high school" in s and "some" not in s:
        return 1
    if "some college" in s or "no degree" in s:
        return 2
    if "associate" in s:
        return 3
    if "bachelor" in s:
        return 4
    if "graduate" in s or "master" in s or "doctor" in s:
        return 5
    return -1


def _time_ord(v):
    return TIME_MAP.get(_norm(v).replace(" ", ""), -1)


def _time_hour(v):
    return TIME_HOUR.get(_norm(v).replace(" ", ""), -1)


class CouponFeatureEngineer:
    """Deterministic engineering + freq + CV-safe target-encoded interactions."""

    def __init__(self):
        self.freq_maps_ = {}
        self.freq_encode_cols_ = []
        self.freq_col_names_ = []
        self.te_maps_ = {}
        self.te_prior_ = 0.5
        self.te_cols_ = []
        self.categorical_columns = []
        self.numeric_columns = []
        self._cat_base = []
        self._num_base = []
        self.fitted = False

    def _build_base(self, X):
        X = X.copy()
        n = len(X)
        out = pd.DataFrame(index=X.index)
        coupon = X["coupon"].astype(str)
        out["coupon"] = coupon.str.strip()
        out["coupon_canon"] = coupon.map(lambda c: _coupon_to_venue(c) or "unknown")
        for col in RAW_CAT_COLS:
            out[col] = X[col].astype(str).str.strip()
        out["direction"] = np.where(
            pd.to_numeric(X["direction_same"], errors="coerce").fillna(0) == 1,
            "same",
            "opp",
        )
        for col in VENUE_COLS:
            out[col] = (
                X[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": "never", "": "never", "none": "never"})
            )
        out["has_children"] = (
            pd.to_numeric(X["has_children"], errors="coerce").fillna(0).astype(int)
        )
        out["temperature"] = pd.to_numeric(X["temperature"], errors="coerce")
        g5 = pd.to_numeric(X["toCoupon_GEQ5min"], errors="coerce").fillna(0)
        g15 = pd.to_numeric(X["toCoupon_GEQ15min"], errors="coerce").fillna(0)
        g25 = pd.to_numeric(X["toCoupon_GEQ25min"], errors="coerce").fillna(0)
        out["distance_level"] = (
            (g5 > 0).astype(int) + (g15 > 0).astype(int) + (g25 > 0).astype(int)
        )
        out["is_25min"] = (g25 > 0).astype(int)
        out["is_15min"] = (g15 > 0).astype(int)
        out["is_5min_only"] = ((g15 == 0) & (g25 == 0)).astype(int)
        out["direction_same"] = (
            pd.to_numeric(X["direction_same"], errors="coerce").fillna(0).astype(int)
        )
        out["is_2h"] = (
            out["expiration"]
            .str.lower()
            .str.replace(" ", "", regex=False)
            .str.contains("2h")
            .astype(int)
        )
        out["age_ord"] = out["age"].map(_age_ord)
        out["income_ord"] = out["income"].map(_income_ord)
        out["education_ord"] = out["education"].map(_edu_ord)
        out["time_ord"] = out["time"].map(_time_ord)
        out["hour"] = out["time"].map(_time_hour)

        freq_arr = {c: out[c].map(FREQ_MAP).fillna(0).astype(int) for c in VENUE_COLS}
        A = np.column_stack([freq_arr[c].values for c in VENUE_COLS])
        vidx = np.array([VENUE_INDEX.get(v, -1) for v in out["coupon_canon"].values])
        matched = np.where(
            vidx >= 0, A[np.arange(n), np.clip(vidx, 0, len(VENUE_COLS) - 1)], 0
        )
        out["coupon_venue_freq"] = matched.astype(int)
        out["coupon_venue_visited"] = (matched > 0).astype(int)
        out["total_engagement"] = A.sum(1)
        out["mean_engagement"] = A.mean(1)
        out["engagement_std"] = A.std(1)
        out["matched_minus_mean"] = matched - A.mean(1)
        out["n_venues_visited"] = (A > 0).sum(1)

        occ = out["occupation"].str.lower()
        out["is_student"] = occ.str.contains("student").astype(int)
        out["is_retired"] = occ.str.contains("retired").astype(int)
        out["is_unemployed"] = occ.str.contains("unemploy").astype(int)
        out["gender_male"] = out["gender"].str.lower().str.startswith("m").astype(int)

        out["coupon_x_exp"] = out["coupon_canon"] + "_" + out["expiration"]
        out["coupon_x_dir"] = out["coupon_canon"] + "_" + out["direction"]
        out["gender_x_coupon"] = out["gender"] + "_" + out["coupon_canon"]
        out["children_x_coupon"] = (
            out["has_children"].astype(str) + "_" + out["coupon_canon"]
        )
        out["coupon_x_time"] = out["coupon_canon"] + "_" + out["time"]
        out["coupon_x_age"] = out["coupon_canon"] + "_" + out["age"]
        out["time_x_exp"] = out["time"] + "_" + out["expiration"]
        out["distance_x_coupon"] = (
            out["distance_level"].astype(str) + "_" + out["coupon_canon"]
        )
        return out

    def _fit_te(self, base, y, cols, prior_weight=20.0):
        prior = float(np.mean(y)) if len(y) else 0.5
        maps = {}
        for c in cols:
            grp = pd.DataFrame(
                {"v": base[c].astype(str).values, "y": np.asarray(y, dtype=float)}
            )
            agg = grp.groupby("v")["y"].agg(["mean", "count"])
            smoothed = (agg["mean"] * agg["count"] + prior * prior_weight) / (
                agg["count"] + prior_weight
            )
            maps[c] = smoothed.to_dict()
        return maps, prior

    def fit(self, X, y=None):
        base = self._build_base(X)
        n = max(len(base), 1)
        self._cat_base = [
            c for c in base.columns if not pd.api.types.is_numeric_dtype(base[c])
        ]
        self._num_base = [c for c in base.columns if c not in self._cat_base]
        truly_num, truly_cat = [], list(self._cat_base)
        for c in self._num_base:
            coerced = pd.to_numeric(base[c], errors="coerce")
            if coerced.notna().sum() >= max(1, int(0.5 * len(base))):
                truly_num.append(c)
            else:
                truly_cat.append(c)
        self._cat_base, self._num_base = truly_cat, truly_num

        self.freq_encode_cols_ = list(self._cat_base)
        self.freq_maps_ = {}
        for c in self.freq_encode_cols_:
            vc = base[c].value_counts()
            self.freq_maps_[c] = (vc / n).to_dict()
        self.freq_col_names_ = [f"freq_{c}" for c in self.freq_encode_cols_]

        # TE for interaction cols (only if y provided; else TE off)
        self.te_cols_ = [c for c in INTERACTION_COLS if c in base.columns]
        if y is not None and len(self.te_cols_) > 0:
            self.te_maps_, self.te_prior_ = self._fit_te(base, y, self.te_cols_)
        else:
            self.te_maps_, self.te_prior_ = {}, 0.5

        te_num_names = [f"te_{c}" for c in self.te_cols_]
        self.categorical_columns = self._cat_base
        self.numeric_columns = self._num_base + self.freq_col_names_ + te_num_names
        self.fitted = True
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        out = self.transform(X)
        if y is not None and len(self.te_cols_) > 0:
            from sklearn.model_selection import KFold
            n = len(X)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            oof_te = {c: np.zeros(n) for c in self.te_cols_}
            y_arr = np.asarray(y)
            for tr_idx, va_idx in kf.split(X):
                base_tr = self._build_base(X.iloc[tr_idx])
                maps, prior = self._fit_te(base_tr, y_arr[tr_idx], self.te_cols_)
                base_va = self._build_base(X.iloc[va_idx])
                for c in self.te_cols_:
                    m = maps.get(c, {})
                    oof_te[c][va_idx] = (
                        base_va[c].astype(str).map(m).fillna(prior).astype(float).values
                    )
            for c in self.te_cols_:
                out[f"te_{c}"] = oof_te[c]
        return out

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("fit first")
        base = self._build_base(X)
        out = base.copy()
        for c in self.freq_encode_cols_:
            m = self.freq_maps_.get(c, {})
            out[f"freq_{c}"] = base[c].map(m).fillna(0.0).astype(float)
        for c in self.te_cols_:
            m = self.te_maps_.get(c, {})
            out[f"te_{c}"] = (
                base[c].astype(str).map(m).fillna(self.te_prior_).astype(float)
            )
        return out


def load_train_data(data_dir):
    with open(os.path.join(data_dir, "schema.json")) as f:
        schema = json.load(f)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    labels = (
        labels_df["label"].values
        if "label" in labels_df.columns
        else labels_df.iloc[:, 0].values
    )
    return features, labels, schema


def load_predict_data(data_dir):
    with open(os.path.join(data_dir, "schema.json")) as f:
        schema = json.load(f)
    return pd.read_csv(os.path.join(data_dir, "features.csv")), schema


def make_stratified_folds(y, n_splits=5, seed=42):
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ===========================================================================
# MLP
# ===========================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = float(alpha), float(gamma)

    def forward(self, logits, targets):
        t = targets.float()
        ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(t == 1.0, p, 1.0 - p)
        a_t = torch.where(t == 1.0, self.alpha, 1.0 - self.alpha)
        return (a_t * ((1.0 - pt) ** self.gamma) * ce).mean()


class EmbMLP(nn.Module):
    def __init__(
        self, cat_cards, n_num, emb_dim=10, hidden=(256, 128, 64), dropout=0.3
    ):
        super().__init__()
        self.n_fields = len(cat_cards)
        self.emb = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cat_cards])
        for e in self.emb:
            nn.init.normal_(e.weight, 0.0, 0.02)
        in_dim = self.n_fields * emb_dim + n_num
        layers = []
        cur = in_dim
        for h in hidden:
            layers += [
                nn.Linear(cur, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            cur = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(cur, 1)

    def forward(self, cat, num):
        e = torch.cat([self.emb[i](cat[:, i]) for i in range(self.n_fields)], dim=1)
        if num is not None and num.size(1) > 0:
            e = torch.cat([e, num], dim=1)
        return self.out(self.body(e)).squeeze(-1)


def _train_mlp(Xc, Xn, y, card, epochs=22, bs=512, lr=1e-3, seed=0, wd=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(y)
    model = EmbMLP(card, Xn.shape[1])
    crit = FocalLoss(0.5, 2.0)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999)
    )
    step_epoch = max(1, math.ceil(n / bs))
    tot = epochs * step_epoch
    warm = max(1, int(0.1 * tot))

    def lr_fn(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, tot - warm)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    Xc_t = torch.from_numpy(Xc).long()
    Xn_t = torch.from_numpy(Xn).float()
    y_t = torch.from_numpy(y.astype(np.float32))
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            if len(b) <= 1:
                continue
            opt.zero_grad()
            logits = model(Xc_t[b], Xn_t[b])
            crit(logits, y_t[b]).backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def _predict_mlp(model, Xc, Xn, bs=1024):
    model.eval()
    out = np.zeros(len(Xc), dtype=np.float32)
    for i in range(0, len(Xc), bs):
        xc = torch.from_numpy(Xc[i : i + bs]).long()
        xn = torch.from_numpy(Xn[i : i + bs]).float()
        out[i : i + bs] = torch.sigmoid(model(xc, xn)).cpu().numpy()
    return out


# ===========================================================================
# Prep helpers
# ===========================================================================
def _to_str_list(series):
    out = []
    for v in series.tolist():
        if v is None:
            out.append("nan")
        elif isinstance(v, str):
            out.append(v)
        else:
            try:
                out.append("nan" if pd.isna(v) else str(v))
            except (TypeError, ValueError):
                out.append(str(v))
    return out


def _numeric_block(fe_df, num_cols):
    if not num_cols:
        return np.zeros((len(fe_df), 0), dtype=np.float32)
    df = pd.DataFrame(
        {
            c: pd.to_numeric(fe_df[c], errors="coerce").astype(np.float64)
            for c in num_cols
        },
        index=fe_df.index,
    )
    return df.fillna(0.0).values.astype(np.float32)


def _prepare_lgb_frame(fe_df, cat_cols, num_cols, cat_categories, fit):
    data = {}
    for c in cat_cols:
        vals = _to_str_list(fe_df[c])
        if fit:
            cat_categories[c] = sorted(set(vals))
        cats = cat_categories.get(c, [])
        data[c] = pd.Categorical(vals, categories=cats)
    for c in num_cols:
        data[c] = (
            pd.to_numeric(fe_df[c], errors="coerce")
            .fillna(0.0)
            .values.astype(np.float64)
        )
    df = pd.DataFrame(data, index=fe_df.index)
    df.attrs["cat_cols"] = cat_cols
    return df


def _prepare_cb_frame(fe_df, cb_cols):
    df = fe_df[cb_cols].reset_index(drop=True).copy()
    for c in cb_cols:
        df[c] = _to_str_list(df[c])
    return df


def _prepare_mlp_arrays(fe_df, cat_cols, num_cols, card_map, fit):
    n = len(fe_df)
    Xc = np.zeros((n, len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        vals = _to_str_list(fe_df[c])
        if fit:
            uniq = sorted(set(vals))
            card_map[c] = {u: i + 1 for i, u in enumerate(uniq)}
        m = card_map[c]
        Xc[:, j] = np.array([m.get(v, 0) for v in vals], dtype=np.int64)
    Xn = _numeric_block(fe_df, num_cols)
    cards = [max(2, len(card_map[c]) + 1) for c in cat_cols]
    return Xc, Xn, cards


# ===========================================================================
# Base model wrappers
# ===========================================================================
def _make_lgb(seed, n_estimators=1200):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=2,
        verbose=-1,
    )


def _train_lgb(X_df, y, cat_cols, seed=0):
    m = _make_lgb(seed)
    X_df = X_df.copy()
    for c in cat_cols:
        if c in X_df.columns and not isinstance(X_df[c].dtype, pd.CategoricalDtype):
            X_df[c] = X_df[c].astype("category")
    m.fit(X_df, y, categorical_feature=cat_cols)
    return m


def _predict_lgb(m, X_df, cat_cols):
    X_df = X_df.copy()
    for c in cat_cols:
        if c in X_df.columns and not isinstance(X_df[c].dtype, pd.CategoricalDtype):
            X_df[c] = X_df[c].astype("category")
    return m.predict_proba(X_df)[:, 1]


try:
    from catboost import CatBoostClassifier

    _HAS_CB = True
except Exception:
    _HAS_CB = False


def _make_cb(seed=0):
    if not _HAS_CB:
        return None
    return CatBoostClassifier(
        iterations=800,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        thread_count=2,
        loss_function="Logloss",
    )


# ===========================================================================
# TRAIN
# ===========================================================================
def run_training(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    features, labels, schema = load_train_data(data_dir)
    y_raw = np.asarray(labels)
    uniq = sorted(set(y_raw.tolist()))
    label_map = {u: i for i, u in enumerate(uniq)}
    y = np.array([label_map[v] for v in y_raw], dtype=np.int64)

    folds = make_stratified_folds(y, n_splits=5, seed=42)
    n = len(y)
    oof_lgb = np.zeros(n)
    oof_cb = np.zeros(n)
    oof_mlp = np.zeros(n)
    has_any_cb = False

    for fi, (tr, va) in enumerate(folds):
        fe_fold = CouponFeatureEngineer()
        fe_tr = fe_fold.fit_transform(features.iloc[tr], y[tr])
        fe_va = fe_fold.transform(features.iloc[va])
        cat_cols = list(fe_fold.categorical_columns)
        num_cols = list(fe_fold.numeric_columns)
        cat_categories = {}
        X_tr = _prepare_lgb_frame(fe_tr, cat_cols, num_cols, cat_categories, fit=True)
        X_va = _prepare_lgb_frame(fe_va, cat_cols, num_cols, cat_categories, fit=False)

        # LGBM
        lgb_m = _train_lgb(X_tr, y[tr], cat_cols, seed=fi)
        oof_lgb[va] = _predict_lgb(lgb_m, X_va, cat_cols)

        # CatBoost
        if _HAS_CB:
            cb_cols = cat_cols + [c for c in num_cols]
            cb_tr = _prepare_cb_frame(fe_tr, cb_cols)
            cb_va = _prepare_cb_frame(fe_va, cb_cols)
            # numeric columns must be numeric in CB frame
            for c in num_cols:
                cb_tr[c] = pd.to_numeric(cb_tr[c], errors="coerce").fillna(0.0)
                cb_va[c] = pd.to_numeric(cb_va[c], errors="coerce").fillna(0.0)
            cb = _make_cb(seed=fi)
            try:
                cb.fit(cb_tr, y[tr], cat_features=cat_cols)
                oof_cb[va] = cb.predict_proba(cb_va)[:, 1]
                has_any_cb = True
            except Exception as e:
                print(f"[fold {fi}] CB failed: {e}")
                oof_cb[va] = oof_lgb[va]

        # MLP
        card_map = {}
        Xc_tr, Xn_tr, cards = _prepare_mlp_arrays(
            fe_tr, cat_cols, num_cols, card_map, fit=True
        )
        Xc_va, Xn_va, _ = _prepare_mlp_arrays(
            fe_va, cat_cols, num_cols, card_map, fit=False
        )
        mlp = _train_mlp(Xc_tr, Xn_tr, y[tr], cards, seed=fi)
        oof_mlp[va] = _predict_mlp(mlp, Xc_va, Xn_va)

        print(
            f"[fold {fi}] lgb={accuracy_score(y[va],(oof_lgb[va]>=0.5).astype(int)):.4f} "
            f"cb={accuracy_score(y[va],(oof_cb[va]>=0.5).astype(int)):.4f} "
            f"mlp={accuracy_score(y[va],(oof_mlp[va]>=0.5).astype(int)):.4f}"
        )

    # Stacker
    stack_cols = [oof_lgb, oof_mlp] + ([oof_cb] if has_any_cb else [])
    S = np.column_stack(stack_cols)
    if LogisticRegression is not None:
        stacker = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        stacker.fit(S, y)
        oof_blend = stacker.predict_proba(S)[:, 1]
        use_stacker = True
    else:
        # simple mean fallback
        stacker = None
        oof_blend = S.mean(axis=1)
        use_stacker = False

    # Threshold tuning on OOF
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 121):
        a = accuracy_score(y, (oof_blend >= t).astype(int))
        if a > best_acc:
            best_acc, best_t = a, t
    print(f"[OOF blend] acc={best_acc:.6f} @ t={best_t:.2f}")

    # Individual OOF scores (for insight)
    print(
        f"[OOF] lgb={accuracy_score(y,(oof_lgb>=0.5).astype(int)):.4f} "
        f"cb={accuracy_score(y,(oof_cb>=0.5).astype(int)) if has_any_cb else float('nan'):.4f} "
        f"mlp={accuracy_score(y,(oof_mlp>=0.5).astype(int)):.4f}"
    )

    # Refit on full training data
    fe = CouponFeatureEngineer()
    fe_train = fe.fit_transform(features, y)
    cat_cols = list(fe.categorical_columns)
    num_cols = list(fe.numeric_columns)
    cat_categories = {}
    X_df = _prepare_lgb_frame(fe_train, cat_cols, num_cols, cat_categories, fit=True)

    lgb_models = [_train_lgb(X_df, y, cat_cols, seed=s) for s in range(3)]
    cb_models = []
    if _HAS_CB:
        cb_cols = cat_cols + num_cols
        cb_df = _prepare_cb_frame(fe_train, cb_cols)
        for c in num_cols:
            cb_df[c] = pd.to_numeric(cb_df[c], errors="coerce").fillna(0.0)
        for s in range(3):
            cb = _make_cb(seed=100 + s)
            try:
                cb.fit(cb_df, y, cat_features=cat_cols)
                cb_models.append(cb)
            except Exception as e:
                print(f"[refit] CB failed: {e}")

    card_map_full = {}
    Xc_full, Xn_full, cards_full = _prepare_mlp_arrays(
        fe_train, cat_cols, num_cols, card_map_full, fit=True
    )
    mlp_models = [
        _train_mlp(Xc_full, Xn_full, y, cards_full, seed=200 + s) for s in range(3)
    ]

    bundle = {
        "fe": fe,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_categories": cat_categories,
        "lgb_models": lgb_models,
        "cb_models": cb_models,
        "mlp_models": mlp_models,
        "card_map": card_map_full,
        "stacker": stacker,
        "use_stacker": use_stacker,
        "threshold": best_t,
        "label_map": label_map,
        "has_cb": has_any_cb,
    }
    joblib.dump(bundle, os.path.join(model_dir, "bundle.joblib"))
    print(f"[train] saved to {model_dir}")
    print(f"Final Validation Score: {best_acc:.6f}")


# ===========================================================================
# PREDICT
# ===========================================================================
def run_predict(data_dir, model_dir, output_path):
    features, schema = load_predict_data(data_dir)
    bundle = joblib.load(os.path.join(model_dir, "bundle.joblib"))

    fe = bundle["fe"]
    cat_cols = bundle["cat_cols"]
    num_cols = bundle["num_cols"]
    cat_categories = bundle["cat_categories"]
    label_map = bundle["label_map"]
    threshold = bundle["threshold"]
    stacker = bundle.get("stacker")
    use_stacker = bundle.get("use_stacker", False)
    card_map = bundle["card_map"]

    fe_p = fe.transform(features)
    X_df = _prepare_lgb_frame(fe_p, cat_cols, num_cols, cat_categories, fit=False)

    lgb_models = bundle["lgb_models"]
    p_lgb = np.mean([_predict_lgb(m, X_df, cat_cols) for m in lgb_models], axis=0)

    if bundle.get("has_cb") and len(bundle.get("cb_models", [])) > 0:
        cb_cols = cat_cols + num_cols
        cb_df = _prepare_cb_frame(fe_p, cb_cols)
        for c in num_cols:
            cb_df[c] = pd.to_numeric(cb_df[c], errors="coerce").fillna(0.0)
        p_cb = np.mean(
            [m.predict_proba(cb_df)[:, 1] for m in bundle["cb_models"]], axis=0
        )
    else:
        p_cb = p_lgb

    Xc_p, Xn_p, _ = _prepare_mlp_arrays(fe_p, cat_cols, num_cols, card_map, fit=False)
    p_mlp = np.mean([_predict_mlp(m, Xc_p, Xn_p) for m in bundle["mlp_models"]], axis=0)

    S = np.column_stack([p_lgb, p_mlp] + ([p_cb] if bundle.get("has_cb") else []))
    if use_stacker and stacker is not None:
        probs = stacker.predict_proba(S)[:, 1]
    else:
        probs = S.mean(axis=1)

    preds_bin = (probs >= threshold).astype(int)
    inv_map = {int(v): k for k, v in label_map.items()}
    preds = np.array([inv_map[int(p)] for p in preds_bin])

    out_df = pd.DataFrame(
        {"row_id": np.arange(len(preds), dtype=np.int64), "prediction": preds}
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[predict] wrote {len(out_df)} rows")


# ===========================================================================
# CLI
# ===========================================================================
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
        run_training(args.data, args.model)
    else:
        run_predict(args.data, args.model, args.output)