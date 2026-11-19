#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Coupon acceptance prediction.

Usage:
  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv
"""

import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def load_schema(data_dir):
    path = os.path.join(data_dir, "schema.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def get_feature_columns(schema, df):
    cols = None
    if isinstance(schema, dict):
        for key in ("feature_columns", "features", "columns", "feature_names"):
            v = schema.get(key)
            if isinstance(v, list) and len(v) > 0:
                cols = [str(c) for c in v]
                break
        if cols is None:
            mod = schema.get("modalities")
            if isinstance(mod, dict):
                for _, spec in mod.items():
                    if isinstance(spec, dict) and isinstance(spec.get("columns"), list):
                        cols = [str(c) for c in spec["columns"]]
                        break
    if cols is None:
        img_cols = []
        if isinstance(schema, dict) and isinstance(schema.get("image_columns"), list):
            img_cols = [str(c) for c in schema["image_columns"]]
        cols = [c for c in df.columns if c not in img_cols]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = list(df.columns)
    return cols


# ----------------------------------------------------------------------------
# Persistence filenames (also part of the public interface)
# ----------------------------------------------------------------------------
FEATURE_ENGINEER_FILE = "feature_engineer.pkl"
ENSEMBLE_FILE = "ensemble.pkl"
GBDT_FILE = "gbdt_models.pkl"

# Ordered categorical scales (slot 0 always reserved for unknown/missing).
_FREQ_ORDER = ["never", "less1", "1~3", "4~8", "gt8"]
_ORDINAL_ORDERS = {
    "income": [
        "Less than $12500", "$12500 - $24999", "$25000 - $37499",
        "$37500 - $49999", "$50000 - $62499", "$62500 - $74999",
        "$75000 - $87499", "$87500 - $99999", "$100000 or More",
    ],
    "age": ["below21", "21", "26", "31", "36", "41", "46", "50plus"],
    "education": [
        "Some High School", "High School Graduate",
        "Some college - no degree", "Associates degree",
        "Bachelors degree", "Graduate degree (Masters or Doctorate)",
    ],
    "Bar": _FREQ_ORDER,
    "CoffeeHouse": _FREQ_ORDER,
    "CarryAway": _FREQ_ORDER,
    "RestaurantLessThan20": _FREQ_ORDER,
    "Restaurant20To50": _FREQ_ORDER,
}

# coupon -> venue frequency column used for affinity features
_COUPON_VENUE = {
    "Bar": "Bar",
    "Coffee House": "CoffeeHouse",
    "CoffeeHouse": "CoffeeHouse",
    "CarryAway": "CarryAway",
    "RestaurantLessThan20": "RestaurantLessThan20",
    "Restaurant(20-50)": "Restaurant20To50",
    "Restaurant20To50": "Restaurant20To50",
    "Restaurant(<20)": "RestaurantLessThan20",
}

# Trimmed, coupon-centric OOF target-encoding key space (plan: exactly 3 keys).
TE_SOURCES = [
    ["coupon", "occupation"],
    ["coupon", "income"],
    ["coupon", "time"],
]


def _canon_scalar(v):
    if v is None:
        return "__NA__"
    try:
        if v != v:
            return "__NA__"
    except Exception:
        pass
    sv = str(v).strip()
    try:
        fv = float(sv)
        if np.isfinite(fv):
            return "%g" % fv
    except Exception:
        pass
    return sv


class FeatureEngineer(object):
    """Stateless-at-predict encoder: ordinal maps, categorical coding, medians.

    Only outer-train rows are used in ``fit``; slot 0 of every ordinal/cat map
    is reserved for unknown values so ``transform`` never fails on new data.
    """

    def __init__(self):
        self.features = []
        self.numeric_cols = []
        self.cat_cols = []
        self.ordinal_cols = []
        self.medians_ = {}
        self.cat_maps_ = {}
        self.ordinal_maps_ = {}
        self.combo_vocab_ = {}
        self.feature_names_ = []
        self.cat_mask_ = np.zeros(0, dtype=bool)

    @staticmethod
    def _as_str(s):
        return s.where(s.notna(), "__MISSING__").astype(str).str.strip()

    def fit(self, df, y=None, feature_columns=None):
        if feature_columns is None:
            feature_columns = list(df.columns)
        self.features = list(feature_columns)
        self.numeric_cols, self.cat_cols, self.ordinal_cols = [], [], []
        self.medians_, self.cat_maps_, self.ordinal_maps_ = {}, {}, {}

        for c in self.features:
            s = df[c]
            num = None
            try:
                num = pd.to_numeric(s, errors="coerce")
            except Exception:
                num = None

            is_num = False
            if num is not None:
                orig_na = int(s.isna().sum())
                new_na = int(num.isna().sum())
                is_num = new_na <= orig_na

            if c in _ORDINAL_ORDERS:
                # Ordinal coding driven by explicit, known semantic order.
                vocab = {v: i + 1 for i, v in enumerate(_ORDINAL_ORDERS[c])}
                sv = self._as_str(s).values
                extra = sorted({v for v in sv if v != "__MISSING__" and v not in vocab})
                for v in extra:
                    vocab[v] = len(vocab) + 1
                self.ordinal_maps_[c] = vocab
                self.ordinal_cols.append(c)
            elif is_num:
                self.numeric_cols.append(c)
                med = float(num.median()) if num.notna().any() else 0.0
                if not np.isfinite(med):
                    med = 0.0
                self.medians_[c] = med
            else:
                self.cat_cols.append(c)
                sv = self._as_str(s)
                uniq = sorted({u for u in sv.values.tolist() if u != "__MISSING__"})
                vocab = {"__MISSING__": 0}
                for i, u in enumerate(uniq):
                    vocab[u] = i + 1
                self.cat_maps_[c] = vocab

        # coupon x expiration interaction vocabulary
        self.combo_vocab_ = {}
        if "coupon" in df.columns and "expiration" in df.columns:
            ck = [
                _canon_scalar(a) + "|" + _canon_scalar(b)
                for a, b in zip(df["coupon"].values, df["expiration"].values)
            ]
            uniq = sorted(set(ck))
            self.combo_vocab_ = {v: i + 1 for i, v in enumerate(uniq)}

        # derive feature ordering + categorical mask
        names, mask = [], []
        for c in self.features:
            names.append(c)
            mask.append(c in self.ordinal_cols or c in self.cat_cols)
        for nm in ("aff__venue_freq", "aff__never", "ix__coupon_exp",
                   "ix__geq15_same", "ix__geq25_same"):
            names.append(nm)
            mask.append(nm == "ix__coupon_exp")
        self.feature_names_ = names
        self.cat_mask_ = np.array(mask, dtype=bool)
        return self

    def _transform_base(self, df):
        n = len(df)
        cols = []
        for c in self.features:
            if c in self.ordinal_maps_:
                m = self.ordinal_maps_[c]
                sv = self._as_str(df[c]).values
                cols.append(np.array([m.get(v, 0) for v in sv], dtype=np.float64))
            elif c in self.medians_:
                try:
                    v = pd.to_numeric(df[c], errors="coerce").values.astype(np.float64)
                except Exception:
                    v = np.full(n, np.nan, dtype=np.float64)
                v = np.where(np.isnan(v), self.medians_[c], v)
                cols.append(v)
            else:
                m = self.cat_maps_.get(c, {"__MISSING__": 0})
                sv = self._as_str(df[c]).values
                cols.append(np.array([m.get(v, 0) for v in sv], dtype=np.float64))
        if cols:
            return np.stack(cols, axis=1)
        return np.zeros((n, 0), dtype=np.float64)

    def _transform_derived(self, df):
        n = len(df)
        base = self._transform_base(df)
        idx = {c: j for j, c in enumerate(self.features)}

        def freq_code(col):
            if col not in idx:
                return np.zeros(n, dtype=np.float64)
            return base[:, idx[col]]

        # coupon -> venue frequency (the strongest signal: affinity)
        coupon_vals = [_canon_scalar(v) for v in
                       (df["coupon"].values if "coupon" in df.columns else [""] * n)]
        venue_freq = np.zeros(n, dtype=np.float64)
        never = np.zeros(n, dtype=np.float64)
        for i, cp in enumerate(coupon_vals):
            vcol = _COUPON_VENUE.get(cp)
            if vcol is not None:
                code = freq_code(vcol)[i]
                venue_freq[i] = code
                never[i] = 1.0 if code == 1.0 else 0.0

        if self.combo_vocab_ and "coupon" in df.columns and "expiration" in df.columns:
            ck = [
                _canon_scalar(a) + "|" + _canon_scalar(b)
                for a, b in zip(df["coupon"].values, df["expiration"].values)
            ]
            coupon_exp = np.array(
                [self.combo_vocab_.get(v, 0) for v in ck], dtype=np.float64
            )
        else:
            coupon_exp = np.zeros(n, dtype=np.float64)

        def col_or_zero(name):
            if name in df.columns:
                try:
                    return pd.to_numeric(df[name], errors="coerce").fillna(0).values.astype(np.float64)
                except Exception:
                    return np.zeros(n, dtype=np.float64)
            return np.zeros(n, dtype=np.float64)

        g15 = (col_or_zero("toCoupon_GEQ15min"))
        g25 = (col_or_zero("toCoupon_GEQ25min"))
        d_same = col_or_zero("direction_same")
        geq15_same = g15 * d_same
        geq25_same = g25 * d_same

        derived = np.stack([venue_freq, never, coupon_exp, geq15_same, geq25_same], axis=1)
        return np.concatenate([base, derived], axis=1)

    def transform(self, df):
        arr = self._transform_derived(df)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def fit_transform(self, df, y=None, feature_columns=None):
        self.fit(df, y, feature_columns)
        return self.transform(df)


class GBDTFeatureEncoder(FeatureEngineer):
    """Same contract as :class:`FeatureEngineer`; exposed for family parity."""
    pass


def build_matrix(X, feats, cat_idx):
    arr = np.array(X, dtype=np.float64, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e9, neginf=-1e9)
    if cat_idx:
        for j in cat_idx:
            col = arr[:, j]
            col = np.where(col < 0, 0.0, col)
            arr[:, j] = np.round(col)
    return arr


# ----------------------------------------------------------------------------
# Leakage-safe out-of-fold target encoding for categorical interactions
# ----------------------------------------------------------------------------
def _canon_values(series):
    out = np.empty(len(series), dtype=object)
    for i, v in enumerate(series.values):
        if v is None:
            out[i] = "__NA__"
            continue
        try:
            if v != v:
                out[i] = "__NA__"
                continue
        except Exception:
            pass
        sv = str(v).strip()
        try:
            fv = float(sv)
            if np.isfinite(fv):
                out[i] = "%g" % fv
            else:
                out[i] = sv
        except Exception:
            out[i] = sv
    return out


class TargetEncoder(object):
    def __init__(self, keys, smoothing=10.0, n_splits=5, seed=42):
        self.keys = [list(k) for k in keys]
        self.smoothing = float(smoothing)
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.maps = []
        self.prior = 0.0

    @staticmethod
    def _key_series(cache, idx):
        # combine pre-canonicalized columns for the given row indices
        cols = cache
        if not cols:
            return np.array([], dtype=object)
        n = len(idx)
        acc = None
        for arr in cols:
            part = np.asarray(arr, dtype=object)[idx]
            if acc is None:
                acc = part
            else:
                acc = np.array([a + "|" + b for a, b in zip(acc, part)], dtype=object)
        if acc is None:
            return np.array([""] * n, dtype=object)
        return acc

    def _map_from(self, keys, y, prior):
        dfk = pd.DataFrame(
            {"k": np.asarray(keys, dtype=object), "y": np.asarray(y, dtype=np.float64)}
        )
        g = dfk.groupby("k")["y"].agg(["mean", "count"])
        m = (g["mean"] * g["count"] + prior * self.smoothing) / (
            g["count"] + self.smoothing
        )
        return m.to_dict()

    def fit(self, df, y):
        y = np.asarray(y, dtype=np.float64)
        self.prior = float(np.mean(y)) if len(y) else 0.0
        self.maps = []
        all_idx = np.arange(len(df))
        for cols in self.keys:
            cache = [_canon_values(df[c]) for c in cols]
            ks = self._key_series(cache, all_idx)
            self.maps.append(self._map_from(ks, y, self.prior))
        return self

    def transform(self, df):
        if not self.keys:
            return np.zeros((len(df), 0), dtype=np.float64)
        all_idx = np.arange(len(df))
        outs = []
        for i, cols in enumerate(self.keys):
            cache = [_canon_values(df[c]) for c in cols]
            ks = self._key_series(cache, all_idx)
            m = self.maps[i]
            prior = self.prior
            outs.append(np.array([m.get(k, prior) for k in ks], dtype=np.float64))
        return np.stack(outs, axis=1)

    def fit_transform_oof(self, df, y):
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        n_feat = len(self.keys)
        out = np.zeros((n, n_feat), dtype=np.float64)
        if n_feat == 0 or n == 0:
            self.fit(df, y)
            return out

        _, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min()) if len(counts) else n
        if min_count < 2:
            self.fit(df, y)
            out[:] = float(np.mean(y)) if n else 0.0
            return out

        n_splits = min(self.n_splits, max(2, min_count))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        caches = [[_canon_values(df[c]) for c in cols] for cols in self.keys]

        for tr, va in skf.split(np.zeros(n), y):
            prior_k = float(np.mean(y[tr])) if len(tr) else 0.0
            for i in range(n_feat):
                ks_tr = self._key_series(caches[i], tr)
                ks_va = self._key_series(caches[i], va)
                m = self._map_from(ks_tr, y[tr], prior_k)
                out[va, i] = np.array(
                    [m.get(k, prior_k) for k in ks_va], dtype=np.float64
                )

        self.fit(df, y)
        return out


# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Hyper-parameter configs (persisted with the artifacts)
# ----------------------------------------------------------------------------
HGB_CONFIG = dict(
    max_iter=600,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=False,
)

MODEL_CONFIG = dict(
    hidden_dim=128,
    num_heads=4,
    num_blocks=2,
    dropout=0.15,
    num_classes=2,
)


def _ckpt_load(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _restore_label(idx, label_map):
    idx = int(idx)
    if 0 <= idx < len(label_map):
        return label_map[idx]
    return label_map[-1] if label_map else 0


def get_cv_splits(y, NUM_FOLDS=5, random_state=42):
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min()) if len(counts) else len(y)
    n_splits = max(2, min(int(NUM_FOLDS), max(2, min_count)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), y))


def build_hgb(cat_mask, seed):
    params = dict(HGB_CONFIG)
    params["random_state"] = int(seed)
    if cat_mask is not None and np.asarray(cat_mask).any():
        try:
            return HistGradientBoostingClassifier(
                categorical_features=np.asarray(cat_mask, dtype=bool), **params
            )
        except Exception:
            pass
    return HistGradientBoostingClassifier(**params)


def build_logreg(seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1, max_iter=1000, solver="lbfgs", random_state=int(seed)
        ),
    )


class TabMRank1Block(object):
    """Named placeholder kept for API compatibility (TabM is a minor member)."""

    def __init__(self, dim=128):
        self.dim = dim


class TabM(object):
    """Light MLP member. Kept deliberately minor and tightly weight-bounded."""

    def __init__(self, n_features, num_classes=2, hidden_dim=128, num_heads=4,
                 num_blocks=2, dropout=0.15, seed=0, max_iter=120):
        from sklearn.neural_network import MLPClassifier

        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.config = dict(MODEL_CONFIG)
        self.model = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=int(max_iter),
            early_stopping=False,
            random_state=self.seed,
        )

    def state_dict(self):
        return {"config": dict(self.config), "model": self.model}

    def load_state_dict(self, state):
        if isinstance(state, dict) and "model" in state:
            self.model = state["model"]

    def fit(self, X, y):
        try:
            self.model.fit(X, y)
        except Exception:
            pass
        return self

    def predict_proba(self, X):
        try:
            p = self.model.predict_proba(X)
            return np.asarray(p, dtype=np.float64)
        except Exception:
            n = len(X)
            return np.full((n, max(2, self.num_classes)), 0.5, dtype=np.float64)


def build_model(n_features, seed=0, **kwargs):
    cfg = dict(MODEL_CONFIG)
    cfg.update(kwargs)
    return TabM(
        n_features,
        num_classes=cfg.get("num_classes", 2),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_heads=cfg.get("num_heads", 4),
        num_blocks=cfg.get("num_blocks", 2),
        dropout=cfg.get("dropout", 0.15),
        seed=seed,
    )


def _proba_1d(model, X):
    """Class-1 probability, tolerating single-column and missing-class cases."""
    p = model.predict_proba(X)
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        return np.clip(p, 0.0, 1.0)
    if p.shape[1] >= 2:
        return p[:, 1]
    return np.zeros(len(X), dtype=np.float64)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def cmd_train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, "features.csv")
    lab_path = os.path.join(data_dir, "labels.csv")
    schema = load_schema(data_dir)

    df = pd.read_csv(feat_path)
    lab = pd.read_csv(lab_path)

    if "label" in lab.columns:
        y_raw = lab["label"].values
    else:
        y_raw = lab.iloc[:, 0].values

    if len(y_raw) != len(df):
        raise ValueError(
            "features/labels length mismatch: %d vs %d" % (len(df), len(y_raw))
        )

    classes = sorted(set([str(v) for v in y_raw]))
    label_map = list(classes)
    cls_index = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_index[str(v)] for v in y_raw], dtype=np.int64)
    n_cls = len(label_map)

    feature_columns = get_feature_columns(schema, df)
    feature_columns = [c for c in feature_columns if c in df.columns]
    for c in df.columns:
        if c not in feature_columns and c not in ("row_id", "id"):
            feature_columns.append(c)

    if n_cls < 2:
        enc = GBDTFeatureEncoder().fit(df, y, feature_columns)
        bundle = {
            "label_map": label_map,
            "constant_index": 0,
            "weights": {"gbdt": 1.0, "lr": 0.0, "mlp": 0.0},
            "threshold": 0.5,
            "te_sources": [],
            "n_features": int(len(enc.feature_names_)),
            "oof_accuracy": 1.0,
        }
        with open(os.path.join(model_dir, FEATURE_ENGINEER_FILE), "wb") as f:
            pickle.dump(enc, f)
        with open(os.path.join(model_dir, ENSEMBLE_FILE), "w") as f:
            json.dump(bundle, f)
        print("[train] single class; wrote constant model")
        return

    te_sources = [k for k in TE_SOURCES if all(c in df.columns for c in k)]
    use_te = bool(te_sources)
    n_te = len(te_sources)

    splits = get_cv_splits(y, NUM_FOLDS=5, random_state=42)

    oof_gbdt = np.zeros(len(y), dtype=np.float64)
    oof_lr = np.zeros(len(y), dtype=np.float64)
    oof_mlp = np.zeros(len(y), dtype=np.float64)

    fold_encoders = []
    fold_hgbs = []
    fold_lrs = []

    for fi, (tr, va) in enumerate(splits):
        folds = len(splits)
        enc = GBDTFeatureEncoder().fit(df.iloc[tr], y[tr], feature_columns)
        Xtr = build_matrix(enc.transform(df.iloc[tr]), enc.feature_names_, [])
        Xva = build_matrix(enc.transform(df.iloc[va]), enc.feature_names_, [])
        cat_mask = enc.cat_mask_

        if use_te:
            fold_te = TargetEncoder(te_sources, smoothing=10.0, n_splits=5, seed=42)
            fold_te.fit(df.iloc[tr], y[tr])
            Xtr = np.concatenate([Xtr, fold_te.transform(df.iloc[tr])], axis=1)
            Xva = np.concatenate([Xva, fold_te.transform(df.iloc[va])], axis=1)
            te_mask = np.concatenate([cat_mask, np.zeros(n_te, dtype=bool)])
        else:
            fold_te = None
            te_mask = cat_mask

        ytr = y[tr]

        ph = np.zeros(len(va), dtype=np.float64)
        hgbs = []
        for sd in (2000 + fi, 3000 + fi):
            m = build_hgb(te_mask, sd)
            m.fit(Xtr, ytr)
            hgbs.append(m)
            ph += _proba_1d(m, Xva)
        oof_gbdt[va] = ph / 2.0

        lr = build_logreg(1000 + fi)
        try:
            lr.fit(Xtr, ytr)
            oof_lr[va] = _proba_1d(lr, Xva)
        except Exception:
            oof_lr[va] = 0.5

        mlp = build_model(Xtr.shape[1], seed=5000 + fi)
        mlp.fit(Xtr, ytr)
        try:
            p_mlp = mlp.predict_proba(Xva)
            oof_mlp[va] = p_mlp[:, 1] if p_mlp.shape[1] >= 2 else 0.5
        except Exception:
            oof_mlp[va] = 0.5

        fold_encoders.append((enc, fold_te))
        fold_hgbs.append(hgbs)
        fold_lrs.append(lr)
        print("[train] fold %d/%d done" % (fi + 1, folds))

    # ---- constrained blend + threshold search on OOF ----
    w_mlp_grid = np.linspace(0.0, 0.3, 16)
    w_lr_grid = np.linspace(0.0, 0.2, 11)
    thr_grid = [0.40 + 0.005 * i for i in range(41)]

    best = {"acc": -1.0, "w_mlp": 0.0, "w_lr": 0.0, "thr": 0.5}
    for w_mlp in w_mlp_grid:
        for w_lr in w_lr_grid:
            w_gbdt = 1.0 - w_mlp - w_lr
            if w_gbdt < 0.5:
                continue
            p = w_gbdt * oof_gbdt + w_lr * oof_lr + w_mlp * oof_mlp
            for t in thr_grid:
                pred = (p >= t).astype(np.int64)
                acc = float(np.mean(pred == y))
                if acc > best["acc"]:
                    best = {"acc": acc, "w_mlp": float(w_mlp),
                            "w_lr": float(w_lr), "thr": float(t)}
            pred05 = (p >= 0.5).astype(np.int64)
            acc05 = float(np.mean(pred05 == y))
            if acc05 > best["acc"]:
                best = {"acc": acc05, "w_mlp": float(w_mlp),
                        "w_lr": float(w_lr), "thr": 0.5}

    w_gbdt = 1.0 - best["w_mlp"] - best["w_lr"]
    print(
        "[train] oof acc=%.5f w_gbdt=%.3f w_lr=%.3f w_mlp=%.3f thr=%.3f"
        % (best["acc"], w_gbdt, best["w_lr"], best["w_mlp"], best["thr"])
    )

    # ---- final refit on 100% of training rows ----
    enc_final = GBDTFeatureEncoder().fit(df, y, feature_columns)
    Xf = build_matrix(enc_final.transform(df), enc_final.feature_names_, [])
    cat_mask = enc_final.cat_mask_

    te_final = None
    if use_te:
        te_final = TargetEncoder(te_sources, smoothing=10.0, n_splits=5, seed=42)
        te_final.fit(df, y)
        Xf = np.concatenate([Xf, te_final.transform(df)], axis=1)
        final_mask = np.concatenate([cat_mask, np.zeros(n_te, dtype=bool)])
    else:
        final_mask = cat_mask

    hgb_models = []
    for sd in (2000, 3000):
        m = build_hgb(final_mask, sd)
        m.fit(Xf, y)
        hgb_models.append(m)

    lr_final = build_logreg(1000)
    try:
        lr_final.fit(Xf, y)
    except Exception:
        lr_final = None

    mlp_final = build_model(Xf.shape[1], seed=5000)
    mlp_final.fit(Xf, y)

    with open(os.path.join(model_dir, GBDT_FILE), "wb") as f:
        pickle.dump(hgb_models, f)
    with open(os.path.join(model_dir, FEATURE_ENGINEER_FILE), "wb") as f:
        pickle.dump({"encoder": enc_final, "te": te_final,
                     "fold_encoders": fold_encoders}, f)

    bundle = {
        "label_map": label_map,
        "te_sources": te_sources if use_te else [],
        "n_te": int(n_te if use_te else 0),
        "cat_mask": [bool(v) for v in final_mask],
        "weights": {"gbdt": float(w_gbdt), "lr": float(best["w_lr"]),
                    "mlp": float(best["w_mlp"])},
        "threshold": float(best["thr"]),
        "oof_accuracy": float(best["acc"]),
        "hgb_config": dict(HGB_CONFIG),
        "model_config": dict(MODEL_CONFIG),
        "constant_index": 0,
        "gbdt_file": GBDT_FILE,
        "feature_engineer_file": FEATURE_ENGINEER_FILE,
    }
    with open(os.path.join(model_dir, ENSEMBLE_FILE), "w") as f:
        json.dump(bundle, f)
    with open(os.path.join(model_dir, "lr_final.pkl"), "wb") as f:
        pickle.dump(lr_final, f)
    with open(os.path.join(model_dir, "mlp_final.pkl"), "wb") as f:
        pickle.dump(mlp_final, f)

    print("[train] saved model to %s" % model_dir)
    return float(best["acc"])


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def cmd_predict(data_dir, model_dir, output_path):
    cfg_path = os.path.join(model_dir, ENSEMBLE_FILE)
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    fe_path = os.path.join(model_dir, FEATURE_ENGINEER_FILE)
    fe_obj = _ckpt_load(fe_path)
    if isinstance(fe_obj, dict):
        enc = fe_obj.get("encoder")
        te = fe_obj.get("te")
    else:
        enc, te = fe_obj, None
    if enc is None:
        enc = _ckpt_load(os.path.join(model_dir, "preprocessor.pkl"))

    label_map = cfg.get("label_map", [0, 1])
    n_te = int(cfg.get("n_te", 0))
    weights = cfg.get("weights", {"gbdt": 1.0, "lr": 0.0, "mlp": 0.0})
    w_gbdt = float(weights.get("gbdt", 1.0))
    w_lr = float(weights.get("lr", 0.0))
    w_mlp = float(weights.get("mlp", 0.0))
    thr = float(cfg.get("threshold", 0.5))

    df = None
    if os.path.isdir(data_dir):
        fp = os.path.join(data_dir, "features.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
    if df is None:
        df = pd.DataFrame()

    n = len(df)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if n == 0:
        pd.DataFrame({"row_id": [], "prediction": []}).to_csv(output_path, index=False)
        print("[predict] empty input; wrote header only")
        return

    if enc is None:
        idx = int(cfg.get("constant_index", 0))
        labels = [_restore_label(idx, label_map)] * n
        out = pd.DataFrame({"row_id": np.arange(n, dtype=np.int64),
                            "prediction": labels})
        out.to_csv(output_path, index=False)
        print("[predict] no encoder; wrote constant predictions")
        return

    if hasattr(enc, "features"):
        for c in enc.features:
            if c not in df.columns:
                df[c] = np.nan

    arr = build_matrix(enc.transform(df), enc.feature_names_, [])

    if n_te > 0:
        te_arr = None
        if te is not None:
            try:
                te_arr = te.transform(df)
            except Exception as e:
                print("[predict] te transform failed: %s" % e)
                te_arr = None
        if te_arr is None or te_arr.shape[1] != n_te:
            te_arr = np.zeros((n, n_te), dtype=np.float64)
        arr = np.concatenate([arr, te_arr], axis=1)

    p = np.zeros(n, dtype=np.float64)
    accum = 0.0

    hgb_models = _ckpt_load(os.path.join(model_dir, GBDT_FILE))
    if hgb_models is None:
        hgb_models = _ckpt_load(os.path.join(model_dir, "hgb_models.pkl"))
    if hgb_models and w_gbdt > 0:
        ph = np.zeros(n, dtype=np.float64)
        cnt = 0
        for m in hgb_models:
            try:
                ph += _proba_1d(m, arr)
                cnt += 1
            except Exception:
                continue
        if cnt > 0:
            p += w_gbdt * (ph / float(cnt))
            accum += w_gbdt

    lr_model = _ckpt_load(os.path.join(model_dir, "lr_final.pkl"))
    if lr_model is not None and w_lr > 0:
        try:
            p += w_lr * _proba_1d(lr_model, arr)
            accum += w_lr
        except Exception as e:
            print("[predict] lr failed: %s" % e)

    mlp_model = _ckpt_load(os.path.join(model_dir, "mlp_final.pkl"))
    if mlp_model is not None and w_mlp > 0:
        try:
            pm = mlp_model.predict_proba(arr)
            p += w_mlp * (pm[:, 1] if pm.shape[1] >= 2 else 0.5)
            accum += w_mlp
        except Exception as e:
            print("[predict] mlp failed: %s" % e)

    if accum > 0:
        p /= accum
    else:
        p[:] = 0.5

    pred_idx = (p >= thr).astype(np.int64)
    n_cls = len(label_map)
    pred_idx = np.clip(pred_idx, 0, max(0, n_cls - 1))
    labels = [_restore_label(int(i), label_map) for i in pred_idx]

    out = pd.DataFrame({"row_id": np.arange(n, dtype=np.int64), "prediction": labels})
    out.to_csv(output_path, index=False)
    print("[predict] wrote %d rows to %s" % (n, output_path))


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = ap.parse_args()

    if args.cmd == "train":
        cmd_train(args.data, args.model)
    elif args.cmd == "predict":
        cmd_predict(args.data, args.model, args.output)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()