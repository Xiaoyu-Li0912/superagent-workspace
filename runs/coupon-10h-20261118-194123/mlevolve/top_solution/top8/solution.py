# solution.py
"""
Coupon acceptance prediction — improved merged pipeline.

Stages:
  1) Leakage-safe feature engineering (fit on TRAIN rows only).
  2) Base learners: LGBM (with fold-fold target-encoded interactions),
     CatBoost (raw categoricals), DeepFM (diversity via MLP).
  3) 5-fold stratified train-only CV -> OOF probs -> metric-aligned selection
     among {lgb_avg, cb_avg, soft_vote, meta_lgb} + threshold tuning;
     final refit; full bundle persisted; fresh-process predict.

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ===========================================================================
# Step 1: Data loading, cleaning and reusable feature engineering
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

# Interaction categoricals that get CV-safe target encoding.
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
        if "<" in s or "less" in s or "under" in s:
            return "RestaurantLessThan20"
        if "20" in s and "50" in s:
            return "Restaurant20To50"
        if "20" in s:
            return "Restaurant20To50"
        return "RestaurantLessThan20"
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
    """Deterministic engineering + target-free freq encodings (fit on train)."""

    def __init__(self):
        self.freq_maps_ = {}
        self.freq_encode_cols_ = []
        self.freq_col_names_ = []
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
        venue_names = out["coupon_canon"].values
        vidx = np.array([VENUE_INDEX.get(v, -1) for v in venue_names])
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

    def fit(self, X):
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
        self._cat_base = truly_cat
        self._num_base = truly_num

        self.freq_encode_cols_ = list(self._cat_base)
        self.freq_maps_ = {}
        for c in self.freq_encode_cols_:
            vc = base[c].value_counts()
            self.freq_maps_[c] = (vc / n).to_dict()

        self.freq_col_names_ = [f"freq_{c}" for c in self.freq_encode_cols_]
        self.categorical_columns = self._cat_base
        self.numeric_columns = self._num_base + self.freq_col_names_
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("CouponFeatureEngineer must be fit before transform().")
        base = self._build_base(X)
        out = base.copy()
        for c in self.freq_encode_cols_:
            m = self.freq_maps_.get(c, {})
            out[f"freq_{c}"] = base[c].map(m).fillna(0.0).astype(float)
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "feature_engineer.joblib"))

    @classmethod
    def load(cls, path):
        obj = joblib.load(os.path.join(path, "feature_engineer.joblib"))
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not a CouponFeatureEngineer.")
        return obj


def load_train_data(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        schema = json.load(f)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    if "label" in labels_df.columns:
        labels = labels_df["label"].values
    else:
        labels = labels_df.iloc[:, 0].values
    return features, labels, schema


def load_predict_data(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        schema = json.load(f)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    return features, schema


def make_stratified_folds(y, n_splits=5, seed=42):
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]


# ===========================================================================
# Target encoding (leakage-safe; fit inside fold / full-train refit)
# ===========================================================================
def _fit_target_encoders(df, y, cols, prior_weight: float = 20.0):
    prior = float(np.mean(y))
    encoders = {}
    for c in cols:
        grp = pd.DataFrame({"v": df[c].astype(str).values, "y": np.asarray(y)})
        agg = grp.groupby("v")["y"].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + prior * prior_weight) / (
            agg["count"] + prior_weight
        )
        encoders[c] = (smoothed.to_dict(), prior)
    return encoders


def _apply_target_encoders(df, cols, encoders):
    n = len(df)
    out = np.zeros((n, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        m, prior = encoders[c]
        out[:, j] = df[c].astype(str).map(m).fillna(prior).values.astype(np.float32)
    return out


# ===========================================================================
# Step 2: Model design
# ===========================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets.float()
        ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(t == 1.0, p, 1.0 - p)
        focal = ((1.0 - pt) ** self.gamma) * ce
        alpha_t = torch.where(t == 1.0, self.alpha, 1.0 - self.alpha)
        return (alpha_t * focal).mean()


class DeepFM(nn.Module):
    def __init__(
        self,
        cat_cardinalities: Sequence[int],
        n_numeric: int,
        emb_dim: int = 8,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.3,
        use_bn: bool = True,
    ):
        super().__init__()
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        self.n_fields = len(self.cat_cardinalities)
        self.n_numeric = int(n_numeric)
        self.emb_dim = int(emb_dim)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, self.emb_dim) for card in self.cat_cardinalities]
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

        self.linear_cats = nn.ModuleList(
            [nn.Embedding(card, 1) for card in self.cat_cardinalities]
        )
        for lin in self.linear_cats:
            nn.init.zeros_(lin.weight)

        self.linear_numeric = (
            nn.Linear(self.n_numeric, 1, bias=False) if self.n_numeric > 0 else None
        )
        self.bias = nn.Parameter(torch.zeros(1))

        in_dim = self.n_fields * self.emb_dim + self.n_numeric
        layers: List[nn.Module] = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, int(h)))
            if use_bn:
                layers.append(nn.BatchNorm1d(int(h)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            cur = int(h)
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(cur, 1)

    def forward(self, cat: torch.Tensor, numeric: Optional[torch.Tensor] = None):
        B = cat.size(0)
        linear = self.bias.expand(B)
        for i in range(self.n_fields):
            linear = linear + self.linear_cats[i](cat[:, i]).squeeze(-1)
        if self.n_numeric > 0 and numeric is not None:
            linear = linear + self.linear_numeric(numeric).squeeze(-1)

        embs = torch.stack(
            [self.embeddings[i](cat[:, i]) for i in range(self.n_fields)], dim=1
        )
        s = embs.sum(dim=1)
        sq = (embs**2).sum(dim=1)
        fm = 0.5 * (s**2 - sq).sum(dim=1)

        deep_in = embs.reshape(B, -1)
        if self.n_numeric > 0 and numeric is not None:
            deep_in = torch.cat([deep_in, numeric], dim=1)
        deep = self.deep_out(self.deep(deep_in)).squeeze(-1)
        return linear + fm + deep


def build_optimizer(model, lr=1e-3, weight_decay=1e-4):
    return torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )


def build_scheduler(optimizer, epochs: int, mode="cosine", warmup_frac=0.1):
    epochs = max(int(epochs), 1)
    warm = max(int(round(epochs * warmup_frac)), 0)

    def lr_lambda(step: int) -> float:
        if warm > 0 and step < warm:
            return float(step + 1) / float(warm + 1)
        progress = (step - warm) / max(1, epochs - warm)
        progress = min(max(progress, 0.0), 1.0)
        if mode == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_lightgbm(seed: int = 42, n_estimators: int = 900, n_jobs: int = 2):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )


def make_meta_lgb(seed: int = 42):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=2,
        verbose=-1,
    )


try:
    from catboost import CatBoostClassifier  # noqa: F401

    _HAS_CB = True
except Exception:
    _HAS_CB = False


def _make_cb(seed: int = 42):
    if not _HAS_CB:
        return None
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        thread_count=2,
        loss_function="Logloss",
    )


# ===========================================================================
# Step 3: Training / evaluation helpers
# ===========================================================================
def _fit_cat_maps(fe_df, cat_cols):
    maps = {}
    for c in cat_cols:
        vals = fe_df[c].astype(str).unique().tolist()
        maps[c] = {v: (i + 1) for i, v in enumerate(vals)}
    return maps


def _transform_cat(fe_df, cat_cols, cat_maps):
    n = len(fe_df)
    arr = np.zeros((n, len(cat_cols)), dtype=np.int64)
    for i, c in enumerate(cat_cols):
        m = cat_maps[c]
        arr[:, i] = fe_df[c].astype(str).map(m).fillna(0).astype(np.int64).values
    return arr


def _numeric_block(fe_df, num_cols):
    if not num_cols:
        return np.zeros((len(fe_df), 0), dtype=np.float32)
    num_block = pd.DataFrame(
        {
            c: pd.to_numeric(fe_df[c], errors="coerce").astype(np.float64)
            for c in num_cols
        },
        index=fe_df.index,
    )
    return num_block.fillna(0.0).values.astype(np.float32)


def _prepare_features(fe_df, cat_cols, num_cols, cat_maps, scaler, fit):
    X_cat = _transform_cat(fe_df, cat_cols, cat_maps)
    num_arr = _numeric_block(fe_df, num_cols)
    if fit:
        scaler.fit(num_arr)
    X_num = scaler.transform(num_arr).astype(np.float32)
    return X_cat, X_num


def _make_lgb_frame(X_cat, X_num, cat_maps, cat_cols, X_extra=None):
    data = {}
    for i, c in enumerate(cat_cols):
        levels = list(range(len(cat_maps[c]) + 1))
        data[f"c{i}"] = pd.Categorical(X_cat[:, i], categories=levels)
    for j in range(X_num.shape[1]):
        data[f"n{j}"] = X_num[:, j]
    if X_extra is not None and X_extra.shape[1] > 0:
        for j in range(X_extra.shape[1]):
            data[f"t{j}"] = X_extra[:, j]
    return pd.DataFrame(data)


def _train_lgbm(
    X_cat, X_num, y, cat_maps, cat_cols, X_extra=None, seed=0, n_estimators=900
):
    model = make_lightgbm(seed=seed, n_estimators=n_estimators, n_jobs=2)
    df = _make_lgb_frame(X_cat, X_num, cat_maps, cat_cols, X_extra)
    cat_names = [f"c{i}" for i in range(len(cat_cols))]
    model.fit(df, y, categorical_feature=cat_names)
    return model


def _predict_lgbm(model, X_cat, X_num, cat_maps, cat_cols, X_extra=None):
    df = _make_lgb_frame(X_cat, X_num, cat_maps, cat_cols, X_extra)
    return model.predict_proba(df)[:, 1]


def _cb_frame(fe_df, cb_cols):
    return fe_df[cb_cols].astype(str).reset_index(drop=True)


def _train_cb(fe_df, cb_cols, y, seed=0):
    m = _make_cb(seed=seed)
    if m is None:
        return None
    df = _cb_frame(fe_df, cb_cols)
    m.fit(df, y, cat_features=cb_cols)
    return m


def _predict_cb(model, fe_df, cb_cols):
    if model is None:
        return None
    df = _cb_frame(fe_df, cb_cols)
    return model.predict_proba(df)[:, 1]


def _train_deepfm(X_cat, X_num, y, card, epochs=18, batch_size=512, seed=0, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"
    n = len(y)
    model = DeepFM(
        cat_cardinalities=card,
        n_numeric=X_num.shape[1],
        emb_dim=8,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
        use_bn=True,
    ).to(device)
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = build_optimizer(model, lr=lr, weight_decay=1e-4)
    sched = build_scheduler(optimizer, epochs=epochs, mode="cosine")

    Xc = torch.from_numpy(X_cat).long()
    Xn = torch.from_numpy(X_num).float()
    yt = torch.from_numpy(y.astype(np.float32))

    model.train()
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            optimizer.zero_grad()
            logits = model(Xc[b], Xn[b])
            loss = criterion(logits, yt[b])
            loss.backward()
            optimizer.step()
        sched.step()
    return model


@torch.no_grad()
def _predict_deepfm(model, X_cat, X_num, batch_size=1024):
    model.eval()
    n = len(X_cat)
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        xc = torch.from_numpy(X_cat[i : i + batch_size]).long()
        xn = torch.from_numpy(X_num[i : i + batch_size]).float()
        out[i : i + batch_size] = torch.sigmoid(model(xc, xn)).cpu().numpy()
    return out


def _tune_threshold(probs, y):
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 61):
        acc = accuracy_score(y, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = float(acc), float(t)
    return best_t, best_acc


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def run_training(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    features, labels, schema = load_train_data(data_dir)
    y_raw = np.asarray(labels)
    uniq = sorted(set(y_raw.tolist()))
    label_map = {u: i for i, u in enumerate(uniq)}
    y = np.array([label_map[v] for v in y_raw], dtype=np.int64)

    fe = CouponFeatureEngineer()
    fe.fit(features)
    fe_train = fe.transform(features)
    cat_cols = list(fe.categorical_columns)
    num_cols = list(fe.numeric_columns)
    # Target-encode only the interaction cats that exist in this dataset.
    te_cols = [c for c in INTERACTION_COLS if c in fe_train.columns]
    # CatBoost / raw categoricals = interaction cats + key base cats.
    cb_cols = te_cols + [
        c
        for c in [
            "coupon_canon",
            "expiration",
            "time",
            "gender",
            "age",
            "income",
            "education",
            "occupation",
            "maritalStatus",
            "direction",
        ]
        if c in fe_train.columns
    ]
    # dedupe, preserve order
    seen = set()
    cb_cols = [c for c in cb_cols if not (c in seen or seen.add(c))]

    cat_maps = _fit_cat_maps(fe_train, cat_cols)
    scaler = StandardScaler()
    X_cat, X_num = _prepare_features(
        fe_train, cat_cols, num_cols, cat_maps, scaler, fit=True
    )
    card = [len(cat_maps[c]) + 1 for c in cat_cols]
    print(
        f"[train] rows={len(y)} cat={len(cat_cols)} num={len(num_cols)} "
        f"te={len(te_cols)} cb={len(cb_cols)} pos_rate={y.mean():.4f} sum_card={sum(card)}"
    )

    folds = make_stratified_folds(y, n_splits=5, seed=42)
    oof_lgb = np.zeros(len(y))
    oof_cb = np.zeros(len(y))
    oof_fm = np.zeros(len(y))

    for fi, (tr, va) in enumerate(folds):
        # Fit all learned preprocessing on the fold's TRAIN rows only.
        fe_f = CouponFeatureEngineer().fit(features.iloc[tr])
        fe_tr = fe_f.transform(features.iloc[tr])
        fe_va = fe_f.transform(features.iloc[va])
        cat_maps_f = _fit_cat_maps(fe_tr, cat_cols)
        sc_f = StandardScaler()
        Xc_tr, Xn_tr = _prepare_features(
            fe_tr, cat_cols, num_cols, cat_maps_f, sc_f, fit=True
        )
        Xc_va, Xn_va = _prepare_features(
            fe_va, cat_cols, num_cols, cat_maps_f, sc_f, fit=False
        )
        # Fold-internal target encoders (TRAIN rows only).
        te_enc_f = _fit_target_encoders(fe_tr, y[tr], te_cols)
        TE_tr = _apply_target_encoders(fe_tr, te_cols, te_enc_f)
        TE_va = _apply_target_encoders(fe_va, te_cols, te_enc_f)

        card_f = [len(cat_maps_f[c]) + 1 for c in cat_cols]

        # LGBM (with TE interactions)
        lgb_f = _train_lgbm(
            Xc_tr, Xn_tr, y[tr], cat_maps_f, cat_cols, X_extra=TE_tr, seed=fi
        )
        oof_lgb[va] = _predict_lgbm(
            lgb_f, Xc_va, Xn_va, cat_maps_f, cat_cols, X_extra=TE_va
        )

        # CatBoost (raw categoricals)
        if _HAS_CB:
            cb_f = _train_cb(fe_tr, cb_cols, y[tr], seed=fi)
            p_cb = _predict_cb(cb_f, fe_va, cb_cols)
            if p_cb is not None:
                oof_cb[va] = p_cb
            else:
                oof_cb[va] = oof_lgb[va]
        else:
            oof_cb[va] = oof_lgb[va]

        # DeepFM (diversity)
        fm_f = _train_deepfm(Xc_tr, Xn_tr, y[tr], card_f, epochs=18, seed=fi)
        oof_fm[va] = _predict_deepfm(fm_f, Xc_va, Xn_va)

        print(
            f"[fold {fi}] "
            f"lgb={accuracy_score(y[va], (oof_lgb[va] >= 0.5).astype(int)):.4f} "
            f"cb={accuracy_score(y[va], (oof_cb[va] >= 0.5).astype(int)):.4f} "
            f"fm={accuracy_score(y[va], (oof_fm[va] >= 0.5).astype(int)):.4f}"
        )

    t_lgb, a_lgb = _tune_threshold(oof_lgb, y)
    t_cb, a_cb = _tune_threshold(oof_cb, y)
    t_fm, a_fm = _tune_threshold(oof_fm, y)
    soft = 0.4 * oof_lgb + 0.4 * oof_cb + 0.2 * oof_fm
    t_sv, a_sv = _tune_threshold(soft, y)
    print(f"[OOF] LGB    acc={a_lgb:.6f} @ t={t_lgb:.2f}")
    print(f"[OOF] CB     acc={a_cb:.6f} @ t={t_cb:.2f}")
    print(f"[OOF] FM     acc={a_fm:.6f} @ t={t_fm:.2f}")
    print(f"[OOF] SoftV  acc={a_sv:.6f} @ t={t_sv:.2f}")

    # Meta-LGBM stack, OOF via fold-wise fit.
    meta_X = np.column_stack([oof_lgb, oof_cb, oof_fm])
    meta_oof = np.zeros(len(y))
    for tr, va in folds:
        mm = make_meta_lgb(seed=42)
        mm.fit(meta_X[tr], y[tr])
        meta_oof[va] = mm.predict_proba(meta_X[va])[:, 1]
    t_meta, a_meta = _tune_threshold(meta_oof, y)
    print(f"[OOF] MetaL  acc={a_meta:.6f} @ t={t_meta:.2f}")

    candidates = [
        ("lgb", a_lgb, t_lgb),
        ("cb", a_cb, t_cb),
        ("softvote", a_sv, t_sv),
        ("meta_lgb", a_meta, t_meta),
    ]
    best_name, best_score, best_thr = max(candidates, key=lambda x: x[1])
    print(f"[select] chosen={best_name} acc={best_score:.6f} @ t={best_thr:.2f}")

    # Final refit on ALL training rows.
    te_enc_full = _fit_target_encoders(fe_train, y, te_cols)
    TE_full = _apply_target_encoders(fe_train, te_cols, te_enc_full)

    final_lgb_models = [
        _train_lgbm(X_cat, X_num, y, cat_maps, cat_cols, X_extra=TE_full, seed=s)
        for s in (0, 1, 2)
    ]
    final_cb_models = []
    if _HAS_CB:
        final_cb_models = [_train_cb(fe_train, cb_cols, y, seed=s) for s in (0, 1, 2)]
    final_fm = _train_deepfm(X_cat, X_num, y, card, epochs=18, seed=0)

    final_meta = None
    if best_name == "meta_lgb":
        final_meta = make_meta_lgb(seed=42)
        final_meta.fit(meta_X, y)

    bundle = {
        "fe": fe,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "te_cols": te_cols,
        "cb_cols": cb_cols,
        "cat_maps": cat_maps,
        "scaler": scaler,
        "card": card,
        "te_encoders": te_enc_full,
        "lgb_models": final_lgb_models,
        "cb_models": final_cb_models,
        "use_cb": _HAS_CB,
        "fm_state": final_fm.state_dict(),
        "fm_config": {
            "card": card,
            "n_numeric": int(X_num.shape[1]),
            "emb_dim": 8,
            "hidden_dims": [256, 128, 64],
            "dropout": 0.3,
        },
        "meta_model": final_meta,
        "choice": best_name,
        "soft_w": (0.4, 0.4, 0.2),
        "threshold": best_thr,
        "label_map": label_map,
    }
    joblib.dump(bundle, os.path.join(model_dir, "bundle.joblib"))
    print(f"[train] saved bundle to {model_dir}")
    print(f"Final Validation Score: {best_score:.6f}")


# ---------------------------------------------------------------------------
# PREDICT (fresh process, read-only model)
# ---------------------------------------------------------------------------
def run_predict(data_dir, model_dir, output_path):
    features, schema = load_predict_data(data_dir)
    bundle = joblib.load(os.path.join(model_dir, "bundle.joblib"))

    fe = bundle["fe"]
    cat_cols = bundle["cat_cols"]
    num_cols = bundle["num_cols"]
    te_cols = bundle["te_cols"]
    cb_cols = bundle["cb_cols"]
    cat_maps = bundle["cat_maps"]
    scaler = bundle["scaler"]
    label_map = bundle["label_map"]
    threshold = bundle["threshold"]
    choice = bundle["choice"]
    soft_w = bundle["soft_w"]

    fe_p = fe.transform(features)
    X_cat, X_num = _prepare_features(
        fe_p, cat_cols, num_cols, cat_maps, scaler, fit=False
    )
    TE = _apply_target_encoders(fe_p, te_cols, bundle["te_encoders"])

    lgb_probs = np.mean(
        [
            _predict_lgbm(m, X_cat, X_num, cat_maps, cat_cols, X_extra=TE)
            for m in bundle["lgb_models"]
        ],
        axis=0,
    )

    if bundle.get("use_cb") and bundle.get("cb_models"):
        cb_probs = np.mean(
            [_predict_cb(m, fe_p, cb_cols) for m in bundle["cb_models"]], axis=0
        )
    else:
        cb_probs = lgb_probs

    fm_cfg = bundle["fm_config"]
    fm = DeepFM(
        cat_cardinalities=fm_cfg["card"],
        n_numeric=fm_cfg["n_numeric"],
        emb_dim=fm_cfg["emb_dim"],
        hidden_dims=fm_cfg["hidden_dims"],
        dropout=fm_cfg["dropout"],
        use_bn=True,
    )
    fm.load_state_dict(bundle["fm_state"])
    fm.eval()
    fm_probs = _predict_deepfm(fm, X_cat, X_num)

    if choice == "lgb":
        probs = lgb_probs
    elif choice == "cb":
        probs = cb_probs
    elif choice == "softvote":
        probs = soft_w[0] * lgb_probs + soft_w[1] * cb_probs + soft_w[2] * fm_probs
    else:  # meta_lgb
        meta_X = np.column_stack([lgb_probs, cb_probs, fm_probs])
        probs = bundle["meta_model"].predict_proba(meta_X)[:, 1]

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
    print(f"[predict] wrote {len(out_df)} rows via {choice}")


# ---------------------------------------------------------------------------
# CLI
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
        run_training(args.data, args.model)
    else:
        run_predict(args.data, args.model, args.output)
