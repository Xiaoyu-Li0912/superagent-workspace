"""
Coupon acceptance prediction — merged pipeline.

Stages:
  1) Leakage-safe feature engineering (fit on TRAIN rows only).
  2) Model design: DeepFM + FocalLoss + optimizers + heterogeneous base learners.
  3) Training: 5-fold stratified train-only CV -> OOF probs -> threshold tuned
     stack (LightGBM + DeepFM + LogisticRegression -> meta LR); final refit;
     complete bundle persisted; fresh-process predict.

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

        self._cat_base = [c for c in base.columns if base[c].dtype == object]
        self._num_base = [c for c in base.columns if c not in self._cat_base]

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
# Step 2: Model design
# ===========================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, p, 1.0 - p)
        focal = ((1.0 - pt) ** self.gamma) * ce
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * focal
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCEWithLogits(nn.Module):
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = None
        if self.pos_weight is not None:
            pw = torch.as_tensor(
                [self.pos_weight], dtype=logits.dtype, device=logits.device
            )
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=pw
        )


def build_loss(name: str = "focal", pos_weight: Optional[float] = None) -> nn.Module:
    name = (name or "focal").lower()
    if name == "focal":
        return FocalLoss(alpha=0.5, gamma=2.0)
    if name in ("wce", "weighted_bce", "bce"):
        return WeightedBCEWithLogits(pos_weight=pos_weight)
    raise ValueError(f"Unknown loss: {name}")


class DeepFM(nn.Module):
    """DeepFM: linear + 2nd-order FM interaction + deep MLP over cat/numeric."""

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

    def forward(
        self, cat: torch.Tensor, numeric: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    mode: str = "cosine",
    warmup_frac: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    epochs = max(int(epochs), 1)
    warm = max(int(round(epochs * warmup_frac)), 0)

    def lr_lambda(step: int) -> float:
        if warm > 0 and step < warm:
            return float(step + 1) / float(warm + 1)
        progress = (step - warm) / max(1, epochs - warm)
        progress = min(max(progress, 0.0), 1.0)
        if mode == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if mode == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_lightgbm(seed: int = 42, n_estimators: int = 800, n_jobs: int = 2):
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


def make_logistic_regression(seed: int = 42):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=0.7, max_iter=1000, solver="lbfgs")


def make_complement_nb():
    from sklearn.naive_bayes import ComplementNB

    return ComplementNB(alpha=1.0)


class StackingMetaModel:
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        self.fitted_ = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.fitted_ = True
        return self

    def predict_proba_pos(self, X):
        if not self.fitted_:
            raise RuntimeError("StackingMetaModel must be fit before predict.")
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


def build_nn_bundle(
    cat_cardinalities: Sequence[int],
    n_numeric: int,
    emb_dim: int = 8,
    hidden_dims: Sequence[int] = (256, 128, 64),
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 40,
    device: str = "cpu",
) -> Tuple[
    DeepFM, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]:
    model = DeepFM(
        cat_cardinalities=cat_cardinalities,
        n_numeric=n_numeric,
        emb_dim=emb_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_bn=True,
    ).to(device)
    criterion = build_loss("focal")
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, epochs=epochs, mode="cosine")
    return model, criterion, optimizer, scheduler


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


def _prepare_features(fe_df, cat_cols, num_cols, cat_maps, scaler, fit):
    X_cat = _transform_cat(fe_df, cat_cols, cat_maps)
    num_arr = fe_df[num_cols].fillna(0.0).values.astype(np.float32)
    if fit:
        scaler.fit(num_arr)
    X_num = scaler.transform(num_arr).astype(np.float32)
    return X_cat, X_num


def _make_lgb_frame(X_cat, X_num, cat_maps, cat_cols):
    """Build a DataFrame with stable categorical dtypes (0..cardinality-1+1)."""
    data = {}
    for i, c in enumerate(cat_cols):
        levels = list(range(len(cat_maps[c]) + 1))
        data[f"c{i}"] = pd.Categorical(X_cat[:, i], categories=levels)
    for j in range(X_num.shape[1]):
        data[f"n{j}"] = X_num[:, j]
    return pd.DataFrame(data)


def _train_lgbm(X_cat, X_num, y, cat_maps, cat_cols, seed=0, n_estimators=500):
    model = make_lightgbm(seed=seed, n_estimators=n_estimators, n_jobs=2)
    df = _make_lgb_frame(X_cat, X_num, cat_maps, cat_cols)
    cat_names = [f"c{i}" for i in range(len(cat_cols))]
    model.fit(df, y, categorical_feature=cat_names)
    return model


def _predict_lgbm(model, X_cat, X_num, cat_maps, cat_cols):
    df = _make_lgb_frame(X_cat, X_num, cat_maps, cat_cols)
    return model.predict_proba(df)[:, 1]


def _train_lr(X_cat, X_num, y, seed=0):
    model = make_logistic_regression(seed=seed)
    X = np.hstack([X_cat.astype(np.float32), X_num])
    model.fit(X, y)
    return model


def _predict_lr(model, X_cat, X_num):
    X = np.hstack([X_cat.astype(np.float32), X_num])
    return model.predict_proba(X)[:, 1]


def _train_deepfm(X_cat, X_num, y, card, epochs=25, batch_size=512, seed=0, lr=1e-3):
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
        logits = model(xc, xn)
        out[i : i + batch_size] = torch.sigmoid(logits).cpu().numpy()
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

    cat_maps = _fit_cat_maps(fe_train, cat_cols)
    scaler = StandardScaler()
    X_cat, X_num = _prepare_features(
        fe_train, cat_cols, num_cols, cat_maps, scaler, fit=True
    )
    card = [len(cat_maps[c]) + 1 for c in cat_cols]
    print(
        f"[train] rows={len(y)} cat={len(cat_cols)} num={len(num_cols)} "
        f"pos_rate={y.mean():.4f} sum_card={sum(card)}"
    )

    folds = make_stratified_folds(y, n_splits=5, seed=42)
    oof_lgb = np.zeros(len(y))
    oof_fm = np.zeros(len(y))
    oof_lr = np.zeros(len(y))

    for fi, (tr, va) in enumerate(folds):
        m = _train_lgbm(X_cat[tr], X_num[tr], y[tr], cat_maps, cat_cols, seed=fi)
        oof_lgb[va] = _predict_lgbm(m, X_cat[va], X_num[va], cat_maps, cat_cols)
        fm = _train_deepfm(X_cat[tr], X_num[tr], y[tr], card, epochs=25, seed=fi)
        oof_fm[va] = _predict_deepfm(fm, X_cat[va], X_num[va])
        lr_m = _train_lr(X_cat[tr], X_num[tr], y[tr], seed=fi)
        oof_lr[va] = _predict_lr(lr_m, X_cat[va], X_num[va])
        print(
            f"[fold {fi}] "
            f"lgb={accuracy_score(y[va], (oof_lgb[va] >= 0.5).astype(int)):.4f} "
            f"fm={accuracy_score(y[va], (oof_fm[va] >= 0.5).astype(int)):.4f} "
            f"lr={accuracy_score(y[va], (oof_lr[va] >= 0.5).astype(int)):.4f}"
        )

    t_lgb, a_lgb = _tune_threshold(oof_lgb, y)
    t_fm, a_fm = _tune_threshold(oof_fm, y)
    t_lr, a_lr = _tune_threshold(oof_lr, y)
    print(f"[OOF] LGB acc={a_lgb:.6f} @ t={t_lgb:.2f}")
    print(f"[OOF] FM  acc={a_fm:.6f} @ t={t_fm:.2f}")
    print(f"[OOF] LR  acc={a_lr:.6f} @ t={t_lr:.2f}")

    meta_X = np.column_stack([oof_lgb, oof_fm, oof_lr])
    meta_oof = np.zeros(len(y))
    for tr, va in folds:
        mm = StackingMetaModel(C=1.0, max_iter=500)
        mm.fit(meta_X[tr], y[tr])
        meta_oof[va] = mm.predict_proba_pos(meta_X[va])
    t_st, a_st = _tune_threshold(meta_oof, y)
    print(f"[OOF] Stack acc={a_st:.6f} @ t={t_st:.2f}")

    candidates = [("lgb", a_lgb, t_lgb), ("stack", a_st, t_st)]
    best_name, best_score, best_t = max(candidates, key=lambda x: x[1])
    use_stack = best_name == "stack"
    print(f"[select] chosen={best_name} acc={best_score:.6f} @ t={best_t:.2f}")

    # Final fits on ALL training rows
    final_lgb_models = [
        _train_lgbm(X_cat, X_num, y, cat_maps, cat_cols, seed=s) for s in (0, 1, 2)
    ]
    final_fm = _train_deepfm(X_cat, X_num, y, card, epochs=25, seed=0)
    final_lr = _train_lr(X_cat, X_num, y, seed=0)
    final_meta = StackingMetaModel(C=1.0, max_iter=500)
    final_meta.fit(meta_X, y)

    bundle = {
        "fe": fe,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_maps": cat_maps,
        "scaler": scaler,
        "card": card,
        "lgb_models": final_lgb_models,
        "lr_model": final_lr,
        "meta_model": final_meta,
        "use_stack": use_stack,
        "threshold": best_t,
        "label_map": label_map,
    }
    joblib.dump(bundle, os.path.join(model_dir, "bundle.joblib"))
    torch.save(
        {
            "state_dict": final_fm.state_dict(),
            "card": card,
            "n_numeric": int(X_num.shape[1]),
            "emb_dim": 8,
            "hidden_dims": [256, 128, 64],
            "dropout": 0.3,
        },
        os.path.join(model_dir, "fm.pt"),
    )
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
    cat_maps = bundle["cat_maps"]
    scaler = bundle["scaler"]
    label_map = bundle["label_map"]
    threshold = bundle["threshold"]
    use_stack = bundle["use_stack"]

    fe_p = fe.transform(features)
    X_cat = _transform_cat(fe_p, cat_cols, cat_maps)
    num_arr = fe_p[num_cols].fillna(0.0).values.astype(np.float32)
    X_num = scaler.transform(num_arr).astype(np.float32)

    lgb_probs = np.mean(
        [
            _predict_lgbm(m, X_cat, X_num, cat_maps, cat_cols)
            for m in bundle["lgb_models"]
        ],
        axis=0,
    )

    if use_stack:
        ck = torch.load(os.path.join(model_dir, "fm.pt"), map_location="cpu")
        fm = DeepFM(
            cat_cardinalities=ck["card"],
            n_numeric=ck["n_numeric"],
            emb_dim=ck["emb_dim"],
            hidden_dims=ck["hidden_dims"],
            dropout=ck["dropout"],
            use_bn=True,
        )
        fm.load_state_dict(ck["state_dict"])
        fm.eval()
        fm_probs = _predict_deepfm(fm, X_cat, X_num)
        lr_probs = _predict_lr(bundle["lr_model"], X_cat, X_num)
        meta_X = np.column_stack([lgb_probs, fm_probs, lr_probs])
        probs = bundle["meta_model"].predict_proba_pos(meta_X)
    else:
        probs = lgb_probs

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
    print(f"[predict] wrote {len(out_df)} rows to {output_path}")


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
