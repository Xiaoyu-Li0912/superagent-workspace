"""
solution.py
Leakage-safe DCN-V2 + Feature-Gating pipeline for in-vehicle coupon acceptance.

Implements:
    python solution.py train   --data /data --model /model
    python solution.py predict --data /data --model /model --output /output/predictions.csv

All stateful preprocessing, fold models, encoders, label mapping and the decision
threshold are fit on TRAINING ROWS ONLY and persisted under --model.  Prediction
loads artifacts read-only (no refitting) and writes row_id,prediction with the
original label.  Official metric: accuracy (threshold tuned on OOF).
"""

from __future__ import annotations

import os
import re
import json
import copy
import pickle
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

# ============================================================================
# STEP 1: Data loading & feature engineering (leakage-safe)
# ============================================================================
FREQUENCY_MAP = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}

VENUE_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]

CAT_RAW = ["coupon", "time", "expiration", "gender", "maritalStatus", "occupation"]

EDUCATION_MAP = {
    "some high school": 0,
    "high school graduate": 1,
    "some college - no degree": 2,
    "associates degree": 3,
    "bachelors degree": 4,
    "graduate degree (masters, doctorate)": 5,
}


def _parse_age(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if "below" in s:
        return 20.0
    if "plus" in s:
        return 50.0
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


def _parse_income(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace(",", "")
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


def _parse_time(v):
    if pd.isna(v):
        return np.nan
    s = str(v).upper().replace(" ", "")
    m = re.match(r"(\d+)(AM|PM)", s)
    if not m:
        return np.nan
    h = int(m.group(1))
    ap = m.group(2)
    if ap == "PM" and h != 12:
        h += 12
    if ap == "AM" and h == 12:
        h = 0
    return float(h)


def _parse_expiration(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if "h" in s:
        return 2.0
    if "d" in s:
        return 24.0
    return np.nan


def _parse_freq(v):
    if pd.isna(v):
        return np.nan
    return FREQUENCY_MAP.get(str(v).strip().lower(), np.nan)


def _parse_education(v):
    if pd.isna(v):
        return np.nan
    return EDUCATION_MAP.get(str(v).strip().lower(), np.nan)


def _coupon_venue(coupon):
    """Map coupon label -> the customer-habit column describing that venue."""
    if pd.isna(coupon):
        return None
    s = str(coupon).lower()
    if "bar" in s:
        return "Bar"
    if "coffee" in s:
        return "CoffeeHouse"
    if "carry" in s or "take" in s:
        return "CarryAway"
    if "20" in s and "50" in s:
        return "Restaurant20To50"
    if "20" in s:
        return "RestaurantLessThan20"
    return None


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def load_features(data_dir):
    return pd.read_csv(os.path.join(data_dir, "features.csv"))


def load_labels(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    col = "label" if "label" in df.columns else df.columns[0]
    return df[col]


class FeatureBuilder:
    """Entirely stateful feature pipeline; fit on TRAIN rows only."""

    def __init__(self):
        self.fill_values_: Dict[str, object] = {}
        self.num_medians_: Dict[str, float] = {}
        self.const_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.num_cols_: List[str] = []
        self.feature_cols_: List[str] = []
        self.fitted_ = False

    def _engineer(self, df):
        df = df.copy()
        out = pd.DataFrame(index=df.index)

        present_venues = [c for c in VENUE_COLS if c in df.columns]
        if present_venues:
            out["n_missing_venues"] = (
                df[present_venues].isna().sum(axis=1).astype(float)
            )

        for c, v in self.fill_values_.items():
            if c in df.columns:
                df[c] = df[c].fillna(v)

        if "temperature" in df.columns:
            out["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        if "age" in df.columns:
            out["age"] = df["age"].map(_parse_age)
        if "income" in df.columns:
            out["income"] = df["income"].map(_parse_income)
        if "education" in df.columns:
            out["education_ord"] = df["education"].map(_parse_education)
        if "has_children" in df.columns:
            out["has_children"] = pd.to_numeric(df["has_children"], errors="coerce")
        if "expiration" in df.columns:
            out["expiration_hours"] = df["expiration"].map(_parse_expiration)
        if "time" in df.columns:
            out["time_hour"] = df["time"].map(_parse_time)

        for c in VENUE_COLS:
            if c in df.columns:
                out[c] = df[c].map(_parse_freq)

        freq_present = [c for c in VENUE_COLS if c in out.columns]
        if freq_present:
            out["total_venue_freq"] = out[freq_present].sum(axis=1, min_count=1)

        if "coupon" in df.columns and freq_present:
            venue_freq = pd.Series(np.nan, index=df.index, dtype=float)
            venue_name = df["coupon"].map(_coupon_venue)
            for col in present_venues:
                mask = venue_name == col
                if mask.any():
                    venue_freq.loc[mask] = out.loc[mask, col]
            out["venue_affinity"] = venue_freq
            out["is_venue_frequent"] = (venue_freq >= 2).astype(float)

        geq15 = (
            pd.to_numeric(df["toCoupon_GEQ15min"], errors="coerce")
            if "toCoupon_GEQ15min" in df.columns
            else None
        )
        geq25 = (
            pd.to_numeric(df["toCoupon_GEQ25min"], errors="coerce")
            if "toCoupon_GEQ25min" in df.columns
            else None
        )
        if geq15 is not None:
            out["toCoupon_GEQ15min"] = geq15
        if geq25 is not None:
            out["toCoupon_GEQ25min"] = geq25
        if geq15 is not None and geq25 is not None:
            tier = np.where(geq15 == 0, 0.0, np.where(geq25 == 0, 1.0, 2.0))
            out["distance_tier"] = tier
        if "direction_same" in df.columns:
            out["direction_same"] = pd.to_numeric(df["direction_same"], errors="coerce")

        for c in CAT_RAW:
            if c in df.columns:
                out[c] = df[c].astype(object).where(df[c].notna(), "missing")

        if "coupon" in df.columns and "time" in df.columns:
            out["coupon_time"] = df["coupon"].astype(str) + "|" + df["time"].astype(str)
        if "coupon" in df.columns and "expiration" in df.columns:
            out["coupon_exp"] = (
                df["coupon"].astype(str) + "|" + df["expiration"].astype(str)
            )

        return out

    def fit(self, df):
        df = df.copy()

        mode_cols = [c for c in (CAT_RAW + VENUE_COLS) if c in df.columns]
        for c in mode_cols:
            vc = df[c].dropna()
            self.fill_values_[c] = vc.mode().iloc[0] if len(vc) else "missing"

        eng = self._engineer(df)

        num_candidates = [
            c for c in eng.columns if c not in (CAT_RAW + ["coupon_time", "coupon_exp"])
        ]
        for c in num_candidates:
            self.num_medians_[c] = (
                float(eng[c].median()) if eng[c].notna().any() else 0.0
            )
            eng[c] = eng[c].fillna(self.num_medians_[c])
        self.const_cols_ = [
            c for c in num_candidates if eng[c].nunique(dropna=True) <= 1
        ]

        self.num_cols_ = [c for c in num_candidates if c not in self.const_cols_]
        self.cat_cols_ = [
            c for c in eng.columns if c in (CAT_RAW + ["coupon_time", "coupon_exp"])
        ]
        self.feature_cols_ = self.cat_cols_ + self.num_cols_
        self.fitted_ = True
        return self

    def transform(self, df):
        if not self.fitted_:
            raise RuntimeError("FeatureBuilder must be fit() on training data first.")
        eng = self._engineer(df)

        for c in self.num_cols_:
            if c not in eng.columns:
                eng[c] = np.nan
            eng[c] = pd.to_numeric(eng[c], errors="coerce").fillna(
                self.num_medians_.get(c, 0.0)
            )

        for c in self.cat_cols_:
            if c not in eng.columns:
                eng[c] = "missing"
            eng[c] = eng[c].astype(object).where(pd.notna(eng[c]), "missing")

        return eng[self.feature_cols_].copy()

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def make_cv_splits(y, n_splits=5, seed=42):
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def prepare_training(data_dir):
    schema = load_schema(data_dir)
    feats = load_features(data_dir)
    y = load_labels(data_dir)
    builder = FeatureBuilder().fit(feats)
    X = builder.transform(feats)
    return X, y, builder, schema


def prepare_inference(data_dir, builder):
    schema = load_schema(data_dir)
    feats = load_features(data_dir)
    X = builder.transform(feats)
    return X, schema


def _to_float_tensor(arr):
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


# ============================================================================
# STEP 2: DCN-V2 + Feature-Gating architecture
# ============================================================================
class TensorEncoder:
    """Maps engineered DataFrame -> integer-categorical / standardised numeric."""

    def __init__(self, cat_cols: Optional[List[str]] = None):
        self.cat_cols_: List[str] = list(cat_cols) if cat_cols else []
        self.num_cols_: List[str] = []
        self.maps_: Dict[str, Dict[str, int]] = {}
        self.cardinalities: List[int] = []
        self.num_dim: int = 0
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self.fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
    ) -> "TensorEncoder":
        if cat_cols is not None:
            self.cat_cols_ = list(cat_cols)
        if num_cols is not None:
            self.num_cols_ = list(num_cols)
        else:
            self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]

        self.maps_, self.cardinalities = {}, []
        for c in self.cat_cols_:
            vals = X[c].astype(object)
            vals = vals.where(pd.notna(vals), "missing").astype(str)
            uniq = sorted(vals.unique().tolist())
            self.maps_[c] = {v: i + 1 for i, v in enumerate(uniq)}
            self.cardinalities.append(len(uniq) + 1)

        if self.num_cols_:
            num = (
                X[self.num_cols_]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float64)
            )
            num = np.nan_to_num(num, nan=0.0)
            self._mean = num.mean(axis=0)
            std = num.std(axis=0)
            std[std < 1e-8] = 1.0
            self._std = std
        else:
            self._mean = np.zeros(0)
            self._std = np.ones(0)
        self.num_dim = len(self.num_cols_)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("TensorEncoder.fit() must be called on training data.")

        n = len(X)
        cat = np.zeros((n, len(self.cat_cols_)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols_):
            mapping = self.maps_[c]
            if c in X.columns:
                vals = X[c].astype(object)
                vals = vals.where(pd.notna(vals), "missing").astype(str)
                cat[:, j] = vals.map(lambda v: mapping.get(v, 0)).to_numpy(
                    dtype=np.int64
                )
        if self.num_dim:
            num = (
                X.reindex(columns=self.num_cols_)
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float64)
            )
            num = np.nan_to_num(num, nan=0.0)
            num = (num - self._mean) / self._std
            num = np.clip(num, -8.0, 8.0).astype(np.float32)
        else:
            num = np.zeros((n, 0), dtype=np.float32)
        return cat, num

    def as_tensors(
        self, X: pd.DataFrame, device: torch.device | str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cat, num = self.transform(X)
        return (torch.from_numpy(cat).to(device), torch.from_numpy(num).to(device))


class FeatureGate(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or max(dim // 2, 8)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x) * 2.0


class CrossLayerV2(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(x) + x


class DeepBranch(nn.Module):
    def __init__(
        self, in_dim: int, hidden: Tuple[int, ...] = (128, 64), dropout: float = 0.2
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, hidden[-1] if hidden else in_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = prev if not hidden else hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DCNConfig:
    emb_dim: int = 8
    d_model: int = 64
    n_cross: int = 3
    deep_hidden: Tuple[int, ...] = (128, 64)
    dropout: float = 0.20
    gate_dropout: float = 0.10
    label_smoothing: float = 0.01
    lr: float = 2e-3
    weight_decay: float = 1e-4
    min_lr_ratio: float = 0.05


class DCNv2Gated(nn.Module):
    def __init__(
        self, cardinalities: List[int], num_dim: int, cfg: Optional[DCNConfig] = None
    ):
        super().__init__()
        cfg = cfg or DCNConfig()
        self.cfg = cfg
        self.num_dim = num_dim
        self.num_cat = len(cardinalities)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(card, cfg.emb_dim, padding_idx=None)
                for card in cardinalities
            ]
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.05)

        cat_dim = self.num_cat * cfg.emb_dim
        self.cat_proj = nn.Linear(cat_dim, cfg.d_model) if cat_dim else None
        self.num_proj = nn.Linear(num_dim, cfg.d_model) if num_dim else None
        self._fallback_dim = max(num_dim, cat_dim, 1)
        self.gate = FeatureGate(cfg.d_model, dropout=cfg.gate_dropout)

        self.cross_layers = nn.ModuleList(
            [CrossLayerV2(cfg.d_model) for _ in range(cfg.n_cross)]
        )
        self.deep = DeepBranch(cfg.d_model, cfg.deep_hidden, cfg.dropout)

        self.head = nn.Linear(cfg.d_model + self.deep.out_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        if self.cat_proj is not None:
            embs = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
            parts.append(self.cat_proj(torch.cat(embs, dim=-1)))
        if self.num_proj is not None:
            parts.append(self.num_proj(num))
        if not parts:
            raise RuntimeError("Model received neither categorical nor numeric input.")
        x0 = parts[0] if len(parts) == 1 else parts[0] + parts[1]
        x0 = self.gate(x0)

        x = x0
        for layer in self.cross_layers:
            x = layer(x0, x)

        deep = self.deep(x0)
        logit = self.head(torch.cat([x, deep], dim=-1)).squeeze(-1)
        return logit


def smooth_binary_targets(y: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        return y
    return y * (1.0 - eps) + 0.5 * eps


def build_optimizer(model: nn.Module, cfg: DCNConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embedding" in name or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )


def build_scheduler(
    opt: torch.optim.Optimizer, epochs: int, cfg: DCNConfig
) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs, 1), eta_min=cfg.lr * cfg.min_lr_ratio
    )


def build_model(
    cardinalities: List[int],
    num_dim: int,
    epochs: int,
    cfg: Optional[DCNConfig] = None,
    device: str | torch.device = "cpu",
):
    cfg = cfg or DCNConfig()
    torch.manual_seed(42)
    model = DCNv2Gated(cardinalities, num_dim, cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, epochs, cfg)
    return model, criterion, optimizer, scheduler, cfg


# ============================================================================
# STEP 3: Training / evaluation & CLI
# ============================================================================
def _predict_logits(model, cat, num):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(cat), torch.from_numpy(num))


def _train_one_fold(
    X_tr,
    y_tr,
    X_va,
    y_va,
    cat_cols,
    num_cols,
    fold,
    epochs=50,
    patience=10,
    seed=42,
):
    torch.set_num_threads(2)
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)

    enc = TensorEncoder().fit(X_tr, cat_cols=cat_cols, num_cols=num_cols)
    cat_tr, num_tr = enc.transform(X_tr)
    cat_va, num_va = enc.transform(X_va)

    y_tr_f = np.asarray(y_tr, dtype=np.float32)
    y_va_f = np.asarray(y_va, dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(cat_tr),
        torch.from_numpy(num_tr),
        torch.from_numpy(y_tr_f),
    )
    loader = DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=False
    )

    model, criterion, optimizer, scheduler, cfg = build_model(
        cardinalities=enc.cardinalities,
        num_dim=enc.num_dim,
        epochs=epochs,
        cfg=DCNConfig(),
        device="cpu",
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for cat_b, num_b, y_b in loader:
            optimizer.zero_grad()
            logits = model(cat_b, num_b)
            smoothed_y = smooth_binary_targets(y_b, cfg.label_smoothing)
            loss = criterion(logits, smoothed_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        train_loss = total_loss / max(n_batches, 1)

        val_logits = _predict_logits(model, cat_va, num_va)
        with torch.no_grad():
            val_loss = float(criterion(val_logits, torch.from_numpy(y_va_f)).item())
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_acc = float(((val_prob >= 0.5).astype(int) == y_va).mean())

        print(
            f"Fold {fold} Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            print(
                f"Fold {fold}: early stopping at epoch {epoch + 1} "
                f"(best epoch {best_epoch + 1})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits = _predict_logits(model, cat_va, num_va)
    with torch.no_grad():
        best_val_prob = torch.sigmoid(val_logits).cpu().numpy()

    return model, enc, best_val_loss, best_val_prob, cfg


def train_command(args):
    torch.set_num_threads(2)
    np.random.seed(42)
    torch.manual_seed(42)

    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)

    X, y, builder, schema = prepare_training(args.data)

    y_series = pd.Series(np.asarray(y))
    label_classes = sorted(y_series.dropna().unique().tolist(), key=lambda v: str(v))
    class_to_idx = {v: i for i, v in enumerate(label_classes)}
    y_idx = np.asarray([class_to_idx[v] for v in y_series.tolist()], dtype=np.int64)

    cat_cols = list(builder.cat_cols_)
    num_cols = list(builder.num_cols_)

    n_splits = 5
    splits = make_cv_splits(y_idx, n_splits=n_splits, seed=42)

    oof_prob = np.zeros(len(y_idx), dtype=np.float64)
    fold_artifacts = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_tr = y_idx[tr_idx]
        y_va = y_idx[va_idx]

        model, enc, best_val_loss, val_prob, cfg = _train_one_fold(
            X_tr,
            y_tr,
            X_va,
            y_va,
            cat_cols=cat_cols,
            num_cols=num_cols,
            fold=fold,
            epochs=50,
            patience=10,
            seed=42,
        )

        oof_prob[va_idx] = val_prob
        fold_artifacts.append(
            {
                "state_dict": copy.deepcopy(model.state_dict()),
                "encoder": enc,
                "cardinalities": list(enc.cardinalities),
                "num_dim": int(enc.num_dim),
                "cfg": cfg,
                "best_val_loss": float(best_val_loss),
            }
        )

    oof_pred = (oof_prob >= 0.5).astype(int)
    base_acc = float((oof_pred == y_idx).mean())

    best_threshold = 0.5
    best_acc = base_acc
    for t in np.linspace(0.30, 0.70, 81):
        acc = float(((oof_prob >= t).astype(int) == y_idx).mean())
        if acc > best_acc + 1e-12:
            best_acc = acc
            best_threshold = float(t)

    print(
        f"\nOOF accuracy @0.5 = {base_acc:.6f}; "
        f"tuned threshold = {best_threshold:.4f}; "
        f"OOF accuracy @tuned = {best_acc:.6f}"
    )

    artifact = {
        "builder": builder,
        "label_classes": label_classes,
        "threshold": float(best_threshold),
        "folds": fold_artifacts,
        "oof_accuracy": float(best_acc),
        "default_threshold_accuracy": float(base_acc),
    }
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(artifact, f)

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "n_folds": len(fold_artifacts),
                "threshold": float(best_threshold),
                "oof_accuracy": float(best_acc),
                "label_classes": [str(v) for v in label_classes],
            },
            f,
            indent=2,
        )

    print(f"Final Validation Score: {best_acc}")


def predict_command(args):
    torch.set_num_threads(2)

    with open(os.path.join(args.model, "model.pkl"), "rb") as f:
        artifact = pickle.load(f)

    builder = artifact["builder"]
    label_classes = artifact["label_classes"]
    threshold = float(artifact.get("threshold", 0.5))
    folds = artifact["folds"]

    X, _ = prepare_inference(args.data, builder)
    n = len(X)

    prob_sum = np.zeros(n, dtype=np.float64)
    for fold_art in folds:
        enc = fold_art["encoder"]
        cat, num = enc.transform(X)

        model = DCNv2Gated(
            cardinalities=fold_art["cardinalities"],
            num_dim=fold_art["num_dim"],
            cfg=fold_art["cfg"],
        )
        model.load_state_dict(fold_art["state_dict"])
        model.eval()

        logits = _predict_logits(model, cat, num)
        with torch.no_grad():
            prob_sum += torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    probs = prob_sum / max(len(folds), 1)
    pred_idx = (probs >= threshold).astype(int)
    pred_labels = [label_classes[i] for i in pred_idx]

    out = pd.DataFrame(
        {"row_id": np.arange(n, dtype=np.int64), "prediction": pred_labels}
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.output, index=False)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)
    p_train.set_defaults(func=train_command)

    p_predict = subparsers.add_parser("predict")
    p_predict.add_argument("--data", required=True)
    p_predict.add_argument("--model", required=True)
    p_predict.add_argument("--output", required=True)
    p_predict.set_defaults(func=predict_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
