"""
solution.py
===========
Single-file pipeline for coupon-acceptance prediction (accuracy metric).

  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv

All learned state (feature-engineer maps, numeric normalizer, categorical
cardinalities, label map, feature order, model weights) is fit ONLY on training
rows and persisted under --model. The `predict` path loads that state in a fresh
process and never refits.
"""

from __future__ import annotations

import os
import json
import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(2)

BUNDLE_NAME = "bundle.pt"


# ===========================================================================
# PART 1 — Data processing & feature engineering
# ===========================================================================
FREQ_ORDER = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
AGE_ORDER = {
    "below21": 0,
    "21": 1,
    "26": 2,
    "31": 3,
    "36": 4,
    "41": 5,
    "46": 6,
    "50plus": 7,
}
INCOME_ORDER = {
    "less than $12500": 0,
    "$12500 - $24999": 1,
    "$25000 - $37499": 2,
    "$37500 - $49999": 3,
    "$50000 - $62499": 4,
    "$62500 - $74999": 5,
    "$75000 - $87499": 6,
    "$87500 - $99999": 7,
    "$100000 or more": 8,
}
EDU_ORDER = {
    "some high school": 0,
    "high school graduate": 1,
    "some college - no degree": 2,
    "associates degree": 3,
    "bachelors degree": 4,
    "graduate degree (masters or doctorate)": 5,
}
TIME_ORDER = {"7am": 0, "10am": 1, "2pm": 2, "6pm": 3, "10pm": 4}
TEMP_MAP = {30.0: 0.0, 55.0: 1.0, 80.0: 2.0}
GENDER_MAP = {"male": 0, "female": 1}
EXPIRATION_MAP = {"2h": 0, "1d": 1}

FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]

COMBO_COLS: List[Tuple[str, ...]] = [
    ("coupon", "occupation"),
    ("coupon", "time"),
    ("coupon", "expiration"),
    ("coupon", "maritalStatus", "has_children"),
    ("age", "maritalStatus", "has_children"),
    ("income", "occupation"),
    ("income", "education"),
]

CAT_SUFFIX = "_le"
UNK_TOKEN = -1


def _s(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _sl(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _map_ordinal(series: pd.Series, mapping: Dict[Any, float]) -> pd.Series:
    return _sl(series).map(mapping).astype(float)


def _safe_col(df: pd.DataFrame, name: str, default: Any = "") -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def _venue_col_for_coupon(c) -> Optional[str]:
    c = str(c).strip().lower()
    if "carry" in c or "take away" in c or "take-away" in c:
        return "CarryAway"
    if "coffee" in c:
        return "CoffeeHouse"
    if "<20" in c:
        return "RestaurantLessThan20"
    if "20" in c and "50" in c:
        return "Restaurant20To50"
    if "bar" in c:
        return "Bar"
    return None


class CouponFeatureEngineer:
    """Stateful, leakage-safe feature transformer (fit on train, reuse at predict)."""

    def __init__(self) -> None:
        self.label_maps: Dict[str, Dict[str, int]] = {}
        self.combo_counts: Dict[str, Dict[str, int]] = {}
        self.categorical_cols: List[str] = ["coupon", "occupation", "maritalStatus"]
        self.feature_names: List[str] = []

    @staticmethod
    def _combo_key(df: pd.DataFrame, combo: Tuple[str, ...]) -> pd.Series:
        parts = [_s(_safe_col(df, c, "")) for c in combo]
        if len(parts) == 1:
            return parts[0]
        out = parts[0]
        for p in parts[1:]:
            out = out.str.cat(p, sep="|")
        return out

    def fit(self, df: pd.DataFrame) -> "CouponFeatureEngineer":
        for col in self.categorical_cols:
            vals = _s(_safe_col(df, col, "")).unique().tolist()
            self.label_maps[col] = {v: i for i, v in enumerate(sorted(vals))}
        for combo in COMBO_COLS:
            key = "|".join(combo)
            k = self._combo_key(df, combo)
            self.combo_counts[key] = k.value_counts().to_dict()
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        temp_num = pd.to_numeric(
            _s(_safe_col(df, "temperature", np.nan)), errors="coerce"
        )
        out["temperature_num"] = temp_num
        out["temperature_ord"] = temp_num.map(TEMP_MAP)

        out["time_ord"] = _map_ordinal(_safe_col(df, "time", ""), TIME_ORDER)
        out["age_ord"] = _map_ordinal(_safe_col(df, "age", ""), AGE_ORDER)
        out["income_ord"] = _map_ordinal(_safe_col(df, "income", ""), INCOME_ORDER)
        out["education_ord"] = _map_ordinal(_safe_col(df, "education", ""), EDU_ORDER)
        out["gender_bin"] = _map_ordinal(_safe_col(df, "gender", ""), GENDER_MAP)
        out["expiration_bin"] = _map_ordinal(
            _safe_col(df, "expiration", ""), EXPIRATION_MAP
        )

        for col in FREQ_COLS:
            out[col + "_ord"] = _map_ordinal(_safe_col(df, col, ""), FREQ_ORDER)

        out["has_children"] = (
            pd.to_numeric(_s(_safe_col(df, "has_children", 0)), errors="coerce")
            .fillna(0)
            .astype(int)
        )
        for b in [
            "toCoupon_GEQ5min",
            "toCoupon_GEQ15min",
            "toCoupon_GEQ25min",
            "direction_same",
            "direction_opp",
        ]:
            out[b] = (
                pd.to_numeric(_s(_safe_col(df, b, 0)), errors="coerce")
                .fillna(0)
                .astype(int)
            )

        g15 = out["toCoupon_GEQ15min"].values
        g25 = out["toCoupon_GEQ25min"].values
        out["distance_level"] = np.where(
            g15 == 1, 0.0, np.where(g25 == 1, 1.0, 2.0)
        ).astype(float)

        venue_col = _s(_safe_col(df, "coupon", "")).map(_venue_col_for_coupon)
        venue_ord = {col: out[col + "_ord"].values for col in FREQ_COLS}
        matched = np.full(len(df), np.nan)
        for col in FREQ_COLS:
            mask = (venue_col == col).values
            if mask.any():
                matched[mask] = venue_ord[col][mask]
        out["venue_freq_ord"] = matched
        out["venue_never"] = np.where(np.isnan(matched), 0, (matched == 0).astype(int))
        out["venue_freq_missing"] = np.isnan(matched).astype(int)

        freq_ord_cols = [c + "_ord" for c in FREQ_COLS]
        acc = out[freq_ord_cols[0]].fillna(0.0)
        for c in freq_ord_cols[1:]:
            acc = acc + out[c].fillna(0.0)
        out["venue_sum_freq"] = acc
        out["venue_mean_freq"] = acc / float(len(FREQ_COLS))
        out["venue_max_freq"] = pd.concat([out[c] for c in freq_ord_cols], axis=1).max(
            axis=1
        )

        out["temp_x_time"] = out["temperature_ord"] * out["time_ord"]
        out["temp_x_distance"] = out["temperature_ord"] * out["distance_level"]
        out["age_x_children"] = out["age_ord"] * out["has_children"]
        out["income_x_distance"] = out["income_ord"] * out["distance_level"]
        out["matched_x_distance"] = (
            out["venue_freq_ord"].fillna(-1.0) * out["distance_level"]
        )

        is_married = (
            _sl(_safe_col(df, "maritalStatus", ""))
            .isin(["married", "unmarried partner"])
            .astype(int)
        )
        out["is_married"] = is_married
        out["married_with_children"] = is_married * out["has_children"]

        for col in self.categorical_cols:
            m = self.label_maps.get(col, {})
            out[col + "_le"] = _s(_safe_col(df, col, "")).map(m).fillna(-1).astype(int)

        for combo in COMBO_COLS:
            key = "|".join(combo)
            k = self._combo_key(df, combo)
            cmap = self.combo_counts.get(key, {})
            out["cnt_" + key] = k.map(cmap).fillna(0).astype(float)

        for col in out.columns:
            if out[col].dtype == object:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        self.feature_names = list(out.columns)
        return out


# ===========================================================================
# Data loaders + CV split
# ===========================================================================
def load_schema(data_dir: str) -> Dict[str, Any]:
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def load_training_data(
    data_dir: str,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    schema = load_schema(data_dir)
    label_col = "label" if "label" in labels_df.columns else labels_df.columns[0]
    y = labels_df[label_col].values
    if len(y) != len(features):
        raise ValueError(
            f"labels ({len(y)}) vs features ({len(features)}) row mismatch"
        )
    return features, y, schema


def load_predict_data(data_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    schema = load_schema(data_dir)
    return features, schema


def make_cv_folds(y: np.ndarray, n_splits: int = 5, seed: int = 42):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    dummy = np.zeros(len(y))
    for tr, va in skf.split(dummy, y):
        folds.append((tr, va))
    return folds


# ===========================================================================
# PART 2 — Model design (entity-embedding residual MLP)
# ===========================================================================
def split_feature_columns(feature_names: Sequence[str]) -> Tuple[List[str], List[str]]:
    cats = [c for c in feature_names if c.endswith(CAT_SUFFIX)]
    nums = [c for c in feature_names if not c.endswith(CAT_SUFFIX)]
    return cats, nums


class NumericNormalizer:
    """Persistable mean/std scaler fitted on TRAIN rows only."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "NumericNormalizer":
        x = np.nan_to_num(
            np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        self.std_ = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(
            np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("NumericNormalizer used before fit()")
        return np.clip((x - self.mean_) / self.std_, -8.0, 8.0).astype(np.float32)

    def state_dict(self) -> Dict[str, list]:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    def load_state_dict(self, state: Dict[str, list]) -> "NumericNormalizer":
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.std_ = np.asarray(state["std"], dtype=np.float64)
        return self


def embedding_dim(cardinality: int, max_dim: int = 16) -> int:
    return int(min(max_dim, max(2, round(1.6 * (max(2, cardinality) ** 0.56)))))


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class CouponEmbeddingNet(nn.Module):
    def __init__(
        self,
        cat_cardinalities: Sequence[int],
        n_numeric: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.25,
        emb_dropout: float = 0.05,
        emb_max_dim: int = 16,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_numeric = int(n_numeric)
        self.n_classes = int(n_classes)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(int(c), embedding_dim(int(c), emb_max_dim))
                for c in cat_cardinalities
            ]
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.num_norm = nn.BatchNorm1d(self.n_numeric) if self.n_numeric > 0 else None

        in_dim = sum(e.embedding_dim for e in self.embeddings) + self.n_numeric
        if in_dim <= 0:
            raise ValueError("CouponEmbeddingNet received zero input features")

        trunk: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            trunk += [
                nn.Linear(prev, int(h)),
                nn.BatchNorm1d(int(h)),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = int(h)
        self.trunk = nn.Sequential(*trunk) if trunk else nn.Identity()

        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(prev, dropout) for _ in range(max(1, len(hidden_dims)))]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(prev) if prev > 1 else nn.Identity(),
            nn.Linear(prev, self.n_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)

    def forward(
        self, x_cat: Optional[torch.Tensor], x_num: Optional[torch.Tensor]
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        if self.n_cat > 0:
            x_cat = x_cat.long()
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            parts.append(self.emb_dropout(torch.cat(embs, dim=1)))
        if self.num_norm is not None:
            parts.append(self.num_norm(x_num.float()))
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        x = self.trunk(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


@dataclass
class ModelConfig:
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.25
    emb_dropout: float = 0.05
    emb_max_dim: int = 16
    label_smoothing: float = 0.03
    lr: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 60
    pct_start: float = 0.25
    grad_clip: float = 1.0
    seed: int = 42
    n_classes: int = 2


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cat_cardinalities, n_numeric, cfg: ModelConfig) -> CouponEmbeddingNet:
    return CouponEmbeddingNet(
        cat_cardinalities=cat_cardinalities,
        n_numeric=n_numeric,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        emb_dropout=cfg.emb_dropout,
        emb_max_dim=cfg.emb_max_dim,
        n_classes=cfg.n_classes,
    )


def build_criterion(cfg: ModelConfig) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))


def build_optimizer(model: nn.Module, cfg: ModelConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or "embedding" in name) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": float(cfg.weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=float(cfg.lr), betas=(0.9, 0.99), eps=1e-8)


def build_scheduler(
    optimizer, cfg: ModelConfig, steps_per_epoch: int, epochs: Optional[int] = None
):
    n_epochs = int(epochs if epochs is not None else cfg.max_epochs)
    total_steps = max(1, int(steps_per_epoch) * n_epochs)
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(cfg.lr),
        total_steps=total_steps,
        pct_start=float(cfg.pct_start),
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )


def build_train_components(
    cat_cardinalities, n_numeric, steps_per_epoch, cfg=None, epochs=None
):
    cfg = cfg or ModelConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device()
    model = build_model(cat_cardinalities, n_numeric, cfg).to(device)
    criterion = build_criterion(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch, epochs)
    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
        "config": cfg,
    }


# ===========================================================================
# PART 3 — Training / evaluation / persistence / prediction
# ===========================================================================
def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shift_cat(
    mat: np.ndarray, cards: Optional[List[int]] = None
) -> Tuple[np.ndarray, List[int]]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = np.where(np.isnan(arr), -1.0, arr)
    shifted = (arr.astype(np.int64) + 1).astype(np.int64)
    if cards is None:
        cards = [
            max(2, int(shifted[:, j].max()) + 1) if shifted.shape[0] else 2
            for j in range(shifted.shape[1])
        ]
    for j, c in enumerate(cards):
        np.clip(shifted[:, j], 0, int(c) - 1, out=shifted[:, j])
    return shifted, list(cards)


def _encode_labels(y: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
    unique, y_idx = np.unique(np.asarray(y), return_inverse=True)
    return y_idx.astype(np.int64), unique.tolist()


def _prepare_fold(fe: CouponFeatureEngineer, df: pd.DataFrame):
    Xfe = fe.transform(df).reindex(columns=fe.feature_names, fill_value=0.0)
    cat_cols, num_cols = split_feature_columns(fe.feature_names)
    cat = (
        np.nan_to_num(Xfe[cat_cols].values, nan=-1.0)
        if cat_cols
        else np.zeros((len(Xfe), 0))
    )
    num = (
        Xfe[num_cols].values.astype(np.float32)
        if num_cols
        else np.zeros((len(Xfe), 0), dtype=np.float32)
    )
    return cat, num, cat_cols, num_cols


def _apply_norm(num: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if num.shape[1] == 0:
        return num.astype(np.float32)
    x = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


def _fit_es(
    cat_tr, num_tr, y_tr, cards, cfg, epochs, seed, cat_va, num_va, y_va
) -> Tuple[CouponEmbeddingNet, int, float]:
    _set_seed(seed)
    device = get_device()
    n = len(y_tr)
    steps_per_epoch = max(1, int(np.ceil(n / cfg.batch_size)))
    parts = build_train_components(
        cards, num_tr.shape[1], steps_per_epoch, cfg=cfg, epochs=epochs
    )
    model, criterion = parts["model"], parts["criterion"]
    optimizer, scheduler = parts["optimizer"], parts["scheduler"]

    xc = torch.from_numpy(cat_tr).long()
    xn = torch.from_numpy(num_tr.astype(np.float32))
    yt = torch.from_numpy(y_tr).long()
    loader = DataLoader(
        TensorDataset(xc, xn, yt), batch_size=cfg.batch_size, shuffle=True
    )

    xv = torch.from_numpy(cat_va).long().to(device)
    nv = torch.from_numpy(num_va.astype(np.float32)).to(device)

    best_acc, best_ep, best_state = -1.0, 0, None
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for xb, nb, yb in loader:
            xb, nb, yb = xb.to(device), nb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, nb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            tl += float(loss.item()) * len(yb)
        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(xv, nv), dim=1).cpu().numpy()
        acc = float((pred == y_va).mean())
        if acc > best_acc:
            best_acc, best_ep, best_state = (
                acc,
                ep + 1,
                copy.deepcopy(model.state_dict()),
            )
        print(
            f"    epoch {ep+1}/{epochs} loss={tl/n:.4f} val_acc={acc:.4f} best={best_acc:.4f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_ep, best_acc


def _fit_full(
    cat_full, num_full, y_idx, cards, cfg, epochs, seed
) -> CouponEmbeddingNet:
    _set_seed(seed)
    device = get_device()
    n = len(y_idx)
    steps_per_epoch = max(1, int(np.ceil(n / cfg.batch_size)))
    parts = build_train_components(
        cards, num_full.shape[1], steps_per_epoch, cfg=cfg, epochs=epochs
    )
    model, criterion = parts["model"], parts["criterion"]
    optimizer, scheduler = parts["optimizer"], parts["scheduler"]

    xc = torch.from_numpy(cat_full).long()
    xn = torch.from_numpy(num_full.astype(np.float32))
    yt = torch.from_numpy(y_idx).long()
    loader = DataLoader(
        TensorDataset(xc, xn, yt), batch_size=cfg.batch_size, shuffle=True
    )

    for ep in range(epochs):
        model.train()
        tl = 0.0
        for xb, nb, yb in loader:
            xb, nb, yb = xb.to(device), nb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, nb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            tl += float(loss.item()) * len(yb)
        print(f"    [seed {seed}] epoch {ep+1}/{epochs} loss={tl/n:.4f}")
    model.eval()
    return model


def _save_bundle(
    model_dir, fe, normalizer, cat_cols, num_cols, cards, label_map, cfg, models, epochs
) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, BUNDLE_NAME)
    bundle = {
        "fe": fe,
        "feature_names": list(fe.feature_names),
        "normalizer_state": normalizer.state_dict(),
        "cat_cols": list(cat_cols),
        "num_cols": list(num_cols),
        "cards": list(cards),
        "label_map": list(label_map),
        "cfg": dict(cfg.__dict__),
        "state_dicts": [m.state_dict() for m in models],
        "epochs": int(epochs),
    }
    torch.save(bundle, path)
    return path


def _load_bundle(model_dir):
    path = os.path.join(model_dir, BUNDLE_NAME)
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")

    fe: CouponFeatureEngineer = bundle["fe"]
    normalizer = NumericNormalizer().load_state_dict(bundle["normalizer_state"])
    cfg_fields = ModelConfig.__dataclass_fields__
    cfg = ModelConfig(**{k: v for k, v in bundle["cfg"].items() if k in cfg_fields})

    models: List[CouponEmbeddingNet] = []
    for sd in bundle["state_dicts"]:
        m = CouponEmbeddingNet(
            cat_cardinalities=bundle["cards"],
            n_numeric=len(bundle["num_cols"]),
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            emb_dropout=cfg.emb_dropout,
            emb_max_dim=cfg.emb_max_dim,
            n_classes=cfg.n_classes,
        )
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return fe, normalizer, bundle, cfg, models


def _ensemble_proba(models, cat_shifted, num_mat, batch_size=512) -> np.ndarray:
    device = get_device()
    for m in models:
        m.to(device).eval()
    proba = np.zeros((len(cat_shifted), 2), dtype=np.float64)
    for start in range(0, len(cat_shifted), batch_size):
        end = min(start + batch_size, len(cat_shifted))
        xc = torch.from_numpy(cat_shifted[start:end]).long().to(device)
        xn = torch.from_numpy(num_mat[start:end].astype(np.float32)).to(device)
        with torch.no_grad():
            for m in models:
                proba[start:end] += torch.softmax(m(xc, xn), dim=1).cpu().numpy()
    proba /= max(1, len(models))
    return proba


def run_training_evaluation(
    data_dir,
    model_dir,
    n_splits=5,
    seed=42,
    cv_epochs=20,
    prod_epochs=20,
    n_prod_seeds=3,
) -> float:
    features, y, _schema = load_training_data(data_dir)
    y_idx, label_map = _encode_labels(y)
    print(f"[data] train rows={len(features)} classes={len(label_map)} {label_map}")

    cfg = ModelConfig()
    folds = make_cv_folds(y_idx, n_splits=n_splits, seed=seed)

    fold_accs: List[float] = []
    best_epochs: List[int] = []
    oof_pred = np.full(len(y_idx), -1, dtype=np.int64)

    for fi, (tr, va) in enumerate(folds):
        df_tr = features.iloc[tr].reset_index(drop=True)
        df_va = features.iloc[va].reset_index(drop=True)

        fe = CouponFeatureEngineer().fit(df_tr)
        cat_tr, num_tr, cat_cols, num_cols = _prepare_fold(fe, df_tr)
        norm_mean = (
            np.nan_to_num(num_tr, nan=0.0).mean(axis=0) if num_cols else np.zeros(0)
        )
        norm_std = (
            np.nan_to_num(num_tr, nan=0.0).std(axis=0) if num_cols else np.zeros(0)
        )
        norm_std = np.where(norm_std < 1e-6, 1.0, norm_std)
        num_tr_n = _apply_norm(num_tr, norm_mean, norm_std)

        cat_va, num_va, _, _ = _prepare_fold(fe, df_va)
        num_va_n = _apply_norm(num_va, norm_mean, norm_std)

        cat_tr_s, cards = _shift_cat(cat_tr, None)
        cat_va_s, _ = _shift_cat(cat_va, cards)

        print(
            f"[CV fold {fi+1}/{n_splits}] train={len(df_tr)} val={len(df_va)} cards={cards}"
        )
        model, best_ep, val_acc = _fit_es(
            cat_tr_s,
            num_tr_n,
            y_idx[tr],
            cards,
            cfg,
            epochs=cv_epochs,
            seed=seed + fi,
            cat_va=cat_va_s,
            num_va=num_va_n,
            y_va=y_idx[va],
        )
        with torch.no_grad():
            device = get_device()
            xv = torch.from_numpy(cat_va_s).long().to(device)
            nv = torch.from_numpy(num_va_n.astype(np.float32)).to(device)
            oof_pred[va] = torch.argmax(model(xv, nv), dim=1).cpu().numpy()

        fold_accs.append(val_acc)
        best_epochs.append(best_ep)
        print(f"[CV fold {fi+1}/{n_splits}] acc={val_acc:.4f} best_epoch={best_ep}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv_acc = float(np.mean(fold_accs))
    cv_std = float(np.std(fold_accs))
    oof_acc = float((oof_pred == y_idx).mean())
    print(f"[CV] fold accs={[round(a, 4) for a in fold_accs]}")
    print(f"[CV] mean={cv_acc:.4f} std={cv_std:.4f} | OOF acc={oof_acc:.4f}")

    # Production: refit featurizer + normalizer on FULL training data.
    fe_full = CouponFeatureEngineer().fit(features)
    cat_full, num_full, cat_cols, num_cols = _prepare_fold(fe_full, features)
    norm_mean = (
        np.nan_to_num(num_full, nan=0.0).mean(axis=0) if num_cols else np.zeros(0)
    )
    norm_std = np.nan_to_num(num_full, nan=0.0).std(axis=0) if num_cols else np.zeros(0)
    norm_std = np.where(norm_std < 1e-6, 1.0, norm_std)
    num_full_n = _apply_norm(num_full, norm_mean, norm_std)
    cat_full_s, cards_full = _shift_cat(cat_full, None)

    normalizer = NumericNormalizer()
    normalizer.mean_ = norm_mean
    normalizer.std_ = norm_std

    prod_epochs_eff = int(
        max(prod_epochs, int(np.median(best_epochs)) if best_epochs else prod_epochs)
    )
    print(f"[prod] cards={cards_full} epochs={prod_epochs_eff} seeds={n_prod_seeds}")

    models: List[CouponEmbeddingNet] = []
    for k in range(n_prod_seeds):
        m = _fit_full(
            cat_full_s,
            num_full_n,
            y_idx,
            cards_full,
            cfg,
            epochs=prod_epochs_eff,
            seed=seed + 100 + k * 7,
        )
        models.append(m)

    path = _save_bundle(
        model_dir,
        fe_full,
        normalizer,
        cat_cols,
        num_cols,
        cards_full,
        label_map,
        cfg,
        models,
        prod_epochs_eff,
    )
    print(f"[save] bundle -> {path} ({len(models)} models)")
    return cv_acc


def run_predict(data_dir, model_dir, output_path) -> None:
    fe, normalizer, bundle, cfg, models = _load_bundle(model_dir)
    features, _schema = load_predict_data(data_dir)

    Xfe = fe.transform(features).reindex(
        columns=bundle["feature_names"], fill_value=0.0
    )
    cat_cols = bundle["cat_cols"]
    num_cols = bundle["num_cols"]

    cat = (
        np.nan_to_num(Xfe[cat_cols].values, nan=-1.0)
        if cat_cols
        else np.zeros((len(Xfe), 0))
    )
    num = (
        Xfe[num_cols].values.astype(np.float32)
        if num_cols
        else np.zeros((len(Xfe), 0), dtype=np.float32)
    )
    if num_cols:
        num = normalizer.transform(num)

    cat_shifted, _ = _shift_cat(cat, bundle["cards"])
    proba = _ensemble_proba(models, cat_shifted, num, batch_size=512)
    idx_pred = np.argmax(proba, axis=1)
    label_map = bundle["label_map"]
    preds = [label_map[int(i)] for i in idx_pred]

    out = pd.DataFrame(
        {
            "row_id": np.arange(len(features), dtype=np.int64),
            "prediction": preds,
        }
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[predict] wrote {len(out)} rows -> {output_path}")


# ===========================================================================
# CLI
# ===========================================================================
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Coupon acceptance model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Fit and persist full pipeline")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict", help="Load pipeline and produce predictions")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.cmd == "train":
        score = run_training_evaluation(args.data, args.model)
        print(f"Final Validation Score: {score}")
    elif args.cmd == "predict":
        run_predict(args.data, args.model, args.output)


if __name__ == "__main__":
    _main()
