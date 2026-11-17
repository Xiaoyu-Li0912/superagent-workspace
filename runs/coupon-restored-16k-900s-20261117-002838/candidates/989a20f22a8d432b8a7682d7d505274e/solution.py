"""
solution.py
===========
Coupon-acceptance pipeline: leakage-safe categorical feature engineering +
FT-Transformer (per-feature tokenizer + self-attention) with 5->3 fold bagging,
OOF threshold tuning for accuracy, artifact persistence, and fresh-process
inference.

CLI:
    python solution.py train   --data /data --model /model
    python solution.py predict --data /data --model /model --output /output/predictions.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

torch.set_num_threads(2)


# =========================================================================== #
# STEP 1 : DATA PROCESSING & FEATURE ENGINEERING
# =========================================================================== #

AGE_ORDER = ["below21", "21", "26", "31", "36", "41", "50plus"]

INCOME_ORDER = [
    "less than $12500",
    "$12500 - $24999",
    "$25000 - $37499",
    "$37500 - $49999",
    "$50000 - $62499",
    "$62500 - $74999",
    "$75000 - $87499",
    "$87500 - $99999",
    "$100000 or more",
]

EDUCATION_ORDER = [
    "some high school",
    "high school graduate",
    "some college - no degree",
    "associates degree",
    "bachelors degree",
    "graduate degree (masters, doctorate)",
]

FREQ_ORDER = ["never", "less1", "1~3", "4~8", "gt8"]

TIME_HOUR = {"7am": 7, "10am": 10, "2pm": 14, "6pm": 18, "10pm": 22}

EXPIRATION_ORDER = ["2h", "1d"]

FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]

BINARY_COLS = [
    "has_children",
    "toCoupon_GEQ5min",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
    "direction_opp",
]

NOMINAL_BASE = ["coupon", "gender", "maritalStatus", "occupation"]

_NA_TOKENS = {"", "nan", "none", "null", "missing", "?"}


def _as_string(series):
    s = series.astype("string").str.strip()
    s = s.mask(s.isna() | s.str.lower().isin(_NA_TOKENS))
    return s


def _map_ordered(series, order):
    lookup = {str(v).strip().lower(): i for i, v in enumerate(order)}
    mapped = series.map(
        lambda x: lookup.get(str(x).strip().lower(), np.nan) if pd.notna(x) else np.nan
    )
    return pd.to_numeric(mapped, errors="coerce").astype("float64")


class FeatureBuilder:
    """Stateful, leakage-safe feature pipeline for the coupon task."""

    def __init__(self, max_onehot_cardinality: int = 200):
        self.max_onehot_cardinality = max_onehot_cardinality
        self.nominal_cols_ = list(NOMINAL_BASE) + ["coupon_time", "coupon_exp"]
        self.num_cols_ = None
        self.num_fill_ = None
        self.count_maps_ = {}
        self.encoder_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        out = pd.DataFrame(index=idx)

        def s(col):
            if col in df.columns:
                return _as_string(df[col])
            return pd.Series(pd.NA, index=idx, dtype="string")

        def num(col):
            return pd.to_numeric(s(col), errors="coerce").astype("float64")

        out["temperature"] = num("temperature")
        out["time_hour"] = s("time").map(
            lambda x: (
                TIME_HOUR.get(str(x).strip().lower(), np.nan) if pd.notna(x) else np.nan
            )
        )
        out["expiration_num"] = _map_ordered(s("expiration"), EXPIRATION_ORDER)

        out["age"] = _map_ordered(s("age"), AGE_ORDER)
        bare_age = pd.to_numeric(s("age"), errors="coerce")
        out["age"] = out["age"].fillna(bare_age)
        out["income"] = _map_ordered(s("income"), INCOME_ORDER)
        out["education"] = _map_ordered(s("education"), EDUCATION_ORDER)

        for c in FREQ_COLS:
            out[c] = _map_ordered(s(c), FREQ_ORDER).fillna(-1.0)

        for c in BINARY_COLS:
            out[c] = num(c).fillna(0.0).clip(lower=0, upper=1)

        for c in ["coupon", "gender", "maritalStatus", "occupation"]:
            out[c] = s(c)

        out["travel_dist"] = (
            5.0 + 10.0 * out["toCoupon_GEQ15min"] + 10.0 * out["toCoupon_GEQ25min"]
        )
        out["is_short_expiration"] = (out["expiration_num"] < 0.5).astype("float64")
        out["is_evening"] = (out["time_hour"] >= 17.0).astype("float64")
        out["is_late_night"] = (out["time_hour"] >= 21.0).astype("float64")

        freq_block = out[FREQ_COLS]
        out["freq_sum"] = freq_block.clip(lower=0).sum(axis=1)
        out["freq_mean"] = freq_block.clip(lower=0).mean(axis=1)
        out["freq_never_count"] = (freq_block == 0.0).sum(axis=1).astype("float64")
        out["freq_missing_count"] = (freq_block == -1.0).sum(axis=1).astype("float64")
        out["freq_max"] = freq_block.max(axis=1)

        out["age_income"] = out["age"] * out["income"]
        out["child_x_married"] = out["has_children"] * out["maritalStatus"].map(
            lambda x: 1.0 if str(x).strip().lower() in ("married", "divorced") else 0.0
        )
        out["direction_same_only"] = out["direction_same"] * (
            1.0 - out["direction_opp"]
        )

        out["coupon_time"] = (
            out["coupon"].fillna("MISSING") + "|" + s("time").fillna("MISSING")
        )
        out["coupon_exp"] = (
            out["coupon"].fillna("MISSING") + "|" + s("expiration").fillna("MISSING")
        )
        out["coupon_time"] = out["coupon_time"].astype("string")
        out["coupon_exp"] = out["coupon_exp"].astype("string")
        return out

    def _add_count_features(self, clean: pd.DataFrame) -> pd.DataFrame:
        out = clean.copy()
        counts = pd.Series(0.0, index=out.index)
        for c in ["occupation", "coupon_time"]:
            mp = self.count_maps_.get(c, {})
            encoded = (
                out[c]
                .map(lambda x: mp.get(str(x), 1.0) if pd.notna(x) else 1.0)
                .astype("float64")
            )
            counts = counts + encoded
        out["cat_count_mean"] = counts / 2.0
        return out

    def fit(self, df: pd.DataFrame, y=None):
        clean = self._clean(df)
        self.count_maps_ = {
            c: clean[c].value_counts().to_dict() for c in ["occupation", "coupon_time"]
        }
        clean = self._add_count_features(clean)

        self.num_cols_ = [c for c in clean.columns if c not in self.nominal_cols_]
        num = clean[self.num_cols_].replace([np.inf, -np.inf], np.nan)
        self.num_fill_ = num.median(numeric_only=True).fillna(0.0)

        nom = clean[self.nominal_cols_].astype("string").fillna("MISSING")
        try:
            self.encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float32
            )
        except TypeError:
            self.encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse=False, dtype=np.float32
            )
        self.encoder_.fit(nom)

        oh_names = list(self.encoder_.get_feature_names_out(self.nominal_cols_))
        self.feature_names_ = list(self.num_cols_) + oh_names
        self.is_fitted_ = True
        return self

    def _prepared(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FeatureBuilder must be fitted before transform().")
        clean = self._clean(df)
        clean = self._add_count_features(clean)
        return clean

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        clean = self._prepared(df)
        num = clean[self.num_cols_].replace([np.inf, -np.inf], np.nan)
        num = num.fillna(self.num_fill_).fillna(0.0).to_numpy(dtype=np.float32)
        nom = clean[self.nominal_cols_].astype("string").fillna("MISSING")
        oh = self.encoder_.transform(nom).astype(np.float32)
        return np.hstack([num, oh]).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(df, y).transform(df)

    def get_feature_names(self):
        return list(self.feature_names_) if self.feature_names_ else []


def load_schema(data_dir: str) -> dict:
    path = os.path.join(data_dir, "schema.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_data(data_dir: str):
    schema = load_schema(data_dir)

    feat_path = os.path.join(data_dir, "features.csv")
    features = pd.read_csv(feat_path)
    features.reset_index(drop=True, inplace=True)

    labels = None
    lab_path = os.path.join(data_dir, "labels.csv")
    if os.path.exists(lab_path):
        lab = pd.read_csv(lab_path)
        label_col = "label" if "label" in lab.columns else lab.columns[-1]
        if "row_id" in lab.columns and "row_id" in features.columns:
            lookup = lab.drop_duplicates("row_id").set_index("row_id")[label_col]
            labels = features["row_id"].map(lookup)
        else:
            labels = lab[label_col]
        labels = pd.Series(labels).reset_index(drop=True)
        if len(labels) != len(features):
            raise ValueError(
                f"Label/feature row mismatch: {len(labels)} vs {len(features)}"
            )

    declared = schema.get("feature_columns")
    if declared:
        present = [c for c in declared if c in features.columns]
        rest = [c for c in features.columns if c not in present and c != "row_id"]
        if present:
            features = features[present + rest]

    return features, labels, schema


def make_cv_splits(y, n_splits: int = 5, seed: int = 42):
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.ravel()
    counts = np.bincount(y.astype(int))
    counts = counts[counts > 0]
    if counts.size < 2:
        idx = np.arange(len(y))
        return [(idx, idx)]
    n_splits = int(max(2, min(n_splits, counts.min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y.astype(int)))


# =========================================================================== #
# STEP 2 : MODEL DESIGN (FT-Transformer)
# =========================================================================== #


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int = 64, bias: bool = True):
        super().__init__()
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = int(n_features)
        self.d_token = int(d_token)

        self.weight = nn.Parameter(torch.empty(self.n_features, self.d_token))
        self.bias = (
            nn.Parameter(torch.zeros(self.n_features, self.d_token)) if bias else None
        )
        self.cls = nn.Parameter(torch.zeros(1, self.d_token))

        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.n_features:
            raise ValueError(
                f"expected (B, {self.n_features}) input, got {tuple(x.shape)}"
            )
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0)
        if self.bias is not None:
            tokens = tokens + self.bias.unsqueeze(0)
        cls = self.cls.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.2,
        ff_mult: float = 2.0,
        d_out: int = 1,
    ):
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("d_token must be divisible by n_heads")

        self.tokenizer = FeatureTokenizer(n_features, d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=max(d_token, int(d_token * ff_mult)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, d_out),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0]
        logits = self.head(cls_out).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


class LabelSmoothBCE(nn.Module):
    def __init__(self, smoothing: float = 0.02, pos_weight=None):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = float(smoothing)
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.tensor([float(pos_weight)]),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if self.smoothing > 0.0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        pw = None if self.pos_weight is None else self.pos_weight
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


DEFAULT_CONFIG = {
    "d_token": 64,
    "n_layers": 2,
    "n_heads": 4,
    "dropout": 0.2,
    "ff_mult": 2.0,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.02,
    "pos_weight": None,
    "warmup_frac": 0.1,
    "grad_clip": 1.0,
}


def build_model(n_features: int, config=None, device: str = "cpu"):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    model = FTTransformer(
        n_features=n_features,
        d_token=cfg["d_token"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"],
        ff_mult=cfg["ff_mult"],
    ).to(device)

    criterion = LabelSmoothBCE(
        smoothing=cfg["label_smoothing"], pos_weight=cfg["pos_weight"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.98),
    )
    return model, criterion, optimizer


def make_scheduler(optimizer, total_steps: int, warmup_frac: float = 0.1):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# =========================================================================== #
# STEP 3 : TRAINING / EVALUATION / PERSISTENCE / INFERENCE
# =========================================================================== #


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_one_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    n_features: int,
    *,
    seed: int,
    max_epochs: int = 20,
    patience: int = 4,
    batch_size: int = 256,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.02,
    device: str = "cpu",
    deadline: float | None = None,
):
    _set_seed(seed)

    model, criterion, optimizer = build_model(
        n_features=n_features,
        config={
            "lr": lr,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
        },
        device=device,
    )

    n_train = len(X_tr)
    steps_per_epoch = max(1, n_train // batch_size)
    total_steps = steps_per_epoch * max_epochs
    scheduler = make_scheduler(optimizer, total_steps, warmup_frac=0.1)

    X_tr_t = torch.from_numpy(X_tr.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
    X_va_t = torch.from_numpy(X_va.astype(np.float32))
    y_va_int = y_va.astype(np.int64)

    g = torch.Generator().manual_seed(seed)
    best_acc = -1.0
    best_state = None
    bad = 0

    for epoch in range(max_epochs):
        if deadline is not None and time.time() > deadline:
            print(f"  [fold] time budget reached at epoch {epoch}; stopping")
            break
        model.train()
        perm = torch.randperm(n_train, generator=g)
        running = 0.0
        nb = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach().item())
            nb += 1

        model.eval()
        with torch.no_grad():
            va_probs = torch.sigmoid(model(X_va_t)).cpu().numpy()
        va_pred = (va_probs > 0.5).astype(np.int64)
        acc = float((va_pred == y_va_int).mean())

        if acc > best_acc:
            best_acc = acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

        print(
            f"  epoch {epoch+1:02d} loss={running/max(1,nb):.4f} "
            f"val_acc={acc:.4f} (best={max(best_acc,acc):.4f})"
        )
        if bad >= patience:
            break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_probs = torch.sigmoid(model(X_va_t)).cpu().numpy()
    best_pred = (best_probs > 0.5).astype(np.int64)
    best_acc = float((best_pred == y_va_int).mean())
    return best_state, best_probs, best_acc


def train(data_dir: str, model_dir: str) -> float:
    os.makedirs(model_dir, exist_ok=True)
    _set_seed(42)
    t0 = time.time()

    features, labels, _schema = load_data(data_dir)
    if labels is None:
        raise RuntimeError("train() requires labels.csv alongside features.csv")
    features = features.reset_index(drop=True)

    le = LabelEncoder()
    y_int = le.fit_transform(np.asarray(labels))
    n_rows = len(features)
    print(
        f"Training FT-Transformer | rows={n_rows} "
        f"classes={list(le.classes_)} pos_rate={float(np.mean(y_int)):.3f}"
    )

    splits = make_cv_splits(y_int, n_splits=3, seed=42)
    oof_probs = np.zeros(n_rows, dtype=np.float64)
    covered = np.zeros(n_rows, dtype=bool)
    fold_artifacts: list[dict] = []
    fold_accs: list[float] = []

    TIME_BUDGET = 240.0

    for fidx, (tr_idx, va_idx) in enumerate(splits):
        if time.time() - t0 > TIME_BUDGET:
            print(f"[stop] time budget reached before fold {fidx}")
            break

        tr_df = features.iloc[tr_idx].reset_index(drop=True)
        va_df = features.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y_int[tr_idx], y_int[va_idx]

        builder = FeatureBuilder()
        X_tr = builder.fit_transform(tr_df, y_tr)
        X_va = builder.transform(va_df)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)
        n_feat = X_tr_s.shape[1]

        print(
            f"[fold {fidx}] n_train={len(X_tr_s)} n_val={len(X_va_s)} n_feat={n_feat}"
        )

        best_state, probs, acc = _train_one_fold(
            X_tr_s,
            y_tr,
            X_va_s,
            y_va,
            n_feat,
            seed=42 + fidx,
            max_epochs=18,
            deadline=t0 + TIME_BUDGET,
        )
        oof_probs[va_idx] = probs
        covered[va_idx] = True
        fold_accs.append(acc)
        fold_artifacts.append(
            {
                "builder": builder,
                "scaler": scaler,
                "state_dict": best_state,
                "n_features": int(n_feat),
            }
        )
        print(f"[fold {fidx}] held-out accuracy = {acc:.4f}")

    if not fold_artifacts:
        raise RuntimeError("All folds failed; no model produced.")

    if covered.any():
        mask = covered
        _p = oof_probs[mask]
        _y = y_int[mask]
        best_thr, best_thr_acc = 0.5, -1.0
        for thr in np.linspace(0.30, 0.70, 81):
            a = float(((_p > thr).astype(np.int64) == _y).mean())
            if a > best_thr_acc:
                best_thr_acc, best_thr = a, float(thr)
        print(
            f"OOF accuracy = {best_thr_acc:.4f} at threshold = {best_thr:.3f} "
            f"(fold accs = {[round(a,4) for a in fold_accs]})"
        )
    else:
        best_thr, best_thr_acc = 0.5, float(np.mean(fold_accs))

    artifact_path = os.path.join(model_dir, "artifacts.pt")
    bundle = {
        "fold_artifacts": fold_artifacts,
        "threshold": float(best_thr),
        "classes": le.classes_,
        "n_features": int(fold_artifacts[0]["n_features"]),
        "oof_acc": float(best_thr_acc),
    }
    torch.save(bundle, artifact_path)

    with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_folds": len(fold_artifacts),
                "threshold": float(best_thr),
                "oof_acc": float(best_thr_acc),
                "fold_accs": [float(a) for a in fold_accs],
                "classes": [str(c) for c in le.classes_.tolist()],
            },
            f,
            indent=2,
        )

    print(f"Saved artifact bundle to {artifact_path} (elapsed={time.time()-t0:.1f}s)")
    return float(best_thr_acc)


def _safe_torch_load(path: str, device: str = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict(data_dir: str, model_dir: str, output_path: str) -> None:
    _set_seed(42)
    features, _labels, _schema = load_data(data_dir)
    features = features.reset_index(drop=True)
    n = len(features)

    bundle = _safe_torch_load(os.path.join(model_dir, "artifacts.pt"))
    fold_artifacts = bundle["fold_artifacts"]
    threshold = float(bundle["threshold"])
    classes = np.asarray(bundle["classes"])
    print(f"Loaded {len(fold_artifacts)} fold models, threshold={threshold:.3f}")

    probs_sum = np.zeros(n, dtype=np.float64)
    for fa in fold_artifacts:
        builder = fa["builder"]
        scaler = fa["scaler"]
        n_feat = int(fa["n_features"])

        X = builder.transform(features).astype(np.float32)
        X = scaler.transform(X).astype(np.float32)

        model, _crit, _opt = build_model(n_feat, config={"lr": 1e-3}, device="cpu")
        model.load_state_dict(fa["state_dict"])
        model.eval()

        with torch.no_grad():
            chunks = []
            for i in range(0, n, 512):
                xb = torch.from_numpy(X[i : i + 512].astype(np.float32))
                chunks.append(torch.sigmoid(model(xb)).cpu().numpy())
            probs = np.concatenate(chunks) if chunks else np.zeros(0)
        probs_sum += probs

    probs_mean = probs_sum / max(1, len(fold_artifacts))
    pred_idx = (probs_mean > threshold).astype(np.int64)
    pred_labels = classes[pred_idx]

    out = pd.DataFrame(
        {"row_id": np.arange(n, dtype=np.int64), "prediction": pred_labels}
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {n} predictions -> {output_path}")


# =========================================================================== #
# CLI
# =========================================================================== #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Coupon acceptance FT-Transformer pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train the model and persist artifacts")
    pt.add_argument("--data", required=True)
    pt.add_argument("--model", required=True)

    pp = sub.add_parser("predict", help="Load artifacts and predict on unlabeled data")
    pp.add_argument("--data", required=True)
    pp.add_argument("--model", required=True)
    pp.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.cmd == "train":
        score = train(args.data, args.model)
        print(f"Final Validation Score: {score}")
    elif args.cmd == "predict":
        predict(args.data, args.model, args.output)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.cmd}")
