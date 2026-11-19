```python
# Coupon acceptance prediction model.
# - Leakage-safe coupon feature engineering (fit on TRAIN rows only).
# - FT-Transformer-lite with a coupon-conditioned Mixture-of-Experts head.
# - Stratified train-only CV, multi-seed refit, fresh-process predict.
# - CLI: `train --data --model`  and  `predict --data --model --output`.
# Keep this header as plain ASCII comments (no module docstring) so that
# `from __future__ import annotations` is the first executable statement.

from __future__ import annotations

import os
import sys
import json
import math
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Domain dictionaries (semantic ordinal scales)
# ---------------------------------------------------------------------------
FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]
FREQ_MAP = {"never": 0.0, "less1": 0.5, "1~3": 2.0, "4~8": 6.0, "gt8": 9.0}

AGE_MAP = {
    "below21": 19.0,
    "21": 21.0,
    "26": 26.0,
    "31": 31.0,
    "36": 36.0,
    "41": 41.0,
    "46": 46.0,
    "50plus": 52.0,
}

TIME_MAP = {"7AM": 7.0, "10AM": 10.0, "2PM": 14.0, "6PM": 18.0, "10PM": 22.0}

EXPIRATION_MAP = {"2h": 2.0, "1d": 24.0}

EDUCATION_MAP = {
    "some high school": 0.0,
    "high school graduate": 1.0,
    "some college - no degree": 2.0,
    "associates degree": 3.0,
    "bachelors degree": 4.0,
    "graduate degree (masters, doctorate)": 5.0,
}

INCOME_MAP = {
    "less than $12500": 6250.0,
    "$12500 - $24999": 18750.0,
    "$25000 - $37499": 31250.0,
    "$37500 - $49999": 43750.0,
    "$50000 - $62499": 56250.0,
    "$62500 - $74999": 68750.0,
    "$75000 - $87499": 81250.0,
    "$87500 - $99999": 93750.0,
    "$100000 or more": 110000.0,
}

COUPON_TO_FREQCOL = {
    "bar": "Bar",
    "coffee house": "CoffeeHouse",
    "carry out & take away": "CarryAway",
    "restaurant(<20)": "RestaurantLessThan20",
    "restaurant(20-50)": "Restaurant20To50",
}

RAW_CAT_COLS = ["gender", "maritalStatus", "has_children", "occupation", "coupon"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_data(data_dir, with_labels=True):
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    schema = None
    schema_path = os.path.join(data_dir, "schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema = json.load(f)
    labels = None
    if with_labels:
        labels_path = os.path.join(data_dir, "labels.csv")
        if os.path.exists(labels_path):
            labels = pd.read_csv(labels_path)
    return features, labels, schema


def get_feature_columns(df, schema):
    if schema and isinstance(schema.get("feature_columns"), list):
        cols = [c for c in schema["feature_columns"] if c in df.columns]
        if cols:
            return cols
    return [c for c in df.columns if c not in ("label", "target", "row_id")]


def get_labels_array(labels):
    if labels is None:
        return None
    col = "label" if "label" in labels.columns else labels.columns[0]
    return labels[col].values


def get_stratified_folds(y, n_splits=5, seed=42):
    """Local stratified k-fold (no sklearn dependency)."""
    y = np.asarray(y)
    n = len(y)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % n_splits].append(int(i))
    out = []
    all_idx = set(range(n))
    for k in range(n_splits):
        va = np.array(sorted(folds[k]), dtype=int)
        tr = np.array(sorted(all_idx - set(folds[k].tolist())), dtype=int)
        out.append((tr, va))
    return out


# ---------------------------------------------------------------------------
# Robust scalar parsers
# ---------------------------------------------------------------------------
def _norm(s):
    return str(s).strip().lower() if s is not None else ""


def _parse_age(v):
    return AGE_MAP.get(_norm(v), 36.0)


def _parse_time(v):
    return TIME_MAP.get(_norm(v), np.nan)


def _parse_expiration(v):
    return EXPIRATION_MAP.get(_norm(v), np.nan)


def _parse_education(v):
    return EDUCATION_MAP.get(_norm(v), np.nan)


def _parse_income(v):
    key = _norm(v)
    if key in INCOME_MAP:
        return INCOME_MAP[key]
    import re

    nums = re.findall(r"\d+", key.replace(",", ""))
    if nums:
        vals = [float(x) for x in nums]
        return float(np.mean(vals))
    return np.nan


def _parse_freq(v):
    return FREQ_MAP.get(_norm(v), np.nan)


def _parse_numeric(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Feature engineering pipeline (stateful, leakage-safe)
# ---------------------------------------------------------------------------
class CouponFeatureEngineer:
    def __init__(self, smoothing=10.0, te_cols=None):
        self.smoothing = float(smoothing)
        self.te_cols = (
            te_cols
            if te_cols is not None
            else ["occupation", "coupon", "income_raw", "education_raw"]
        )
        self.cat_columns_ = []
        self.num_columns_ = []
        self._medians_ = {}
        self._te_maps_ = {}
        self._te_prior_ = 0.5
        self._has_te_ = False

    @staticmethod
    def _build_core(df):
        d = pd.DataFrame(index=df.index)

        def _col(name, default):
            if name in df.columns:
                return df[name]
            return pd.Series([default] * len(df), index=df.index, dtype=object)

        d["age_years"] = _col("age", np.nan).map(_parse_age)
        d["time_hour"] = _col("time", np.nan).map(_parse_time)
        d["expiration_hours"] = _col("expiration", np.nan).map(_parse_expiration)
        d["temperature_num"] = _col("temperature", np.nan).map(_parse_numeric)
        d["education_level"] = _col("education", np.nan).map(_parse_education)
        d["income_mid"] = _col("income", np.nan).map(_parse_income)

        for c in FREQ_COLS:
            if c in df.columns:
                d[f"freq_{c}"] = df[c].map(_parse_freq)
            else:
                d[f"freq_{c}"] = np.nan

        coupon_series = (
            df["coupon"]
            if "coupon" in df.columns
            else pd.Series([""] * len(df), index=df.index, dtype=object)
        )
        coupon_norm = coupon_series.astype(str).map(_norm)
        freq_matrix = d[[f"freq_{c}" for c in FREQ_COLS]]
        match_col = coupon_norm.map(lambda v: COUPON_TO_FREQCOL.get(v, None))

        affinity = pd.Series(np.nan, index=df.index, dtype=float)
        for col in FREQ_COLS:
            sel = match_col == col
            if sel.any():
                affinity[sel] = d.loc[sel, f"freq_{col}"]
        d["venue_affinity"] = affinity
        d["venue_match"] = (d["venue_affinity"].fillna(0) > 0).astype(float)

        d["total_venue_activity"] = freq_matrix.fillna(0).sum(axis=1)

        g5 = _col("toCoupon_GEQ5min", 1).map(_parse_numeric).fillna(1)
        g15 = _col("toCoupon_GEQ15min", 0).map(_parse_numeric).fillna(0)
        g25 = _col("toCoupon_GEQ25min", 0).map(_parse_numeric).fillna(0)
        min_dist = np.where(
            g5 < 1, 0.0, np.where(g15 < 1, 5.0, np.where(g25 < 1, 15.0, 25.0))
        )
        d["min_distance_min"] = min_dist.astype(float)
        d["direction_same"] = _col("direction_same", 0).map(_parse_numeric).fillna(0)

        return d

    @staticmethod
    def _build_categoricals(df):
        c = pd.DataFrame(index=df.index)
        for col in RAW_CAT_COLS:
            if col in df.columns:
                c[col] = df[col].astype(str).str.strip()
            else:
                c[col] = "missing"

        def _col(name, default):
            if name in df.columns:
                return df[name].astype(str).str.strip()
            return pd.Series([default] * len(df), index=df.index, dtype=object)

        coupon = c["coupon"].map(_norm)
        expir = _col("expiration", "missing")
        time_ = _col("time", "missing")

        c["coupon_x_expiration"] = coupon + "|" + expir.map(_norm)
        c["coupon_x_time"] = coupon + "|" + time_.map(_norm)

        age = _col("age", "missing")
        kids = c["has_children"]
        c["age_x_children"] = age.map(_norm) + "|" + kids.map(_norm)
        c["marital_x_children"] = c["maritalStatus"].map(_norm) + "|" + kids.map(_norm)

        income = _col("income", "missing")
        c["income_x_coupon"] = income.map(_norm) + "|" + coupon

        c = c.fillna("missing").replace({"": "missing", "nan": "missing"})
        return c

    def fit(self, df, y=None):
        core = self._build_core(df)
        cats = self._build_categoricals(df)

        for col in core.columns:
            med = core[col].median()
            if pd.isna(med):
                med = 0.0
            self._medians_[col] = float(med)

        self._has_te_ = y is not None
        if self._has_te_:
            y = np.asarray(y, dtype=float)
            self._te_prior_ = float(np.mean(y)) if len(y) else 0.5
            te_source = cats.copy()
            te_source["income_raw"] = (
                df["income"] if "income" in df.columns
                else pd.Series(["missing"] * len(df), index=df.index)
            ).astype(str)
            te_source["education_raw"] = (
                df["education"] if "education" in df.columns
                else pd.Series(["missing"] * len(df), index=df.index)
            ).astype(str)
            for col in self.te_cols:
                if col not in te_source.columns:
                    continue
                key = te_source[col].astype(str)
                agg = pd.DataFrame({"k": key.values, "y": y}).groupby("k")["y"]
                sm = agg.sum()
                cnt = agg.count()
                m = self.smoothing
                rates = (sm + self._te_prior_ * m) / (cnt + m)
                self._te_maps_[col] = rates.to_dict()

        self.num_columns_ = list(core.columns) + [
            f"te_{col}" for col in self.te_cols if col in self._te_maps_
        ]
        self.cat_columns_ = list(cats.columns)
        return self

    def transform(self, df):
        core = self._build_core(df)
        cats = self._build_categoricals(df)

        for col in list(core.columns):
            med = self._medians_.get(col, 0.0)
            core[col] = core[col].fillna(med)

        if self._has_te_:
            te_source = cats.copy()
            te_source["income_raw"] = (
                df["income"] if "income" in df.columns
                else pd.Series(["missing"] * len(df), index=df.index)
            ).astype(str)
            te_source["education_raw"] = (
                df["education"] if "education" in df.columns
                else pd.Series(["missing"] * len(df), index=df.index)
            ).astype(str)
            for col in self.te_cols:
                if col not in self._te_maps_:
                    continue
                mp = self._te_maps_[col]
                core[f"te_{col}"] = (
                    te_source[col]
                    .astype(str)
                    .map(mp)
                    .fillna(self._te_prior_)
                    .astype(float)
                )

        out = pd.concat([cats[self.cat_columns_], core[self.num_columns_]], axis=1)
        out = out.reset_index(drop=True)
        return out

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def to_state(self):
        return {
            "smoothing": self.smoothing,
            "te_cols": self.te_cols,
            "cat_columns_": self.cat_columns_,
            "num_columns_": self.num_columns_,
            "medians_": self._medians_,
            "te_maps_": self._te_maps_,
            "te_prior_": self._te_prior_,
            "has_te_": self._has_te_,
        }

    @classmethod
    def from_state(cls, state):
        obj = cls(smoothing=state.get("smoothing", 10.0), te_cols=state.get("te_cols"))
        obj.cat_columns_ = state.get("cat_columns_", [])
        obj.num_columns_ = state.get("num_columns_", [])
        obj._medians_ = state.get("medians_", {})
        obj._te_maps_ = state.get("te_maps_", {})
        obj._te_prior_ = state.get("te_prior_", 0.5)
        obj._has_te_ = state.get("has_te_", False)
        return obj


# ---------------------------------------------------------------------------
# Model: FT-Transformer-lite + coupon-conditioned MoE head
# ---------------------------------------------------------------------------
DEFAULT_MODEL_CONFIG = {
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 128,
    "dropout": 0.15,
    "attn_dropout": 0.10,
    "stochastic_depth": 0.05,
    "n_experts": 5,
    "expert_hidden": 96,
    "gate_hidden": 32,
    "label_smoothing": 0.05,
    "lr": 2.0e-3,
    "weight_decay": 1.0e-4,
    "warmup_frac": 0.10,
}


class NumericStandardizer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    @torch.no_grad()
    def set_stats(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(std, dtype=torch.float32).flatten()
        std = torch.where(torch.isfinite(std) & (std > 1e-6), std, torch.ones_like(std))
        assert mean.numel() == self.mean.numel() == std.numel()
        self.mean.copy_(mean)
        self.std.copy_(std)

    def forward(self, x_num):
        return (x_num - self.mean) / self.std


class FeatureTokenizer(nn.Module):
    def __init__(self, cat_cardinalities, n_num_features, d_model):
        super().__init__()
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(int(c), d_model) for c in cat_cardinalities]
        )
        self.num_weight = nn.Parameter(torch.randn(n_num_features, d_model) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(n_num_features, d_model))
        n_tokens = len(cat_cardinalities) + n_num_features
        self.token_pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.cat_dropout = nn.Dropout(0.05)
        self.n_num = n_num_features

    def forward(self, x_cat, x_num):
        cat_tokens = []
        for j, emb in enumerate(self.cat_embeddings):
            tok = emb(x_cat[:, j])
            cat_tokens.append(self.cat_dropout(tok))
        if cat_tokens:
            cat_tokens = torch.stack(cat_tokens, dim=1)
        else:
            cat_tokens = torch.zeros(
                (x_num.shape[0], 0, self.num_weight.shape[1]), device=x_num.device
            )
        num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        tokens = torch.cat([cat_tokens, num_tokens], dim=1)
        return tokens + self.token_pos.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, attn_dropout, stochastic_depth):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.sd = float(stochastic_depth)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        h = self.norm2(x)
        f = self.ff(h)
        if self.training and self.sd > 0.0:
            keep = torch.empty((x.shape[0], 1, 1), device=x.device).bernoulli_(
                1.0 - self.sd
            )
            f = f * keep / (1.0 - self.sd)
        return x + f


class CouponMoEHead(nn.Module):
    def __init__(
        self, d_model, n_experts, expert_hidden, gate_hidden, n_classes=2, dropout=0.15
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, n_classes),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, n_classes),
                )
                for _ in range(n_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, n_experts),
        )
        self.n_experts = n_experts

    def forward(self, z, coupon_emb, return_gate=False):
        g = torch.softmax(self.gate(torch.cat([z, coupon_emb], dim=-1)), dim=-1)
        out = self.shared(z)
        for k, expert in enumerate(self.experts):
            out = out + g[:, k : k + 1] * expert(z)
        if return_gate:
            return out, g
        return out


class CouponMoETransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        n_num_features,
        n_classes=2,
        coupon_token_index=0,
        config=None,
    ):
        super().__init__()
        cfg = dict(DEFAULT_MODEL_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg
        d = cfg["d_model"]
        self.coupon_token_index = int(coupon_token_index)

        self.standardizer = NumericStandardizer(n_num_features)
        self.tokenizer = FeatureTokenizer(cat_cardinalities, n_num_features, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d,
                    cfg["n_heads"],
                    cfg["d_ff"],
                    cfg["dropout"],
                    cfg["attn_dropout"],
                    cfg["stochastic_depth"],
                )
                for _ in range(cfg["n_layers"])
            ]
        )
        self.norm_out = nn.LayerNorm(d)
        self.head = CouponMoEHead(
            d_model=d,
            n_experts=cfg["n_experts"],
            expert_hidden=cfg["expert_hidden"],
            gate_hidden=cfg["gate_hidden"],
            n_classes=n_classes,
            dropout=cfg["dropout"],
        )

    def set_numeric_stats(self, mean, std):
        self.standardizer.set_stats(mean, std)

    def forward(self, x_cat, x_num, return_gate=False):
        x_num = self.standardizer(x_num)
        tokens = self.tokenizer(x_cat, x_num)
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        z = x[:, 0]
        coupon_emb = x[:, 1 + self.coupon_token_index]
        return self.head(z, coupon_emb, return_gate=return_gate)

    @torch.no_grad()
    def predict_proba(self, x_cat, x_num):
        self.eval()
        return torch.softmax(self.forward(x_cat, x_num), dim=-1)


def infer_cat_maps(df, cat_columns):
    maps = {}
    for col in cat_columns:
        vals = sorted(set(df[col].astype(str).values))
        maps[col] = {v: i for i, v in enumerate(vals)}
    return maps


def cat_cardinalities(cat_maps, cat_columns):
    return [len(cat_maps[c]) for c in cat_columns]


def encode_inputs(df, cat_columns, cat_maps, num_columns):
    if cat_columns:
        cols = []
        for c in cat_columns:
            m = cat_maps[c]
            cols.append(df[c].astype(str).map(m).fillna(0).astype("int64").values)
        x_cat = torch.as_tensor(np.stack(cols, axis=1), dtype=torch.long)
    else:
        x_cat = torch.zeros((len(df), 0), dtype=torch.long)
    if num_columns:
        num = (
            df[num_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype="float32")
        )
    else:
        num = np.zeros((len(df), 0), dtype="float32")
    num = np.asarray(num, dtype="float32")
    num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
    x_num = torch.as_tensor(num, dtype=torch.float32)
    return x_cat, x_num


def build_criterion(class_counts=None, label_smoothing=0.05):
    weight = None
    if class_counts is not None:
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp(min=1.0)
        weight = counts.sum() / (len(counts) * counts)
        weight = weight / weight.mean()
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=float(label_smoothing))


def build_optimizer(model, lr=2.0e-3, weight_decay=1.0e-4, betas=(0.9, 0.98), eps=1e-8):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name
            or "embeddings" in name
            or "token_pos" in name
            or "cls_token" in name
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(lr),
        betas=betas,
        eps=eps,
    )


def build_scheduler(optimizer, total_steps, warmup_frac=0.10, min_lr_ratio=0.05):
    total_steps = max(int(total_steps), 1)
    warmup = max(int(total_steps * float(warmup_frac)), 1)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        prog = min(max(prog, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * prog)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), base_lrs


def build_model(
    cat_maps, cat_columns, n_num_features, n_classes=2, class_counts=None, config=None
):
    cfg = dict(DEFAULT_MODEL_CONFIG)
    if config:
        cfg.update(config)
    coupon_idx = cat_columns.index("coupon") if "coupon" in cat_columns else 0
    model = CouponMoETransformer(
        cat_cardinalities=cat_cardinalities(cat_maps, cat_columns),
        n_num_features=n_num_features,
        n_classes=n_classes,
        coupon_token_index=coupon_idx,
        config=cfg,
    )
    criterion = build_criterion(
        class_counts=class_counts, label_smoothing=cfg["label_smoothing"]
    )
    optimizer = build_optimizer(model, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    return model, criterion, optimizer


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
BATCH_SIZE = 256
MAX_EPOCHS = 18
PATIENCE = 4
N_FOLDS = 4
N_SEEDS = 2
CV_SEED = 42
BUNDLE_NAME = "bundle.pt"


@torch.no_grad()
def _eval_acc(model, x_cat, x_num, y):
    model.eval()
    if len(y) == 0:
        return 0.0
    correct = 0
    for i in range(0, len(y), 512):
        logits = model(x_cat[i : i + 512], x_num[i : i + 512])
        correct += (logits.argmax(dim=1) == y[i : i + 512]).sum().item()
    return correct / float(len(y))


@torch.no_grad()
def _predict_proba(model, x_cat, x_num):
    model.eval()
    outs = []
    for i in range(0, len(x_cat), 512):
        logits = model(x_cat[i : i + 512], x_num[i : i + 512])
        outs.append(torch.softmax(logits, dim=1))
    if not outs:
        return np.zeros((0, 2), dtype=np.float32)
    return torch.cat(outs, dim=0).cpu().numpy()


def _train_one(
    x_cat_tr,
    x_num_tr,
    y_tr,
    cat_maps,
    cat_columns,
    config,
    seed,
    epochs,
    class_counts,
    val=None,
    patience=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_num = x_num_tr.shape[1]
    model, criterion, optimizer = build_model(
        cat_maps,
        cat_columns,
        n_num,
        n_classes=2,
        class_counts=class_counts,
        config=config,
    )
    model.set_numeric_stats(
        x_num_tr.mean(dim=0).cpu().numpy(),
        x_num_tr.std(dim=0).cpu().numpy(),
    )

    steps_per_epoch = max(1, math.ceil(len(y_tr) / BATCH_SIZE))
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler, _ = build_scheduler(
        optimizer, total_steps, warmup_frac=config.get("warmup_frac", 0.1)
    )

    best_state = None
    best_acc = -1.0
    best_epoch = 0
    bad = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y_tr))
        for i in range(0, len(y_tr), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(x_cat_tr[idx], x_num_tr[idx])
            loss = criterion(logits, y_tr[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()

        if val is not None:
            acc = _eval_acc(model, val[0], val[1], val[2])
            if acc > best_acc + 1e-6:
                best_acc = acc
                best_epoch = ep
                bad = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if patience is not None and bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_acc = _eval_acc(model, val[0], val[1], val[2]) if val is not None else 0.0
        best_epoch = epochs - 1

    return model, best_acc, best_epoch


def run_training(data_dir, model_dir):
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    np.random.seed(CV_SEED)
    torch.manual_seed(CV_SEED)
    os.makedirs(model_dir, exist_ok=True)

    features, labels, schema = load_data(data_dir, with_labels=True)
    if labels is None:
        raise RuntimeError("labels.csv required for training")
    cols = get_feature_columns(features, schema)
    X_raw = features[cols].reset_index(drop=True)
    y_orig = get_labels_array(labels)
    classes = np.unique(y_orig)
    label_to_idx = {label: i for i, label in enumerate(classes)}
    y_np = np.array([label_to_idx[v] for v in y_orig], dtype=np.int64)
    y = torch.as_tensor(y_np, dtype=torch.long)

    class_counts = np.bincount(y_np, minlength=len(classes)).tolist()

    config = dict(DEFAULT_MODEL_CONFIG)

    folds = get_stratified_folds(y_np, n_splits=N_FOLDS, seed=CV_SEED)
    oof_prob = np.zeros(len(y_np), dtype=np.float64)
    best_epochs = []

    for fi, (tr_idx, va_idx) in enumerate(folds):
        eng = CouponFeatureEngineer()
        Xtr = eng.fit_transform(X_raw.iloc[tr_idx], y_np[tr_idx]).reset_index(drop=True)
        Xva = eng.transform(X_raw.iloc[va_idx]).reset_index(drop=True)

        cat_columns = eng.cat_columns_
        num_columns = eng.num_columns_
        cat_maps = infer_cat_maps(Xtr, cat_columns)
        x_cat_tr, x_num_tr = encode_inputs(Xtr, cat_columns, cat_maps, num_columns)
        x_cat_va, x_num_va = encode_inputs(Xva, cat_columns, cat_maps, num_columns)

        y_tr = y[tr_idx]
        y_va = y[va_idx]
        fold_class_counts = np.bincount(
            y_np[tr_idx], minlength=len(classes)
        ).tolist()

        model, val_acc, best_ep = _train_one(
            x_cat_tr,
            x_num_tr,
            y_tr,
            cat_maps,
            cat_columns,
            config,
            seed=CV_SEED + fi,
            epochs=MAX_EPOCHS,
            class_counts=fold_class_counts,
            val=(x_cat_va, x_num_va, y_va),
            patience=PATIENCE,
        )
        probs = _predict_proba(model, x_cat_va, x_num_va)
        oof_prob[va_idx] = probs[:, 1]
        best_epochs.append(max(1, best_ep))
        print(f"[fold {fi}] val_acc={val_acc:.4f} best_epoch={best_ep}")

    oof_pred = (oof_prob >= 0.5).astype(int)
    oof_acc = float((oof_pred == y_np).mean())
    print(f"OOF accuracy (train-only CV): {oof_acc:.4f}")

    final_engineer = CouponFeatureEngineer()
    X_full = final_engineer.fit_transform(X_raw, y_np).reset_index(drop=True)
    cat_columns = final_engineer.cat_columns_
    num_columns = final_engineer.num_columns_
    cat_maps = infer_cat_maps(X_full, cat_columns)
    x_cat_full, x_num_full = encode_inputs(X_full, cat_columns, cat_maps, num_columns)

    final_epochs = max(8, int(round(np.mean(best_epochs)))) if best_epochs else 20
    print(f"Refitting on full train with {final_epochs} epochs, {N_SEEDS} seeds")

    state_dicts = []
    for s in range(N_SEEDS):
        model, _, _ = _train_one(
            x_cat_full,
            x_num_full,
            y,
            cat_maps,
            cat_columns,
            config,
            seed=1000 + s,
            epochs=final_epochs,
            class_counts=class_counts,
            val=None,
            patience=None,
        )
        state_dicts.append({k: v.cpu() for k, v in model.state_dict().items()})

    bundle = {
        "engineer": final_engineer.to_state(),
        "cat_columns": cat_columns,
        "num_columns": num_columns,
        "cat_maps": cat_maps,
        "config": config,
        "classes": classes.tolist(),
        "state_dicts": state_dicts,
        "cv_accuracy": oof_acc,
    }
    bundle_path = os.path.join(model_dir, BUNDLE_NAME)
    torch.save(bundle, bundle_path)
    print(f"Saved bundle -> {bundle_path} ({len(state_dicts)} models)")

    return oof_acc


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_prediction(data_dir, model_dir, output_path):
    torch.set_num_threads(2)
    bundle_path = os.path.join(model_dir, BUNDLE_NAME)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Missing model bundle: {bundle_path}")
    bundle = _safe_torch_load(bundle_path)

    engineer = CouponFeatureEngineer.from_state(bundle["engineer"])
    cat_columns = bundle["cat_columns"]
    num_columns = bundle["num_columns"]
    cat_maps = bundle["cat_maps"]
    config = bundle["config"]
    classes = np.asarray(bundle["classes"])

    features, _, schema = load_data(data_dir, with_labels=False)
    cols = get_feature_columns(features, schema)
    X_raw = features[cols].reset_index(drop=True)
    n = len(X_raw)
    if n == 0:
        # No rows to predict: still emit a valid, empty prediction file.
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(
            {"row_id": np.array([], dtype=int), "prediction": np.array([], dtype=int)}
        ).to_csv(output_path, index=False)
        return output_path

    X = engineer.transform(X_raw).reset_index(drop=True)
    x_cat, x_num = encode_inputs(X, cat_columns, cat_maps, num_columns)

    n_num = x_num.shape[1]
    probs = np.zeros((n, len(classes)), dtype=np.float64)
    for sd in bundle["state_dicts"]:
        model, _, _ = build_model(
            cat_maps,
            cat_columns,
            n_num,
            n_classes=len(classes),
            config=config,
        )
        model.load_state_dict(sd)
        model.eval()
        probs += _predict_proba(model, x_cat, x_num)
    if len(bundle["state_dicts"]) == 0:
        raise RuntimeError("Model bundle contains no state dicts")
    probs /= max(1, len(bundle["state_dicts"]))

    pred_idx = probs.argmax(axis=1)
    predictions = classes[pred_idx]

    out_df = pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=int),
            "prediction": predictions,
        }
    )
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df["row_id"] = out_df["row_id"].astype(int)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {n} predictions -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Coupon acceptance model")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train model and persist bundle")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict", help="Load bundle and write predictions")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "train":
        score = run_training(args.data, args.model)
        print(f"Final Validation Score: {score}")
    elif args.command == "predict":
        run_prediction(args.data, args.model, args.output)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
```