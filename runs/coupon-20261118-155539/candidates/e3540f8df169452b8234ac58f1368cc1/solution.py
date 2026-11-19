"""
Coupon-acceptance solution: leakage-safe feature engineering + coupon-conditioned
mixture-of-experts (DCN-v2 crosses) blended with a LightGBM anchor.

Merged from:
  - Step 1: loaders + FeatureEngineer (fitted on training rows only)
  - Step 2: CouponMoE / SinglePathNet + loss / optimizer / scheduler
  - Step 3: 5-fold stratified CV training, bagged inference, fresh-process predict

CLI:
  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv
"""

import os
import re
import copy
import json
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Sequence

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


# =========================================================================
# Step 1: data loading + leakage-safe feature engineering
# =========================================================================
FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]

FREQ_MAP = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}

AGE_ORD = {
    "below21": 0,
    "21": 1,
    "26": 2,
    "31": 3,
    "36": 4,
    "41": 5,
    "46": 6,
    "50plus": 7,
}

EDU_ORD = {
    "some high school": 0,
    "high school graduate": 1,
    "some college - no degree": 2,
    "associates degree": 3,
    "bachelors degree": 4,
    "graduate degree (masters, doctorate)": 5,
}

TIME_ORD = {"7am": 0, "10am": 1, "2pm": 2, "6pm": 3, "10pm": 4}
TIME_HOUR = {"7am": 7, "10am": 10, "2pm": 14, "6pm": 18, "10pm": 22}

EXPIRATION_MAP = {"2h": 0, "1d": 1}

INCOME_ORD = {
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

NOMINAL_COLS = ["gender", "maritalStatus", "occupation", "coupon"]


def _norm(x):
    return str(x).strip().lower()


def _map_ord(series, mapping):
    return series.map(lambda x: mapping.get(_norm(x), -1)).astype(float)


def _income_mid(x):
    if pd.isna(x):
        return np.nan
    nums = re.findall(r"\d+", str(x))
    if not nums:
        return np.nan
    return float(np.mean([float(n) for n in nums]))


def _venue_col_for_coupon(coupon):
    c = _norm(coupon)
    if "coffee" in c:
        return "CoffeeHouse"
    if "carry" in c:
        return "CarryAway"
    if "bar" in c:
        return "Bar"
    if "restaurant" in c:
        if "-" in c and "50" in c:
            return "Restaurant20To50"
        return "RestaurantLessThan20"
    return None


def load_schema(data_dir):
    path = os.path.join(data_dir, "schema.json")
    with open(path, "r") as f:
        return json.load(f)


def load_features(data_dir):
    return pd.read_csv(os.path.join(data_dir, "features.csv"))


def load_train(data_dir):
    """Return (features_df, y_encoded, label_classes, schema). Row order preserved."""
    schema = load_schema(data_dir)
    X = load_features(data_dir)
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    label_col = "label" if "label" in labels.columns else labels.columns[0]
    raw = labels[label_col]
    classes = sorted(pd.unique(raw).tolist(), key=lambda v: str(v))
    mapping = {c: i for i, c in enumerate(classes)}
    y = raw.map(mapping).to_numpy(dtype=np.int64)
    return X, y, classes, schema


def make_cv_folds(y, n_splits=5, seed=42):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]


class FeatureEngineer:
    """Leakage-safe feature builder. fit() on train only, transform() anytime."""

    def __init__(self):
        self.nominal_vocab = {}
        self.income_median = 0.0
        self.income_ord_median = 0.0
        self.feature_columns = None

    @staticmethod
    def _fit_vocab(series):
        vals = sorted({_norm(v) for v in series.astype(str).unique().tolist()})
        return {v: i for i, v in enumerate(vals)}

    def fit(self, df):
        self.nominal_vocab = {
            c: self._fit_vocab(df[c]) for c in NOMINAL_COLS if c in df.columns
        }
        inc = df["income"].map(_income_mid)
        self.income_median = float(inc.median()) if inc.notna().any() else 0.0
        inc_ord = _map_ord(df["income"], INCOME_ORD)
        inc_ord = inc_ord.where(inc_ord >= 0)
        self.income_ord_median = (
            float(inc_ord.median()) if inc_ord.notna().any() else 0.0
        )
        self.feature_columns = self.transform(df).columns.tolist()
        return self

    def transform(self, df):
        df = df.copy()
        out = pd.DataFrame(index=df.index)

        out["temperature"] = pd.to_numeric(df["temperature"], errors="coerce").fillna(
            -1
        )
        out["expiration_num"] = _map_ord(df["expiration"], EXPIRATION_MAP)
        out["has_children"] = pd.to_numeric(df["has_children"], errors="coerce").fillna(
            -1
        )
        out["age_ord"] = _map_ord(df["age"], AGE_ORD)
        out["edu_ord"] = _map_ord(df["education"], EDU_ORD)

        time_norm = df["time"].map(_norm)
        out["time_ord"] = time_norm.map(lambda x: TIME_ORD.get(x, -1)).astype(float)
        hours = time_norm.map(lambda x: TIME_HOUR.get(x, np.nan))
        ang = 2 * np.pi * hours / 24.0
        out["time_sin"] = np.sin(ang).fillna(0.0)
        out["time_cos"] = np.cos(ang).fillna(1.0)
        is_morning = time_norm.isin(["7am", "10am"]).astype(float)
        is_evening = time_norm.isin(["6pm", "10pm"]).astype(float)
        out["is_morning"] = is_morning
        out["is_evening"] = is_evening

        inc_mid = df["income"].map(_income_mid)
        out["income_mid"] = inc_mid.fillna(self.income_median).astype(float)
        out["income_ord"] = _map_ord(df["income"], INCOME_ORD)
        out.loc[out["income_ord"] < 0, "income_ord"] = np.nan
        out["income_ord"] = out["income_ord"].fillna(self.income_ord_median)

        for c in NOMINAL_COLS:
            if c not in df.columns:
                continue
            vocab = self.nominal_vocab.get(c, {})
            out[c + "_enc"] = df[c].map(lambda x: vocab.get(_norm(x), -1)).astype(float)

        for c in FREQ_COLS:
            out[c + "_ord"] = _map_ord(df[c], FREQ_MAP)

        freq_ord = out[[c + "_ord" for c in FREQ_COLS]]
        present = freq_ord.where(freq_ord >= 0)
        out["freq_sum"] = present.sum(axis=1).fillna(0.0)
        out["freq_mean"] = present.mean(axis=1).fillna(0.0)
        out["freq_nonmissing"] = present.notna().sum(axis=1).astype(float)
        out["freq_never_count"] = (freq_ord == 0).sum(axis=1).astype(float)

        venue_col = df["coupon"].map(_venue_col_for_coupon)
        venue_match = pd.Series(np.nan, index=df.index, dtype=float)
        for c in FREQ_COLS:
            mask = venue_col == c
            if mask.any():
                venue_match.loc[mask] = out.loc[mask, c + "_ord"]
        out["venue_match_freq"] = venue_match.fillna(-1.0)
        gap = venue_match - out["freq_mean"]
        out["venue_pref_gap"] = gap.fillna(0.0)
        out["venue_match_missing"] = (venue_match < 0).astype(float)

        geq = {}
        for flag in ["toCoupon_GEQ5min", "toCoupon_GEQ15min", "toCoupon_GEQ25min"]:
            geq[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0)
        out["geq5"] = geq["toCoupon_GEQ5min"]
        out["geq15"] = geq["toCoupon_GEQ15min"]
        out["geq25"] = geq["toCoupon_GEQ25min"]
        out["dist_tier"] = (
            geq["toCoupon_GEQ5min"]
            + geq["toCoupon_GEQ15min"]
            + geq["toCoupon_GEQ25min"]
        )

        d_same = pd.to_numeric(df["direction_same"], errors="coerce").fillna(0)
        out["direction_same"] = d_same
        out["dir_ge15"] = d_same * geq["toCoupon_GEQ15min"]
        out["dir_ge25"] = d_same * geq["toCoupon_GEQ25min"]

        c_low = df["coupon"].map(_norm)
        c_bar = c_low.str.contains("bar").astype(float)
        c_coffee = c_low.str.contains("coffee").astype(float)
        c_carry = c_low.str.contains("carry").astype(float)
        c_rest = c_low.str.contains("restaurant").astype(float)
        out["coupon_bar"] = c_bar
        out["coupon_coffee"] = c_coffee
        out["coupon_carry"] = c_carry
        out["coupon_restaurant"] = c_rest
        out["bar_x_evening"] = c_bar * is_evening
        out["coffee_x_morning"] = c_coffee * is_morning
        out["rest_x_evening"] = c_rest * is_evening
        out["short_exp_x_inshop"] = (1.0 - out["expiration_num"].clip(0, 1)) * (
            c_rest + c_carry
        )

        out["age_x_children"] = out["age_ord"] * out["has_children"]
        out["income_x_children"] = out["income_ord"] * out["has_children"]
        out["temp_above"] = (out["temperature"] >= 80).astype(float)
        out["temp_cold"] = (out["temperature"] <= 30).astype(float)

        if self.feature_columns is not None:
            for c in self.feature_columns:
                if c not in out.columns:
                    out[c] = 0.0
            out = out[self.feature_columns]

        return out.astype(np.float32)

    def to_dict(self):
        return {
            "nominal_vocab": self.nominal_vocab,
            "income_median": self.income_median,
            "income_ord_median": self.income_ord_median,
            "feature_columns": self.feature_columns,
        }

    @classmethod
    def from_dict(cls, d):
        fe = cls()
        fe.nominal_vocab = {k: dict(v) for k, v in d["nominal_vocab"].items()}
        fe.income_median = float(d["income_median"])
        fe.income_ord_median = float(d.get("income_ord_median", 0.0))
        fe.feature_columns = (
            list(d["feature_columns"]) if d.get("feature_columns") else None
        )
        return fe


# =========================================================================
# Step 2: model architecture, loss, optimizer, scheduler
# =========================================================================
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossLayer(nn.Module):
    """DCN-v2 layer: x_{l+1} = x0 * (W x_l + b) + x_l."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * (self.w(x) + self.b) + x


class CouponMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        num_experts: int = 6,
        expert_hidden: int = 96,
        expert_out: int = 64,
        gate_hidden: int = 48,
        deep_hidden: int = 128,
        cross_layers: int = 2,
        dropout: float = 0.35,
        gate_feature_idx: Optional[Sequence[int]] = None,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.gate_temperature = float(gate_temperature)
        self.gate_feature_idx = (
            list(gate_feature_idx) if gate_feature_idx is not None else None
        )

        self.cross_layers = nn.ModuleList(
            [CrossLayer(self.input_dim) for _ in range(max(1, cross_layers))]
        )
        self.cross_norm = nn.LayerNorm(self.input_dim)

        gate_in = (
            len(self.gate_feature_idx)
            if self.gate_feature_idx is not None
            else self.input_dim
        )
        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, self.num_experts),
        )

        self.experts = nn.ModuleList(
            [
                MLPBlock(self.input_dim, expert_hidden, expert_out, dropout)
                for _ in range(self.num_experts)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(expert_out + self.input_dim, deep_hidden),
            nn.BatchNorm1d(deep_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(deep_hidden, self.num_classes),
        )
        self.wide = nn.Linear(self.input_dim, self.num_classes)

        self.register_buffer("last_gate", torch.zeros(1, self.num_experts))
        self.aux_load_balance = torch.tensor(0.0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _route(self, x: torch.Tensor) -> torch.Tensor:
        gate_in = (
            x[:, self.gate_feature_idx] if self.gate_feature_idx is not None else x
        )
        logits = self.gate(gate_in) / self.gate_temperature
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"expected input (B,{self.input_dim}), got {tuple(x.shape)}"
            )

        x0 = x
        c = x
        for layer in self.cross_layers:
            c = layer(x0, c)
        c = self.cross_norm(c)

        gate = self._route(x)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        moe = (gate.unsqueeze(-1) * expert_outs).sum(dim=1)

        h = torch.cat([moe, c], dim=1)
        logits = self.head(h) + self.wide(x)

        with torch.no_grad():
            self.last_gate = gate.detach()
            mean_gate = gate.mean(dim=0)
            self.aux_load_balance = (
                mean_gate * torch.log(mean_gate.clamp_min(1e-8))
            ).sum() + torch.log(torch.tensor(float(self.num_experts)))
        return logits


class SinglePathNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden: int = 160,
        deep_hidden: int = 96,
        cross_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.cross_layers = nn.ModuleList(
            [CrossLayer(self.input_dim) for _ in range(max(1, cross_layers))]
        )
        self.cross_norm = nn.LayerNorm(self.input_dim)
        self.trunk = MLPBlock(self.input_dim, hidden, deep_hidden, dropout)
        self.head = nn.Sequential(nn.Linear(deep_hidden + self.input_dim, num_classes))
        self.wide = nn.Linear(self.input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, c = x, x
        for layer in self.cross_layers:
            c = layer(x0, c)
        c = self.cross_norm(c)
        h = self.trunk(x)
        return self.head(torch.cat([h, c], dim=1)) + self.wide(x)


DEFAULT_CONFIG = {
    "arch": "moe",
    "num_classes": 2,
    "num_experts": 4,
    "expert_hidden": 64,
    "expert_out": 48,
    "gate_hidden": 32,
    "deep_hidden": 96,
    "cross_layers": 1,
    "dropout": 0.3,
    "gate_temperature": 1.0,
    "lr": 3e-3,
    "weight_decay": 1e-2,
    "label_smoothing": 0.05,
}
=======


def build_model(
    input_dim: int,
    num_classes: int = 2,
    config: Optional[dict] = None,
    gate_feature_idx: Optional[Sequence[int]] = None,
) -> nn.Module:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if cfg.get("arch", "moe") == "single":
        return SinglePathNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden=cfg["deep_hidden"] * 2,
            deep_hidden=cfg["expert_out"] * 2,
            cross_layers=cfg["cross_layers"],
            dropout=cfg["dropout"],
        )
    return CouponMoE(
        input_dim=input_dim,
        num_classes=num_classes,
        num_experts=cfg["num_experts"],
        expert_hidden=cfg["expert_hidden"],
        expert_out=cfg["expert_out"],
        gate_hidden=cfg["gate_hidden"],
        deep_hidden=cfg["deep_hidden"],
        cross_layers=cfg["cross_layers"],
        dropout=cfg["dropout"],
        gate_feature_idx=gate_feature_idx,
        gate_temperature=cfg["gate_temperature"],
    )


def build_criterion(
    num_classes: int = 2,
    class_weights: Optional[Sequence[float]] = None,
    label_smoothing: float = 0.05,
) -> nn.Module:
    w = (
        torch.tensor(list(class_weights), dtype=torch.float32)
        if class_weights is not None
        else None
    )
    return nn.CrossEntropyLoss(weight=w, label_smoothing=float(label_smoothing))


def build_optimizer(
    model: nn.Module, lr: float = 3e-3, weight_decay: float = 1e-2
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in name for k in ("bn", "norm", "bias")):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.99), eps=1e-8)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    pct_start: float = 0.25,
    max_lr: Optional[float] = None,
) -> torch.optim.lr_scheduler._LRScheduler:
    total_steps = max(1, int(epochs * steps_per_epoch))
    base_lr = float(optimizer.param_groups[0]["lr"])
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr if max_lr is not None else base_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        cycle_momentum=True,
        div_factor=10.0,
        final_div_factor=100.0,
    )


# =========================================================================
# Step 3: training / internal CV / persistence / fresh-process prediction
# =========================================================================
N_FOLDS = 3
SEED = 42
EPOCHS = 20
BATCH = 256
LR = 3e-3
WD = 1e-2
PATIENCE = 5
LGB_ROUNDS = 350
LGB_ES = 25
BLEND_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
# Hard wall-clock budget (seconds) for the whole train subprocess; the process
# limit is 600s so we stop launching new folds well before that.
TRAIN_TIME_BUDGET = 420.0
=======


def _set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    try:
        torch.set_num_threads(2)
    except Exception:
        pass


def _train_neural_fold(Xtr, ytr, Xva, yva, input_dim, model_config, fold_seed):
    _set_seed(fold_seed)
    mu = Xtr.mean(axis=0).astype(np.float32)
    sd = Xtr.std(axis=0).astype(np.float32) + 1e-6
    Xtr_n = ((Xtr - mu) / sd).astype(np.float32)
    Xva_n = ((Xva - mu) / sd).astype(np.float32)

    model = build_model(input_dim, num_classes=2, config=model_config)
    crit = build_criterion(
        2, class_weights=None, label_smoothing=model_config.get("label_smoothing", 0.05)
    )
    opt = build_optimizer(model, lr=LR, weight_decay=WD)
    steps = max(1, int(np.ceil(len(Xtr_n) / BATCH)))
    sched = build_scheduler(opt, EPOCHS, steps, max_lr=LR)

    Xtr_t = torch.from_numpy(Xtr_n)
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    Xva_t = torch.from_numpy(Xva_n)

    best_acc, best_state, patience = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), BATCH):
            idx = perm[i : i + BATCH]
            if len(idx) < 2:
                continue
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            vp = F.softmax(model(Xva_t), dim=-1).numpy()
        acc = float((vp.argmax(1) == yva).mean())
        print(f"[NN] fold {fold_seed} epoch {ep + 1}: val_acc={acc:.4f}")
        if acc > best_acc + 1e-6:
            best_acc = acc
            # Exclude runtime-only buffers (e.g. `last_gate` is resized every
            # forward pass) so the checkpoint stays shape-stable across processes.
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
                if k != "last_gate" and not k.endswith(".last_gate")
            }
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state, strict=False)
    model.eval()
    with torch.no_grad():
        va_probs = F.softmax(model(Xva_t), dim=-1).numpy()[:, 1]
    return best_state, mu, sd, va_probs, best_acc


def _train_lgb_fold(Xtr, ytr, Xva, yva, fold_seed):
    if lgb is None:
        return None, None
    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_data_in_leaf=20,
        lambda_l1=0.0,
        lambda_l2=1.0,
        verbose=-1,
        num_threads=2,
        seed=fold_seed,
    )
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    bst = lgb.train(
        params,
        dtr,
        num_boost_round=LGB_ROUNDS,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(LGB_ES, verbose=False)],
    )
    va_probs = bst.predict(Xva, num_iteration=bst.best_iteration)
    return bst, va_probs


def run_train(data_dir, model_dir):
    import time

    _t0 = time.time()
    _set_seed(SEED)
    os.makedirs(model_dir, exist_ok=True)

    X_df, y, classes, schema = load_train(data_dir)

    model_config = dict(DEFAULT_CONFIG)
    model_config["num_classes"] = 2

    folds = make_cv_folds(y, n_splits=N_FOLDS, seed=SEED)

    n = len(y)
    oof_nn = np.zeros(n, dtype=np.float64)
    oof_gb = np.zeros(n, dtype=np.float64)
    fold_scalers = []
    fold_fes = []
    nn_files, gb_files = [], []
    input_dim = None

    completed = 0
    covered = np.zeros(n, dtype=bool)
    for fi, (tr, va) in enumerate(folds):
        if (time.time() - _t0) > TRAIN_TIME_BUDGET:
            print(f"[CV] time budget hit after {completed} folds; stopping early")
            break
        fe_fold = FeatureEngineer().fit(X_df.iloc[tr])
        Xtr = np.nan_to_num(
            fe_fold.transform(X_df.iloc[tr]).values.astype(np.float32)
        )
        Xva = np.nan_to_num(
            fe_fold.transform(X_df.iloc[va]).values.astype(np.float32)
        )
        ytr, yva = y[tr], y[va]
        input_dim = int(Xtr.shape[1])
        fold_fes.append(fe_fold.to_dict())

        state, mu, sd, va_probs_nn, acc_nn = _train_neural_fold(
            Xtr, ytr, Xva, yva, input_dim, model_config, SEED + fi
        )
        nn_path = os.path.join(model_dir, f"fold_{fi}_nn.pt")
        torch.save(state, nn_path)
        nn_files.append(os.path.basename(nn_path))
        fold_scalers.append({"mu": mu.tolist(), "sd": sd.tolist()})
        oof_nn[va] = va_probs_nn
        covered[va] = True

        bst, va_probs_gb = _train_lgb_fold(Xtr, ytr, Xva, yva, SEED + fi)
        if bst is not None:
            gb_path = os.path.join(model_dir, f"fold_{fi}_lgb.txt")
            bst.save_model(gb_path)
            gb_files.append(os.path.basename(gb_path))
            oof_gb[va] = va_probs_gb

        completed += 1
        print(
            f"[CV] fold {fi}: nn_acc={acc_nn:.4f} "
            f"elapsed={time.time() - _t0:.1f}s"
        )

    completed = max(1, completed)
    cov = covered if covered.any() else np.ones(n, dtype=bool)
    nn_acc_oof = float(((oof_nn[cov] > 0.5).astype(int) == y[cov]).mean())
    use_gb = lgb is not None and len(gb_files) == completed
    gb_acc_oof = (
        float(((oof_gb[cov] > 0.5).astype(int) == y[cov]).mean()) if use_gb else -1.0
    )

    best_w, best_acc = 1.0, nn_acc_oof
    if use_gb:
        for w in BLEND_GRID:
            p = w * oof_nn[cov] + (1.0 - w) * oof_gb[cov]
            acc = float(((p > 0.5).astype(int) == y[cov]).mean())
            if acc > best_acc + 1e-9:
                best_acc, best_w = acc, float(w)

    meta = {
        "classes": [c if isinstance(c, (int, float, str)) else str(c) for c in classes],
        "input_dim": input_dim,
        "model_config": model_config,
        "blend_w": best_w,
        "use_gb": use_gb,
        "n_folds": completed,
=======
        "fold_scalers": fold_scalers,
        "fold_fes": fold_fes,
        "nn_files": nn_files,
        "gb_files": gb_files,
        "oof_nn_acc": nn_acc_oof,
        "oof_gb_acc": gb_acc_oof,
        "cv_blend_acc": best_acc,
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[CV] oof_nn_acc={nn_acc_oof:.4f} oof_gb_acc={gb_acc_oof:.4f} "
        f"best_blend_w={best_w:.2f} blend_acc={best_acc:.4f}"
    )
    return best_acc


def run_predict(data_dir, model_dir, output_path):
    with open(os.path.join(model_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    X_df = load_features(data_dir)
    N = len(X_df)
    input_dim = int(meta["input_dim"])
    model_config = meta["model_config"]
    n_folds = int(meta["n_folds"])

    nn_probs = np.zeros(N, dtype=np.float64)
    gb_probs = np.zeros(N, dtype=np.float64)
    use_gb = (
        bool(meta.get("use_gb"))
        and lgb is not None
        and len(meta.get("gb_files", [])) == n_folds
    )
    for fi in range(n_folds):
        fe_fold = FeatureEngineer.from_dict(meta["fold_fes"][fi])
        X = np.nan_to_num(fe_fold.transform(X_df).values.astype(np.float32))
        mu = np.asarray(meta["fold_scalers"][fi]["mu"], dtype=np.float32)
        sd = np.asarray(meta["fold_scalers"][fi]["sd"], dtype=np.float32)
        Xs = ((X - mu) / sd).astype(np.float32)
        model = build_model(input_dim, num_classes=2, config=model_config)
        state = torch.load(
            os.path.join(model_dir, meta["nn_files"][fi]), map_location="cpu"
        )
        state = {
            k: v for k, v in state.items()
            if k != "last_gate" and not k.endswith(".last_gate")
        }
        model.load_state_dict(state, strict=False)
        model.eval()
        with torch.no_grad():
            nn_probs += F.softmax(model(torch.from_numpy(Xs)), dim=-1).numpy()[:, 1]
        if use_gb:
            bst = lgb.Booster(
                model_file=os.path.join(model_dir, meta["gb_files"][fi])
            )
            gb_probs += bst.predict(X)
    nn_probs /= max(1, n_folds)

    if use_gb:
        gb_probs /= len(meta["gb_files"])
        w = float(meta.get("blend_w", 1.0))
        probs = w * nn_probs + (1.0 - w) * gb_probs
    else:
        probs = nn_probs

    pred_idx = (probs > 0.5).astype(int)
    classes = meta["classes"]
    labels = [classes[i] for i in pred_idx]

    out = pd.DataFrame({"row_id": np.arange(N, dtype=int), "prediction": labels})
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[PREDICT] wrote {out.shape[0]} rows -> {output_path}")


# =========================================================================
# CLI
# =========================================================================
def _main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--data", default="/data")
    pt.add_argument("--model", default="/model")

    pp = sub.add_parser("predict")
    pp.add_argument("--data", default="/data")
    pp.add_argument("--model", default="/model")
    pp.add_argument("--output", default="/output/predictions.csv")

    args = parser.parse_args()
    if args.command == "train":
        score = run_train(args.data, args.model)
        print(f"Final Validation Score: {score}")
    else:
        run_predict(args.data, args.model, args.output)


if __name__ == "__main__":
    _main()