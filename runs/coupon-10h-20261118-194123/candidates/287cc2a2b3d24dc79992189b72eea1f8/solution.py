"""
Coupon acceptance prediction — merged single-file solution.

  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv

Leakage-safe: every stateful component (median imputer, categorical vocabulary,
out-of-fold target encoders, model weights, GBDT, blend weight, threshold,
label map) is fit ONLY on training rows (per CV fold) and reloaded unchanged
in a fresh predict process. Output is row_id,prediction in authoritative
input order with original labels.

Improvements over the previous plateaued version:
  * OOF target encoding of high-order categorical combos (new numeric features).
  * Additive self-attention token mixer over per-feature embeddings, added to
    the DCN + MoE tower so arbitrary-order interactions can be modelled.
  * Everything else (blend + threshold search, refit, fresh reload) preserved.
"""

import os
import json
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

torch.set_num_threads(2)

# ===========================================================================
# 0. Configuration
# ===========================================================================
SEED = 42
N_FOLDS = 5
MAX_EPOCHS = 60
PATIENCE = 9
BATCH = 256
N_REFIT = 3

MODEL_CFG = dict(
    emb_dim=16,
    n_cross=2,
    hidden=(256, 128),
    n_experts=4,
    expert_hidden=64,
    dropout=0.30,
    n_heads=4,
    n_attn_layers=2,
    lr=2e-3,
    weight_decay=1e-4,
    total_epochs=MAX_EPOCHS,
    label_smoothing=0.05,
)

GBDT_PARAMS = dict(
    max_iter=500,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    min_samples_leaf=20,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=40,
)

TE_SMOOTHING = 20.0

# ===========================================================================
# 1. FEATURE ENGINEERING
# ===========================================================================
FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]
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
EXPIRATION_ORDER = {"2h": 0, "1d": 1}
COUPON_VENUE = {
    "bar": "Bar",
    "coffee house": "CoffeeHouse",
    "carry out & take away": "CarryAway",
    "restaurant(<20)": "RestaurantLessThan20",
    "restaurant(20-50)": "Restaurant20To50",
}
BINARY_COLS = [
    "has_children",
    "toCoupon_GEQ5min",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
    "direction_opp",
]
NOMINAL_COLS = ["coupon", "time", "gender", "maritalStatus", "occupation", "expiration"]

# combos to target-encode (must exist as columns in the engineered frame)
TE_SPECS = [
    ["coupon"],
    ["occupation"],
    ["coupon", "time"],
    ["coupon", "expiration"],
    ["coupon", "occupation"],
    ["coupon", "gender"],
    ["maritalStatus", "has_children"],
    ["age", "has_children"],
    ["direction_opp", "toCoupon_GEQ15min"],
    ["income", "age"],
    ["coupon", "direction_opp"],
    ["occupation", "education"],
]


def _ord_lookup(series, mapping):
    def _f(x):
        if pd.isna(x):
            return np.nan
        return mapping.get(str(x).strip().lower(), np.nan)

    return series.apply(_f)


def _to_float(series):
    return pd.to_numeric(series, errors="coerce")


def engineer_features(df):
    df = df.copy()
    dfe = pd.DataFrame(index=df.index)

    for c in FREQ_COLS:
        if c in df.columns:
            dfe[c + "_ord"] = _ord_lookup(df[c], FREQ_ORDER)
            dfe[c + "_missing"] = df[c].isna().astype(int)
        else:
            dfe[c + "_ord"] = np.nan
            dfe[c + "_missing"] = 0

    if "age" in df.columns:
        dfe["age_ord"] = _ord_lookup(df["age"], AGE_ORDER)
    if "income" in df.columns:
        dfe["income_ord"] = _ord_lookup(df["income"], INCOME_ORDER)
    if "education" in df.columns:
        dfe["education_ord"] = _ord_lookup(df["education"], EDU_ORDER)
    if "temperature" in df.columns:
        dfe["temperature"] = _to_float(df["temperature"])

    for b in BINARY_COLS:
        if b in df.columns:
            dfe[b] = _to_float(df[b]).fillna(0).astype(int)
        else:
            dfe[b] = 0

    if "time" in df.columns:
        dfe["time_ord"] = _ord_lookup(df["time"], TIME_ORDER)
    if "expiration" in df.columns:
        dfe["expiration_ord"] = _ord_lookup(df["expiration"], EXPIRATION_ORDER)

    dfe["distance_level"] = dfe["toCoupon_GEQ15min"] + dfe["toCoupon_GEQ25min"]

    freq_ord_cols = [c + "_ord" for c in FREQ_COLS]
    dfe["freq_mean"] = dfe[freq_ord_cols].mean(axis=1)
    dfe["freq_sum"] = dfe[freq_ord_cols].sum(axis=1)

    affinity = pd.Series(np.nan, index=df.index)
    if "coupon" in df.columns:
        lower_coupon = df["coupon"].apply(
            lambda x: str(x).strip().lower() if not pd.isna(x) else None
        )
        venue_of_row = lower_coupon.map(COUPON_VENUE)
        for venue in venue_of_row.dropna().unique():
            col = venue + "_ord"
            if col in dfe.columns:
                mask = venue_of_row == venue
                affinity[mask] = dfe.loc[mask, col]
    dfe["coupon_venue_affinity"] = affinity
    dfe["coupon_venue_affinity_rel"] = dfe["coupon_venue_affinity"] - dfe["freq_mean"]
    dfe["coupon_venue_high"] = (dfe["coupon_venue_affinity"] >= 3).astype(float)
    dfe["coupon_venue_never"] = (dfe["coupon_venue_affinity"] <= 0).astype(float)

    if "age_ord" in dfe.columns and "has_children" in dfe.columns:
        dfe["age_x_children"] = dfe["age_ord"] * dfe["has_children"]
    if "income_ord" in dfe.columns and "expiration_ord" in dfe.columns:
        dfe["income_x_expiration"] = dfe["income_ord"] * dfe["expiration_ord"]

    for c in NOMINAL_COLS:
        if c in df.columns:
            dfe[c] = df[c].fillna("__missing__").astype(str).str.strip()
        else:
            dfe[c] = "__missing__"

    dfe["coupon_time"] = dfe["coupon"] + "|" + dfe["time"]
    dfe["coupon_expiration"] = dfe["coupon"] + "|" + dfe["expiration"]
    if "temperature" in df.columns:
        dfe["coupon_temp"] = dfe["coupon"] + "|" + df["temperature"].astype(str)
    else:
        dfe["coupon_temp"] = dfe["coupon"]
    return dfe


def load_data(data_dir, train=True):
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    schema = {}
    schema_path = os.path.join(data_dir, "schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema = json.load(f)
    labels = None
    if train:
        labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
        label_col = "label" if "label" in labels_df.columns else labels_df.columns[0]
        labels = labels_df[label_col].values
    return features, labels, schema


class CouponFeatureEngineer:
    """Fit learns only imputation values + column order from training rows."""

    def __init__(self):
        self.feature_order_ = None
        self.num_cols_ = None
        self.cat_cols_ = None
        self.fill_values_ = {}

    def fit(self, df):
        dfe = engineer_features(df)
        self.feature_order_ = list(dfe.columns)
        cat_candidates = set(NOMINAL_COLS) | {
            "coupon_time",
            "coupon_expiration",
            "coupon_temp",
        }
        self.cat_cols_ = [c for c in self.feature_order_ if c in cat_candidates]
        self.num_cols_ = [c for c in self.feature_order_ if c not in self.cat_cols_]
        self.fill_values_ = {}
        for c in self.num_cols_:
            med = pd.to_numeric(dfe[c], errors="coerce").median()
            self.fill_values_[c] = 0.0 if pd.isna(med) else float(med)
        return self

    def transform(self, df):
        if self.feature_order_ is None:
            raise RuntimeError("CouponFeatureEngineer must be fit before transform.")
        dfe = engineer_features(df)
        for c in self.feature_order_:
            if c not in dfe.columns:
                dfe[c] = np.nan if c in self.num_cols_ else "__missing__"
        dfe = dfe[self.feature_order_].copy()
        for c in self.num_cols_:
            dfe[c] = pd.to_numeric(dfe[c], errors="coerce").fillna(self.fill_values_[c])
        for c in self.cat_cols_:
            dfe[c] = dfe[c].fillna("__missing__").astype(str)
        return dfe


class TargetEncoder:
    """Leakage-safe smoothed target encoding of *combinations* of columns.

    The prior (global positive rate) and per-key statistics are learned ONLY
    on the rows passed to ``fit``. Transform never touches targets.
    """

    def __init__(self, specs, smoothing=TE_SMOOTHING):
        self.specs = [list(s) for s in specs]
        self.smoothing = float(smoothing)
        self.prior_ = 0.5
        self.maps_ = []  # list of dicts, one per spec

    @staticmethod
    def _keys(df, spec):
        n = len(df)
        if not spec:
            return pd.Series(["__all__"] * n, index=df.index)
        parts = []
        for c in spec:
            if c in df.columns:
                parts.append(df[c].astype(str))
            else:
                parts.append(pd.Series(["__missing__"] * n, index=df.index))
        key = parts[0]
        for p in parts[1:]:
            key = key + "|" + p
        return key

    def fit(self, df, y):
        yv = np.asarray(y, dtype=np.float64).ravel()
        self.prior_ = float(yv.mean()) if len(yv) else 0.5
        self.maps_ = []
        m = self.smoothing
        for spec in self.specs:
            key = self._keys(df, spec)
            agg = pd.DataFrame({"k": key.values, "y": yv}).groupby("k")["y"]
            s = agg.sum()
            c = agg.count()
            enc = (s + self.prior_ * m) / (c + m)
            self.maps_.append(enc.to_dict())
        return self

    def transform(self, df):
        n = len(df)
        out = np.empty((n, len(self.specs)), dtype=np.float32)
        for j, spec in enumerate(self.specs):
            key = self._keys(df, spec)
            mp = self.maps_[j] if j < len(self.maps_) else {}
            out[:, j] = key.map(lambda v: mp.get(v, self.prior_)).to_numpy(
                dtype=np.float32
            )
        return out


# ===========================================================================
# 2. MODEL
# ===========================================================================
class CategoricalIndexer:
    def __init__(self):
        self.maps_ = {}
        self.cat_cols_ = []
        self.cardinalities_ = []

    def fit(self, df, cat_cols):
        self.cat_cols_ = [c for c in cat_cols if c in df.columns]
        self.maps_ = {}
        self.cardinalities_ = []
        for c in self.cat_cols_:
            vals = sorted(pd.unique(df[c].fillna("__missing__").astype(str)))
            self.maps_[c] = {v: i for i, v in enumerate(vals)}
            self.cardinalities_.append(len(vals) + 1)
        return self

    def transform(self, df):
        n = len(df)
        out = np.zeros((n, len(self.cat_cols_)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols_):
            m = self.maps_[c]
            unk = len(m)
            col = (
                df[c].fillna("__missing__").astype(str)
                if c in df.columns
                else pd.Series(["__missing__"] * n)
            )
            out[:, j] = col.map(lambda v: m.get(v, unk)).to_numpy(dtype=np.int64)
        return torch.from_numpy(out)

    def transform_np(self, df):
        if not self.cat_cols_:
            return np.zeros((len(df), 0), dtype=np.float64)
        return self.transform(df).numpy().astype(np.float64)


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, xl):
        return x0 * (xl @ self.w) + self.b + xl


class AttnBlock(nn.Module):
    """Pre-LN transformer block over per-feature embedding tokens."""

    def __init__(self, dim, n_heads=4, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class CouponDCNMoE(nn.Module):
    def __init__(
        self,
        n_num,
        cat_cardinalities,
        emb_dim=16,
        n_cross=2,
        hidden=(256, 128),
        n_experts=4,
        expert_hidden=64,
        dropout=0.25,
        n_heads=4,
        n_attn_layers=2,
    ):
        super().__init__()
        self.n_num = int(n_num)
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        self.n_cat = len(self.cat_cardinalities)
        self.emb_dim = emb_dim
        self.n_heads = max(1, int(n_heads))
        # emb_dim must be divisible by n_heads
        while self.emb_dim % self.n_heads != 0:
            self.n_heads -= 1
        self.n_heads = max(1, self.n_heads)

        if self.n_cat > 0:
            self.embeddings = nn.ModuleList(
                [nn.Embedding(c, emb_dim) for c in self.cat_cardinalities]
            )
            for emb in self.embeddings:
                nn.init.normal_(emb.weight, std=0.05)
            self.emb_dropout = nn.Dropout(dropout)
        else:
            self.embeddings = None

        d = self.n_num + emb_dim * self.n_cat
        self.d = max(d, 1)
        self.num_norm = nn.LayerNorm(self.n_num) if self.n_num > 0 else nn.Identity()
        self.cross_layers = nn.ModuleList([CrossLayer(self.d) for _ in range(n_cross)])

        # --- attention mixer ---
        self.num_token_proj = nn.Linear(self.n_num, emb_dim) if self.n_num > 0 else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.attn_blocks = nn.ModuleList(
            [AttnBlock(emb_dim, self.n_heads, dropout) for _ in range(n_attn_layers)]
        )
        self.attn_norm = nn.LayerNorm(emb_dim)

        tower_in = 2 * self.d + emb_dim
        tower = []
        prev = tower_in
        for h in hidden:
            tower += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.deep = nn.Sequential(*tower)

        self.n_experts = int(n_experts)
        self.gate = nn.Linear(prev, self.n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(prev, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, 1),
                )
                for _ in range(self.n_experts)
            ]
        )
        self.shared_head = nn.Linear(prev, 1)
        self._init_outputs()

    def _init_outputs(self):
        for m in (self.gate, self.shared_head):
            nn.init.zeros_(m.bias)
            nn.init.normal_(m.weight, std=0.01)
        for expert in self.experts:
            last = expert[-1]
            nn.init.zeros_(last.bias)
            nn.init.normal_(last.weight, std=0.01)

    def _attn_repr(self, xn, xc):
        tokens = []
        if self.num_token_proj is not None and xn is not None and xn.shape[1] > 0:
            tokens.append(self.num_token_proj(xn).unsqueeze(1))
        if self.embeddings is not None and xc is not None:
            tokens.append(
                torch.stack([e(xc[:, j]) for j, e in enumerate(self.embeddings)], dim=1)
            )
        if not tokens:
            return None
        toks = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(toks.shape[0], -1, -1)
        x = torch.cat([cls, toks], dim=1)
        for blk in self.attn_blocks:
            x = blk(x)
        x = self.attn_norm(x)
        # mean-pool the non-CLS tokens plus the CLS token
        pooled = torch.cat([x[:, 0], x[:, 1:].mean(dim=1)], dim=1)
        return pooled

    def forward(self, x_num, x_cat=None):
        parts = []
        xnum_norm = None
        if self.n_num > 0:
            xnum_norm = self.num_norm(x_num)
            parts.append(xnum_norm)
        if self.embeddings is not None and x_cat is not None:
            emb = torch.cat(
                [e(x_cat[:, j]) for j, e in enumerate(self.embeddings)], dim=1
            )
            parts.append(self.emb_dropout(emb))
        if len(parts) == 0:
            raise RuntimeError("CouponDCNMoE received no input features.")
        x0 = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)

        attn = self._attn_repr(x_num if self.n_num > 0 else None, x_cat)
        if attn is None:
            attn = torch.zeros((x0.shape[0], self.emb_dim), device=x0.device)
        h = self.deep(torch.cat([x0, xl, attn], dim=1))

        shared = self.shared_head(h)
        gates = torch.softmax(self.gate(h), dim=1)
        expert_out = torch.cat([e(h) for e in self.experts], dim=1)
        mixed = (gates * expert_out).sum(dim=1, keepdim=True)
        return (shared + mixed).squeeze(1)


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing = float(smoothing)
        self.pos_weight = None if pos_weight is None else float(pos_weight)

    def forward(self, logits, target):
        t = target.float().view(-1)
        if self.smoothing > 0:
            t = t * (1.0 - self.smoothing) + 0.5 * self.smoothing
        pw = (
            None
            if self.pos_weight is None
            else torch.tensor(self.pos_weight, device=logits.device)
        )
        return F.binary_cross_entropy_with_logits(logits.view(-1), t, pos_weight=pw)


def build_optimizer_and_scheduler(
    model, lr=2e-3, weight_decay=1e-4, total_epochs=40, eta_min_ratio=0.02
):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() <= 1 or "embeddings" in name or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(total_epochs)), eta_min=lr * eta_min_ratio
    )
    return optimizer, scheduler


def build_model_and_optimizer(
    n_num,
    cat_cardinalities,
    emb_dim=16,
    n_cross=2,
    hidden=(256, 128),
    n_experts=4,
    expert_hidden=64,
    dropout=0.25,
    n_heads=4,
    n_attn_layers=2,
    lr=2e-3,
    weight_decay=1e-4,
    total_epochs=40,
    label_smoothing=0.05,
    pos_weight=None,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CouponDCNMoE(
        n_num=n_num,
        cat_cardinalities=cat_cardinalities,
        emb_dim=emb_dim,
        n_cross=n_cross,
        hidden=hidden,
        n_experts=n_experts,
        expert_hidden=expert_hidden,
        dropout=dropout,
        n_heads=n_heads,
        n_attn_layers=n_attn_layers,
    )
    criterion = LabelSmoothingBCE(smoothing=label_smoothing, pos_weight=pos_weight)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=lr, weight_decay=weight_decay, total_epochs=total_epochs
    )
    return model, criterion, optimizer, scheduler


# ===========================================================================
# 3. TRAINING / EVALUATION HELPERS
# ===========================================================================
def _num_matrix(fe, te, df_out):
    if fe.num_cols_:
        base = df_out[fe.num_cols_].to_numpy(np.float32)
    else:
        base = np.zeros((len(df_out), 0), dtype=np.float32)
    if te is not None and len(te.specs) > 0:
        ten = te.transform(df_out)
        return np.hstack([base, ten]).astype(np.float32)
    return base


def _to_tensors(fe, idx, te, df_out):
    Xnum = _num_matrix(fe, te, df_out)
    Xn = torch.from_numpy(Xnum)
    Xc = idx.transform(df_out) if idx.cat_cols_ else None
    return Xn, Xc


def _build_gbdt_matrix(fe, idx, te, df_out):
    """Numeric + TE cols + ordinal category codes as single float matrix."""
    Xn = _num_matrix(fe, te, df_out).astype(np.float64)
    Xc = idx.transform_np(df_out)
    X = np.hstack([Xn, Xc]).astype(np.float64)
    n_num_cols = Xn.shape[1]
    n_cat_cols = Xc.shape[1]
    if n_cat_cols == 0:
        return X, None
    cat_mask = np.array([False] * n_num_cols + [True] * n_cat_cols)
    return X, cat_mask


def _fit_gbdt(Xtr, ytr, cat_mask, seed):
    params = dict(GBDT_PARAMS)
    params["random_state"] = seed
    if cat_mask is not None:
        params["categorical_features"] = cat_mask
    gb = HistGradientBoostingClassifier(**params)
    gb.fit(Xtr, ytr.astype(int))
    return gb


@torch.no_grad()
def _ensemble_probs(models, Xn, Xc, bs=2048):
    n = len(Xn)
    out = np.zeros(n, dtype=np.float64)
    for start in range(0, n, bs):
        end = min(start + bs, n)
        xn = Xn[start:end]
        xc = Xc[start:end] if Xc is not None else None
        p = torch.zeros(end - start)
        for m in models:
            m.eval()
            p += torch.sigmoid(m(xn, xc))
        out[start:end] = (p / len(models)).numpy()
    return out


def _train_loop(
    model,
    crit,
    opt,
    sched,
    Xn,
    Xc,
    y,
    val=None,
    epochs=MAX_EPOCHS,
    patience=PATIENCE,
    bs=BATCH,
    verbose=True,
):
    n = len(Xn)
    best_acc = -1.0
    best_state = None
    best_ep = epochs
    wait = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xn = Xn[idx]
            xc = Xc[idx] if Xc is not None else None
            yb = y[idx]
            opt.zero_grad()
            logits = model(xn, xc)
            loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()

        if val is not None:
            p = _ensemble_probs([model], val[0], val[1])
            acc = float(((p >= 0.5).astype(int) == val[2].numpy().astype(int)).mean())
            if verbose:
                print(
                    f"  epoch {ep + 1:02d} loss {tot / max(nb, 1):.4f} val_acc {acc:.4f}"
                )
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
                best_ep = ep + 1
                wait = 0
            else:
                wait += 1
                if patience is not None and wait >= patience:
                    break
        else:
            if verbose:
                print(f"  epoch {ep + 1:02d} loss {tot / max(nb, 1):.4f}")

    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    return (best_acc if best_acc >= 0 else None), best_ep


def _build_models_from_bundle(bundle):
    cfg = bundle["model_cfg"]
    models = []
    for sd in bundle["state_dicts"]:
        m = CouponDCNMoE(
            n_num=bundle["n_num"],
            cat_cardinalities=bundle["cat_cardinalities"],
            emb_dim=cfg["emb_dim"],
            n_cross=cfg["n_cross"],
            hidden=cfg["hidden"],
            n_experts=cfg["n_experts"],
            expert_hidden=cfg["expert_hidden"],
            dropout=cfg["dropout"],
            n_heads=cfg.get("n_heads", 4),
            n_attn_layers=cfg.get("n_attn_layers", 2),
        )
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models


def _py(x):
    try:
        return x.item()
    except Exception:
        return x


# ===========================================================================
# 4. TRAIN ENTRY
# ===========================================================================
def run_train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    features, labels, schema = load_data(data_dir, train=True)
    y_raw = np.asarray(labels).ravel()

    classes = list(np.unique(y_raw))
    neg_label = classes[0]
    pos_label = classes[-1]
    y_bin = (y_raw == pos_label).astype(np.float32)
    y_t = torch.from_numpy(y_bin)
    n = len(features)

    print(f"train rows={n} pos_rate={y_bin.mean():.4f} classes={classes}")

    # ---------------- leakage-safe stratified CV (train-only) --------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_nn = np.zeros(n, dtype=np.float64)
    oof_gb = np.zeros(n, dtype=np.float64)
    best_eps = []

    for fold, (tr, va) in enumerate(skf.split(features, y_bin)):
        # Feature engineering + TE fit ONLY on this fold's TRAIN part
        fe = CouponFeatureEngineer().fit(features.iloc[tr])
        tr_out = fe.transform(features.iloc[tr])
        va_out = fe.transform(features.iloc[va])

        te = TargetEncoder(TE_SPECS, TE_SMOOTHING).fit(tr_out, y_bin[tr])
        idx = CategoricalIndexer().fit(tr_out, fe.cat_cols_)

        Xn_tr, Xc_tr = _to_tensors(fe, idx, te, tr_out)
        Xn_va, Xc_va = _to_tensors(fe, idx, te, va_out)
        y_tr, y_va = y_t[tr], y_t[va]

        model, crit, opt, sched = build_model_and_optimizer(
            n_num=len(fe.num_cols_) + len(te.specs),
            cat_cardinalities=idx.cardinalities_,
            **MODEL_CFG,
            seed=SEED + fold,
        )

        print(f"[fold {fold + 1}/{N_FOLDS}] NN")
        best_acc, bep = _train_loop(
            model,
            crit,
            opt,
            sched,
            Xn_tr,
            Xc_tr,
            y_tr,
            val=(Xn_va, Xc_va, y_va),
            epochs=MAX_EPOCHS,
            patience=PATIENCE,
        )
        best_eps.append(bep)
        oof_nn[va] = _ensemble_probs([model], Xn_va, Xc_va)
        print(f"  -> NN fold {fold + 1} best val_acc {best_acc:.4f} (epoch {bep})")

        # ---- GBDT base learner on the identical engineered matrix ----
        Xg_tr, cat_mask = _build_gbdt_matrix(fe, idx, te, tr_out)
        Xg_va, _ = _build_gbdt_matrix(fe, idx, te, va_out)
        gb = _fit_gbdt(Xg_tr, y_tr.numpy(), cat_mask, seed=SEED + fold)
        oof_gb[va] = gb.predict_proba(Xg_va)[:, 1]
        gb_acc = float(
            ((oof_gb[va] >= 0.5).astype(int) == y_bin[va].astype(int)).mean()
        )
        print(f"  -> GBDT fold {fold + 1} val_acc {gb_acc:.4f}")

    nn_acc_only = float(((oof_nn >= 0.5).astype(int) == y_bin.astype(int)).mean())
    gb_acc_only = float(((oof_gb >= 0.5).astype(int) == y_bin.astype(int)).mean())
    print(f"OOF NN-only accuracy  {nn_acc_only:.6f}")
    print(f"OOF GBDT-only accuracy {gb_acc_only:.6f}")

    # ---------------- joint threshold + blend-weight search on OOF --------
    best_w, best_thr, best_acc_oof = 0.5, 0.5, 0.0
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * oof_nn + (1.0 - w) * oof_gb
        for thr in np.linspace(0.30, 0.70, 41):
            acc = float(((blend >= thr).astype(int) == y_bin.astype(int)).mean())
            if acc > best_acc_oof:
                best_acc_oof, best_thr, best_w = acc, float(thr), float(w)
    oof_acc = best_acc_oof
    print(f"OOF blended accuracy {oof_acc:.6f} @ w_nn={best_w:.2f} thr={best_thr:.3f}")

    # ---------------- full refit NN ensemble (all training rows only) -----
    fe_full = CouponFeatureEngineer().fit(features)
    full_out = fe_full.transform(features)
    te_full = TargetEncoder(TE_SPECS, TE_SMOOTHING).fit(full_out, y_bin)
    idx_full = CategoricalIndexer().fit(full_out, fe_full.cat_cols_)
    Xn_all, Xc_all = _to_tensors(fe_full, idx_full, te_full, full_out)

    refit_eps = int(
        np.clip(np.mean(best_eps) if best_eps else MAX_EPOCHS, 8, MAX_EPOCHS)
    )
    print(f"refitting {N_REFIT} NN seeds x {refit_eps} epochs on all training rows")
    state_dicts = []
    for s in range(N_REFIT):
        model, crit, opt, sched = build_model_and_optimizer(
            n_num=len(fe_full.num_cols_) + len(te_full.specs),
            cat_cardinalities=idx_full.cardinalities_,
            **MODEL_CFG,
            seed=SEED + 100 + s,
        )
        _train_loop(
            model,
            crit,
            opt,
            sched,
            Xn_all,
            Xc_all,
            y_t,
            val=None,
            epochs=refit_eps,
            patience=None,
            verbose=(s == 0),
        )
        state_dicts.append({k: v.detach().cpu() for k, v in model.state_dict().items()})

    # ---------------- full refit GBDT (all training rows only) ------------
    print("refitting GBDT on all training rows")
    Xg_all, cat_mask_full = _build_gbdt_matrix(fe_full, idx_full, te_full, full_out)
    gb_full = _fit_gbdt(Xg_all, y_bin, cat_mask_full, seed=SEED + 200)

    # ---------------- persist a complete, self-describing bundle ----------
    bundle = dict(
        fe=fe_full,
        indexer=idx_full,
        target_encoder=te_full,
        model_cfg=MODEL_CFG,
        state_dicts=state_dicts,
        n_num=len(fe_full.num_cols_) + len(te_full.specs),
        cat_cardinalities=idx_full.cardinalities_,
        num_cols=fe_full.num_cols_,
        cat_cols=fe_full.cat_cols_,
        gbdt_model=gb_full,
        gbdt_cat_mask=cat_mask_full,
        blend_w_nn=float(best_w),
        threshold=float(best_thr),
        pos_label=_py(pos_label),
        neg_label=_py(neg_label),
        label_classes=[_py(c) for c in classes],
        oof_accuracy=oof_acc,
    )
    torch.save(bundle, os.path.join(model_dir, "bundle.pt"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "oof_accuracy": oof_acc,
                "nn_only_oof_accuracy": nn_acc_only,
                "gbdt_only_oof_accuracy": gb_acc_only,
                "blend_w_nn": best_w,
                "threshold": best_thr,
                "refit_epochs": refit_eps,
                "n_refit": N_REFIT,
                "n_te_features": len(te_full.specs),
            },
            f,
            indent=2,
        )
    print(f"saved bundle to {model_dir}")
    return oof_acc


# ===========================================================================
# 5. PREDICT ENTRY (fresh process, read-only model dir, no refit)
# ===========================================================================
def run_predict(data_dir, model_dir, output_path):
    bundle = torch.load(
        os.path.join(model_dir, "bundle.pt"), map_location="cpu", weights_only=False
    )
    fe = bundle["fe"]
    idx = bundle["indexer"]
    te = bundle.get("target_encoder", None)
    thr = bundle["threshold"]
    w = bundle.get("blend_w_nn", 1.0)

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    n = len(features)
    out_df = fe.transform(features)
    Xn, Xc = _to_tensors(fe, idx, te, out_df)

    models = _build_models_from_bundle(bundle)
    prob_nn = _ensemble_probs(models, Xn, Xc)

    if "gbdt_model" in bundle:
        Xg, _ = _build_gbdt_matrix(fe, idx, te, out_df)
        prob_gb = bundle["gbdt_model"].predict_proba(Xg)[:, 1]
        prob = w * prob_nn + (1.0 - w) * prob_gb
    else:
        prob = prob_nn

    pred_bin = (prob >= thr).astype(int)

    pos_label = bundle["pos_label"]
    neg_label = bundle["neg_label"]
    pred = np.where(pred_bin == 1, pos_label, neg_label)

    result = pd.DataFrame({"row_id": np.arange(n), "prediction": pred})
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(
        f"wrote {len(result)} predictions to {output_path} "
        f"(pos_rate={float(pred_bin.mean()):.4f}, thr={thr:.3f}, w_nn={w:.2f})"
    )


# ===========================================================================
# 6. CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Coupon acceptance pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.cmd == "train":
        acc = run_train(args.data, args.model)
        print(f"Final Validation Score: {acc}")
    else:
        run_predict(args.data, args.model, args.output)


if __name__ == "__main__":
    main()
