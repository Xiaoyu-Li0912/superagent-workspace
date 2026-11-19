"""
Coupon acceptance prediction — merged single-file solution.

  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv

Leakage-safe: every stateful component (imputer, categorical vocabulary, model
weights, GBDT, blend weight, threshold, label map) is fit ONLY on training rows
(per CV fold) and reloaded unchanged in a fresh predict process.

Ensemble of two structurally different learners:
  * DCN-MoE neural net (embedding + cross + gated MoE head), N_REFIT-seed average
  * HistGradientBoostingClassifier over the same engineered matrix
Blend weight & decision threshold tuned on out-of-fold predictions (accuracy).
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

try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    _HAS_GBDT = True
except Exception:  # noqa: BLE001
    _HAS_GBDT = False

try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import OneHotEncoder

    _HAS_ET = True
except Exception:  # noqa: BLE001
    _HAS_ET = False

torch.set_num_threads(2)

# ===========================================================================
# 0. Configuration
# ===========================================================================
SEED = 42
N_FOLDS = 5
MAX_EPOCHS = 55
PATIENCE = 8
BATCH = 256
N_REFIT = 3

MODEL_CFG = dict(
    emb_dim=8,
    n_cross=2,
    hidden=(256, 128),
    n_experts=4,
    expert_hidden=64,
    dropout=0.30,
    lr=2e-3,
    weight_decay=1e-4,
    total_epochs=MAX_EPOCHS,
    label_smoothing=0.05,
)

GBDT_CFG = dict(
    max_iter=600,
    learning_rate=0.04,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
)


def _make_gbdt(cat_mask, seed):
    """Construct HGB in a version-tolerant way.

    Newer sklearn expects ``categorical_features`` at construction time (a bool
    mask), *not* as a kwarg to ``.fit()``. We try that first and fall back to a
    plain estimator (numeric-only, still valid) if the installed sklearn does
    not support the argument at all.
    """
    try:
        return HistGradientBoostingClassifier(
            **GBDT_CFG, categorical_features=cat_mask, random_state=seed
        )
    except TypeError:
        return HistGradientBoostingClassifier(**GBDT_CFG, random_state=seed)

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

# ---- Target-encoding specifications (leakage-safe smoothed TE) ----
TE_SINGLE_COLS = [
    "coupon",
    "occupation",
    "education",
    "income",
    "age",
    "maritalStatus",
    "gender",
] + FREQ_COLS
TE_INTERACTION_COLS = [
    ("coupon", "occupation"),
    ("coupon", "time"),
    ("coupon", "expiration"),
    ("coupon", "temperature"),
    ("coupon", "income"),
    ("age", "maritalStatus"),
]


def _te_key(df, cols):
    """Build the composite string key for a TE spec from the raw dataframe."""
    out = None
    for c in cols:
        if c in df.columns:
            col = df[c].fillna("__missing__").astype(str)
        else:
            col = pd.Series(["__missing__"] * len(df), index=df.index)
        out = col if out is None else out + "|" + col
    return out


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
        self.base_num_cols_ = None
        self.cat_cols_ = None
        self.fill_values_ = {}
        # target-encoding state (persisted with the bundle)
        self.te_specs_ = []
        self.te_col_names_ = []
        self.te_global_maps_ = None
        self.te_oof_train_ = None
        self.te_prior_ = 0.5
        self.te_m_ = 20.0

    def fit(self, df):
        dfe = engineer_features(df)
        self.feature_order_ = list(dfe.columns)
        cat_candidates = set(NOMINAL_COLS) | {
            "coupon_time",
            "coupon_expiration",
            "coupon_temp",
        }
        self.cat_cols_ = [c for c in self.feature_order_ if c in cat_candidates]
        self.base_num_cols_ = [c for c in self.feature_order_ if c not in self.cat_cols_]
        self.num_cols_ = list(self.base_num_cols_)
        self.fill_values_ = {}
        for c in self.base_num_cols_:
            med = pd.to_numeric(dfe[c], errors="coerce").median()
            self.fill_values_[c] = 0.0 if pd.isna(med) else float(med)
        # TE state is reset until fit_target_encoding is called
        self.te_specs_ = []
        self.te_col_names_ = []
        self.te_global_maps_ = None
        self.te_oof_train_ = None
        return self

    def _te_specs(self):
        specs = [(c,) for c in TE_SINGLE_COLS]
        specs += [tuple(x) for x in TE_INTERACTION_COLS]
        return specs

    def fit_target_encoding(self, df, y, m=20, inner_folds=5, random_state=42):
        """Fit leakage-safe smoothed target encoding.

        - Builds per-key (count, sum) tables on ALL training rows -> ``te_global_maps_``
          for transforming validation / test rows.
        - Builds row-level OOF encodings on the same training rows via an internal
          StratifiedKFold (a row's own label never contributes to its encoding)
          -> ``te_oof_train_``.
        Only training labels are ever touched.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(df)
        prior = float(y.mean()) if n > 0 else 0.5
        self.te_m_ = float(m)
        self.te_prior_ = prior
        self.te_specs_ = self._te_specs()
        self.te_col_names_ = ["te_" + "_x_".join(s) for s in self.te_specs_]

        # --- global maps (fit on all training rows) ---
        self.te_global_maps_ = []
        for spec in self.te_specs_:
            keys = _te_key(df, list(spec))
            tmp = pd.DataFrame({"k": keys.values, "y": y})
            grp = tmp.groupby("k")["y"].agg(["count", "sum"])
            enc = (grp["sum"] + prior * m) / (grp["count"] + m)
            self.te_global_maps_.append(enc.to_dict())

        # --- out-of-fold encodings for the training rows ---
        oof = np.zeros((n, len(self.te_specs_)), dtype=np.float64)
        yi = (y >= 0.5).astype(int)
        splits = []
        try:
            skf_inner = StratifiedKFold(
                n_splits=int(inner_folds), shuffle=True, random_state=random_state
            )
            splits = list(skf_inner.split(np.zeros(n), yi))
        except Exception:  # noqa: BLE001
            splits = []

        if not splits:
            # Fallback: use global maps for the training rows as well.
            for j, spec in enumerate(self.te_specs_):
                keys = _te_key(df, list(spec))
                mp = self.te_global_maps_[j]
                oof[:, j] = keys.map(
                    lambda kk, _mp=mp, _p=prior: _mp.get(kk, _p)
                ).to_numpy(np.float64)
        else:
            for tr_i, va_i in splits:
                y_tr = y[tr_i]
                p_tr = float(y_tr.mean()) if len(y_tr) else prior
                for j, spec in enumerate(self.te_specs_):
                    keys_tr = _te_key(df.iloc[tr_i], list(spec))
                    tmp = pd.DataFrame({"k": keys_tr.values, "y": y_tr})
                    grp = tmp.groupby("k")["y"].agg(["count", "sum"])
                    enc = ((grp["sum"] + p_tr * m) / (grp["count"] + m)).to_dict()
                    keys_va = _te_key(df.iloc[va_i], list(spec))
                    oof[va_i, j] = keys_va.map(
                        lambda kk, _e=enc, _p=p_tr: _e.get(kk, _p)
                    ).to_numpy(np.float64)

        self.te_oof_train_ = oof
        # expose TE as extra numeric block (existing numerics preserved)
        self.num_cols_ = list(self.base_num_cols_) + list(self.te_col_names_)
        return self

    def _append_te(self, raw_df, dfe):
        for j, spec in enumerate(self.te_specs_):
            keys = _te_key(raw_df, list(spec))
            mp = self.te_global_maps_[j]
            prior = self.te_prior_
            dfe[self.te_col_names_[j]] = keys.map(
                lambda kk, _mp=mp, _p=prior: _mp.get(kk, _p)
            ).to_numpy(np.float64)
        return dfe

    def transform_target_encoding(self, df):
        """Append TE columns computed from saved maps (NEVER fit here)."""
        if getattr(self, "te_global_maps_", None) is None or not self.te_col_names_:
            return df
        return self._append_te(df, df.copy())

    def transform(self, df):
        if self.feature_order_ is None:
            raise RuntimeError("CouponFeatureEngineer must be fit before transform.")
        dfe = engineer_features(df)
        for c in self.feature_order_:
            if c not in dfe.columns:
                dfe[c] = np.nan if c in self.num_cols_ else "__missing__"
        dfe = dfe[self.feature_order_].copy()
        for c in self.base_num_cols_:
            dfe[c] = pd.to_numeric(dfe[c], errors="coerce").fillna(self.fill_values_[c])
        for c in self.cat_cols_:
            dfe[c] = dfe[c].fillna("__missing__").astype(str)
        if getattr(self, "te_global_maps_", None) is not None and self.te_col_names_:
            dfe = self._append_te(df, dfe)
        return dfe


# ===========================================================================
# 2. MODEL (neural base learner)
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


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, xl):
        return x0 * (xl @ self.w) + self.b + xl


class CouponDCNMoE(nn.Module):
    def __init__(
        self,
        n_num,
        cat_cardinalities,
        emb_dim=8,
        n_cross=2,
        hidden=(256, 128),
        n_experts=4,
        expert_hidden=64,
        dropout=0.25,
    ):
        super().__init__()
        self.n_num = int(n_num)
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        self.n_cat = len(self.cat_cardinalities)
        self.emb_dim = emb_dim

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

        tower = []
        prev = 2 * self.d
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

    def forward(self, x_num, x_cat=None):
        parts = []
        if self.n_num > 0:
            parts.append(self.num_norm(x_num))
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
        h = self.deep(torch.cat([x0, xl], dim=1))
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
    emb_dim=8,
    n_cross=2,
    hidden=(256, 128),
    n_experts=4,
    expert_hidden=64,
    dropout=0.25,
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
    )
    criterion = LabelSmoothingBCE(smoothing=label_smoothing, pos_weight=pos_weight)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=lr, weight_decay=weight_decay, total_epochs=total_epochs
    )
    return model, criterion, optimizer, scheduler


# ===========================================================================
# 3. TRAINING / EVALUATION HELPERS
# ===========================================================================
def _to_tensors(fe, indexer, df_out):
    if fe.num_cols_:
        Xn = torch.from_numpy(df_out[fe.num_cols_].to_numpy(np.float32))
    else:
        Xn = torch.zeros((len(df_out), 0), dtype=torch.float32)
    Xc = indexer.transform(df_out) if indexer.cat_cols_ else None
    return Xn, Xc


def _gbdt_matrix(fe, indexer, df_out):
    """Same information as _to_tensors, but layout for HistGradientBoosting:
    numeric columns first (float), then ordinal codes of categoricals, with a
    boolean mask declaring the categorical block."""
    n = len(df_out)
    if fe.num_cols_:
        Xn = df_out[fe.num_cols_].to_numpy(np.float32)
    else:
        Xn = np.zeros((n, 0), dtype=np.float32)
    if indexer.cat_cols_:
        Xc = indexer.transform(df_out).numpy().astype(np.float32)
        X = np.concatenate([Xn, Xc], axis=1)
        mask = np.concatenate(
            [np.zeros(Xn.shape[1], dtype=bool), np.ones(Xc.shape[1], dtype=bool)]
        )
    else:
        X = Xn
        mask = np.zeros(X.shape[1], dtype=bool)
    return X, mask


# ---- one-hot view for the ExtraTrees member (no TE, no ordinal ordering) ----
_ET_CAT_POOL = (
    [
        "coupon",
        "time",
        "gender",
        "maritalStatus",
        "occupation",
        "expiration",
        "coupon_time",
        "coupon_expiration",
        "coupon_temp",
    ]
    + [c + "_ord" for c in FREQ_COLS]
    + ["age_ord", "income_ord", "education_ord", "temperature"]
)


class OneHotView:
    """Persistable one-hot view over raw categoricals + binned ordinals.

    Deliberately excludes every ``te_*`` column (no label-derived smoothing) and
    the remaining ordinal numerics (no integer-ordering semantics), so a tree
    learner here splits on discrete indicator conjunctions that the ordinal and
    smoothed-TE encodings blur.
    """

    def __init__(self):
        self.cols_ = []
        self.categories_ = []
        self.encoder_ = None
        self.num_cols_ = []

    @staticmethod
    def _default_cat_cols(df_out):
        return [
            c
            for c in _ET_CAT_POOL
            if c in df_out.columns and not str(c).startswith("te_")
        ]

    def _str_frame(self, df_out):
        n = len(df_out)
        data = {}
        for c in self.cols_:
            if c not in df_out.columns:
                data[c] = pd.Series(["__missing__"] * n, index=df_out.index)
                continue
            if c == "temperature":
                v = pd.to_numeric(df_out[c], errors="coerce")
                data[c] = v.apply(
                    lambda x: "__missing__" if pd.isna(x) else str(int(round(float(x))))
                )
            else:
                data[c] = (
                    df_out[c].fillna("__missing__").astype(str).str.strip()
                )
        return pd.DataFrame(data, index=df_out.index)

    def fit(self, df_out, cat_cols=None, num_cols=None):
        if cat_cols is None:
            cat_cols = self._default_cat_cols(df_out)
        if num_cols is None:
            num_cols = [c for c in BINARY_COLS if c in df_out.columns]
        self.cols_ = list(cat_cols)
        self.num_cols_ = list(num_cols)
        Xcat = self._str_frame(df_out)
        try:
            self.encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float32
            )
        except TypeError:  # older sklearn uses ``sparse``
            self.encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse=False, dtype=np.float32
            )
        self.encoder_.fit(Xcat)
        self.categories_ = list(self.encoder_.categories_)
        return self

    def transform(self, df_out):
        """Return the one-hot block only (never fits)."""
        if self.encoder_ is None:
            raise RuntimeError("OneHotView must be fit before transform.")
        Xcat = self._str_frame(df_out)
        return np.asarray(
            self.encoder_.transform(Xcat), dtype=np.float32
        )


def _et_matrix(view, df_out):
    """One-hot block + binary numeric indicators in fixed column order."""
    Xoh = view.transform(df_out)
    n = len(df_out)
    if not view.num_cols_:
        return Xoh
    Xnum = np.zeros((n, len(view.num_cols_)), dtype=np.float32)
    for j, c in enumerate(view.num_cols_):
        if c in df_out.columns:
            Xnum[:, j] = (
                pd.to_numeric(df_out[c], errors="coerce")
                .fillna(0.0)
                .to_numpy(np.float32)
            )
    return np.concatenate([Xoh, Xnum], axis=1)


def _fit_et(view, X, y_int, seed):
    """Fit an ExtraTrees on the one-hot view; None if sklearn lacks it."""
    if not _HAS_ET:
        return None
    try:
        et = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=2,
            random_state=seed,
            bootstrap=False,
        )
        et.fit(X, y_int)
        return et
    except ImportError:  # noqa: BLE001
        return None


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
    y_int = y_bin.astype(np.int64)
    y_t = torch.from_numpy(y_bin)
    n = len(features)

    print(f"train rows={n} pos_rate={y_bin.mean():.4f} classes={classes}")

    # ---------------- leakage-safe stratified CV (train-only) --------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_nn = np.zeros(n, dtype=np.float64)
    oof_gbdt = np.zeros(n, dtype=np.float64)
    oof_et = np.zeros(n, dtype=np.float64)
    has_et_oof = False
    best_eps = []
    gbdt_mask = None  # filled on first fold; identical for all folds

    for fold, (tr, va) in enumerate(skf.split(features, y_bin)):
        fe = CouponFeatureEngineer().fit(features.iloc[tr])
        # Leakage-safe TE: fit maps + OOF encodings using fold-training rows only.
        fe.fit_target_encoding(
            features.iloc[tr], y_bin[tr], m=20, random_state=SEED + fold
        )
        tr_out = fe.transform(features.iloc[tr])
        va_out = fe.transform(features.iloc[va])
        # Training rows must see out-of-fold encodings (never their own label).
        for j, cn in enumerate(fe.te_col_names_):
            tr_out[cn] = fe.te_oof_train_[:, j]
        idx = CategoricalIndexer().fit(tr_out, fe.cat_cols_)

        Xn_tr, Xc_tr = _to_tensors(fe, idx, tr_out)
        Xn_va, Xc_va = _to_tensors(fe, idx, va_out)
        y_tr, y_va = y_t[tr], y_t[va]

        model, crit, opt, sched = build_model_and_optimizer(
            n_num=len(fe.num_cols_),
            cat_cardinalities=idx.cardinalities_,
            **MODEL_CFG,
            seed=SEED + fold,
        )

        print(f"[fold {fold + 1}/{N_FOLDS}]")
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
        print(f"  -> fold {fold + 1} NN val_acc {best_acc:.4f} (epoch {bep})")

        # ---- GBDT base learner on the identical engineered matrix ----------
        if _HAS_GBDT:
            if gbdt_mask is None:
                _, gbdt_mask = _gbdt_matrix(fe, idx, tr_out)
            Xg_tr, _ = _gbdt_matrix(fe, idx, tr_out)
            Xg_va, _ = _gbdt_matrix(fe, idx, va_out)
            gbdt = _make_gbdt(gbdt_mask, SEED + fold)
            gbdt.fit(Xg_tr, y_int[tr])
            oof_gbdt[va] = gbdt.predict_proba(Xg_va)[:, 1]
            g_acc = float(((oof_gbdt[va] >= 0.5).astype(int) == y_int[va]).mean())
            print(f"  -> fold {fold + 1} GBDT val_acc {g_acc:.4f}")

        # ---- ExtraTrees on a one-hot view (fit on fold-train rows only) ----
        if _HAS_ET:
            view = OneHotView().fit(tr_out)
            Xet_tr = _et_matrix(view, tr_out)
            Xet_va = _et_matrix(view, va_out)
            et = _fit_et(view, Xet_tr, y_int[tr], SEED + fold)
            if et is not None:
                oof_et[va] = et.predict_proba(Xet_va)[:, 1]
                has_et_oof = True
                e_acc = float(
                    ((oof_et[va] >= 0.5).astype(int) == y_int[va]).mean()
                )
                print(f"  -> fold {fold + 1} ET val_acc {e_acc:.4f}")

    # ---------------- blend + threshold tuning on OOF (official metric) ----
    has_gbdt = _HAS_GBDT
    has_et = bool(_HAS_ET and has_et_oof)
    best_thr, best_acc_oof = 0.5, -1.0
    best_w_nn, best_w_gbdt, best_w_et = 1.0, 0.0, 0.0
    thresholds = np.linspace(0.25, 0.75, 101)
    steps = 20  # 0.05 granularity simplex grid

    def _scan_simplex(cands):
        nonlocal best_thr, best_acc_oof
        nonlocal best_w_nn, best_w_gbdt, best_w_et
        for wn, wg, we in cands:
            p = wn * oof_nn
            if has_gbdt and wg > 0.0:
                p = p + wg * oof_gbdt
            if has_et and we > 0.0:
                p = p + we * oof_et
            for thr in thresholds:
                acc = float(((p >= thr).astype(int) == y_int).mean())
                if acc > best_acc_oof:
                    best_acc_oof, best_thr = acc, float(thr)
                    best_w_nn, best_w_gbdt, best_w_et = (
                        float(wn),
                        float(wg),
                        float(we),
                    )

    if has_gbdt and has_et:
        cands = [
            (i / steps, j / steps, (steps - i - j) / steps)
            for i in range(steps + 1)
            for j in range(steps + 1 - i)
        ]
    elif has_gbdt:
        cands = [(i / steps, 1.0 - i / steps, 0.0) for i in range(steps + 1)]
    elif has_et:
        cands = [(i / steps, 0.0, 1.0 - i / steps) for i in range(steps + 1)]
    else:
        cands = [(1.0, 0.0, 0.0)]
    _scan_simplex(cands)
    oof_acc = best_acc_oof
    print(
        f"OOF accuracy {oof_acc:.6f} | w_nn={best_w_nn:.2f} "
        f"w_gbdt={best_w_gbdt:.2f} w_et={best_w_et:.2f} thr={best_thr:.3f}"
    )

    # ---------------- full refit (all training rows only) ------------------
    fe_full = CouponFeatureEngineer().fit(features)
    # Refit TE maps on ALL training rows (labels of training rows only).
    fe_full.fit_target_encoding(features, y_bin, m=20, random_state=SEED)
    full_out = fe_full.transform(features)
    # Full-model training rows see out-of-fold encodings (no self-leak); the
    # persisted global maps are what ``run_predict`` will use for unseen rows.
    for j, cn in enumerate(fe_full.te_col_names_):
        full_out[cn] = fe_full.te_oof_train_[:, j]
    idx_full = CategoricalIndexer().fit(full_out, fe_full.cat_cols_)
    Xn_all, Xc_all = _to_tensors(fe_full, idx_full, full_out)

    refit_eps = int(
        np.clip(np.mean(best_eps) if best_eps else MAX_EPOCHS, 8, MAX_EPOCHS)
    )
    print(f"refitting {N_REFIT} seeds x {refit_eps} epochs on all training rows")
    state_dicts = []
    for s in range(N_REFIT):
        model, crit, opt, sched = build_model_and_optimizer(
            n_num=len(fe_full.num_cols_),
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

    gbdt_full = None
    gbdt_mask_full = None
    if _HAS_GBDT:
        Xg_all, gbdt_mask_full = _gbdt_matrix(fe_full, idx_full, full_out)
        gbdt_full = _make_gbdt(gbdt_mask_full, SEED + 777)
        gbdt_full.fit(Xg_all, y_int)
        print("refit GBDT on all training rows")

    et_view_full = None
    et_model_full = None
    et_cat_cols_full = []
    if _HAS_ET:
        # ET ignores TE columns, so ordering w.r.t. the TE overwrite above is moot.
        et_view_full = OneHotView().fit(full_out)
        Xet_all = _et_matrix(et_view_full, full_out)
        et_model_full = _fit_et(et_view_full, Xet_all, y_int, SEED + 900)
        et_cat_cols_full = list(et_view_full.cols_)
        print("refit ExtraTrees on all training rows")

    # ---------------- persist a complete, self-describing bundle ----------
    bundle = dict(
        fe=fe_full,
        indexer=idx_full,
        model_cfg=MODEL_CFG,
        state_dicts=state_dicts,
        n_num=len(fe_full.num_cols_),
        cat_cardinalities=idx_full.cardinalities_,
        num_cols=fe_full.num_cols_,
        cat_cols=fe_full.cat_cols_,
        te_col_names=list(getattr(fe_full, "te_col_names_", [])),
        gbdt=gbdt_full,
        gbdt_cat_mask=gbdt_mask_full,
        et_view=et_view_full,
        et_model=et_model_full,
        et_cat_cols=et_cat_cols_full,
        blend_w_nn=float(best_w_nn),
        blend_w_gbdt=float(best_w_gbdt),
        blend_w_et=float(best_w_et),
        has_et=bool(et_model_full is not None),
        threshold=float(best_thr),
        pos_label=_py(pos_label),
        neg_label=_py(neg_label),
        label_classes=[_py(c) for c in classes],
        oof_accuracy=oof_acc,
        has_gbdt=bool(_HAS_GBDT),
    )
    torch.save(bundle, os.path.join(model_dir, "bundle.pt"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "oof_accuracy": oof_acc,
                "threshold": best_thr,
                "blend_w_nn": best_w_nn,
                "blend_w_gbdt": best_w_gbdt,
                "blend_w_et": best_w_et,
                "refit_epochs": refit_eps,
                "n_refit": N_REFIT,
                "has_gbdt": bool(_HAS_GBDT),
                "has_et": bool(et_model_full is not None),
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
    thr = bundle["threshold"]
    w_nn = bundle.get("blend_w_nn", 1.0)
    gbdt = bundle.get("gbdt", None)
    has_gbdt = bool(bundle.get("has_gbdt", False)) and gbdt is not None
    et_view = bundle.get("et_view", None)
    et_model = bundle.get("et_model", None)
    has_et = bool(bundle.get("has_et", False)) and (
        et_view is not None and et_model is not None
    )
    if "blend_w_gbdt" in bundle:
        w_gbdt = float(bundle.get("blend_w_gbdt", 0.0))
    else:  # legacy 2-way bundle
        w_gbdt = 1.0 - float(w_nn)
    w_et = float(bundle.get("blend_w_et", 0.0))

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    n = len(features)
    out_df = fe.transform(features)
    Xn, Xc = _to_tensors(fe, idx, out_df)

    models = _build_models_from_bundle(bundle)
    p_nn = _ensemble_probs(models, Xn, Xc)

    components = [(p_nn, w_nn)]
    if has_gbdt:
        Xg, _ = _gbdt_matrix(fe, idx, out_df)
        components.append((gbdt.predict_proba(Xg)[:, 1], w_gbdt))
    if has_et:
        Xet = _et_matrix(et_view, out_df)
        components.append((et_model.predict_proba(Xet)[:, 1], w_et))

    tot_w = float(sum(w for _, w in components))
    if tot_w <= 0.0:
        components = [(components[0][0], 1.0)]
        tot_w = 1.0
    prob = np.zeros(n, dtype=np.float64)
    for p_comp, w_comp in components:
        prob += (w_comp / tot_w) * p_comp

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
        f"(pos_rate={float(pred_bin.mean()):.4f}, thr={thr:.3f}, w_nn={w_nn:.2f})"
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
