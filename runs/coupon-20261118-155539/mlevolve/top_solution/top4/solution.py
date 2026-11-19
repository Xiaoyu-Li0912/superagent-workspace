"""
Merged coupon-acceptance pipeline.

Train:  load train features+labels -> FeatureEngineer (fit on TRAIN only)
        -> per-fold transform -> Field-aware DeepFM (bagged) + TREE ZOO
        -> OOF blend-weight + threshold search -> final fit on all train rows
        -> persist FE + tree zoo + DeepFM state dicts + meta.

Predict: load features + schema -> transform with persisted FE
        -> rebuild DeepFM(s) + tree zoo -> blended probarian -> threshold -> labels
        -> write row_id,prediction.
"""

from __future__ import annotations

import os
import json
import argparse
import pickle
import math
import random
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import StratifiedKFold

torch.set_num_threads(2)
torch.set_grad_enabled(True)


# ==========================================================================
# STEP 1: DATA PROCESSING & FEATURE ENGINEERING
# ==========================================================================
FREQ_MAP = {"never": 0.0, "less1": 1.0, "1~3": 2.0, "4~8": 3.0, "gt8": 4.0}

AGE_MAP = {
    "below21": 0.0,
    "21": 1.0,
    "26": 2.0,
    "31": 3.0,
    "36": 4.0,
    "41": 5.0,
    "46": 6.0,
    "50plus": 7.0,
}
INCOME_MAP = {
    "Less than $12500": 0.0,
    "$12500 - $24999": 1.0,
    "$25000 - $37499": 2.0,
    "$37500 - $49999": 3.0,
    "$50000 - $62499": 4.0,
    "$62500 - $74999": 5.0,
    "$75000 - $87499": 6.0,
    "$87500 - $99999": 7.0,
    "$100000 or More": 8.0,
}
EDUCATION_MAP = {
    "Some High School": 0.0,
    "High School Graduate": 1.0,
    "Some college - no degree": 2.0,
    "Associates degree": 3.0,
    "Bachelors degree": 4.0,
    "Graduate degree (Masters, Doctorate)": 5.0,
}
TIME_MAP = {"7AM": 0.0, "10AM": 1.0, "2PM": 2.0, "6PM": 3.0, "10PM": 4.0}
EXPIRATION_MAP = {"2h": 0.0, "1d": 1.0}

FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]
HABIT_COLUMN = {
    "Bar": "Bar_num",
    "Coffee House": "CoffeeHouse_num",
    "Restaurant(<20)": "RestaurantLessThan20_num",
    "Restaurant(20-50)": "Restaurant20To50_num",
    "Carry out & Take away": "CarryAway_num",
}
NOMINAL_COLS = ["coupon", "gender", "maritalStatus", "occupation"]
ORDINAL_MAPS = {
    "time": TIME_MAP,
    "expiration": EXPIRATION_MAP,
    "age": AGE_MAP,
    "income": INCOME_MAP,
    "education": EDUCATION_MAP,
}
NUMERIC_COLS = [
    "temperature",
    "has_children",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
]


class FeatureEngineer:
    """Stateful, leakage-safe feature transformer."""

    TE_KEYS = [
        "coupon",
        "occupation",
        "coupon_x_occupation",
        "coupon_x_time",
        "coupon_x_expiration",
        "coupon_x_distance",
        "coupon_x_direction",
        "coupon_x_age",
        "coupon_x_children",
    ]
    TE_COLS = ["te__" + k for k in TE_KEYS]

    CC_COLS = [
        "time_num",
        "expiration_num",
        "distance_bucket",
        "direction_same_num",
        "habit_match",
        "age_num",
        "income_num",
        "has_children_num",
    ]

    def __init__(self, te_smoothing: float = 25.0):
        self.fill_values = {}
        self.nominal_categories = {}
        self.feature_columns = None
        self.te_smoothing = float(te_smoothing)
        self.te_prior = 0.5
        self.te_maps = {}
        self.te_active = False
        self._te_recursion = False
        self.coupon_means = {}
        self.coupon_global_means = {}

    @staticmethod
    def _normalize(s):
        s = s.astype("object")
        out = s.astype(str).str.strip()
        return out.where(~out.isin(["nan", "None", "", "NaN"]), np.nan)

    @staticmethod
    def _col_or_nan(df, name, n):
        if name in df.columns:
            return df[name].reset_index(drop=True)
        return pd.Series([np.nan] * n, dtype="object")

    def fit(self, df, y=None):
        df = df.reset_index(drop=True)
        for c in FREQ_COLS:
            self.fill_values[c] = -1.0
        for c, mapping in ORDINAL_MAPS.items():
            s = self._normalize(self._col_or_nan(df, c, len(df)))
            mapped = s.map(mapping)
            mode = mapped.dropna().mode()
            self.fill_values[c] = float(mode.iloc[0]) if len(mode) else 0.0
        for c in NOMINAL_COLS:
            s = self._normalize(self._col_or_nan(df, c, len(df))).fillna("missing")
            self.nominal_categories[c] = sorted(s.dropna().unique().tolist())
        for c in NUMERIC_COLS:
            v = pd.to_numeric(self._col_or_nan(df, c, len(df)), errors="coerce")
            med = v.dropna().median()
            self.fill_values[c] = float(med) if pd.notna(med) else 0.0

        self.transform(df)  # establish feature_columns

        # Coupon-conditional centering: store per-coupon means over TRAIN rows only.
        n_fit = len(df)
        coupon_arr_fit = (
            self._normalize(self._col_or_nan(df, "coupon", n_fit))
            .fillna("missing")
            .to_numpy()
            .astype(str)
        )
        base_df = self.transform(df)
        self.coupon_means = {}
        self.coupon_global_means = {}
        for c in self.CC_COLS:
            if c not in base_df.columns:
                continue
            vals = base_df[c].to_numpy(np.float64)
            finite = np.isfinite(vals)
            gmean = float(vals[finite].mean()) if finite.any() else 0.0
            self.coupon_global_means[c] = gmean
            per = {}
            for cp in np.unique(coupon_arr_fit):
                mask = coupon_arr_fit == cp
                v = vals[mask]
                v = v[np.isfinite(v)]
                per[str(cp)] = float(v.mean()) if len(v) else gmean
            self.coupon_means[c] = per
        self.feature_columns = None
        self.transform(df)

        if y is not None:
            y_arr = np.asarray(y).astype(np.float64).ravel()
            if len(y_arr) == len(df) and len(y_arr) > 0:
                self.te_prior = float(np.clip(y_arr.mean(), 1e-6, 1.0 - 1e-6))
                m = self.te_smoothing
                keys = self._build_te_frame(df)
                self.te_maps = {}
                for k in self.TE_KEYS:
                    kv = keys[k].astype(str)
                    agg = (
                        pd.DataFrame({"k": kv, "y": y_arr})
                        .groupby("k")["y"]
                        .agg(["sum", "count"])
                    )
                    prior = self.te_prior
                    smoothed = (agg["sum"] + m * prior) / (agg["count"] + m)
                    self.te_maps[k] = {
                        str(idx): (float(smoothed.loc[idx]), int(agg["count"].loc[idx]))
                        for idx in agg.index
                    }
                self.te_active = True
                self.feature_columns = None
                self.transform(df)
        return self

    def _raw_te_components(self, df):
        n = len(df)
        out = self.transform(df)
        coupon_arr = (
            self._normalize(self._col_or_nan(df, "coupon", n))
            .fillna("missing")
            .to_numpy()
            .astype(str)
        )
        occ_arr = (
            self._normalize(self._col_or_nan(df, "occupation", n))
            .fillna("missing")
            .to_numpy()
            .astype(str)
        )
        age_num = out["age_num"].to_numpy()
        age_bucket = np.floor(age_num / 2.0).astype(int).astype(str)
        children = out["has_children_num"].to_numpy().astype(int).astype(str)
        time_v = out["time_num"].to_numpy().astype(str)
        exp_v = out["expiration_num"].to_numpy().astype(str)
        dist_v = out["distance_bucket"].astype(str)
        dir_v = out["direction_same_num"].to_numpy().astype(str)
        return {
            "coupon": coupon_arr,
            "occupation": occ_arr,
            "coupon_x_occupation": coupon_arr + "||" + occ_arr,
            "coupon_x_time": coupon_arr + "||" + time_v,
            "coupon_x_expiration": coupon_arr + "||" + exp_v,
            "coupon_x_distance": coupon_arr + "||" + dist_v,
            "coupon_x_direction": coupon_arr + "||" + dir_v,
            "coupon_x_age": coupon_arr + "||" + age_bucket,
            "coupon_x_children": coupon_arr + "||" + children,
        }

    def _build_te_frame(self, df):
        return self._raw_te_components(df)

    def transform(self, df):
        df = df.reset_index(drop=True)
        n = len(df)
        out = {}

        for c, mapping in ORDINAL_MAPS.items():
            s = self._normalize(self._col_or_nan(df, c, n))
            out[c + "_num"] = (
                s.map(mapping).astype(float).fillna(self.fill_values.get(c, 0.0))
            )

        for c in FREQ_COLS:
            s = self._normalize(self._col_or_nan(df, c, n))
            out[c + "_num"] = (
                s.map(FREQ_MAP).astype(float).fillna(self.fill_values.get(c, -1.0))
            )

        for c in NUMERIC_COLS:
            v = pd.to_numeric(self._col_or_nan(df, c, n), errors="coerce").astype(float)
            out[c + "_num"] = v.fillna(self.fill_values.get(c, 0.0))

        g15 = out["toCoupon_GEQ15min_num"].to_numpy()
        g25 = out["toCoupon_GEQ25min_num"].to_numpy()
        bucket = np.where(g15 < 0.5, 0.0, np.where(g25 < 0.5, 1.0, 2.0))
        out["distance_bucket"] = bucket.astype(float)
        out["direction_and_close"] = out["direction_same_num"].to_numpy() * (
            g15 < 0.5
        ).astype(float)

        coupon_s = self._normalize(self._col_or_nan(df, "coupon", n)).fillna("missing")
        coupon_arr = coupon_s.to_numpy()
        habit_val = np.full(n, -1.0, dtype=float)
        for ctype, hcol in HABIT_COLUMN.items():
            mask = coupon_arr == ctype
            if mask.any():
                habit_val[mask] = out[hcol].to_numpy()[mask]
        out["habit_match"] = habit_val
        out["habit_high"] = (habit_val >= 2.0).astype(float)
        out["habit_missing"] = (habit_val < 0.0).astype(float)

        out["expiry_x_distance"] = (
            out["expiration_num"].to_numpy() * out["distance_bucket"]
        )
        out["has_children_x_age"] = (
            out["has_children_num"].to_numpy() * out["age_num"].to_numpy()
        )

        for col in NOMINAL_COLS:
            cats = self.nominal_categories.get(col, ["missing"])
            s = (
                self._normalize(self._col_or_nan(df, col, n))
                .fillna("missing")
                .to_numpy()
            )
            for c in cats:
                out[f"{col}__{c}"] = (s == c).astype(float)
            out[f"{col}__is_missing"] = (
                self._normalize(self._col_or_nan(df, col, n))
                .isna()
                .astype(float)
                .to_numpy()
            )

        if self.coupon_means:
            coupon_arr_s = coupon_arr.astype(str)
            for c in self.CC_COLS:
                if c not in out:
                    continue
                base = np.asarray(out[c], dtype=np.float64)
                gm = self.coupon_global_means.get(c, 0.0)
                per = self.coupon_means.get(c, {})
                means_arr = np.fromiter(
                    (per.get(k, gm) for k in coupon_arr_s),
                    dtype=np.float64,
                    count=len(coupon_arr_s),
                )
                out["cc__" + c] = (base - means_arr).astype(np.float32)

        if self.te_active and self.te_maps and not getattr(self, "_te_recursion", False):
            # Re-entrancy guard: _raw_te_components -> transform must NOT re-enter
            # the TE block (otherwise infinite recursion).  The inner transform
            # only needs the core (non-TE) features to build the composite keys.
            self._te_recursion = True
            try:
                comp = self._raw_te_components(df)
            finally:
                self._te_recursion = False
            default = float(self.te_prior) if self.te_prior is not None else 0.5
            for te_col, k in zip(self.TE_COLS, self.TE_KEYS):
                mp = self.te_maps.get(k, {})
                kv = comp[k]
                vals = np.fromiter(
                    (mp[kk][0] if kk in mp else default for kk in kv),
                    dtype=np.float32,
                    count=len(kv),
                )
                out[te_col] = vals
                p = np.clip(vals, 1e-6, 1.0 - 1e-6)
                out[te_col + "_logit"] = (np.log(p) - np.log1p(-p)).astype(np.float32)

        result = pd.DataFrame(out)
        if self.feature_columns is None and not getattr(self, "_te_recursion", False):
            self.feature_columns = list(result.columns)
        if self.feature_columns is not None:
            result = result.reindex(columns=self.feature_columns, fill_value=0.0)
        return result.astype(np.float32)


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def load_train(data_dir):
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    schema = load_schema(data_dir)
    y = labels["label"].to_numpy()
    return features, y, schema


def load_predict(data_dir):
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    schema = load_schema(data_dir)
    return features, schema


# ==========================================================================
# STEP 2: MODEL DESIGN
# ==========================================================================
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing: float = 0.02, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = float(smoothing)
        self.register_buffer(
            "pos_weight", pos_weight if pos_weight is not None else torch.tensor(1.0)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, t, pos_weight=self.pos_weight)


class FieldAwareDeepFM(nn.Module):
    def __init__(
        self,
        cont_dim,
        cat_cardinalities,
        emb_dim=8,
        deep_dims=(128, 64),
        dropout=0.20,
        emb_dropout=0.05,
        coupon_field_idx=None,
        gate_dropout=0.1,
    ):
        super().__init__()
        self.cont_dim = int(cont_dim)
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        self.emb_dim = int(emb_dim)
        n_fields = len(self.cat_cardinalities)
        self.coupon_field_idx = (
            int(coupon_field_idx) if coupon_field_idx is not None else None
        )
        if self.coupon_field_idx is not None:
            coupon_card = self.cat_cardinalities[self.coupon_field_idx]
            self.coupon_gate = nn.Embedding(
                coupon_card, n_fields * self.emb_dim
            )
            # Bias gate toward sigmoid ~0.88 (identity-ish) at init.
            nn.init.constant_(self.coupon_gate.weight, 2.0)
            self.gate_dropout = nn.Dropout(float(gate_dropout))
        else:
            self.coupon_gate = None
            self.gate_dropout = None

        self.linear_emb = nn.ModuleList(
            [nn.Embedding(c, 1) for c in self.cat_cardinalities]
        )
        self.cont_linear = nn.Linear(self.cont_dim, 1)
        self.fm_emb = nn.ModuleList(
            [nn.Embedding(c, self.emb_dim) for c in self.cat_cardinalities]
        )
        self.cont_emb = nn.Linear(self.cont_dim, self.emb_dim)
        self.emb_dropout = nn.Dropout(emb_dropout)

        deep_in = self.cont_dim + n_fields * self.emb_dim
        layers: List[nn.Module] = []
        prev = deep_in
        for d in deep_dims:
            layers += [
                nn.Linear(prev, d),
                nn.LayerNorm(d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = d
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for emb in list(self.linear_emb) + list(self.fm_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_cont, x_cat):
        linear = self.cont_linear(x_cont).squeeze(-1)
        for i, emb in enumerate(self.linear_emb):
            linear = linear + emb(x_cat[:, i]).squeeze(-1)
        field_embs = torch.stack(
            [self.fm_emb[i](x_cat[:, i]) for i in range(len(self.fm_emb))], dim=1
        )
        if self.coupon_gate is not None:
            cid = x_cat[:, self.coupon_field_idx]
            gate_logits = self.coupon_gate(cid).view(
                -1, field_embs.size(1), self.emb_dim
            )
            gate = torch.sigmoid(gate_logits)
            if self.gate_dropout is not None:
                gate = self.gate_dropout(gate)
            field_embs = field_embs * gate
        field_embs = self.emb_dropout(field_embs)
        cont_emb = self.cont_emb(x_cont).unsqueeze(1)
        all_embs = torch.cat([field_embs, cont_emb], dim=1)
        sum_e = all_embs.sum(dim=1)
        sq_sum = all_embs.pow(2).sum(dim=1).sum(dim=1)
        fm_term = 0.5 * (sum_e.pow(2).sum(dim=1) - sq_sum)
        deep_in = torch.cat([x_cont, field_embs.flatten(1)], dim=1)
        deep_term = self.deep(deep_in).squeeze(-1)
        return linear + fm_term + deep_term

    @torch.no_grad()
    def predict_proba(self, x_cont, x_cat, batch_size=1024):
        self.eval()
        outs = []
        n = x_cont.size(0)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            logits = self.forward(x_cont[start:end], x_cat[start:end])
            outs.append(torch.sigmoid(logits).cpu().numpy())
        if not outs:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(outs, axis=0)


def build_deepfm(
    cont_dim,
    cat_cardinalities,
    *,
    emb_dim=8,
    deep_dims=(128, 64),
    dropout=0.20,
    emb_dropout=0.05,
    lr=2e-3,
    weight_decay=1e-4,
    epochs=30,
    pos_weight=None,
    label_smoothing=0.02,
    device="cpu",
    coupon_field_idx=None,
    gate_dropout=0.1,
):
    model = FieldAwareDeepFM(
        cont_dim,
        cat_cardinalities,
        emb_dim,
        deep_dims,
        dropout,
        emb_dropout,
        coupon_field_idx=coupon_field_idx,
        gate_dropout=gate_dropout,
    ).to(device)
    pw = None
    if pos_weight is not None and pos_weight > 0:
        pw = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    criterion = LabelSmoothingBCE(smoothing=label_smoothing, pos_weight=pw).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs)), eta_min=lr * 0.05
    )
    return model, criterion, optimizer, scheduler


def build_tree_zoo(random_state: int = 42):
    """Diverse tree ensemble. Includes empty optional slots for lgb/xgb/catboost."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    zoo = [
        (
            "hgb_a",
            HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=random_state,
            ),
        ),
        (
            "hgb_b",
            HistGradientBoostingClassifier(
                max_iter=700,
                learning_rate=0.03,
                max_leaf_nodes=63,
                min_samples_leaf=10,
                l2_regularization=3.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=random_state + 1,
            ),
        ),
        (
            "hgb_c",
            HistGradientBoostingClassifier(
                max_iter=450,
                learning_rate=0.08,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=0.5,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=random_state + 2,
            ),
        ),
    ]
    try:
        import xgboost as xgb

        zoo.append(
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.04,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    min_child_weight=3,
                    tree_method="hist",
                    n_jobs=2,
                    random_state=random_state,
                    verbosity=0,
                    eval_metric="logloss",
                ),
            )
        )
    except Exception as e:
        print(f"[zoo] xgboost unavailable: {e}")
    try:
        import lightgbm as lgb

        zoo.append(
            (
                "lgb",
                lgb.LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.04,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    n_jobs=2,
                    random_state=random_state,
                    verbose=-1,
                ),
            )
        )
    except Exception as e:
        print(f"[zoo] lightgbm unavailable: {e}")
    try:
        import catboost as cb

        zoo.append(
            (
                "cat",
                cb.CatBoostClassifier(
                    iterations=400,
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=3.0,
                    random_seed=random_state,
                    verbose=0,
                    thread_count=2,
                ),
            )
        )
    except Exception as e:
        print(f"[zoo] catboost unavailable: {e}")
    return zoo


def compute_pos_weight(y):
    y = np.asarray(y).astype(np.int64).ravel()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return max(1e-3, n_neg / n_pos)


# ==========================================================================
# STEP 3: TRAINING / EVALUATION / PERSISTENCE / PREDICTION
# ==========================================================================
CATEGORICAL_SOURCE_COLS = ["coupon", "gender", "maritalStatus", "occupation"]


def build_cat_block_spec(fe):
    spec = []
    for col in CATEGORICAL_SOURCE_COLS:
        cats = fe.nominal_categories[col]
        cols = [f"{col}__{c}" for c in cats] + [f"{col}__is_missing"]
        spec.append((col, cols))
    return spec


def find_coupon_field_idx(spec):
    for i, (name, _cols) in enumerate(spec):
        if name == "coupon":
            return i
    return None


def split_cont_cat(df, block_spec):
    cat_cols = set()
    for _, cols in block_spec:
        cat_cols.update(cols)
    cont_cols = [c for c in df.columns if c not in cat_cols]
    x_cont = df[cont_cols].to_numpy(np.float32)
    x_cat = np.stack(
        [df[cols].to_numpy(np.float32).argmax(axis=1) for _, cols in block_spec], axis=1
    ).astype(np.int64)
    cards = [len(cols) for _, cols in block_spec]
    return x_cont, x_cat, cont_cols, cards


def fit_scaler(x):
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def encode_labels(y):
    y = np.asarray(y)
    if y.dtype.kind in "OUS":
        uniq = sorted(set(y.tolist()))
    else:
        uniq = sorted(set(y.astype(np.int64).tolist()))
    mapping = {v: i for i, v in enumerate(uniq)}
    yi = np.array([mapping[v] for v in y.tolist()], dtype=np.int64)
    return yi, uniq


def train_deepfm(
    x_cont,
    x_cat,
    y,
    cards,
    epochs=25,
    bs=256,
    lr=2e-3,
    seed=0,
    val=None,
    patience=4,
    coupon_field_idx=None,
    gate_dropout=0.1,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pos_w = compute_pos_weight(y)
    cont_dim = x_cont.shape[1]
    model, crit, opt, sched = build_deepfm(
        cont_dim=cont_dim,
        cat_cardinalities=cards,
        epochs=epochs,
        lr=lr,
        pos_weight=pos_w,
        device="cpu",
        coupon_field_idx=coupon_field_idx,
        gate_dropout=gate_dropout,
    )
    xc = torch.from_numpy(x_cont.astype(np.float32))
    xn = torch.from_numpy(x_cat.astype(np.int64))
    yt = torch.from_numpy(y.astype(np.float32))
    n = len(y)
    idx = np.arange(n)
    if val is not None:
        xv_c, xv_n, yv = val
        xv_c_t = torch.from_numpy(xv_c.astype(np.float32))
        xv_n_t = torch.from_numpy(xv_n.astype(np.int64))
        best_val = math.inf
        best_state = None
        patience_left = patience
    for ep in range(epochs):
        model.train()
        np.random.shuffle(idx)
        for s in range(0, n, bs):
            b = idx[s : s + bs]
            opt.zero_grad()
            logits = model(xc[b], xn[b])
            loss = crit(logits, yt[b])
            loss.backward()
            opt.step()
        sched.step()
        if val is not None:
            p = model.predict_proba(xv_c_t, xv_n_t)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            vloss = float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                patience_left = patience
            else:
                patience_left -= 1
            if patience_left <= 0:
                break
    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model


def _fit_predict_zoo(zoo, X_tr, y_tr, X_va):
    """Fit every model in zoo on (X_tr,y_tr), return averaged proba on X_va."""
    X_tr = X_tr.astype(np.float32)
    X_va = X_va.astype(np.float32)
    p = np.zeros(len(X_va), dtype=np.float64)
    fitted = []
    for name, mdl in zoo:
        mdl.fit(X_tr, y_tr)
        p += mdl.predict_proba(X_va)[:, 1]
        fitted.append(mdl)
    p /= max(1, len(zoo))
    return p, fitted


def run_oof(features_df, y_int, n_splits=5, seed=42, epochs=14, bag=1):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y_int)), y_int))

    oof_deep = np.zeros(len(y_int), dtype=np.float64)
    oof_tree = np.zeros(len(y_int), dtype=np.float64)

    for fi, (tr, va) in enumerate(splits):
        fe = FeatureEngineer().fit(features_df.iloc[tr], y=y_int[tr])
        spec = build_cat_block_spec(fe)
        df_tr = build_inner_kfold_te(features_df.iloc[tr], y_int[tr], fe, n_splits=5, seed=42)
        df_va = fe.transform(features_df.iloc[va])
        xc_tr, xn_tr, _, cards = split_cont_cat(df_tr, spec)
        xc_va, xn_va, _, _ = split_cont_cat(df_va, spec)

        mean, std = fit_scaler(xc_tr)
        xc_tr_s = (xc_tr - mean) / std
        xc_va_s = (xc_va - mean) / std

        y_va = y_int[va]

        coupon_field_idx = find_coupon_field_idx(spec)
        p_sum = np.zeros(len(va), dtype=np.float64)
        for b in range(bag):
            m = train_deepfm(
                xc_tr_s,
                xn_tr,
                y_int[tr],
                cards,
                epochs=epochs,
                seed=seed + fi * 10 + b,
                val=(xc_va_s, xn_va, y_va.astype(np.float32)),
                coupon_field_idx=coupon_field_idx,
                gate_dropout=float(ARCH["gate_dropout"]),
            )
            p_sum += m.predict_proba(
                torch.from_numpy(xc_va_s.astype(np.float32)),
                torch.from_numpy(xn_va.astype(np.int64)),
            )
        oof_deep[va] = p_sum / bag

        zoo = build_tree_zoo(random_state=seed + fi)
        p_tree, _ = _fit_predict_zoo(
            zoo, df_tr.to_numpy(np.float32), y_int[tr], df_va.to_numpy(np.float32)
        )
        oof_tree[va] = p_tree

        acc_d = ((oof_deep[va] >= 0.5).astype(int) == y_va).mean()
        acc_t = ((oof_tree[va] >= 0.5).astype(int) == y_va).mean()
        print(
            f"fold {fi} deep_acc={acc_d:.4f} tree_acc={acc_t:.4f} "
            f"zoo={[n for n, _ in zoo]}"
        )

    return oof_deep, oof_tree


def search_blend_threshold(oof_deep, oof_tree, y_int):
    ws = np.arange(0.0, 1.0001, 0.05)
    ts = np.arange(0.30, 0.701, 0.005)
    best = (-1.0, 0.5, 0.5)
    for w in ws:
        blend = w * oof_deep + (1.0 - w) * oof_tree
        for t in ts:
            acc = float(((blend >= t).astype(int) == y_int).mean())
            if acc > best[0]:
                best = (acc, float(w), float(t))
    _, w0, _ = best
    for w in np.arange(max(0.0, w0 - 0.05), min(1.0, w0 + 0.05) + 1e-9, 0.01):
        blend = w * oof_deep + (1.0 - w) * oof_tree
        for t in ts:
            acc = float(((blend >= t).astype(int) == y_int).mean())
            if acc > best[0]:
                best = (acc, float(w), float(t))
    return best


def run_oof_multi_seed(
    features_df, y_int, seeds=(42, 43, 44), n_splits=5, epochs=18, bag=1
):
    """Run run_oof() once per seed and return list of (oof_deep, oof_tree, acc)."""
    results = []
    for seed in seeds:
        oof_deep, oof_tree = run_oof(
            features_df, y_int, n_splits=n_splits, seed=seed, epochs=epochs, bag=bag
        )
        acc_d = float(((oof_deep >= 0.5).astype(int) == y_int).mean())
        acc_t = float(((oof_tree >= 0.5).astype(int) == y_int).mean())
        acc_b = float(
            (((0.5 * oof_deep + 0.5 * oof_tree) >= 0.5).astype(int) == y_int).mean()
        )
        print(
            f"[multi-seed] seed={seed} deep(t=0.5)={acc_d:.5f} "
            f"tree(t=0.5)={acc_t:.5f} blend(t=0.5)={acc_b:.5f}"
        )
        results.append((oof_deep, oof_tree, acc_b))
    return results


def search_blend_threshold_median(oof_list, y_int):
    """Select (w, t) maximizing MEDIAN accuracy across seed-specific OOFs."""
    ws = np.arange(0.0, 1.0001, 0.05)
    ts = np.arange(0.30, 0.701, 0.005)
    best = (-1.0, 0.5, 0.5)
    for w in ws:
        for t in ts:
            accs = [
                float(((w * od + (1.0 - w) * ot >= t).astype(int) == y_int).mean())
                for od, ot in oof_list
            ]
            med = float(np.median(accs))
            if med > best[0]:
                best = (med, float(w), float(t))
    _, w0, _ = best
    for w in np.arange(max(0.0, w0 - 0.05), min(1.0, w0 + 0.05) + 1e-9, 0.01):
        for t in ts:
            accs = [
                float(
                    ((w * od + (1.0 - w) * ot >= t).astype(int) == y_int).mean()
                )
                for od, ot in oof_list
            ]
            med = float(np.median(accs))
            if med > best[0]:
                best = (med, float(w), float(t))
    return best


def build_inner_kfold_te(features_df, y_int, fe_full, n_splits=5, seed=42):
    df_full = fe_full.transform(features_df)
    te_cols = [c for c in df_full.columns if c.startswith("te__")]
    if not te_cols:
        return df_full
    te_vals = np.zeros((len(features_df), len(te_cols)), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(len(y_int)), y_int):
        fe_fold = FeatureEngineer().fit(features_df.iloc[tr], y=y_int[tr])
        part = fe_fold.transform(features_df.iloc[va])
        for j, c in enumerate(te_cols):
            if c in part.columns:
                te_vals[va, j] = part[c].to_numpy(np.float32)
            else:
                te_vals[va, j] = np.float32(fe_fold.te_prior)
    df_oof = df_full.copy()
    for j, c in enumerate(te_cols):
        df_oof[c] = te_vals[:, j]
    return df_oof


def final_fit(features_df, y_int, n_bag=2, epochs=18, seed=100):
    fe = FeatureEngineer().fit(features_df, y=y_int)
    spec = build_cat_block_spec(fe)
    print("building inner-KFold OOF target encodings for training rows")
    df = build_inner_kfold_te(features_df, y_int, fe, n_splits=5, seed=42)
    xc, xn, cont_cols, cards = split_cont_cat(df, spec)
    mean, std = fit_scaler(xc)
    xc = (xc - mean) / std

    coupon_field_idx = find_coupon_field_idx(spec)
    deep_models = []
    for b in range(n_bag):
        m = train_deepfm(
            xc,
            xn,
            y_int,
            cards,
            epochs=epochs,
            seed=seed + b,
            val=None,
            coupon_field_idx=coupon_field_idx,
            gate_dropout=float(ARCH["gate_dropout"]),
        )
        deep_models.append(m)

    zoo = build_tree_zoo(random_state=seed)
    X = df.to_numpy(np.float32)
    tree_models = []
    for name, mdl in zoo:
        mdl.fit(X, y_int)
        tree_models.append((name, mdl))
        print(f"  fitted tree: {name}")
    return fe, spec, cont_cols, mean, std, cards, deep_models, tree_models


ARCH = dict(
    emb_dim=8,
    deep_dims=[128, 64],
    dropout=0.20,
    emb_dropout=0.05,
    gate_dropout=0.15,
)


def save_artifacts(
    model_dir,
    fe,
    spec,
    cont_cols,
    mean,
    std,
    cards,
    blend_w,
    threshold,
    deep_models,
    tree_models,
    code_to_label,
):
    os.makedirs(model_dir, exist_ok=True)
    coupon_field_idx = find_coupon_field_idx(spec)
    arch_meta = dict(ARCH)
    arch_meta["coupon_field_idx"] = coupon_field_idx
    meta = dict(
        block_spec=[(c, list(cols)) for c, cols in spec],
        cont_cols=list(cont_cols),
        mean=np.asarray(mean, np.float32).tolist(),
        std=np.asarray(std, np.float32).tolist(),
        cards=[int(c) for c in cards],
        cont_dim=int(len(cont_cols)),
        blend_weight=float(blend_w),
        threshold=float(threshold),
        n_bagged=int(len(deep_models)),
        n_trees=int(len(tree_models)),
        arch=arch_meta,
        code_to_label=list(code_to_label),
    )
    with open(os.path.join(model_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"meta": meta, "fe": fe, "tree_models": tree_models}, f)
    for i, m in enumerate(deep_models):
        torch.save(m.state_dict(), os.path.join(model_dir, f"deepfm_{i}.pt"))


def load_artifacts(model_dir):
    with open(os.path.join(model_dir, "meta.pkl"), "rb") as f:
        payload = pickle.load(f)
    return payload["meta"], payload["fe"], payload["tree_models"]


def predict_pipeline(data_dir, model_dir, output_path):
    features, _schema = load_predict(data_dir)
    meta, fe, tree_models = load_artifacts(model_dir)

    spec = meta["block_spec"]
    df = fe.transform(features)
    xc, xn, _, _ = split_cont_cat(df, spec)

    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)
    xc = (xc - mean) / std
    xc_t = torch.from_numpy(xc.astype(np.float32))
    xn_t = torch.from_numpy(xn.astype(np.int64))

    n_nn = int(meta["n_bagged"])
    p_deep = np.zeros(len(xc), dtype=np.float64)
    for i in range(n_nn):
        m, _, _, _ = build_deepfm(
            cont_dim=int(meta["cont_dim"]),
            cat_cardinalities=meta["cards"],
            emb_dim=int(meta["arch"]["emb_dim"]),
            deep_dims=tuple(meta["arch"]["deep_dims"]),
            dropout=float(meta["arch"]["dropout"]),
            emb_dropout=float(meta["arch"]["emb_dropout"]),
            coupon_field_idx=meta["arch"].get("coupon_field_idx"),
            gate_dropout=float(meta["arch"].get("gate_dropout", 0.0) or 0.0),
            device="cpu",
        )
        state = torch.load(
            os.path.join(model_dir, f"deepfm_{i}.pt"), map_location="cpu"
        )
        m.load_state_dict(state)
        p_deep += m.predict_proba(xc_t, xn_t)
    p_deep /= max(1, n_nn)

    X = df.to_numpy(np.float32)
    p_tree = np.zeros(len(X), dtype=np.float64)
    for _name, mdl in tree_models:
        p_tree += mdl.predict_proba(X)[:, 1]
    p_tree /= max(1, len(tree_models))

    w = float(meta["blend_weight"])
    blended = w * p_deep + (1.0 - w) * p_tree
    pred_code = (blended >= float(meta["threshold"])).astype(np.int64)

    code_to_label = meta["code_to_label"]
    pred_label = np.array([code_to_label[c] for c in pred_code.tolist()])

    out = pd.DataFrame(
        {"row_id": np.arange(len(pred_label), dtype=np.int64), "prediction": pred_label}
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"wrote {len(out)} predictions to {output_path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train")
    pt.add_argument("--data", required=True)
    pt.add_argument("--model", required=True)
    pp = sub.add_parser("predict")
    pp.add_argument("--data", required=True)
    pp.add_argument("--model", required=True)
    pp.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.cmd == "train":
        features, y_raw, _schema = load_train(args.data)
        y_int, label_values = encode_labels(y_raw)
        code_to_label = label_values
        print(f"TE smoothing strength = {FeatureEngineer().te_smoothing}")
        print("OOF TE inner-KFold seed = 42 (5 folds, shuffle=True)")

        seeds_list = (42,)
        oof_results = run_oof_multi_seed(
            features, y_int, seeds=seeds_list, n_splits=5, epochs=18, bag=1
        )
        oof_pairs = [(od, ot) for od, ot, _acc in oof_results]
        for i, (_od, _ot, acc_i) in enumerate(oof_results):
            print(f"[seed {seeds_list[i]}] OOF blend(t=0.5) acc = {acc_i:.5f}")

        best_acc, best_w, best_t = search_blend_threshold_median(oof_pairs, y_int)
        print(
            f"OOF median blend acc w={best_w:.2f} t={best_t:.3f} = {best_acc:.5f}"
        )

        fe, spec, cont_cols, mean, std, cards, deep_models, tree_models = final_fit(
            features, y_int, n_bag=1, epochs=15, seed=100
        )
        save_artifacts(
            args.model,
            fe,
            spec,
            cont_cols,
            mean,
            std,
            cards,
            best_w,
            best_t,
            deep_models,
            tree_models,
            code_to_label,
        )
        print("model artifacts saved")
        print(f"Final Validation Score: {best_acc:.5f}")
    else:
        predict_pipeline(args.data, args.model, args.output)


if __name__ == "__main__":
    main()