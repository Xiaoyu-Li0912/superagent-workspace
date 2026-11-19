"""
Merged coupon-acceptance pipeline.

Train:  load train features+labels -> FeatureEngineer (fit on TRAIN only)
        -> per-fold transform -> Field-aware DeepFM (bagged) + HistGBM anchor
        -> OOF blend-weight + threshold search -> final fit on all train rows
        -> persist FeatureEngineer + GBM + DeepFM state dicts + meta.

Predict: load features (no labels) + schema -> transform with persisted FE
        -> rebuild DeepFM(s) + GBM -> blend -> threshold -> original labels
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

    def __init__(self):
        self.fill_values = {}
        self.nominal_categories = {}
        self.feature_columns = None

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

    def fit(self, df):
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

        self.transform(df)
        return self

    def transform(self, df):
        df = df.reset_index(drop=True)
        n = len(df)
        out = {}

        for c, mapping in ORDINAL_MAPS.items():
            s = self._normalize(self._col_or_nan(df, c, n))
            v = s.map(mapping).astype(float).fillna(self.fill_values.get(c, 0.0))
            out[c + "_num"] = v

        for c in FREQ_COLS:
            s = self._normalize(self._col_or_nan(df, c, n))
            v = s.map(FREQ_MAP).astype(float).fillna(self.fill_values.get(c, -1.0))
            out[c + "_num"] = v

        for c in NUMERIC_COLS:
            v = pd.to_numeric(self._col_or_nan(df, c, n), errors="coerce").astype(float)
            v = v.fillna(self.fill_values.get(c, 0.0))
            out[c + "_num"] = v

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

        result = pd.DataFrame(out)
        if self.feature_columns is None:
            self.feature_columns = list(result.columns)
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
        cont_dim: int,
        cat_cardinalities: Sequence[int],
        emb_dim: int = 8,
        deep_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.20,
        emb_dropout: float = 0.05,
    ):
        super().__init__()
        self.cont_dim = int(cont_dim)
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        self.emb_dim = int(emb_dim)
        n_fields = len(self.cat_cardinalities)

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

    def _init_weights(self) -> None:
        for emb in list(self.linear_emb) + list(self.fm_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.dim() != 2 or x_cat.size(1) != len(self.cat_cardinalities):
            raise ValueError(
                f"x_cat expected [B,{len(self.cat_cardinalities)}], got {tuple(x_cat.shape)}"
            )
        if x_cont.dim() != 2 or x_cont.size(1) != self.cont_dim:
            raise ValueError(
                f"x_cont expected [B,{self.cont_dim}], got {tuple(x_cont.shape)}"
            )

        linear = self.cont_linear(x_cont).squeeze(-1)
        for i, emb in enumerate(self.linear_emb):
            linear = linear + emb(x_cat[:, i]).squeeze(-1)

        field_embs = torch.stack(
            [self.fm_emb[i](x_cat[:, i]) for i in range(len(self.fm_emb))], dim=1
        )
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
    def predict_proba(
        self, x_cont: torch.Tensor, x_cat: torch.Tensor, batch_size: int = 1024
    ) -> np.ndarray:
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
    cont_dim: int,
    cat_cardinalities: Sequence[int],
    *,
    emb_dim: int = 8,
    deep_dims: Tuple[int, ...] = (128, 64),
    dropout: float = 0.20,
    emb_dropout: float = 0.05,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    epochs: int = 30,
    pos_weight: float | None = None,
    label_smoothing: float = 0.02,
    device: str = "cpu",
):
    model = FieldAwareDeepFM(
        cont_dim=cont_dim,
        cat_cardinalities=cat_cardinalities,
        emb_dim=emb_dim,
        deep_dims=deep_dims,
        dropout=dropout,
        emb_dropout=emb_dropout,
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


def build_gbm_anchor(
    *,
    max_iter: int = 400,
    learning_rate: float = 0.06,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    random_state: int = 42,
):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )


def compute_pos_weight(y: np.ndarray) -> float:
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
    x_cont, x_cat, y, cards, epochs=25, bs=256, lr=2e-3, seed=0, val=None, patience=5
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
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, bs):
            b = idx[s : s + bs]
            opt.zero_grad()
            logits = model(xc[b], xn[b])
            loss = crit(logits, yt[b])
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        sched.step()
        msg = f"epoch {ep + 1}/{epochs} loss={ep_loss / max(1, nb):.4f}"

        if val is not None:
            p = model.predict_proba(xv_c_t, xv_n_t)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            vloss = float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
            vacc = float(((p > 0.5).astype(int) == yv.astype(int)).mean())
            msg += f" val_loss={vloss:.4f} val_acc={vacc:.4f}"
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                patience_left = patience
            else:
                patience_left -= 1
            print(msg)
            if patience_left <= 0:
                print(f"early stop at epoch {ep + 1}")
                break
        else:
            print(msg)

    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_oof(features_df, y_int, n_splits=5, seed=42, epochs=22, bag=2):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y_int)), y_int))

    oof_deep = np.zeros(len(y_int), dtype=np.float64)
    oof_gbm = np.zeros(len(y_int), dtype=np.float64)

    for fi, (tr, va) in enumerate(splits):
        fe = FeatureEngineer().fit(features_df.iloc[tr])
        spec = build_cat_block_spec(fe)
        df_tr = fe.transform(features_df.iloc[tr])
        df_va = fe.transform(features_df.iloc[va])
        xc_tr, xn_tr, _, cards = split_cont_cat(df_tr, spec)
        xc_va, xn_va, _, _ = split_cont_cat(df_va, spec)

        mean, std = fit_scaler(xc_tr)
        xc_tr = (xc_tr - mean) / std
        xc_va = (xc_va - mean) / std

        y_va = y_int[va]

        p_sum = np.zeros(len(va), dtype=np.float64)
        for b in range(bag):
            m = train_deepfm(
                xc_tr,
                xn_tr,
                y_int[tr],
                cards,
                epochs=epochs,
                seed=seed + fi * 10 + b,
                val=(xc_va, xn_va, y_va.astype(np.float32)),
            )
            p_sum += m.predict_proba(
                torch.from_numpy(xc_va.astype(np.float32)),
                torch.from_numpy(xn_va.astype(np.int64)),
            )
        oof_deep[va] = p_sum / bag

        gbm = build_gbm_anchor(random_state=seed + fi)
        gbm.fit(df_tr.to_numpy(np.float32), y_int[tr])
        oof_gbm[va] = gbm.predict_proba(df_va.to_numpy(np.float32))[:, 1]

        acc_d = float(((oof_deep[va] >= 0.5).astype(int) == y_va).mean())
        acc_g = float(((oof_gbm[va] >= 0.5).astype(int) == y_va).mean())
        print(f"fold {fi} deep_acc={acc_d:.4f} gbm_acc={acc_g:.4f}")

    return oof_deep, oof_gbm


def search_blend_threshold(oof_deep, oof_gbm, y_int):
    ws = np.arange(0.0, 1.0001, 0.05)
    ts = np.arange(0.30, 0.701, 0.005)
    best = (-1.0, 0.5, 0.5)
    for w in ws:
        blend = w * oof_deep + (1.0 - w) * oof_gbm
        for t in ts:
            acc = float(((blend >= t).astype(int) == y_int).mean())
            if acc > best[0]:
                best = (acc, float(w), float(t))
    _, w0, _ = best
    for w in np.arange(max(0.0, w0 - 0.05), min(1.0, w0 + 0.05) + 1e-9, 0.01):
        blend = w * oof_deep + (1.0 - w) * oof_gbm
        for t in ts:
            acc = float(((blend >= t).astype(int) == y_int).mean())
            if acc > best[0]:
                best = (acc, float(w), float(t))
    return best


def final_fit(features_df, y_int, n_bag=3, epochs=28, seed=100):
    fe = FeatureEngineer().fit(features_df)
    spec = build_cat_block_spec(fe)
    df = fe.transform(features_df)
    xc, xn, cont_cols, cards = split_cont_cat(df, spec)
    mean, std = fit_scaler(xc)
    xc = (xc - mean) / std

    deep_models = []
    for b in range(n_bag):
        m = train_deepfm(xc, xn, y_int, cards, epochs=epochs, seed=seed + b, val=None)
        deep_models.append(m)

    gbm = build_gbm_anchor(random_state=seed)
    gbm.fit(df.to_numpy(np.float32), y_int)
    return fe, spec, cont_cols, mean, std, cards, deep_models, gbm


ARCH = dict(emb_dim=8, deep_dims=[128, 64], dropout=0.20, emb_dropout=0.05)


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
    gbm,
    code_to_label,
):
    os.makedirs(model_dir, exist_ok=True)
    meta = dict(
        block_spec=[(c, list(cols)) for c, cols in spec],
        cont_cols=list(cont_cols),
        mean=mean.astype(np.float32).tolist(),
        std=std.astype(np.float32).tolist(),
        cards=[int(c) for c in cards],
        cont_dim=int(len(cont_cols)),
        blend_weight=float(blend_w),
        threshold=float(threshold),
        n_bagged=int(len(deep_models)),
        arch=ARCH,
        code_to_label=list(code_to_label),
    )
    with open(os.path.join(model_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"meta": meta, "fe": fe, "gbm": gbm}, f)
    for i, m in enumerate(deep_models):
        torch.save(m.state_dict(), os.path.join(model_dir, f"deepfm_{i}.pt"))


def load_artifacts(model_dir):
    with open(os.path.join(model_dir, "meta.pkl"), "rb") as f:
        payload = pickle.load(f)
    return payload["meta"], payload["fe"], payload["gbm"]


def predict_pipeline(data_dir, model_dir, output_path):
    features, _schema = load_predict(data_dir)
    meta, fe, gbm = load_artifacts(model_dir)

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
            device="cpu",
        )
        state = torch.load(
            os.path.join(model_dir, f"deepfm_{i}.pt"), map_location="cpu"
        )
        m.load_state_dict(state)
        p_deep += m.predict_proba(xc_t, xn_t)
    p_deep /= max(1, n_nn)

    p_gbm = gbm.predict_proba(df.to_numpy(np.float32))[:, 1]

    w = float(meta["blend_weight"])
    blended = w * p_deep + (1.0 - w) * p_gbm
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

        oof_deep, oof_gbm = run_oof(
            features, y_int, n_splits=5, seed=42, epochs=22, bag=2
        )

        acc_deep = float(((oof_deep >= 0.5).astype(int) == y_int).mean())
        acc_gbm = float(((oof_gbm >= 0.5).astype(int) == y_int).mean())
        print(f"OOF deep alone acc (t=0.5) = {acc_deep:.5f}")
        print(f"OOF gbm alone  acc (t=0.5) = {acc_gbm:.5f}")

        best_acc, best_w, best_t = search_blend_threshold(oof_deep, oof_gbm, y_int)
        print(f"OOF blend acc w={best_w:.2f} t={best_t:.3f} = {best_acc:.5f}")

        fe, spec, cont_cols, mean, std, cards, deep_models, gbm = final_fit(
            features,
            y_int,
            n_bag=3,
            epochs=28,
            seed=100,
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
            gbm,
            code_to_label,
        )
        print("model artifacts saved")
        print(f"Final Validation Score: {best_acc:.5f}")
    else:
        predict_pipeline(args.data, args.model, args.output)


if __name__ == "__main__":
    main()
