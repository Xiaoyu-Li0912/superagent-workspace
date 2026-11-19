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
    best_eps = []
    gbdt_mask = None  # filled on first fold; identical for all folds

    for fold, (tr, va) in enumerate(skf.split(features, y_bin)):
        fe = CouponFeatureEngineer().fit(features.iloc[tr])
        tr_out = fe.transform(features.iloc[tr])
        va_out = fe.transform(features.iloc[va])
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
            gbdt = HistGradientBoostingClassifier(**GBDT_CFG, random_state=SEED + fold)
            gbdt.fit(Xg_tr, y_int[tr], categorical_features=gbdt_mask)
            oof_gbdt[va] = gbdt.predict_proba(Xg_va)[:, 1]
            g_acc = float(((oof_gbdt[va] >= 0.5).astype(int) == y_int[va]).mean())
            print(f"  -> fold {fold + 1} GBDT val_acc {g_acc:.4f}")

    # ---------------- blend + threshold tuning on OOF (official metric) ----
    has_gbdt = _HAS_GBDT
    best_w, best_thr, best_acc_oof = 1.0, 0.5, 0.0
    w_grid = np.linspace(0.0, 1.0, 21) if has_gbdt else [1.0]
    for w in w_grid:
        p = w * oof_nn + (1.0 - w) * oof_gbdt if has_gbdt else oof_nn
        for thr in np.linspace(0.25, 0.75, 101):
            acc = float(((p >= thr).astype(int) == y_int).mean())
            if acc > best_acc_oof:
                best_acc_oof, best_thr, best_w = acc, float(thr), float(w)
    oof_acc = best_acc_oof
    print(f"OOF accuracy {oof_acc:.6f} | blend_w(nn)={best_w:.2f} thr={best_thr:.3f}")

    # ---------------- full refit (all training rows only) ------------------
    fe_full = CouponFeatureEngineer().fit(features)
    full_out = fe_full.transform(features)
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
        gbdt_full = HistGradientBoostingClassifier(**GBDT_CFG, random_state=SEED + 777)
        gbdt_full.fit(Xg_all, y_int, categorical_features=gbdt_mask_full)
        print("refit GBDT on all training rows")

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
        gbdt=gbdt_full,
        gbdt_cat_mask=gbdt_mask_full,
        blend_w_nn=float(best_w),
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
                "blend_w_nn": best_w,
                "refit_epochs": refit_eps,
                "n_refit": N_REFIT,
                "has_gbdt": bool(_HAS_GBDT),
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

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    n = len(features)
    out_df = fe.transform(features)
    Xn, Xc = _to_tensors(fe, idx, out_df)

    models = _build_models_from_bundle(bundle)
    p_nn = _ensemble_probs(models, Xn, Xc)

    if has_gbdt:
        Xg, _ = _gbdt_matrix(fe, idx, out_df)
        p_gbdt = gbdt.predict_proba(Xg)[:, 1]
        prob = w_nn * p_nn + (1.0 - w_nn) * p_gbdt
    else:
        prob = p_nn

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
