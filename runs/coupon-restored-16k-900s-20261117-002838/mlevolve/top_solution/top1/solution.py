"""
solution.py  — coupon-acceptance (binary classification, accuracy)

Improvements over baseline:
  #1 CatBoost fed RAW string categoricals via cat_features (native categorical
     splits / ordered target statistics) instead of integer ordinal codes.
  #2 Weighted ensemble (weights + threshold tuned on an honest training holdout).

CLI:
  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv
"""

import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Optional third-party imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    _HAS_TORCH = False

try:
    from lightgbm import LGBMClassifier

    _HAS_LGB = True
except Exception:  # pragma: no cover
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier, Pool

    _HAS_CB = True
except Exception:  # pragma: no cover
    CatBoostClassifier = None
    Pool = None
    _HAS_CB = False


# ===========================================================================
# STEP 1: Feature engineering (leakage-safe) + loading helpers
# ===========================================================================
FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]
FREQ_MAP = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}

AGE_ORDER = ["below21", "21", "26", "31", "36", "41", "46", "50plus"]
AGE_MAP = {v: i for i, v in enumerate(AGE_ORDER)}

INCOME_ORDER = [
    "Less than $12500",
    "$12500 - $24999",
    "$25000 - $37499",
    "$37500 - $49999",
    "$50000 - $62499",
    "$62500 - $74999",
    "$75000 - $87499",
    "$87500 - $99999",
    "$100000 or More",
]
INCOME_MAP = {v: i for i, v in enumerate(INCOME_ORDER)}

EDU_ORDER = [
    "Some High School",
    "High School Graduate",
    "Some college - no degree",
    "Associates degree",
    "Bachelors degree",
    "Graduate degree (Masters or Doctorate)",
]
EDU_MAP = {v: i for i, v in enumerate(EDU_ORDER)}

TIME_HOUR = {"7AM": 7, "10AM": 10, "2PM": 14, "6PM": 18, "10PM": 22}

COUPON_AFFINITY = {
    "Bar": "Bar",
    "Coffee House": "CoffeeHouse",
    "Restaurant(<20)": "RestaurantLessThan20",
    "Restaurant(20-50)": "Restaurant20To50",
    "Carry out & Take away": "CarryAway",
}

BIN_COLS = [
    "toCoupon_GEQ5min",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
    "direction_opp",
    "has_children",
]

CAT_BASE = [
    "time",
    "coupon",
    "expiration",
    "gender",
    "age",
    "maritalStatus",
    "education",
    "occupation",
    "income",
] + FREQ_COLS

CAT_INTERACTIONS = ["coupon_time"]


class FeatureBuilder:
    """Fit-on-train, transform-anywhere feature pipeline."""

    def __init__(self):
        self.cat_maps = {}
        self.num_medians = {}
        self.cat_cols = []
        self.num_cols = []
        self.feature_names = []

    @staticmethod
    def _series(df, name):
        if name in df.columns:
            return df[name]
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    @staticmethod
    def _clean_str(s):
        return s.astype("string").str.strip()

    def _base_engineer(self, df):
        out = pd.DataFrame(index=df.index)

        out["temperature"] = pd.to_numeric(
            self._series(df, "temperature"), errors="coerce"
        )

        out["time_hour"] = pd.to_numeric(
            self._clean_str(self._series(df, "time")).map(TIME_HOUR), errors="coerce"
        )

        out["age_ord"] = pd.to_numeric(
            self._clean_str(self._series(df, "age")).map(AGE_MAP), errors="coerce"
        )
        out["income_ord"] = pd.to_numeric(
            self._clean_str(self._series(df, "income")).map(INCOME_MAP), errors="coerce"
        )
        out["edu_ord"] = pd.to_numeric(
            self._clean_str(self._series(df, "education")).map(EDU_MAP), errors="coerce"
        )

        freq_vals = {}
        miss_sum = pd.Series(0, index=df.index, dtype="float64")
        for c in FREQ_COLS:
            fv = pd.to_numeric(
                self._clean_str(self._series(df, c)).map(FREQ_MAP), errors="coerce"
            )
            freq_vals[c] = fv
            out["freq_" + c] = fv
            miss_sum = miss_sum + fv.isna().astype("float64")
        out["n_missing_freq"] = miss_sum

        coupon_norm = self._clean_str(self._series(df, "coupon"))
        affinity = pd.Series(np.nan, index=df.index, dtype="float64")
        for coupon_val, fcol in COUPON_AFFINITY.items():
            mask = coupon_norm == coupon_val
            if fcol in freq_vals and bool(mask.any()):
                affinity = affinity.where(~mask, freq_vals[fcol])
        out["coupon_affinity"] = affinity

        for c in BIN_COLS:
            out[c] = pd.to_numeric(self._series(df, c), errors="coerce")

        out["distance_level"] = (
            out["toCoupon_GEQ5min"].fillna(0)
            + out["toCoupon_GEQ15min"].fillna(0)
            + out["toCoupon_GEQ25min"].fillna(0)
        )

        for c in CAT_BASE:
            out[c] = self._clean_str(self._series(df, c))

        out["coupon_time"] = (
            coupon_norm.astype("string").fillna("NA")
            + "|"
            + self._clean_str(self._series(df, "time")).astype("string").fillna("NA")
        )

        return out

    def fit(self, df, schema=None, y=None):
        engineered = self._base_engineer(df)
        self.cat_cols = [
            c for c in engineered.columns if not is_numeric_dtype(engineered[c])
        ]
        self.num_cols = [
            c for c in engineered.columns if is_numeric_dtype(engineered[c])
        ]

        self.cat_maps = {}
        for c in self.cat_cols:
            levels = engineered[c].dropna().astype(str).unique().tolist()
            levels = sorted(levels)
            self.cat_maps[c] = {lvl: i for i, lvl in enumerate(levels)}

        self.num_medians = {}
        for c in self.num_cols:
            v = pd.to_numeric(engineered[c], errors="coerce")
            med = v.median()
            self.num_medians[c] = 0.0 if pd.isna(med) else float(med)

        self.feature_names = list(self.num_cols) + list(self.cat_cols)
        return self

    def transform(self, df):
        """Integer-ordinal encoded matrix (for LGBM / MLP)."""
        if not self.feature_names:
            raise RuntimeError("FeatureBuilder.transform() called before fit().")
        engineered = self._base_engineer(df)
        out = pd.DataFrame(index=df.index)

        for c in self.num_cols:
            v = pd.to_numeric(engineered[c], errors="coerce").fillna(
                self.num_medians[c]
            )
            out[c] = v.astype("float32")

        for c in self.cat_cols:
            mapping = self.cat_maps.get(c, {})
            codes = engineered[c].map(mapping)
            out[c] = codes.fillna(-1).astype("int32")

        return out[self.feature_names]

    def transform_catboost(self, df):
        """Numeric features as-is + RAW string categorical columns (native cats)."""
        if not self.feature_names:
            raise RuntimeError("FeatureBuilder.transform_catboost() before fit().")
        engineered = self._base_engineer(df)
        out = pd.DataFrame(index=df.index)

        for c in self.num_cols:
            v = pd.to_numeric(engineered[c], errors="coerce").fillna(
                self.num_medians[c]
            )
            out[c] = v.astype("float32")

        for c in self.cat_cols:
            out[c] = engineered[c].fillna("NA").astype(str)

        return out[self.feature_names]

    def fit_transform(self, df, schema=None, y=None):
        return self.fit(df, schema=schema, y=y).transform(df)


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def load_training(data_dir):
    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    label_col = "label" if "label" in labels.columns else labels.columns[-1]
    y = labels[label_col].reset_index(drop=True)
    if len(features) != len(y):
        raise ValueError(
            "features.csv and labels.csv row counts differ: %d vs %d"
            % (len(features), len(y))
        )
    return features.reset_index(drop=True), y, schema


def load_predict(data_dir):
    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    return features.reset_index(drop=True), schema


# ===========================================================================
# STEP 2: Model builders
# ===========================================================================
def build_lgbm(
    n_classes: int = 2,
    seed: int = 42,
    regularization_scale: float = 1.0,
    n_estimators: int = 400,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 30,
    reg_lambda: float = 1.5,
    learning_rate: float = 0.05,
):
    """Global defaults unchanged; specialists pass explicit overrides via kwargs."""
    if not _HAS_LGB:
        raise RuntimeError("lightgbm is not available in this environment.")
    s = float(max(0.1, regularization_scale))
    return LGBMClassifier(
        objective="binary" if n_classes == 2 else "multiclass",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=int(num_leaves),
        max_depth=int(max_depth),
        min_child_samples=int(max(1, round(min_child_samples * s))),
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1 * s,
        reg_lambda=float(reg_lambda) * s,
        n_jobs=2,
        random_state=seed,
        verbose=-1,
    )


def build_catboost(
    cat_feature_names=None,
    seed: int = 42,
    regularization_scale: float = 1.0,
    iterations: int = 400,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    learning_rate: float = 0.06,
):
    """Global defaults unchanged; specialists pass explicit overrides via kwargs."""
    if not _HAS_CB:
        raise RuntimeError("catboost is not available in this environment.")
    s = float(max(0.1, regularization_scale))
    kwargs = dict(
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        l2_leaf_reg=float(l2_leaf_reg) * s,
        loss_function="Logloss",
        random_seed=seed,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
    if cat_feature_names is not None:
        kwargs["cat_features"] = list(cat_feature_names)
    return CatBoostClassifier(**kwargs)


if _HAS_TORCH:

    class EntityEmbeddingMLP(nn.Module):
        def __init__(
            self,
            cat_cardinalities,
            n_continuous: int,
            n_classes: int = 2,
            embed_dim_cap: int = 16,
            hidden_dims=(256, 128, 64),
            dropout: float = 0.20,
        ):
            super().__init__()
            self.embeddings = nn.ModuleList()
            emb_total = 0
            for card in cat_cardinalities:
                card = max(1, int(card))
                dim = int(min(embed_dim_cap, max(2, round(1.6 * card**0.56))))
                self.embeddings.append(
                    nn.Embedding(num_embeddings=card + 1, embedding_dim=dim)
                )
                emb_total += dim

            in_dim = emb_total + int(n_continuous)

            layers = []
            prev = in_dim
            for hd in hidden_dims:
                layers += [
                    nn.Linear(prev, hd),
                    nn.LayerNorm(hd),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
                prev = hd
            layers += [nn.Linear(prev, n_classes)]
            self.mlp = nn.Sequential(*layers)

            for emb in self.embeddings:
                with torch.no_grad():
                    emb.weight[0].zero_()

        def forward(self, x_cat, x_cont):
            shifted = x_cat + 1
            shifted = torch.clamp(shifted, min=0)
            embs = [emb(shifted[:, i]) for i, emb in enumerate(self.embeddings)]
            h = torch.cat(embs + [x_cont], dim=1)
            return self.mlp(h)

    def build_embedding_mlp(
        cat_cardinalities, n_continuous: int, n_classes: int = 2, dropout: float = 0.20
    ):
        return EntityEmbeddingMLP(
            cat_cardinalities=cat_cardinalities,
            n_continuous=n_continuous,
            n_classes=n_classes,
            dropout=dropout,
        )

    def build_nn_criterion(class_weights=None):
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    def build_nn_optimizer(model, lr: float = 1e-3, weight_decay: float = 1e-4):
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99)
        )

else:  # pragma: no cover
    EntityEmbeddingMLP = None  # type: ignore

    def build_embedding_mlp(*a, **k):  # type: ignore
        raise RuntimeError("torch is not available in this environment.")

    def build_nn_criterion(*a, **k):  # type: ignore
        raise RuntimeError("torch is not available in this environment.")

    def build_nn_optimizer(*a, **k):  # type: ignore
        raise RuntimeError("torch is not available in this environment.")


# ===========================================================================
# STEP 3: MLP wrapper
# ===========================================================================
class MLPClassifierWrapper:
    """Trains/predicts an EntityEmbeddingMLP from array layout [cont | cat]."""

    def __init__(
        self,
        cat_cardinalities,
        n_cont,
        cat_start_idx,
        n_classes=2,
        seed=42,
        dropout=0.20,
        device="cpu",
    ):
        self.cat_cardinalities = list(cat_cardinalities)
        self.n_cont = int(n_cont)
        self.cat_start_idx = int(cat_start_idx)
        self.n_classes = int(n_classes)
        self.device = device
        self.model = build_embedding_mlp(
            self.cat_cardinalities,
            self.n_cont,
            n_classes=self.n_classes,
            dropout=dropout,
        ).to(device)
        self.criterion = build_nn_criterion()
        self.optimizer = build_nn_optimizer(self.model)
        self.best_state = None

    def _split(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        cont = X[:, : self.cat_start_idx].astype(np.float32)
        cat = X[:, self.cat_start_idx :].astype(np.int64)
        return cat, cont

    def _forward_probs(self, cat, cont):
        self.model.eval()
        with torch.no_grad():
            ct = torch.from_numpy(cat).to(self.device)
            ft = torch.from_numpy(cont).to(self.device)
            logits = self.model(ct, ft)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=30,
        batch_size=512,
        patience=5,
        verbose=False,
        seed=42,
    ):
        torch.manual_seed(seed)
        cat, cont = self._split(X)
        y = np.asarray(y).astype(np.int64)

        ct_all = torch.from_numpy(cat).to(self.device)
        ft_all = torch.from_numpy(cont).to(self.device)
        yt_all = torch.from_numpy(y).to(self.device)
        n = len(y)

        best_acc = -1.0
        bad = 0
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                self.optimizer.zero_grad()
                logits = self.model(ct_all[idx], ft_all[idx])
                loss = self.criterion(logits, yt_all[idx])
                loss.backward()
                self.optimizer.step()
                total += loss.item() * len(idx)
            tr_loss = total / n

            if X_val is not None:
                vcat, vcont = self._split(X_val)
                vprobs = self._forward_probs(vcat, vcont)
                vacc = float((vprobs.argmax(1) == np.asarray(y_val)).mean())
                if verbose:
                    print(
                        f"  [mlp] epoch {epoch:02d} loss={tr_loss:.4f} val_acc={vacc:.4f}"
                    )
                if vacc > best_acc:
                    best_acc = vacc
                    bad = 0
                    self.best_state = {
                        k: v.detach().clone()
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    bad += 1
                    if bad >= patience:
                        break
            elif verbose:
                print(f"  [mlp] epoch {epoch:02d} loss={tr_loss:.4f}")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self

    def predict_proba(self, X, cat_split=None):
        cat, cont = self._split(X)
        return self._forward_probs(cat, cont)

    def predict(self, X, cat_split=None):
        return self.predict_proba(X, cat_split).argmax(1)


def _cat_cards(fb):
    return [max(1, len(fb.cat_maps.get(c, {}))) for c in fb.cat_cols]


# ---------------------------------------------------------------------------
# Weight + threshold search (leakage-free: holdout rows only)
# ---------------------------------------------------------------------------
def _weight_grid(n_members, step=10):
    """Yield tuples of weights (integers summing to step) over n_members."""
    if n_members == 1:
        yield (step,)
        return

    def rec(prefix, remaining, slots):
        if slots == 1:
            yield tuple(prefix + [remaining])
            return
        for v in range(remaining + 1):
            yield from rec(prefix + [v], remaining - v, slots - 1)

    yield from rec([], step, n_members)


def tune_weights_threshold(probs_list, y, thresholds=None):
    """probs_list: list of N-length positive-class prob arrays aligned to y."""
    P = np.stack(probs_list, axis=1)  # N x M
    y = np.asarray(y).astype(int)
    M = P.shape[1]
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.70, 41)

    best_acc = -1.0
    best_w = np.ones(M, dtype=float) / M
    best_t = 0.5

    for w_int in _weight_grid(M, step=10):
        w = np.array(w_int, dtype=float) / float(sum(w_int))
        pos = P.dot(w)
        for t in thresholds:
            acc = float(((pos >= t).astype(int) == y).mean())
            if acc > best_acc:
                best_acc = acc
                best_w = w
                best_t = float(t)
    return best_w, best_t, best_acc


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def train(data_dir, model_dir, seed=42):
    os.makedirs(model_dir, exist_ok=True)
    X, y, schema = load_training(data_dir)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    classes = sorted(pd.unique(y).tolist())
    label_to_int = {c: i for i, c in enumerate(classes)}
    int_to_label = {i: c for c, i in label_to_int.items()}
    y_int = y.map(label_to_int).to_numpy().astype(np.int64)
    n = len(y_int)

    # --- honest holdout on TRAINING rows only (weight + threshold tuning) ---
    tr_idx, va_idx = train_test_split(
        np.arange(n), test_size=0.15, stratify=y_int, random_state=seed
    )
    fb_hold = FeatureBuilder().fit(X.iloc[tr_idx], schema=schema, y=y.iloc[tr_idx])
    X_h_tr = fb_hold.transform(X.iloc[tr_idx])
    X_h_va = fb_hold.transform(X.iloc[va_idx])
    C_h_tr = fb_hold.transform_catboost(X.iloc[tr_idx])
    C_h_va = fb_hold.transform_catboost(X.iloc[va_idx])
    y_h_tr = y_int[tr_idx]
    y_h_va = y_int[va_idx]

    cat_cards_h = _cat_cards(fb_hold)
    cat_start_h = len(fb_hold.num_cols)
    cb_cats_h = list(fb_hold.cat_cols)

    print("[holdout] fitting members ...")
    hold_probs = []
    member_names = []

    if _HAS_LGB:
        m = build_lgbm(n_classes=2, seed=seed)
        m.fit(X_h_tr.values, y_h_tr)
        hold_probs.append(np.asarray(m.predict_proba(X_h_va.values))[:, 1])
        member_names.append("lgbm")

    if _HAS_CB:
        m = build_catboost(cat_feature_names=cb_cats_h, seed=seed)
        m.fit(C_h_tr, y_h_tr)
        hold_probs.append(np.asarray(m.predict_proba(C_h_va))[:, 1])
        member_names.append("catboost")

    if _HAS_TORCH:
        mlp_h = MLPClassifierWrapper(
            cat_cardinalities=cat_cards_h,
            n_cont=cat_start_h,
            cat_start_idx=cat_start_h,
            n_classes=2,
            seed=seed,
        )
        mlp_h.fit(
            X_h_tr.values,
            y_h_tr,
            X_val=X_h_va.values,
            y_val=y_h_va,
            epochs=30,
            patience=5,
            seed=seed,
        )
        hold_probs.append(np.asarray(mlp_h.predict_proba(X_h_va.values))[:, 1])
        member_names.append("mlp")

    if not hold_probs:  # absolute fallback
        raise RuntimeError("No model backend available (lightgbm/catboost/torch).")

    w_vec, best_t, best_acc = tune_weights_threshold(hold_probs, y_h_va)
    weights = {name: float(w) for name, w in zip(member_names, w_vec)}
    print(
        f"[holdout] tuned weights={weights} threshold={best_t:.3f} acc={best_acc:.4f}"
    )

    # ------------------------------------------------------------------
    # Per-coupon specialists: small, guarded by min subgroup size, and
    # fitted on the SAME FeatureBuilder output with the `coupon` column
    # masked out (they cannot see the identity they are conditioned on).
    # Blend weight `alpha` is tuned on the same honest holdout.
    # ------------------------------------------------------------------
    MIN_SPECIALIST_ROWS = 400
    spec_kind = "lgbm" if _HAS_LGB else ("catboost" if _HAS_CB else None)
    spec_mask_names = [c for c in fb_hold.feature_names if c != "coupon"]
    spec_cb_cats = [c for c in fb_hold.cat_cols if c != "coupon"]

    hold_coupon_tr = X.iloc[tr_idx]["coupon"].astype(str).to_numpy()
    hold_coupon_va = X.iloc[va_idx]["coupon"].astype(str).to_numpy()

    specialists_hold = {}
    spec_va_probs = np.full(len(va_idx), np.nan, dtype=float)
    skipped_coupons = []
    if spec_kind is not None:
        print(f"[holdout] fitting per-coupon specialists ({spec_kind}) ...")
        for cv in sorted(set(hold_coupon_tr.tolist())):
            tr_mask = hold_coupon_tr == cv
            va_mask = hold_coupon_va == cv
            n_tr = int(tr_mask.sum())
            if n_tr < MIN_SPECIALIST_ROWS or int(va_mask.sum()) == 0:
                skipped_coupons.append(cv)
                continue
            scale = float(max(0.5, min(3.0, 1500.0 / max(1, n_tr))))
            try:
                if spec_kind == "lgbm":
                    mdl = build_lgbm(
                        n_classes=2,
                        seed=seed,
                        regularization_scale=scale,
                        n_estimators=250,
                        num_leaves=15,
                        max_depth=5,
                        min_child_samples=40,
                        reg_lambda=3.0,
                    )
                    mdl.fit(
                        X_h_tr[spec_mask_names].values[tr_mask], y_h_tr[tr_mask]
                    )
                    p = np.asarray(
                        mdl.predict_proba(X_h_va[spec_mask_names].values[va_mask])
                    )[:, 1]
                else:
                    mdl = build_catboost(
                        cat_feature_names=spec_cb_cats,
                        seed=seed,
                        regularization_scale=scale,
                        iterations=250,
                        depth=4,
                        l2_leaf_reg=5.0,
                    )
                    mdl.fit(C_h_tr[spec_mask_names][tr_mask], y_h_tr[tr_mask])
                    p = np.asarray(
                        mdl.predict_proba(C_h_va[spec_mask_names][va_mask])
                    )[:, 1]
                specialists_hold[cv] = mdl
                spec_va_probs[va_mask] = p
            except Exception as e:  # pragma: no cover
                print(f"[holdout] specialist {cv} failed ({e}); routing to global")
                skipped_coupons.append(cv)
        print(
            f"[holdout] specialists={list(specialists_hold)} "
            f"skipped={skipped_coupons}"
        )

    # Coarse alpha-blend search on the same holdout (reuses threshold sweep).
    base_pos = np.stack(hold_probs, axis=1).dot(np.asarray(w_vec, dtype=float))
    best_alpha = 0.0
    best_alpha_acc = best_acc
    if specialists_hold:
        has_spec = ~np.isnan(spec_va_probs)
        spec_filled = np.where(has_spec, spec_va_probs, base_pos)
        for a_int in range(0, 11):
            a = a_int / 10.0
            fused = a * spec_filled + (1.0 - a) * base_pos
            for t in np.linspace(0.30, 0.70, 41):
                acc_a = float(((fused >= t).astype(int) == y_h_va).mean())
                if acc_a > best_alpha_acc + 1e-9:
                    best_alpha_acc = acc_a
                    best_alpha = a
                    best_t = float(t)
        print(
            f"[holdout] specialist alpha={best_alpha:.2f} "
            f"acc={best_alpha_acc:.4f} threshold={best_t:.3f}"
        )
    best_acc = max(best_acc, best_alpha_acc)

    # --- final refit on ALL training rows ---
    fb_final = FeatureBuilder().fit(X, schema=schema, y=y)
    X_full = fb_final.transform(X)
    C_full = fb_final.transform_catboost(X)
    cat_cards_final = _cat_cards(fb_final)
    cat_start_final = len(fb_final.num_cols)
    cb_cats_final = list(fb_final.cat_cols)

    lgbm_final = None
    if _HAS_LGB:
        print("[final] fitting LightGBM ...")
        lgbm_final = build_lgbm(n_classes=2, seed=seed)
        lgbm_final.fit(X_full.values, y_int)

    cb_final = None
    if _HAS_CB:
        print("[final] fitting CatBoost (native categoricals) ...")
        cb_final = build_catboost(cat_feature_names=cb_cats_final, seed=seed)
        cb_final.fit(C_full, y_int)

    mlp_final = None
    if _HAS_TORCH:
        print("[final] fitting entity-embedding MLP ...")
        Xtr, Xva, ytr, yva = train_test_split(
            X_full.values, y_int, test_size=0.12, stratify=y_int, random_state=seed
        )
        mlp_final = MLPClassifierWrapper(
            cat_cardinalities=cat_cards_final,
            n_cont=cat_start_final,
            cat_start_idx=cat_start_final,
            n_classes=2,
            seed=seed,
        )
        mlp_final.fit(Xtr, ytr, X_val=Xva, y_val=yva, epochs=30, patience=5, seed=seed)

    # ---- final specialists refit on ALL training rows (same size guard) ----
    specialists_final = {}
    if spec_kind is not None:
        full_coupon = X["coupon"].astype(str).to_numpy()
        print("[final] refitting per-coupon specialists ...")
        for cv in sorted(set(full_coupon.tolist())):
            m_cv = full_coupon == cv
            n_cv = int(m_cv.sum())
            if n_cv < MIN_SPECIALIST_ROWS:
                continue
            scale = float(max(0.5, min(3.0, 1500.0 / max(1, n_cv))))
            try:
                if spec_kind == "lgbm":
                    mdl = build_lgbm(
                        n_classes=2,
                        seed=seed,
                        regularization_scale=scale,
                        n_estimators=250,
                        num_leaves=15,
                        max_depth=5,
                        min_child_samples=40,
                        reg_lambda=3.0,
                    )
                    mdl.fit(X_full[spec_mask_names].values[m_cv], y_int[m_cv])
                else:
                    mdl = build_catboost(
                        cat_feature_names=spec_cb_cats,
                        seed=seed,
                        regularization_scale=scale,
                        iterations=250,
                        depth=4,
                        l2_leaf_reg=5.0,
                    )
                    mdl.fit(C_full[spec_mask_names][m_cv], y_int[m_cv])
                specialists_final[cv] = mdl
            except Exception as e:  # pragma: no cover
                print(f"[final] specialist {cv} failed: {e}")
        print(f"[final] specialists built for {list(specialists_final)}")

    artifact = {
        "feature_builder": fb_final,
        "feature_names": list(fb_final.feature_names),
        "num_cols": list(fb_final.num_cols),
        "cat_cols": list(fb_final.cat_cols),
        "cat_start_idx": cat_start_final,
        "cat_cardinalities": cat_cards_final,
        "cb_cat_features": cb_cats_final,
        "label_to_int": label_to_int,
        "int_to_label": int_to_label,
        "lgbm": lgbm_final,
        "catboost": cb_final,
        "mlp": mlp_final,
        "weights": weights,
        "threshold": best_t,
        "specialists": specialists_final,
        "specialist_feature_names": list(spec_mask_names),
        "specialist_alpha": float(best_alpha),
        "specialist_blend_weights": dict(weights),
        "min_specialist_rows": int(MIN_SPECIALIST_ROWS),
        "specialist_kind": spec_kind,
    }
    with open(os.path.join(model_dir, "artifact.pkl"), "wb") as f:
        pickle.dump(artifact, f)

    print(f"Final Validation Score: {best_acc}")


# ---------------------------------------------------------------------------
# PREDICT (fresh process, --model read-only)
# ---------------------------------------------------------------------------
def predict(data_dir, model_dir, output_path):
    with open(os.path.join(model_dir, "artifact.pkl"), "rb") as f:
        art = pickle.load(f)

    fb = art["feature_builder"]
    X, schema = load_predict(data_dir)
    X = X.reset_index(drop=True)

    Xv = fb.transform(X)[art["feature_names"]].values

    members = {}
    if art.get("lgbm") is not None:
        members["lgbm"] = np.asarray(art["lgbm"].predict_proba(Xv))[:, 1]
    if art.get("catboost") is not None:
        try:
            Cv = fb.transform_catboost(X)[art["feature_names"]]
            members["catboost"] = np.asarray(art["catboost"].predict_proba(Cv))[:, 1]
        except Exception as e:
            print(f"[predict] skipping catboost member: {e}")
    if art.get("mlp") is not None and _HAS_TORCH:
        try:
            members["mlp"] = np.asarray(art["mlp"].predict_proba(Xv))[:, 1]
        except Exception as e:
            print(f"[predict] skipping MLP member: {e}")

    stored_w = art.get("weights", {})
    # renormalize over present members
    present = list(members.keys())
    wsum = sum(float(stored_w.get(k, 0.0)) for k in present)
    if wsum <= 0:
        w = {k: 1.0 / len(present) for k in present}
    else:
        w = {k: float(stored_w.get(k, 0.0)) / wsum for k in present}

    base_pos = np.zeros(len(Xv), dtype=float)
    for k in present:
        base_pos += w[k] * members[k]

    # ---- per-coupon specialist routing (falls back to global blend) ----
    pos = base_pos
    sp_alpha = float(art.get("specialist_alpha", 0.0) or 0.0)
    specialists = art.get("specialists") or {}
    spec_kind_p = art.get("specialist_kind")
    if sp_alpha > 0.0 and specialists and spec_kind_p is not None:
        spec_names = art.get("specialist_feature_names") or []
        coupon_raw = (
            fb._clean_str(X["coupon"]).fillna("NA").astype(str).to_numpy()
            if "coupon" in X.columns
            else None
        )
        if coupon_raw is not None and spec_names:
            try:
                if spec_kind_p == "lgbm":
                    Xs = fb.transform(X)[spec_names].values
                else:
                    Xs = fb.transform_catboost(X)[spec_names]
                spec_pos = base_pos.copy()
                for cv, mdl in specialists.items():
                    m = coupon_raw == str(cv)
                    if not bool(m.any()):
                        continue
                    try:
                        spec_pos[m] = np.asarray(mdl.predict_proba(Xs[m]))[:, 1]
                    except Exception as e:  # pragma: no cover
                        print(f"[predict] specialist {cv} skipped: {e}")
                pos = sp_alpha * spec_pos + (1.0 - sp_alpha) * base_pos
            except Exception as e:  # pragma: no cover
                print(f"[predict] specialist block skipped: {e}")
                pos = base_pos

    y_hat_int = (pos >= art["threshold"]).astype(int)
    int_to_label = art["int_to_label"]
    y_hat = [int_to_label[int(v)] for v in y_hat_int]

    out = pd.DataFrame({"row_id": np.arange(len(y_hat)), "prediction": y_hat})
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[predict] wrote {len(out)} rows -> {output_path}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def _build_argparser():
    parser = argparse.ArgumentParser(description="Coupon acceptance: train / predict")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="fit pipeline on labelled data")
    p_tr.add_argument("--data", required=True)
    p_tr.add_argument("--model", required=True)

    p_pr = sub.add_parser("predict", help="infer on unlabelled data")
    p_pr.add_argument("--data", required=True)
    p_pr.add_argument("--model", required=True)
    p_pr.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.cmd == "train":
        train(args.data, args.model)
    else:
        predict(args.data, args.model, args.output)
