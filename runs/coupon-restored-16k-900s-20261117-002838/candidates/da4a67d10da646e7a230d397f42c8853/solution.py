"""
solution.py — coupon-acceptance (binary classification, accuracy)

Improvements over previous version:
  #1 Richer leakage-safe feature engineering (history aggregates, affinity gap,
     coupon x time/expiration interactions, life-stage interactions).
  #2 3-fold out-of-fold stacking to estimate blend weights + decision threshold
     on the FULL training set (instead of a noisy 15% holdout). MLP early
     stopping uses an inner split so OOF predictions stay honest.
  #3 Added ExtraTrees member for diversity; stronger/slower-boosted
     CatBoost & LightGBM configs. Each member is fail-safe.

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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression  # kept for optional use

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

CAT_INTERACTIONS = ["coupon_time", "exp_coupon", "is_alone"]


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
        fsum = pd.Series(0.0, index=df.index, dtype="float64")
        fcnt = pd.Series(0, index=df.index, dtype="float64")
        n_zero = pd.Series(0, index=df.index, dtype="float64")
        for c in FREQ_COLS:
            fv = pd.to_numeric(
                self._clean_str(self._series(df, c)).map(FREQ_MAP), errors="coerce"
            )
            freq_vals[c] = fv
            out["freq_" + c] = fv
            miss_sum = miss_sum + fv.isna().astype("float64")
            fsum = fsum + fv.fillna(0.0)
            fcnt = fcnt + fv.notna().astype("float64")
            n_zero = n_zero + (fv.fillna(-1) == 0).astype("float64")
        out["n_missing_freq"] = miss_sum
        out["freq_sum"] = fsum
        out["freq_mean"] = fsum / fcnt.replace(0, np.nan)
        out["n_zero_freq"] = n_zero

        coupon_norm = self._clean_str(self._series(df, "coupon"))
        affinity = pd.Series(np.nan, index=df.index, dtype="float64")
        for coupon_val, fcol in COUPON_AFFINITY.items():
            mask = coupon_norm == coupon_val
            if fcol in freq_vals and bool(mask.any()):
                affinity = affinity.where(~mask, freq_vals[fcol])
        out["coupon_affinity"] = affinity
        # affinity relative to the customer's average habit level for this coupon
        out["affinity_gap"] = affinity - out["freq_mean"]

        for c in BIN_COLS:
            out[c] = pd.to_numeric(self._series(df, c), errors="coerce")

        out["distance_level"] = (
            out["toCoupon_GEQ5min"].fillna(0)
            + out["toCoupon_GEQ15min"].fillna(0)
            + out["toCoupon_GEQ25min"].fillna(0)
        )

        # life-stage interaction
        out["age_income"] = out["age_ord"] * out["income_ord"]

        for c in CAT_BASE:
            out[c] = self._clean_str(self._series(df, c))

        out["coupon_time"] = (
            coupon_norm.astype("string").fillna("NA")
            + "|"
            + self._clean_str(self._series(df, "time")).astype("string").fillna("NA")
        )
        out["exp_coupon"] = (
            self._clean_str(self._series(df, "expiration"))
            .astype("string")
            .fillna("NA")
            + "|"
            + coupon_norm.astype("string").fillna("NA")
        )

        mar = self._clean_str(self._series(df, "maritalStatus")).astype("string")
        kids = pd.to_numeric(self._series(df, "has_children"), errors="coerce")
        alone = mar.isin(["Single", "Divorced", "Widowed"]).fillna(False) & (
            kids.fillna(0) == 0
        )
        out["is_alone"] = alone.map({True: "yes", False: "no"}).astype("string")

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
        """Integer-ordinal encoded matrix (for LGBM / MLP / ExtraTrees)."""
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
def build_lgbm(n_classes: int = 2, seed: int = 42):
    if not _HAS_LGB:
        raise RuntimeError("lightgbm is not available in this environment.")
    return LGBMClassifier(
        objective="binary" if n_classes == 2 else "multiclass",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=2.0,
        n_jobs=2,
        random_state=seed,
        verbose=-1,
    )


def build_catboost(cat_feature_names=None, seed: int = 42):
    if not _HAS_CB:
        raise RuntimeError("catboost is not available in this environment.")
    kwargs = dict(
        iterations=700,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        random_seed=seed,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        one_hot_max_size=6,
    )
    if cat_feature_names is not None:
        kwargs["cat_features"] = list(cat_feature_names)
    return CatBoostClassifier(**kwargs)


def build_et(seed: int = 42):
    return ExtraTreesClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=2,
        random_state=seed,
        class_weight=None,
    )


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
            shifted = torch.clamp(x_cat + 1, min=0)
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
# Weight + threshold search (leakage-free: OOF predictions on training rows)
# ---------------------------------------------------------------------------
def _weight_grid(n_members, step=10):
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
        thresholds = np.linspace(0.30, 0.70, 51)

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


def _fit_mlp_clean(Xtr, ytr, seed, cat_cards, cat_start):
    """Fit MLP with early stopping on an INNER split of fold-train (honest OOF)."""
    X_in, X_vin, y_in, y_vin = train_test_split(
        Xtr, ytr, test_size=0.10, stratify=ytr, random_state=seed
    )
    mlp = MLPClassifierWrapper(
        cat_cardinalities=cat_cards,
        n_cont=cat_start,
        cat_start_idx=cat_start,
        n_classes=2,
        seed=seed,
    )
    mlp.fit(
        X_in,
        y_in,
        X_val=X_vin,
        y_val=y_vin,
        epochs=30,
        patience=5,
        seed=seed,
    )
    return mlp


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

    # -------- 3-fold OOF stacking (train rows only) --------
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(n), y_int))

    member_names_all = ["lgbm", "catboost", "mlp", "et"]
    oof = {m: np.full(n, np.nan) for m in member_names_all}
    ok = {m: True for m in member_names_all}
    if not _HAS_LGB:
        ok["lgbm"] = False
    if not _HAS_CB:
        ok["catboost"] = False
    if not _HAS_TORCH:
        ok["mlp"] = False

    for fi, (tr, va) in enumerate(folds):
        print(f"[oof] fold {fi + 1}/{n_splits}")
        fb = FeatureBuilder().fit(X.iloc[tr], schema=schema)
        Xtr = fb.transform(X.iloc[tr]).values.astype(np.float32)
        Xva = fb.transform(X.iloc[va]).values.astype(np.float32)
        Ctr = fb.transform_catboost(X.iloc[tr])
        Cva = fb.transform_catboost(X.iloc[va])
        cats = _cat_cards(fb)
        cat_start = len(fb.num_cols)
        cbf = list(fb.cat_cols)
        ytr = y_int[tr]
        yva = y_int[va]

        if ok["lgbm"]:
            try:
                m = build_lgbm(2, seed)
                m.fit(Xtr, ytr)
                oof["lgbm"][va] = m.predict_proba(Xva)[:, 1]
            except Exception as e:
                print(f"[oof] lgbm failed: {e}")
                ok["lgbm"] = False

        if ok["catboost"]:
            try:
                m = build_catboost(cat_feature_names=cbf, seed=seed)
                m.fit(Ctr, ytr)
                oof["catboost"][va] = m.predict_proba(Cva)[:, 1]
            except Exception as e:
                print(f"[oof] catboost failed: {e}")
                ok["catboost"] = False

        if ok["mlp"]:
            try:
                m = _fit_mlp_clean(Xtr, ytr, seed, cats, cat_start)
                oof["mlp"][va] = m.predict_proba(Xva)[:, 1]
            except Exception as e:
                print(f"[oof] mlp failed: {e}")
                ok["mlp"] = False

        if ok["et"]:
            try:
                m = build_et(seed)
                m.fit(Xtr, ytr)
                oof["et"][va] = m.predict_proba(Xva)[:, 1]
            except Exception as e:
                print(f"[oof] et failed: {e}")
                ok["et"] = False

    present = [m for m in member_names_all if ok[m] and not np.isnan(oof[m]).any()]
    if not present:
        raise RuntimeError("No model backend succeeded during OOF.")
    print(f"[oof] members present: {present}")

    probs_list = [oof[m] for m in present]
    w_vec, best_t, best_acc = tune_weights_threshold(probs_list, y_int)
    weights = {name: float(w) for name, w in zip(present, w_vec)}
    print(
        f"[oof] tuned weights={weights} threshold={best_t:.3f} oof_acc={best_acc:.4f}"
    )

    # -------- final refit on ALL training rows --------
    fb_final = FeatureBuilder().fit(X, schema=schema, y=y)
    X_full = fb_final.transform(X).values.astype(np.float32)
    C_full = fb_final.transform_catboost(X)
    cat_cards_final = _cat_cards(fb_final)
    cat_start_final = len(fb_final.num_cols)
    cb_cats_final = list(fb_final.cat_cols)

    lgbm_final = cb_final = mlp_final = et_final = None

    if "lgbm" in present:
        print("[final] fitting LightGBM ...")
        try:
            lgbm_final = build_lgbm(2, seed)
            lgbm_final.fit(X_full, y_int)
        except Exception as e:
            print(f"[final] lgbm failed: {e}")
            lgbm_final = None

    if "catboost" in present:
        print("[final] fitting CatBoost ...")
        try:
            cb_final = build_catboost(cat_feature_names=cb_cats_final, seed=seed)
            cb_final.fit(C_full, y_int)
        except Exception as e:
            print(f"[final] catboost failed: {e}")
            cb_final = None

    if "mlp" in present:
        print("[final] fitting entity-embedding MLP ...")
        try:
            mlp_final = _fit_mlp_clean(
                X_full, y_int, seed, cat_cards_final, cat_start_final
            )
        except Exception as e:
            print(f"[final] mlp failed: {e}")
            mlp_final = None

    if "et" in present:
        print("[final] fitting ExtraTrees ...")
        try:
            et_final = build_et(seed)
            et_final.fit(X_full, y_int)
        except Exception as e:
            print(f"[final] et failed: {e}")
            et_final = None

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
        "et": et_final,
        "weights": weights,
        "threshold": best_t,
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

    Xv = fb.transform(X)[art["feature_names"]].values.astype(np.float32)

    members = {}
    if art.get("lgbm") is not None:
        try:
            members["lgbm"] = np.asarray(art["lgbm"].predict_proba(Xv))[:, 1]
        except Exception as e:
            print(f"[predict] skipping lgbm member: {e}")
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
    if art.get("et") is not None:
        try:
            members["et"] = np.asarray(art["et"].predict_proba(Xv))[:, 1]
        except Exception as e:
            print(f"[predict] skipping ET member: {e}")

    if not members:
        raise RuntimeError("No members available at prediction time.")

    stored_w = art.get("weights", {})
    present = list(members.keys())
    wsum = sum(float(stored_w.get(k, 0.0)) for k in present)
    if wsum <= 0:
        w = {k: 1.0 / len(present) for k in present}
    else:
        w = {k: float(stored_w.get(k, 0.0)) / wsum for k in present}

    pos = np.zeros(len(Xv), dtype=float)
    for k in present:
        pos += w[k] * members[k]

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
