"""
solution.py

Merged solution for the coupon-acceptance task (binary, accuracy).

Pipeline:
  * Step 1: leakage-safe FeatureBuilder (fit-on-train) + loaders + CV splits
  * Step 2: model builders   -> LightGBM, CatBoost, entity-embedding MLP
  * Step 3: MLPClassifierWrapper + training loop (honest OOF) + predict CLI

CLI:
  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv

Metric: accuracy (supervisor evaluates held-out accuracy independently).
"""

import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Optional third-party imports (guarded; env reports them installed)
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
    from catboost import CatBoostClassifier

    _HAS_CB = True
except Exception:  # pragma: no cover
    _HAS_CB = False


# ===========================================================================
# STEP 1: Feature engineering (leakage-safe) + loading helpers
# ===========================================================================
# "How often do you go to X" columns share one ordinal scale.
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


def make_cv_splits(y, n_splits=3, seed=42):
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ===========================================================================
# STEP 2: Model design (LGBM, CatBoost, entity-embedding MLP)
# ===========================================================================
def build_lgbm(n_classes: int = 2, seed: int = 42):
    if not _HAS_LGB:
        raise RuntimeError("lightgbm is not available in this environment.")
    return LGBMClassifier(
        objective="binary" if n_classes == 2 else "multiclass",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        n_jobs=2,
        random_state=seed,
        verbose=-1,
    )


def build_catboost(cat_feature_indices=None, seed: int = 42):
    if not _HAS_CB:
        raise RuntimeError("catboost is not available in this environment.")
    kwargs = dict(
        iterations=300,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        random_seed=seed,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
    if cat_feature_indices is not None:
        kwargs["cat_features"] = list(cat_feature_indices)
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


def build_stacker(seed: int = 42):
    return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)


# ===========================================================================
# STEP 3: MLP sklearn-style wrapper + training + predict
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
                        if verbose:
                            print(
                                f"  [mlp] early stop @ epoch {epoch} "
                                f"(best val_acc={best_acc:.4f})"
                            )
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
# TRAIN
# ---------------------------------------------------------------------------
def train(data_dir, model_dir, n_splits=3, seed=42):
    os.makedirs(model_dir, exist_ok=True)
    X, y, schema = load_training(data_dir)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    classes = sorted(pd.unique(y).tolist())
    label_to_int = {c: i for i, c in enumerate(classes)}
    int_to_label = {i: c for c, i in label_to_int.items()}
    y_int = y.map(label_to_int).to_numpy().astype(np.int64)
    n = len(y_int)

    # Single honest holdout (train rows only) to tune the decision threshold.
    # Much cheaper than K-fold x 3 models and still leakage-free.
    tr_idx, va_idx = train_test_split(
        np.arange(n), test_size=0.15, stratify=y_int, random_state=seed
    )
    fb_hold = FeatureBuilder().fit(X.iloc[tr_idx], schema=schema, y=y.iloc[tr_idx])
    X_h_tr = fb_hold.transform(X.iloc[tr_idx]).values
    X_h_va = fb_hold.transform(X.iloc[va_idx]).values
    y_h_tr = y_int[tr_idx]
    y_h_va = y_int[va_idx]

    hold_lgb = build_lgbm(n_classes=2, seed=seed)
    hold_lgb.fit(X_h_tr, y_h_tr)
    hold_pos = np.asarray(hold_lgb.predict_proba(X_h_va))[:, 1]
    hold_acc = float(((hold_pos >= 0.5).astype(int) == y_h_va).mean())
    print(f"[holdout] lgbm accuracy: {hold_acc:.4f}")

    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.35, 0.65, 31):
        acc = float(((hold_pos >= t).astype(int) == y_h_va).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    print(f"[holdout] lgbm best threshold={best_t:.3f} acc={best_acc:.4f}")

    # --- final refit on ALL training rows ---
    fb_final = FeatureBuilder().fit(X, schema=schema, y=y)
    X_full = fb_final.transform(X)
    cat_cards_final = _cat_cards(fb_final)
    cat_start_final = len(fb_final.num_cols)

    print("[final] fitting LightGBM ...")
    lgbm_final = build_lgbm(n_classes=2, seed=seed)
    lgbm_final.fit(X_full.values, y_int)

    print("[final] fitting CatBoost ...")
    cb_final = build_catboost(cat_feature_indices=None, seed=seed)
    cb_final.fit(X_full.values, y_int)

    mlp_final = None
    if _HAS_TORCH:
        print("[final] fitting entity-embedding MLP ...")
        Xtr, Xva, ytr, yva = train_test_split(
            X_full.values,
            y_int,
            test_size=0.12,
            stratify=y_int,
            random_state=seed,
        )
        mlp_final = MLPClassifierWrapper(
            cat_cardinalities=cat_cards_final,
            n_cont=cat_start_final,
            cat_start_idx=cat_start_final,
            n_classes=2,
            seed=seed,
        )
        mlp_final.fit(
            Xtr,
            ytr,
            X_val=Xva,
            y_val=yva,
            epochs=30,
            patience=5,
            verbose=False,
            seed=seed,
        )

    artifact = {
        "feature_builder": fb_final,
        "feature_names": list(fb_final.feature_names),
        "num_cols": list(fb_final.num_cols),
        "cat_cols": list(fb_final.cat_cols),
        "cat_start_idx": cat_start_final,
        "cat_cardinalities": cat_cards_final,
        "label_to_int": label_to_int,
        "int_to_label": int_to_label,
        "lgbm": lgbm_final,
        "catboost": cb_final,
        "mlp": mlp_final,
        "threshold": best_t,
        "member_names": ["lgbm", "catboost"] + (["mlp"] if _HAS_TORCH else []),
    }
    with open(os.path.join(model_dir, "artifact.pkl"), "wb") as f:
        pickle.dump(artifact, f)

    score = best_acc
    print(f"Final Validation Score: {score}")


# ---------------------------------------------------------------------------
# PREDICT (fresh process, --model read-only)
# ---------------------------------------------------------------------------
def predict(data_dir, model_dir, output_path):
    with open(os.path.join(model_dir, "artifact.pkl"), "rb") as f:
        art = pickle.load(f)

    fb = art["feature_builder"]
    X, schema = load_predict(data_dir)
    X = X.reset_index(drop=True)
    Xt = fb.transform(X)[art["feature_names"]]
    Xv = Xt.values

    probs = [np.asarray(art["lgbm"].predict_proba(Xv))[:, 1]]
    if art.get("catboost") is not None:
        probs.append(np.asarray(art["catboost"].predict_proba(Xv))[:, 1])
    if art.get("mlp") is not None and _HAS_TORCH:
        try:
            probs.append(np.asarray(art["mlp"].predict_proba(Xv))[:, 1])
        except Exception as e:
            print(f"[predict] skipping MLP member: {e}")

    pos = np.mean(probs, axis=0)
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
