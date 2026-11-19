"""
solution.py

In-vehicle coupon acceptance prediction (binary classification, metric = accuracy).

Pipeline:
  1. Data processing / feature engineering (leakage-safe, deterministic, stateless
     domain mappings; all stateful transforms are fit only on training rows).
  2. Model design: CatBoost with native categorical handling (ordered target
     statistics) + an optional entity-embedding MLP architecture (unused by the
     final predictor but kept for ensemble diversity).
  3. Training/evaluation: StratifiedKFold OOF on TRAIN rows only, threshold tuned
     on OOF probabilities, complete bundle persisted under --model.
  4. Predict: loads the bundle (no refit) and writes row_id,prediction.

Usage:
  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier, Pool

import torch
import torch.nn as nn

# =========================================================================== #
# STEP 1: DATA PROCESSING & FEATURE ENGINEERING
# =========================================================================== #
AGE_MAP = {
    "below21": 0,
    "21": 1,
    "26": 2,
    "31": 3,
    "36": 4,
    "41": 5,
    "46": 6,
    "50plus": 7,
}

INCOME_MAP = {
    "Less than $12500": 0,
    "$12500 - $24999": 1,
    "$25000 - $37499": 2,
    "$37500 - $49999": 3,
    "$50000 - $62499": 4,
    "$62500 - $74999": 5,
    "$75000 - $87499": 6,
    "$87500 - $99999": 7,
    "$100000 or More": 8,
}

EDU_MAP = {
    "Some High School": 0,
    "High School Graduate": 1,
    "Some college - no degree": 2,
    "Associates degree": 3,
    "Bachelors degree": 4,
    "Graduate degree (Masters or Doctorate)": 5,
}

FREQ_MAP = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}

TIME_MAP = {"7AM": 0, "10AM": 1, "2PM": 2, "6PM": 3, "10PM": 4}

EXPIRATION_HOURS = {"2h": 2.0, "1d": 24.0}

COUPON_VENUE = {
    "Bar": "Bar",
    "Coffee House": "CoffeeHouse",
    "Carry out & Take away": "CarryAway",
    "Restaurant(<20)": "RestaurantLessThan20",
    "Restaurant(20-50)": "Restaurant20To50",
}

FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]
FREQ_ORD = [c + "_freq" for c in FREQ_COLS]

TEXT_COLS = [
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

BINARY_COLS = [
    "has_children",
    "toCoupon_GEQ5min",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
    "direction_opp",
]

BASE_NUM = [
    "temperature",
    "has_children",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
]
ORD_NUM = ["age_ord", "income_ord", "edu_ord", "time_ord", "expiration_hours"]
ENG_NUM = FREQ_ORD + ["social_index", "matched_venue_freq", "distance_level"]
NUM_FEATS = BASE_NUM + ORD_NUM + ENG_NUM

CAT_FEATS = [
    "time",
    "coupon",
    "expiration",
    "gender",
    "maritalStatus",
    "education",
    "occupation",
    "income",
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
    "coupon_time",
    "coupon_exp",
    "matched_bucket",
]


def load_schema(data_dir):
    path = os.path.join(data_dir, "schema.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _restrict_to_schema(df, schema):
    cols = schema.get("feature_columns")
    if cols:
        keep = [c for c in cols if c in df.columns]
        df = df[keep].copy()
    return df


def clean_features(df):
    """Type-cast, normalise strings, impute obvious missingness (no target)."""
    df = df.copy()

    for c in ["temperature"] + BINARY_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)

    for c in FREQ_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("never")

    for c in ["gender", "maritalStatus", "education", "occupation", "income"]:
        if c in df.columns:
            df[c] = df[c].fillna("missing")

    for c in BINARY_COLS:
        if c in df.columns:
            df[c] = df[c].astype(float)

    return df


def _matched_venue_freq(df):
    """Habitual frequency code for the venue that matches the offered coupon."""
    out = pd.Series(np.full(len(df), np.nan), index=df.index)
    if "coupon" not in df.columns:
        return out
    venue_col = df["coupon"].map(COUPON_VENUE)
    for c in FREQ_COLS:
        if c not in df.columns:
            continue
        mask = (venue_col == c).values
        if mask.any():
            codes = df.loc[mask, c].map(FREQ_MAP).astype(float)
            out.loc[df.index[mask]] = codes.values
    return out


def engineer_features(df):
    """Create ordinal + interaction + domain-match features."""
    df = df.copy()

    if "age" in df.columns:
        df["age_ord"] = df["age"].map(AGE_MAP).astype(float)
    if "income" in df.columns:
        df["income_ord"] = df["income"].map(INCOME_MAP).astype(float)
    if "education" in df.columns:
        df["edu_ord"] = df["education"].map(EDU_MAP).astype(float)
    if "time" in df.columns:
        df["time_ord"] = df["time"].map(TIME_MAP).astype(float)
    if "expiration" in df.columns:
        df["expiration_hours"] = df["expiration"].map(EXPIRATION_HOURS).astype(float)

    for c in FREQ_COLS:
        if c in df.columns:
            df[c + "_freq"] = df[c].map(FREQ_MAP).astype(float)

    existing_freq_ord = [c for c in FREQ_ORD if c in df.columns]
    if existing_freq_ord:
        df["social_index"] = df[existing_freq_ord].fillna(0.0).sum(axis=1)
    else:
        df["social_index"] = 0.0

    df["matched_venue_freq"] = _matched_venue_freq(df)
    mvf = df["matched_venue_freq"]
    df["matched_bucket"] = np.select(
        [mvf.isna(), mvf == 0, mvf <= 2, mvf >= 3],
        ["unknown", "never", "rare", "regular"],
        default="unknown",
    )

    g15 = df["toCoupon_GEQ15min"].fillna(0.0) if "toCoupon_GEQ15min" in df else 0.0
    g25 = df["toCoupon_GEQ25min"].fillna(0.0) if "toCoupon_GEQ25min" in df else 0.0
    df["distance_level"] = g15 + g25

    if {"coupon", "time"}.issubset(df.columns):
        df["coupon_time"] = df["coupon"].astype(str) + "|" + df["time"].astype(str)
    if {"coupon", "expiration"}.issubset(df.columns):
        df["coupon_exp"] = df["coupon"].astype(str) + "|" + df["expiration"].astype(str)

    for c in NUM_FEATS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_FEATS:
        if c not in df.columns:
            df[c] = "missing"
        df[c] = df[c].astype(str)

    return df


def build_preprocessor():
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, NUM_FEATS), ("cat", cat_pipe, CAT_FEATS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_train(data_dir):
    """Return (raw_features_df, y, schema). Does NOT touch validation/test."""
    features_path = os.path.join(data_dir, "features.csv")
    labels_path = os.path.join(data_dir, "labels.csv")
    schema = load_schema(data_dir)

    feats = pd.read_csv(features_path)
    feats = _restrict_to_schema(feats, schema)

    labels = pd.read_csv(labels_path)
    label_col = "label" if "label" in labels.columns else labels.columns[-1]
    y = labels[label_col].values

    assert len(feats) == len(y), "features/labels row mismatch"
    return feats, y, schema


def load_predict_features(data_dir):
    schema = load_schema(data_dir)
    feats = pd.read_csv(os.path.join(data_dir, "features.csv"))
    feats = _restrict_to_schema(feats, schema)
    return feats, schema


def prepare_train(data_dir):
    raw, y, schema = load_train(data_dir)
    cleaned = clean_features(raw)
    engineered = engineer_features(cleaned)
    return engineered, y, schema


def prepare_predict(data_dir):
    raw, schema = load_predict_features(data_dir)
    cleaned = clean_features(raw)
    engineered = engineer_features(cleaned)
    return engineered, schema


# =========================================================================== #
# STEP 2: MODEL DESIGN
# =========================================================================== #
DEFAULT_CATBOOST_PARAMS = dict(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=4.0,
    random_strength=1.0,
    bagging_temperature=1.0,
    loss_function="Logloss",
    eval_metric="Logloss",
    od_type="Iter",
    od_wait=100,
    thread_count=2,
    verbose=False,
    random_seed=42,
)


def build_catboost(**overrides) -> CatBoostClassifier:
    params = dict(DEFAULT_CATBOOST_PARAMS)
    params.update(overrides)
    return CatBoostClassifier(**params)


def _default_embed_dim(cardinality: int) -> int:
    return int(min(50, max(2, (cardinality + 1) // 2)))


class CouponEmbeddingMLP(nn.Module):
    """Entity-embedding MLP over categorical + standardised numeric features."""

    def __init__(
        self,
        cat_cardinalities,
        n_numeric: int,
        hidden_dims=(256, 128),
        dropout: float = 0.3,
        embed_dim_fn=None,
    ):
        super().__init__()
        embed_dim_fn = embed_dim_fn or _default_embed_dim

        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embed_dim_fn(card)) for card in cat_cardinalities]
        )
        emb_total = sum(embed_dim_fn(c) for c in cat_cardinalities)

        layers = []
        prev = emb_total + n_numeric
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.LongTensor, x_num: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat(embs + [x_num], dim=1)
        else:
            x = x_num
        return self.mlp(x).squeeze(-1)


def build_mlp(
    cat_cardinalities,
    n_numeric: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 30,
    dropout: float = 0.3,
    device: str = "cpu",
):
    model = CouponEmbeddingMLP(
        cat_cardinalities=cat_cardinalities,
        n_numeric=n_numeric,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    return model, criterion, optimizer, scheduler


def accuracy_threshold(y_true, y_proba, thresholds=None) -> float:
    """Best accuracy over a threshold sweep (train-derived OOF probabilities)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).ravel()
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.70, 81)
    best = 0.0
    for t in thresholds:
        acc = float(((y_proba >= t).astype(int) == y_true).mean())
        if acc > best:
            best = acc
    return best


# =========================================================================== #
# STEP 3: TRAINING / EVALUATION
# =========================================================================== #
FEATURES = NUM_FEATS + CAT_FEATS
BUNDLE_NAME = "coupon_model_bundle.joblib"
META_NAME = "model_metadata.json"
PREPROC_NAME = "preprocessor.joblib"


def _make_X(engineered_df, feature_order=None, cat_features=None):
    """Select + order features; force categoricals to str for CatBoost."""
    feature_order = feature_order or FEATURES
    cat_features = cat_features or CAT_FEATS
    X = engineered_df[feature_order].copy()
    for c in cat_features:
        X[c] = X[c].astype(str)
    return X.reset_index(drop=True)


def train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    engineered, y, schema = prepare_train(data_dir)
    y_raw = np.asarray(y).ravel()
    classes = np.unique(y_raw).tolist()  # original label values
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.asarray([label_to_idx[label] for label in y_raw], dtype=int)
    X = _make_X(engineered)

    # Fit reference preprocessor on training rows only (persisted for completeness).
    pre = build_preprocessor()
    pre.fit(engineered)
    joblib.dump(pre, os.path.join(model_dir, PREPROC_NAME))

    # ---- leakage-safe internal CV: TRAIN rows only --------------------- #
    n_splits = 5 if len(y) >= 50 else max(2, min(5, max(2, len(y) // 2)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(y), dtype=float)
    fold_models = []
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        model = build_catboost(random_seed=42 + fold)
        train_pool = Pool(X_tr, y_tr, cat_features=CAT_FEATS)
        val_pool = Pool(X_va, y_va, cat_features=CAT_FEATS)

        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
        oof[va_idx] = model.predict_proba(val_pool)[:, 1]

        fold_models.append(model)
        it = model.get_best_iteration()
        best_iters.append(int(it) if it is not None else -1)

        fold_acc = float(((oof[va_idx] >= 0.5).astype(int) == y_va).mean())
        print(
            f"[train] fold {fold}: best_iter={best_iters[-1]} val_acc@0.5={fold_acc:.4f}"
        )

    # ---- threshold tuning on OOF probabilities (train-derived only) ---- #
    thresholds = np.linspace(0.30, 0.70, 81)
    accs = [((oof >= t).astype(int) == y).mean() for t in thresholds]
    best_idx = int(np.argmax(accs))
    best_thr = float(thresholds[best_idx])
    best_acc = float(accs[best_idx])
    acc_05 = float(((oof >= 0.5).astype(int) == y).mean())

    print(f"[train] OOF accuracy @0.5 = {acc_05:.4f}")
    print(f"[train] OOF accuracy @best({best_thr:.3f}) = {best_acc:.4f}")

    # ---- persist a COMPLETE usable bundle ----------------------------- #
    bundle = {
        "models": fold_models,
        "threshold": best_thr,
        "classes": classes,
        "cat_features": list(CAT_FEATS),
        "num_features": list(NUM_FEATS),
        "feature_order": list(FEATURES),
        "oof_accuracy_05": acc_05,
        "oof_accuracy_best": best_acc,
    }
    joblib.dump(bundle, os.path.join(model_dir, BUNDLE_NAME))

    meta = {
        "model_type": "catboost_native_categorical_ensemble",
        "n_folds": len(fold_models),
        "best_iters": best_iters,
        "threshold": best_thr,
        "classes": classes,
        "cat_features": list(CAT_FEATS),
        "num_features": list(NUM_FEATS),
        "feature_order": list(FEATURES),
        "n_train_rows": int(len(y)),
        "oof_accuracy_05": acc_05,
        "oof_accuracy_best": best_acc,
    }
    with open(os.path.join(model_dir, META_NAME), "w") as f:
        json.dump(meta, f, indent=2)

    return best_acc


def predict(data_dir, model_dir, output_path):
    bundle = joblib.load(os.path.join(model_dir, BUNDLE_NAME))
    models = bundle["models"]
    thr = float(bundle["threshold"])
    classes = bundle["classes"]
    cat_features = bundle["cat_features"]
    feature_order = bundle["feature_order"]

    engineered, schema = prepare_predict(data_dir)
    X = _make_X(engineered, feature_order=feature_order, cat_features=cat_features)

    n = len(X)
    if n == 0:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame({"row_id": [], "prediction": []}).to_csv(output_path, index=False)
        print("[predict] empty input -> wrote empty predictions")
        return

    pred_pool = Pool(X, cat_features=cat_features)
    proba = np.zeros(n, dtype=float)
    for model in models:
        proba += model.predict_proba(pred_pool)[:, 1]
    proba /= len(models)

    pred_idx = (proba >= thr).astype(int)

    classes_arr = np.asarray(classes)
    predictions = classes_arr[pred_idx]

    out = pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=int),
            "prediction": predictions,
        }
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(
        f"[predict] wrote {len(out)} predictions to {output_path} (threshold={thr:.3f})"
    )


# =========================================================================== #
# CLI
# =========================================================================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "train":
        score = train(args.data, args.model)
        print(f"Final Validation Score: {score}")
    elif args.command == "predict":
        predict(args.data, args.model, args.output)