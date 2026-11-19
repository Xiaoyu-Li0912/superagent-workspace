#!/usr/bin/env python3
"""Coupon acceptance prediction.

Pipeline:
  * robust tabular feature engineering (ordinal maps + one-hot + numeric block),
  * ensemble of HistGradientBoosting / ExtraTrees / (optional) LightGBM,
  * OOF-optimized simplex blend weights,
  * single pickled bundle persisted under --model, loaded at predict time.

Usage:
  python solution.py train   --data /data  --model /model
  python solution.py predict --data /data  --model /model --output /output/predictions.csv
"""

import argparse
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

try:
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:
    HAS_LGB = False


# ---------------------------------------------------------------------------
# Feature-engineering configuration
# ---------------------------------------------------------------------------
ORDINAL_CATS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
    "income",
    "age",
]
PLAIN_CATS = [
    "time",
    "coupon",
    "expiration",
    "gender",
    "maritalStatus",
    "education",
    "occupation",
]
NUMERIC_COLS = [
    "temperature",
    "has_children",
    "toCoupon_GEQ5min",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
    "direction_opp",
]

_ORD_LEVELS = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
ORDINAL_MAPS = {
    "Bar": _ORD_LEVELS,
    "CoffeeHouse": _ORD_LEVELS,
    "CarryAway": _ORD_LEVELS,
    "RestaurantLessThan20": _ORD_LEVELS,
    "Restaurant20To50": _ORD_LEVELS,
    "income": {
        "Less than $12500": 0,
        "$12500 - $24999": 1,
        "$25000 - $37499": 2,
        "$37500 - $49999": 3,
        "$50000 - $62499": 4,
        "$62500 - $74999": 5,
        "$75000 - $87499": 6,
        "$87500 - $99999": 7,
        "$100000 or More": 8,
    },
    "age": {
        "below21": 0,
        "21": 1,
        "26": 2,
        "31": 3,
        "36": 4,
        "41": 5,
        "46": 6,
        "50plus": 7,
    },
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _load_schema(data_dir):
    p = os.path.join(data_dir, "schema.json")
    if os.path.exists(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _read_features(data_dir, schema=None):
    p = os.path.join(data_dir, "features.csv")
    df = pd.read_csv(p)
    if schema:
        cols = schema.get("feature_columns")
        if cols:
            keep = [c for c in cols if c in df.columns]
            if keep:
                df = df[keep]
    return df.reset_index(drop=True)


def _read_labels(data_dir):
    p = os.path.join(data_dir, "labels.csv")
    lbl = pd.read_csv(p)
    col = "label" if "label" in lbl.columns else lbl.columns[0]
    return lbl[col].tolist()


# ---------------------------------------------------------------------------
# String-safe column extraction (NaN/None -> "nan")
# ---------------------------------------------------------------------------
def _safe_str(v):
    if v is None:
        return "nan"
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        if v == int(v):
            return str(int(v))
        return str(v)
    return str(v)


def _col_str(df, c):
    n = len(df)
    if c not in df.columns:
        return ["nan"] * n
    return [_safe_str(v) for v in df[c].tolist()]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_features(df, ohe=None, fit=False):
    n = len(df)

    # 1) plain categorical block
    cat_data = {c: _col_str(df, c) for c in PLAIN_CATS}
    cat_df = pd.DataFrame(cat_data)

    # 2) ordinal-encoded block -> numeric
    num_data = {}
    for c in ORDINAL_CATS:
        mp = ORDINAL_MAPS.get(c, {})
        vals = _col_str(df, c)
        num_data[c + "_ord"] = np.array([mp.get(v, -1) for v in vals], dtype=np.float32)
    # 3) direct numeric block
    for c in NUMERIC_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            num_data[c + "_num"] = s.fillna(-1).astype(np.float32).values
        else:
            num_data[c + "_num"] = np.full(n, -1, dtype=np.float32)

    # 4) cheap interactions
    income = num_data["income_ord"]
    age = num_data["age_ord"]
    num_data["income_x_age"] = (income * age).astype(np.float32)
    dist = (
        num_data["toCoupon_GEQ5min_num"]
        + num_data["toCoupon_GEQ15min_num"]
        + num_data["toCoupon_GEQ25min_num"]
    )
    num_data["dist_sum"] = dist.astype(np.float32)
    num_data["dist_x_same"] = (dist * num_data["direction_same_num"]).astype(np.float32)

    num_df = pd.DataFrame(num_data)

    # 5) one-hot for plain categoricals
    if fit or ohe is None:
        ohe = _make_ohe()
        X_cat = ohe.fit_transform(cat_df)
    else:
        X_cat = ohe.transform(cat_df)
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()
    X_cat = np.asarray(X_cat, dtype=np.float32)

    X = np.hstack([X_cat, num_df.values.astype(np.float32)])
    return X, ohe


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def _make_models(seed, n_classes):
    models = []
    models.append(
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.06,
                max_leaf_nodes=31,
                min_samples_leaf=25,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=int(seed),
            ),
        )
    )
    models.append(
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                n_jobs=2,
                random_state=int(seed),
            ),
        )
    )
    if HAS_LGB:
        models.append(
            (
                "lgb",
                lgb.LGBMClassifier(
                    n_estimators=600,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=25,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=int(seed),
                    n_jobs=2,
                    verbose=-1,
                ),
            )
        )
    return models


def _fit_predict(model, X_tr, y_tr, X_te, n_classes):
    model.fit(X_tr, y_tr)
    p = np.asarray(model.predict_proba(X_te), dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != n_classes:
        idx = p.argmax(axis=1) if p.ndim == 2 else np.zeros(len(X_te), dtype=int)
        p = np.eye(n_classes)[idx]
    return model, p


# ---------------------------------------------------------------------------
# Blend-weight search (coordinate ascent on the simplex)
# ---------------------------------------------------------------------------
def _learn_blend_weights(oof, names, y):
    n = len(names)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    w = np.ones(n, dtype=np.float64) / n

    def _acc(weights):
        prob = np.zeros_like(oof[names[0]])
        for i, nm in enumerate(names):
            prob += oof[nm] * weights[i]
        return float((prob.argmax(1) == y).mean())

    best = _acc(w)
    for _ in range(60):
        improved = False
        for i in range(n):
            for delta in (0.20, -0.20, 0.10, -0.10, 0.05, -0.05):
                w2 = w.copy()
                w2[i] = max(0.0, w2[i] + delta)
                s = float(w2.sum())
                if s <= 1e-9:
                    continue
                w2 /= s
                acc = _acc(w2)
                if acc > best + 1e-6:
                    best = acc
                    w = w2
                    improved = True
        if not improved:
            break
    return [float(x) for x in w]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    schema = _load_schema(data_dir)
    df = _read_features(data_dir, schema)
    y_raw = _read_labels(data_dir)
    if len(y_raw) != len(df):
        raise RuntimeError(
            f"[train] features rows ({len(df)}) != labels rows ({len(y_raw)})"
        )

    labels_arr = np.unique(np.asarray(y_raw))
    labels_list = labels_arr.tolist()
    l2i = {l: i for i, l in enumerate(labels_list)}
    y = np.array([l2i[v] for v in y_raw], dtype=np.int64)
    n_classes = len(labels_list)

    X, ohe = build_features(df, ohe=None, fit=True)
    print(
        f"[train] rows={len(df)} n_classes={n_classes} X={X.shape} "
        f"labels={labels_list} lgb={HAS_LGB}"
    )

    # 5-fold OOF for base models
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_ids = np.full(len(y), -1, dtype=int)
    for k, (_, va) in enumerate(skf.split(X, y)):
        fold_ids[va] = k

    model_names = [nm for nm, _ in _make_models(0, n_classes)]
    oof = {nm: np.zeros((len(y), n_classes), dtype=np.float64) for nm in model_names}

    for k in range(5):
        tr = fold_ids != k
        va = fold_ids == k
        if tr.sum() == 0 or va.sum() == 0:
            continue
        for nm, m in _make_models(100 + k, n_classes):
            try:
                _, p = _fit_predict(m, X[tr], y[tr], X[va], n_classes)
                oof[nm][va] = p
            except Exception as exc:
                print(f"[train] fold {k} model {nm} failed: {exc}")

    accs = {nm: float((oof[nm].argmax(1) == y).mean()) for nm in model_names}
    print("[train] OOF " + " | ".join(f"{nm}={accs[nm]:.4f}" for nm in model_names))

    weights = _learn_blend_weights(oof, model_names, y)
    if len(weights) == len(model_names):
        blend_prob = np.zeros((len(y), n_classes), dtype=np.float64)
        for i, nm in enumerate(model_names):
            blend_prob += oof[nm] * weights[i]
        blend_acc = float((blend_prob.argmax(1) == y).mean())
        print(
            f"[train] blend weights={np.round(weights, 3).tolist()} "
            f"acc={blend_acc:.4f}"
        )
    else:
        weights = [1.0 / len(model_names)] * len(model_names)

    # Refit each model on the full training data
    final_models = []
    for nm, m in _make_models(777, n_classes):
        try:
            m.fit(X, y)
            final_models.append((nm, m))
        except Exception as exc:
            print(f"[train] final {nm} failed: {exc}")

    if not final_models:
        raise RuntimeError("[train] no base model trained successfully")

    bundle = {
        "ohe": ohe,
        "models": final_models,
        "model_names": [nm for nm, _ in final_models],
        "blend_weights": weights,
        "labels": labels_list,
        "n_classes": n_classes,
        "has_lgb": HAS_LGB,
    }
    path = os.path.join(model_dir, "bundle.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[train] saved bundle -> {path} " f"(models={[nm for nm,_ in final_models]})"
    )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict(data_dir, model_dir, output_path):
    path = os.path.join(model_dir, "bundle.pkl")
    if not os.path.exists(path):
        raise RuntimeError(f"[predict] missing bundle: {path}")
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    schema = _load_schema(data_dir)
    df = _read_features(data_dir, schema)
    X, _ = build_features(df, ohe=bundle["ohe"], fit=False)
    n_classes = int(bundle["n_classes"])
    labels = bundle["labels"]
    names = bundle["model_names"]
    weights = np.asarray(bundle["blend_weights"], dtype=np.float64)
    models = dict(bundle["models"])

    if len(weights) != len(names):
        weights = np.ones(len(names), dtype=np.float64) / max(len(names), 1)
    if weights.sum() <= 0:
        weights = np.ones(len(names), dtype=np.float64) / max(len(names), 1)
    else:
        weights = weights / weights.sum()

    prob = np.zeros((len(df), n_classes), dtype=np.float64)
    used = []
    for i, nm in enumerate(names):
        m = models.get(nm)
        if m is None:
            continue
        try:
            p = np.asarray(m.predict_proba(X), dtype=np.float64)
            if p.ndim != 2 or p.shape[1] != n_classes:
                idx = p.argmax(axis=1) if p.ndim == 2 else np.zeros(len(X), int)
                p = np.eye(n_classes)[idx]
            prob += p * weights[i]
            used.append(nm)
        except Exception as exc:
            print(f"[predict] model {nm} failed: {exc}")

    if not used:
        print("[predict] no model succeeded; falling back to majority class")
        prob = np.zeros((len(df), n_classes), dtype=np.float64)
        prob[:, 0] = 1.0
    else:
        w_sum = float(weights[[names.index(nm) for nm in used]].sum())
        if w_sum > 0:
            prob = prob / w_sum
        print(f"[predict] used={used} weights={np.round(weights, 3).tolist()}")

    idx = prob.argmax(axis=1)
    pred_labels = [labels[int(i)] for i in idx]

    out_df = pd.DataFrame(
        {
            "row_id": np.arange(len(df), dtype=np.int64),
            "prediction": pred_labels,
        }
    )
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[predict] wrote {output_path} ({len(out_df)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Coupon acceptance prediction")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--model", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.data, args.model)
    elif args.cmd == "predict":
        predict(args.data, args.model, args.output)
    else:
        ap.error("unknown command")


if __name__ == "__main__":
    main()
