#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Coupon acceptance prediction: train / predict CLI.

ASCII-only source. Schema-driven preprocessing fitted on training rows only,
persisted to the model directory and reloaded read-only at prediction time.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    _HAS_LGBM = True
except Exception:
    lgb = None
    _HAS_LGBM = False

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def load_schema(data_dir):
    path = os.path.join(data_dir, "schema.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def get_feature_columns(schema, df):
    cols = None
    if isinstance(schema, dict):
        cols = schema.get("feature_columns")
    if not cols:
        cols = list(df.columns)
    return [c for c in cols if c in df.columns]


# ----------------------------------------------------------------------------
# Preprocessing (fit on training rows only, then pickled)
# ----------------------------------------------------------------------------
class Preprocessor(object):
    def __init__(self):
        self.numeric_cols = []
        self.cat_cols = []
        self.cat_maps = {}
        self.features = []

    @staticmethod
    def _is_missing(v):
        if v is None:
            return True
        if isinstance(v, float) and np.isnan(v):
            return True
        return False

    def fit(self, df, feature_columns):
        self.numeric_cols = []
        self.cat_cols = []
        self.cat_maps = {}
        for c in feature_columns:
            s = df[c]
            num = pd.to_numeric(s, errors="coerce")
            frac_ok = float(num.notna().mean()) if len(s) else 0.0
            if frac_ok >= 0.99:
                self.numeric_cols.append(c)
            else:
                self.cat_cols.append(c)
                vals = []
                for v in s.tolist():
                    if self._is_missing(v):
                        continue
                    vals.append(str(v))
                uniq = sorted(set(vals))
                self.cat_maps[c] = {v: (i + 1) for i, v in enumerate(uniq)}
        self.features = list(self.numeric_cols) + list(self.cat_cols)
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.numeric_cols:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        for c in self.cat_cols:
            mapping = self.cat_maps.get(c, {})
            codes = []
            for v in df[c].tolist():
                if self._is_missing(v):
                    codes.append(-1)
                    continue
                codes.append(int(mapping.get(str(v), -1)))
            out[c] = np.asarray(codes, dtype=np.float64)
        return out[self.features]


def build_matrix(X, features, cat_idx):
    X2 = X.reindex(columns=features)
    arr = X2.to_numpy(dtype=np.float64, copy=True)
    if arr.shape[1] == 0:
        return arr
    if cat_idx:
        for j in cat_idx:
            col = arr[:, j]
            col = np.where(np.isnan(col), -1.0, col)
            arr[:, j] = np.round(col)
    return arr


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def lgbm_params(seed):
    return {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": int(seed),
        "n_jobs": 2,
        "verbosity": -1,
    }


def fit_lgbm_one(arr, y, cat_idx, seed):
    clf = lgb.LGBMClassifier(**lgbm_params(seed))
    if cat_idx:
        clf.fit(arr, y, categorical_feature=cat_idx)
    else:
        clf.fit(arr, y)
    return clf


def fit_hgb_one(arr, y, seed):
    clf = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=int(seed),
    )
    clf.fit(arr, y)
    return clf


def _as_proba(pr, n_cls):
    pr = np.asarray(pr, dtype=np.float64)
    if pr.ndim == 1:
        if n_cls == 2:
            return np.stack([1.0 - pr, pr], axis=1)
        return np.tile(pr.reshape(-1, 1), (1, n_cls))
    return pr


# ----------------------------------------------------------------------------
# Metrics / selection
# ----------------------------------------------------------------------------
def proba_to_pred(P, threshold):
    if P.shape[1] == 2:
        return (P[:, 1] >= threshold).astype(np.int64)
    return np.argmax(P, axis=1)


def accuracy_of(P, y, threshold=0.5):
    if len(y) == 0:
        return 0.0
    pred = proba_to_pred(P, threshold)
    return float(np.mean(pred == y))


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(args):
    data_dir = args.data
    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)

    feats_path = os.path.join(data_dir, "features.csv")
    labels_path = os.path.join(data_dir, "labels.csv")
    df = pd.read_csv(feats_path)
    labels_df = pd.read_csv(labels_path)

    schema = load_schema(data_dir)
    feature_columns = get_feature_columns(schema, df)

    raw_vals = labels_df.iloc[:, 0].tolist()
    uniq = []
    seen = set()
    for v in raw_vals:
        if isinstance(v, float) and np.isnan(v):
            continue
        key = str(v)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v)
    try:
        classes = sorted(uniq)
    except Exception:
        classes = uniq
    cls_map = {}
    for i, c in enumerate(classes):
        cls_map[str(c)] = i
    y = np.asarray([cls_map[str(v)] for v in raw_vals], dtype=np.int64)
    n_cls = len(classes)

    pre = Preprocessor()
    pre.fit(df, feature_columns)
    X = pre.transform(df)
    feats = list(pre.features)
    cat_idx = [feats.index(c) for c in pre.cat_cols if c in feats]
    arr = build_matrix(X, feats, cat_idx)

    print(
        "[train] rows=%d features=%d cat=%d classes=%d"
        % (len(df), len(feats), len(cat_idx), n_cls)
    )

    use_lgbm = bool(_HAS_LGBM)
    use_hgb = True

    if n_cls <= 1:
        # degenerate label set: constant prediction
        bundle = {
            "features": feats,
            "cat_idx": cat_idx,
            "label_classes": [
                c.item() if isinstance(c, np.generic) else c for c in classes
            ],
            "n_classes": int(n_cls),
            "constant": True,
            "w_lgbm": 0.0,
            "w_hgb": 0.0,
            "threshold": 0.5,
            "lgbm_paths": [],
            "hgb_path": None,
        }
        with open(os.path.join(model_dir, "preprocessor.pkl"), "wb") as f:
            pickle.dump(pre, f)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(bundle, f)
        print("[train] single class; wrote constant model")
        return

    # ---- internal cross-validation (training rows only) ----
    n_splits = 5
    min_count = int(np.bincount(y, minlength=n_cls).min()) if n_cls > 0 else 0
    if min_count < n_splits:
        n_splits = max(2, min_count)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_l = np.zeros((len(y), n_cls), dtype=np.float64)
    oof_h = np.zeros((len(y), n_cls), dtype=np.float64)
    have_oof_l = np.zeros(len(y), dtype=bool)

    for k, (tr, va) in enumerate(skf.split(arr, y)):
        X_tr, y_tr = arr[tr], y[tr]
        X_va = arr[va]
        if use_lgbm:
            try:
                m = fit_lgbm_one(X_tr, y_tr, cat_idx, seed=100 + k)
                p = _as_proba(m.predict_proba(X_va), n_cls)
                # align class columns conservatively
                if p.shape[1] == n_cls:
                    oof_l[va] = p
                    have_oof_l[va] = True
            except Exception as e:
                print("[train] lgbm fold %d failed: %s" % (k, e))
        if use_hgb:
            try:
                m = fit_hgb_one(X_tr, y_tr, seed=200 + k)
                p = _as_proba(m.predict_proba(X_va), n_cls)
                if p.shape[1] == n_cls:
                    oof_h[va] = p
            except Exception as e:
                print("[train] hgb fold %d failed: %s" % (k, e))

    acc_l = accuracy_of(oof_l, y) if have_oof_l.any() else 0.0
    acc_h = accuracy_of(oof_h, y) if use_hgb else 0.0
    print("[train] OOF accuracy lgbm=%.4f hgb=%.4f" % (acc_l, acc_h))

    # ---- blend weight + threshold search on OOF only ----
    best = {"w": 0.0, "thr": 0.5, "acc": -1.0}
    weights_grid = np.arange(0.0, 1.0001, 0.05)
    if n_cls == 2:
        thr_grid = np.arange(0.30, 0.7001, 0.01)
    else:
        thr_grid = np.array([0.5])

    for w in weights_grid:
        w = float(w)
        if use_lgbm and use_hgb and have_oof_l.any():
            P = w * oof_l + (1.0 - w) * oof_h
        elif use_lgbm and have_oof_l.any():
            P = oof_l
        elif use_hgb:
            P = oof_h
        else:
            continue
        for thr in thr_grid:
            acc = accuracy_of(P, y, float(thr))
            if acc > best["acc"] + 1e-9:
                best = {"w": w, "thr": float(thr), "acc": float(acc)}

    if best["acc"] < 0:
        best = {"w": 0.0, "thr": 0.5, "acc": max(acc_l, acc_h)}

    if not (use_lgbm and use_hgb):
        w_lgbm = 1.0 if use_lgbm else 0.0
        w_hgb = 1.0 - w_lgbm
    else:
        w_lgbm = best["w"]
        w_hgb = 1.0 - w_lgbm

    print(
        "[train] best OOF acc=%.4f w_lgbm=%.2f w_hgb=%.2f thr=%.2f"
        % (best["acc"], w_lgbm, w_hgb, best["thr"])
    )

    # ---- final fit on all training rows ----
    lgbm_paths = []
    if use_lgbm and w_lgbm > 0.0:
        seeds = [42, 55, 68, 81, 94]
        for i, s in enumerate(seeds):
            try:
                m = fit_lgbm_one(arr, y, cat_idx, seed=s)
                name = "gbdt_lgbm_%d.txt" % i
                m.booster_.save_model(os.path.join(model_dir, name))
                lgbm_paths.append(name)
            except Exception as e:
                print("[train] final lgbm seed %d failed: %s" % (s, e))

    hgb_path = None
    if use_hgb and w_hgb > 0.0:
        seeds = [42, 55, 68, 81, 94]
        models = []
        for s in seeds:
            try:
                models.append(fit_hgb_one(arr, y, seed=s))
            except Exception as e:
                print("[train] final hgb seed %d failed: %s" % (s, e))
        if models:
            hgb_path = "gbdt_hgb.pkl"
            with open(os.path.join(model_dir, hgb_path), "wb") as f:
                pickle.dump(models, f)

    if not lgbm_paths and hgb_path is None:
        raise RuntimeError("no model could be trained")

    if not lgbm_paths:
        w_lgbm, w_hgb = 0.0, 1.0
    if hgb_path is None:
        w_lgbm, w_hgb = 1.0, 0.0

    bundle = {
        "features": feats,
        "cat_idx": cat_idx,
        "label_classes": [
            c.item() if isinstance(c, np.generic) else c for c in classes
        ],
        "n_classes": int(n_cls),
        "constant": False,
        "w_lgbm": float(w_lgbm),
        "w_hgb": float(w_hgb),
        "threshold": float(best["thr"]),
        "lgbm_paths": lgbm_paths,
        "hgb_path": hgb_path,
        "oof_accuracy": float(best["acc"]),
    }
    with open(os.path.join(model_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(pre, f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(bundle, f)

    print(
        "[train] saved model to %s (lgbm_files=%d hgb=%s)"
        % (model_dir, len(lgbm_paths), hgb_path)
    )


# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
def predict(args):
    model_dir = args.model
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as f:
        pre = pickle.load(f)

    df = pd.read_csv(os.path.join(args.data, "features.csv"))
    n = len(df)

    classes = cfg.get("label_classes", [0, 1])
    n_cls = int(cfg.get("n_classes", len(classes)))

    if n == 0:
        pd.DataFrame({"row_id": [], "prediction": []}).to_csv(args.output, index=False)
        return

    if cfg.get("constant") or n_cls <= 1:
        preds = [classes[0]] * n
        out = pd.DataFrame({"row_id": np.arange(n), "prediction": preds})
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        out.to_csv(args.output, index=False)
        return

    X = pre.transform(df)
    feats = cfg["features"]
    cat_idx = cfg.get("cat_idx", [])
    arr = build_matrix(X, feats, cat_idx)

    probs = []
    weights = []

    lgbm_paths = cfg.get("lgbm_paths") or []
    if lgbm_paths and _HAS_LGBM:
        p_list = []
        for name in lgbm_paths:
            try:
                booster = lgb.Booster(model_file=os.path.join(model_dir, name))
                pr = booster.predict(arr)
                pr = _as_proba(pr, n_cls)
                if pr.shape[1] == n_cls:
                    p_list.append(pr)
            except Exception as e:
                print("[predict] lgbm file %s failed: %s" % (name, e))
        if p_list:
            probs.append(np.mean(np.stack(p_list, axis=0), axis=0))
            weights.append(float(cfg.get("w_lgbm", 1.0)))

    hgb_name = cfg.get("hgb_path")
    if hgb_name:
        try:
            with open(os.path.join(model_dir, hgb_name), "rb") as f:
                models = pickle.load(f)
            p_list = []
            for m in models:
                pr = _as_proba(m.predict_proba(arr), n_cls)
                if pr.shape[1] == n_cls:
                    p_list.append(pr)
            if p_list:
                probs.append(np.mean(np.stack(p_list, axis=0), axis=0))
                weights.append(float(cfg.get("w_hgb", 1.0)))
        except Exception as e:
            print("[predict] hgb load failed: %s" % e)

    if not probs:
        # last resort: majority-ish constant from stored classes
        preds = [classes[0]] * n
        out = pd.DataFrame({"row_id": np.arange(n), "prediction": preds})
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        out.to_csv(args.output, index=False)
        return

    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()

    P = np.zeros((n, n_cls), dtype=np.float64)
    for wi, pi in zip(w, probs):
        P += wi * pi

    idx = proba_to_pred(P, float(cfg.get("threshold", 0.5)))
    idx = np.clip(idx, 0, len(classes) - 1)
    preds = [classes[int(i)] for i in idx]

    out = pd.DataFrame({"row_id": np.arange(n), "prediction": preds})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print("[predict] wrote %d rows to %s" % (n, args.output))


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Coupon acceptance predictor")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
