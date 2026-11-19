#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Coupon acceptance prediction.

Usage:
  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv
"""

import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def load_schema(data_dir):
    path = os.path.join(data_dir, "schema.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def get_feature_columns(schema, df):
    cols = None
    if isinstance(schema, dict):
        for key in ("feature_columns", "features", "columns", "feature_names"):
            v = schema.get(key)
            if isinstance(v, list) and len(v) > 0:
                cols = [str(c) for c in v]
                break
        if cols is None:
            mod = schema.get("modalities")
            if isinstance(mod, dict):
                for _, spec in mod.items():
                    if isinstance(spec, dict) and isinstance(spec.get("columns"), list):
                        cols = [str(c) for c in spec["columns"]]
                        break
    if cols is None:
        img_cols = []
        if isinstance(schema, dict) and isinstance(schema.get("image_columns"), list):
            img_cols = [str(c) for c in schema["image_columns"]]
        cols = [c for c in df.columns if c not in img_cols]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = list(df.columns)
    return cols


# ----------------------------------------------------------------------------
# Preprocessor: numeric coercion + categorical label coding, fitted once
# ----------------------------------------------------------------------------
class Preprocessor(object):
    def __init__(self):
        self.features = []
        self.numeric_cols = []
        self.cat_cols = []
        self.num_fill = {}
        self.cat_maps = {}

    @staticmethod
    def _as_str(s):
        return s.where(s.notna(), "__MISSING__").astype(str).str.strip()

    def fit(self, df, feature_columns):
        self.features = list(feature_columns)
        self.numeric_cols = []
        self.cat_cols = []
        self.num_fill = {}
        self.cat_maps = {}

        for c in self.features:
            s = df[c]
            num = None
            try:
                num = pd.to_numeric(s, errors="coerce")
            except Exception:
                num = None

            is_num = False
            if num is not None:
                orig_na = int(s.isna().sum())
                new_na = int(num.isna().sum())
                is_num = new_na <= orig_na

            if is_num:
                self.numeric_cols.append(c)
                med = float(num.median()) if num.notna().any() else 0.0
                if not np.isfinite(med):
                    med = 0.0
                self.num_fill[c] = med
            else:
                self.cat_cols.append(c)
                sv = self._as_str(s)
                uniq = sorted(set(sv.values.tolist()))
                uniq = [u for u in uniq if u != "__MISSING__"]
                vocab = ["__MISSING__"] + uniq
                self.cat_maps[c] = {v: i for i, v in enumerate(vocab)}
        return self

    def transform(self, df):
        n = len(df)
        X = np.zeros((n, len(self.features)), dtype=np.float64)
        for j, c in enumerate(self.features):
            if c in self.num_fill:
                try:
                    v = pd.to_numeric(df[c], errors="coerce").values.astype(np.float64)
                except Exception:
                    v = np.full(n, np.nan, dtype=np.float64)
                v = np.where(np.isnan(v), self.num_fill[c], v)
                X[:, j] = v
            else:
                m = self.cat_maps.get(c, {"__MISSING__": 0})
                sv = self._as_str(df[c]).values
                X[:, j] = np.array([m.get(v, 0) for v in sv], dtype=np.float64)
        return X


def build_matrix(X, feats, cat_idx):
    arr = np.array(X, dtype=np.float64, copy=True)
    arr = np.nan_to_num(arr, nan=-999.0, posinf=1e9, neginf=-1e9)
    if cat_idx:
        for j in cat_idx:
            col = arr[:, j]
            col = np.where(col < 0, 0.0, col)
            arr[:, j] = np.round(col)
    return arr


# ----------------------------------------------------------------------------
# Leakage-safe out-of-fold target encoding for categorical interactions
# ----------------------------------------------------------------------------
def _canon_values(series):
    out = np.empty(len(series), dtype=object)
    for i, v in enumerate(series.values):
        if v is None:
            out[i] = "__NA__"
            continue
        try:
            if v != v:
                out[i] = "__NA__"
                continue
        except Exception:
            pass
        sv = str(v).strip()
        try:
            fv = float(sv)
            if np.isfinite(fv):
                out[i] = "%g" % fv
            else:
                out[i] = sv
        except Exception:
            out[i] = sv
    return out


class TargetEncoder(object):
    def __init__(self, keys, smoothing=10.0, n_splits=5, seed=42):
        self.keys = [list(k) for k in keys]
        self.smoothing = float(smoothing)
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.maps = []
        self.prior = 0.0

    @staticmethod
    def _key_series(cache, idx):
        # combine pre-canonicalized columns for the given row indices
        cols = cache
        if not cols:
            return np.array([], dtype=object)
        n = len(idx)
        acc = None
        for arr in cols:
            part = np.asarray(arr, dtype=object)[idx]
            if acc is None:
                acc = part
            else:
                acc = np.array([a + "|" + b for a, b in zip(acc, part)], dtype=object)
        if acc is None:
            return np.array([""] * n, dtype=object)
        return acc

    def _map_from(self, keys, y, prior):
        dfk = pd.DataFrame(
            {"k": np.asarray(keys, dtype=object), "y": np.asarray(y, dtype=np.float64)}
        )
        g = dfk.groupby("k")["y"].agg(["mean", "count"])
        m = (g["mean"] * g["count"] + prior * self.smoothing) / (
            g["count"] + self.smoothing
        )
        return m.to_dict()

    def fit(self, df, y):
        y = np.asarray(y, dtype=np.float64)
        self.prior = float(np.mean(y)) if len(y) else 0.0
        self.maps = []
        all_idx = np.arange(len(df))
        for cols in self.keys:
            cache = [_canon_values(df[c]) for c in cols]
            ks = self._key_series(cache, all_idx)
            self.maps.append(self._map_from(ks, y, self.prior))
        return self

    def transform(self, df):
        if not self.keys:
            return np.zeros((len(df), 0), dtype=np.float64)
        all_idx = np.arange(len(df))
        outs = []
        for i, cols in enumerate(self.keys):
            cache = [_canon_values(df[c]) for c in cols]
            ks = self._key_series(cache, all_idx)
            m = self.maps[i]
            prior = self.prior
            outs.append(np.array([m.get(k, prior) for k in ks], dtype=np.float64))
        return np.stack(outs, axis=1)

    def fit_transform_oof(self, df, y):
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        n_feat = len(self.keys)
        out = np.zeros((n, n_feat), dtype=np.float64)
        if n_feat == 0 or n == 0:
            self.fit(df, y)
            return out

        _, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min()) if len(counts) else n
        if min_count < 2:
            self.fit(df, y)
            out[:] = float(np.mean(y)) if n else 0.0
            return out

        n_splits = min(self.n_splits, max(2, min_count))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        caches = [[_canon_values(df[c]) for c in cols] for cols in self.keys]

        for tr, va in skf.split(np.zeros(n), y):
            prior_k = float(np.mean(y[tr])) if len(tr) else 0.0
            for i in range(n_feat):
                ks_tr = self._key_series(caches[i], tr)
                ks_va = self._key_series(caches[i], va)
                m = self._map_from(ks_tr, y[tr], prior_k)
                out[va, i] = np.array(
                    [m.get(k, prior_k) for k in ks_va], dtype=np.float64
                )

        self.fit(df, y)
        return out


# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
def fit_lgb(X, y, cat_idx, seed):
    params = dict(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        min_child_weight=1e-3,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        cat_smooth=10.0,
        cat_l2=10.0,
        max_cat_threshold=32,
        random_state=seed,
        n_jobs=2,
        verbose=-1,
    )
    m = lgb.LGBMClassifier(**params)
    m.fit(
        X,
        y,
        categorical_feature=list(cat_idx) if cat_idx else "auto",
    )
    return m


def fit_hgb(X, y, cat_mask, seed):
    params = dict(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        max_bins=255,
        random_state=seed,
        early_stopping=False,
    )
    if cat_mask is not None and cat_mask.any():
        try:
            m = HistGradientBoostingClassifier(categorical_features=cat_mask, **params)
            m.fit(X, y)
            return m
        except Exception:
            pass
    m = HistGradientBoostingClassifier(**params)
    m.fit(X, y)
    return m


def predict_proba_lgb(model, X):
    p = model.predict_proba(X)
    return np.asarray(p[:, 1], dtype=np.float64)


def predict_proba_hgb(model, X):
    p = model.predict_proba(X)
    return np.asarray(p[:, 1], dtype=np.float64)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def cmd_train(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, "features.csv")
    lab_path = os.path.join(data_dir, "labels.csv")
    schema = load_schema(data_dir)

    df = pd.read_csv(feat_path)
    lab = pd.read_csv(lab_path)

    if "label" in lab.columns:
        y_raw = lab["label"].values
    else:
        y_raw = lab.iloc[:, 0].values

    if len(y_raw) != len(df):
        raise ValueError(
            "features/labels length mismatch: %d vs %d" % (len(df), len(y_raw))
        )

    classes = sorted(set([str(v) for v in y_raw]))
    label_map = list(classes)
    cls_index = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_index[str(v)] for v in y_raw], dtype=np.int64)
    n_cls = len(label_map)

    feature_columns = get_feature_columns(schema, df)

    pre = Preprocessor()
    pre.fit(df, feature_columns)
    X = pre.transform(df)
    base_feats = list(pre.features)
    cat_idx = [base_feats.index(c) for c in pre.cat_cols if c in base_feats]
    arr = build_matrix(X, base_feats, cat_idx)
    base_arr = arr

    # ---- leakage-safe OOF target encoding for key interactions ----
    te = None
    n_te = 0
    te_keys = [
        ["coupon"],
        ["occupation"],
        ["maritalStatus", "has_children"],
        ["coupon", "has_children"],
        ["time", "coupon"],
        ["income", "coupon"],
        ["coupon", "expiration"],
        ["age", "coupon"],
    ]
    te_keys = [k for k in te_keys if all(c in df.columns for c in k)]
    if te_keys and n_cls == 2:
        try:
            cand = TargetEncoder(te_keys, smoothing=10.0, n_splits=5, seed=42)
            te_oof = cand.fit_transform_oof(df, y)
            if te_oof.shape[1] > 0:
                arr = np.concatenate([arr, te_oof], axis=1)
                n_te = int(te_oof.shape[1])
                te = cand
        except Exception as e:
            print("[train] target encoding failed: %s" % e)
            te = None
            n_te = 0

    feats = base_feats + ["__te_%d" % i for i in range(n_te)]

    print(
        "[train] rows=%d base=%d cat=%d te=%d classes=%d"
        % (len(df), len(base_feats), len(cat_idx), n_te, n_cls)
    )

    if n_cls < 2:
        bundle = {
            "features": feats,
            "base_features": base_feats,
            "cat_idx": cat_idx,
            "n_te": n_te,
            "label_map": label_map,
            "blend_w": 1.0,
            "threshold": 0.5,
            "lgb_path": None,
            "hgb_path": None,
            "constant_index": 0,
            "oof_accuracy": 1.0,
        }
        with open(os.path.join(model_dir, "preprocessor.pkl"), "wb") as f:
            pickle.dump(pre, f)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(bundle, f)
        print("[train] single class; wrote constant model")
        return

    Xf = arr
    cat_mask = np.zeros(Xf.shape[1], dtype=bool)
    for j in cat_idx:
        cat_mask[j] = True

    seeds = [42, 55, 68]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(y), dtype=np.float64)
    oof_hgb = np.zeros(len(y), dtype=np.float64)
    has_l = HAS_LGB
    use_te = bool(te_keys) and n_cls == 2 and n_te > 0

    for fold_i, (tr, va) in enumerate(skf.split(Xf, y)):
        # Leakage-safe per-fold target encoding: fit on training-fold rows only.
        if use_te:
            fold_te = TargetEncoder(te_keys, smoothing=10.0, n_splits=5, seed=42)
            fold_te.fit(df.iloc[tr], y[tr])
            Xtr = np.concatenate([base_arr[tr], fold_te.transform(df.iloc[tr])], axis=1)
            Xva = np.concatenate([base_arr[va], fold_te.transform(df.iloc[va])], axis=1)
        else:
            Xtr, Xva = Xf[tr], Xf[va]
        ytr = y[tr]

        if has_l:
            pl = np.zeros(len(va), dtype=np.float64)
            for sd in seeds:
                m = fit_lgb(Xtr, ytr, cat_idx, sd)
                pl += predict_proba_lgb(m, Xva)
            oof_lgb[va] = pl / float(len(seeds))

        ph = np.zeros(len(va), dtype=np.float64)
        for sd in seeds:
            m = fit_hgb(Xtr, ytr, cat_mask, sd)
            ph += predict_proba_hgb(m, Xva)
        oof_hgb[va] = ph / float(len(seeds))

        print("[train] fold %d/%d done" % (fold_i + 1, 5))

    # ---- blend weight + threshold search on OOF ----
    if has_l:
        w_grid = [i / 20.0 for i in range(21)]
    else:
        w_grid = [0.0]

    thr_grid = [0.30 + 0.005 * i for i in range(81)]

    best = {"acc": -1.0, "w": 1.0 if has_l else 0.0, "thr": 0.5}
    for w in w_grid:
        p = w * oof_lgb + (1.0 - w) * oof_hgb if has_l else oof_hgb
        # fast sweep over thresholds using sorted probabilities
        for t in thr_grid:
            pred = (p >= t).astype(np.int64)
            acc = float(np.mean(pred == y))
            if acc > best["acc"]:
                best = {"acc": acc, "w": float(w), "thr": float(t)}

    print(
        "[train] oof acc=%.5f w=%.2f thr=%.3f" % (best["acc"], best["w"], best["thr"])
    )

    # ---- final refit on all training rows ----
    lgb_models = []
    if has_l:
        for sd in seeds:
            lgb_models.append(fit_lgb(Xf, y, cat_idx, sd))

    hgb_models = []
    for sd in seeds:
        hgb_models.append(fit_hgb(Xf, y, cat_mask, sd))

    lgb_path = None
    hgb_path = None
    if has_l:
        lgb_path = "lgb_models.pkl"
        with open(os.path.join(model_dir, lgb_path), "wb") as f:
            pickle.dump(lgb_models, f)
    hgb_path = "hgb_models.pkl"
    with open(os.path.join(model_dir, hgb_path), "wb") as f:
        pickle.dump(hgb_models, f)

    bundle = {
        "features": feats,
        "base_features": base_feats,
        "cat_idx": cat_idx,
        "n_te": n_te,
        "label_map": label_map,
        "blend_w": best["w"] if has_l else 0.0,
        "threshold": float(best["thr"]),
        "lgb_path": lgb_path,
        "hgb_path": hgb_path,
        "constant_index": 0,
        "oof_accuracy": float(best["acc"]),
    }

    with open(os.path.join(model_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(pre, f)
    if te is not None and n_te > 0:
        with open(os.path.join(model_dir, "te.pkl"), "wb") as f:
            pickle.dump(te, f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(bundle, f)

    print("[train] saved model to %s" % model_dir)


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def cmd_predict(data_dir, model_dir, output_path):
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as f:
        pre = pickle.load(f)

    label_map = cfg.get("label_map", [0, 1])
    base_feats = cfg.get("base_features", cfg.get("features", []))
    cat_idx = cfg.get("cat_idx", [])
    n_te = int(cfg.get("n_te", 0))
    w = float(cfg.get("blend_w", 0.0))
    thr = float(cfg.get("threshold", 0.5))

    df = None
    if os.path.isdir(data_dir):
        fp = os.path.join(data_dir, "features.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
    if df is None:
        df = pd.DataFrame()

    n = len(df)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if n == 0:
        pd.DataFrame({"row_id": [], "prediction": []}).to_csv(output_path, index=False)
        print("[predict] empty input; wrote header only")
        return

    for c in base_feats:
        if c not in df.columns:
            df[c] = np.nan

    X = pre.transform(df)
    arr = build_matrix(X, base_feats, cat_idx)

    te_path = os.path.join(model_dir, "te.pkl")
    if n_te > 0 and os.path.exists(te_path):
        try:
            with open(te_path, "rb") as f:
                te = pickle.load(f)
            te_arr = te.transform(df)
            if te_arr.shape[1] != n_te:
                raise ValueError("te width mismatch")
            arr = np.concatenate([arr, te_arr], axis=1)
        except Exception as e:
            print("[predict] te load failed: %s" % e)
            pad = np.zeros((n, n_te), dtype=np.float64)
            arr = np.concatenate([arr, pad], axis=1)

    # constant model fallback
    if cfg.get("lgb_path") is None and cfg.get("hgb_path") is None:
        idx = int(cfg.get("constant_index", 0))
        pred_idx = np.full(n, idx, dtype=np.int64)
    else:
        p = np.zeros(n, dtype=np.float64)
        accum = 0.0

        lgb_path = cfg.get("lgb_path")
        if lgb_path and HAS_LGB:
            pth = os.path.join(model_dir, lgb_path)
            if os.path.exists(pth):
                with open(pth, "rb") as f:
                    lgb_models = pickle.load(f)
                pl = np.zeros(n, dtype=np.float64)
                for m in lgb_models:
                    pl += predict_proba_lgb(m, arr)
                pl /= float(len(lgb_models))
                p += w * pl
                accum += w

        hgb_path = cfg.get("hgb_path")
        if hgb_path:
            pth = os.path.join(model_dir, hgb_path)
            if os.path.exists(pth):
                with open(pth, "rb") as f:
                    hgb_models = pickle.load(f)
                ph = np.zeros(n, dtype=np.float64)
                for m in hgb_models:
                    ph += predict_proba_hgb(m, arr)
                ph /= float(len(hgb_models))
                p += (1.0 - w) * ph
                accum += 1.0 - w

        if accum > 0:
            p /= accum
        else:
            p[:] = 0.5

        pred_idx = (p >= thr).astype(np.int64)

    # guard clause index range
    n_cls = len(label_map)
    pred_idx = np.clip(pred_idx, 0, max(0, n_cls - 1))
    labels = [label_map[int(i)] for i in pred_idx]

    out = pd.DataFrame({"row_id": np.arange(n, dtype=np.int64), "prediction": labels})
    out.to_csv(output_path, index=False)
    print("[predict] wrote %d rows to %s" % (n, output_path))


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = ap.parse_args()

    if args.cmd == "train":
        cmd_train(args.data, args.model)
    elif args.cmd == "predict":
        cmd_predict(args.data, args.model, args.output)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()