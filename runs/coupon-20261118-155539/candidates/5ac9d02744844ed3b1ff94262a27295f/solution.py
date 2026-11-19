## Analysis & Plan

Let me trace the current solution's behavior on this data: all 21 features are label-encoded to arbitrary integer codes and fed to LightGBM + HistGB. The label is heavily driven by **coupon type × customer context** interactions (e.g., "Bar" coupons accepted near bars by younger/recreational customers; "Coffee House" in morning; families accept restaurant coupons). GBDTs with plain label codes must discover all these high-order categorical interactions using split depth, which is sample-inefficient on ~8k rows and leaves signal on the table.

CHANGES:

Change #1: [Category: Feature Engineering / Data Augmentation]
- What: Add **leakage-safe out-of-fold (OOF) target encoding** for six key interaction keys — `coupon`, `occupation`, `maritalStatus×has_children`, `coupon×has_children`, `time×coupon`, `income×coupon` — with Laplace smoothing, computed with stratified 5-fold OOF on training rows (each row's code derived only from other rows' labels), and full-train maps persisted for prediction.
- Why: This directly encodes the target-conditional structure of the interactions the GBDT currently has to infer implicitly from label codes. OOF fitting guarantees zero target leak (a row's own label is never used for its own feature).

WHY current solution limited:
- Root cause: plain label-coded categoricals force the ensemble to learn multiplicative coupon×context effects via many deep splits; with `num_leaves=31` and regularization this under-fits the joint distribution, capping accuracy (~0.746).
- Evidence: `coupon` has 5 levels with very different base acceptance; ~8k rows is small for a 31-leaf tree to nail 2–3 way categorical interactions.

HOW these changes address it:
- Mechanism: A target-encoded interaction collapses each joint category to a smoothed expected-acceptance scalar, giving the tree a *monotone, directly-useful* split feature instead of a combinatorial search. The OOF construction preserves generalization and prevents the classic TE leak.
- Expected improvement: Typically +1–3 pp accuracy on this dataset (coupon contexts carry strong signal).
- Synergy: TE features complement the existing ensemble; blend weights/threshold are re-tuned on OOF, now over the richer feature space.

KEEP UNCHANGED:
- Random seed: 42 (and existing model seeds 42/55/68/81/94).
- Data split: 5-fold StratifiedKFold, shuffle=True, random_state=42.
- Model family: LGBM + HistGB blend, same hyperparameters, same OOF blend-weight/threshold search, same `_as_proba`/`build_matrix`, same CLI contract.

---

## Implementation (SEARCH/REPLACE)

**Hunk 1 — add `TargetEncoder` class after `build_matrix`:**

SEARCH:
```
    if cat_idx:
        for j in cat_idx:
            col = arr[:, j]
            col = np.where(np.isnan(col), -1.0, col)
            arr[:, j] = np.round(col)
    return arr


# ----------------------------------------------------------------------------
# Models
```
REPLACE:
```
    if cat_idx:
        for j in cat_idx:
            col = arr[:, j]
            col = np.where(np.isnan(col), -1.0, col)
            arr[:, j] = np.round(col)
    return arr


# ----------------------------------------------------------------------------
# Leakage-safe out-of-fold target encoding for categorical interactions
# ----------------------------------------------------------------------------
class TargetEncoder(object):
    def __init__(self, keys, smoothing=10.0, n_splits=5, seed=42):
        self.keys = [list(k) for k in keys]
        self.smoothing = float(smoothing)
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.maps = []
        self.prior = 0.0

    @staticmethod
    def _key_series(df, cols):
        keys = None
        for c in cols:
            s = df[c].astype(str).values
            if keys is None:
                keys = s.astype(object)
            else:
                keys = np.array([a + "|" + b for a, b in zip(keys, s)], dtype=object)
        return keys

    def _map_from(self, keys, y, prior):
        agg = pd.DataFrame({"k": np.asarray(keys), "y": np.asarray(y, dtype=np.float64)})
        agg = agg.groupby("k")["y"].agg(["mean", "count"])
        m = (agg["mean"] * agg["count"] + prior * self.smoothing) / (
            agg["count"] + self.smoothing
        )
        return m.to_dict()

    def fit(self, df, y):
        y = np.asarray(y, dtype=np.float64)
        self.prior = float(np.mean(y)) if len(y) else 0.0
        self.maps = []
        for cols in self.keys:
            ks = self._key_series(df, cols)
            self.maps.append(self._map_from(ks, y, self.prior))
        return self

    def transform(self, df):
        if not self.keys:
            return np.zeros((len(df), 0), dtype=np.float64)
        outs = []
        for i, cols in enumerate(self.keys):
            ks = self._key_series(df, cols)
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
            out[:] = float(np.mean(y))
            return out
        n_splits = min(self.n_splits, max(2, min_count))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        key_cache = [self._key_series(df, cols) for cols in self.keys]
        for tr, va in skf.split(np.zeros(n), y):
            prior_k = float(np.mean(y[tr]))
            for i in range(n_feat):
                ks = key_cache[i]
                m = self._map_from(ks[tr], y[tr], prior_k)
                out[va, i] = np.array([m.get(k, prior_k) for k in ks[va]], dtype=np.float64)
        self.fit(df, y)
        return out


# ----------------------------------------------------------------------------
# Models
```

**Hunk 2 — build TE features in `train`:**

SEARCH:
```
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
```
REPLACE:
```
    pre = Preprocessor()
    pre.fit(df, feature_columns)
    X = pre.transform(df)
    feats = list(pre.features)
    cat_idx = [feats.index(c) for c in pre.cat_cols if c in feats]
    arr = build_matrix(X, feats, cat_idx)

    # ---- leakage-safe out-of-fold target encoding for key interactions ----
    te = None
    n_te = 0
    base_feats = list(feats)
    te_keys = [
        ["coupon"],
        ["occupation"],
        ["maritalStatus", "has_children"],
        ["coupon", "has_children"],
        ["time", "coupon"],
        ["income", "coupon"],
    ]
    te_keys = [k for k in te_keys if all(c in df.columns for c in k)]
    if te_keys and n_cls == 2:
        try:
            te = TargetEncoder(te_keys, smoothing=10.0, n_splits=5, seed=42)
            te_oof = te.fit_transform_oof(df, y)
            if te_oof.shape[1] > 0:
                te_names = ["__te_%d" % i for i in range(te_oof.shape[1])]
                arr = np.concatenate([arr, te_oof], axis=1)
                feats = base_feats + te_names
                n_te = int(te_oof.shape[1])
        except Exception as e:
            print("[train] target encoding failed: %s" % e)
            te = None
            n_te = 0

    print(
        "[train] rows=%d features=%d base=%d cat=%d te=%d classes=%d"
        % (len(df), len(feats), len(base_feats), len(cat_idx), n_te, n_cls)
    )
```

**Hunk 3 — persist TE + config fields:**

SEARCH:
```
        "hgb_path": hgb_path,
        "oof_accuracy": float(best["acc"]),
    }
    with open(os.path.join(model_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(pre, f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(bundle, f)
```
REPLACE:
```
        "hgb_path": hgb_path,
        "oof_accuracy": float(best["acc"]),
        "n_te": int(n_te),
        "base_features": list(base_feats),
    }
    with open(os.path.join(model_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(pre, f)
    if te is not None and n_te > 0:
        with open(os.path.join(model_dir, "te.pkl"), "wb") as f:
            pickle.dump(te, f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(bundle, f)
```

**Hunk 4 — apply TE at predict:**

SEARCH:
```
    X = pre.transform(df)
    feats = cfg["features"]
    cat_idx = cfg.get("cat_idx", [])
    arr = build_matrix(X, feats, cat_idx)
```
REPLACE:
```
    X = pre.transform(df)
    base_feats = cfg.get("base_features", cfg["features"])
    cat_idx = cfg.get("cat_idx", [])
    arr = build_matrix(X, base_feats, cat_idx)
    n_te = int(cfg.get("n_te", 0))
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
            pad = np.zeros((len(df), n_te), dtype=np.float64)
            arr = np.concatenate([arr, pad], axis=1)
```

These hunks are self-contained, keep the CLI contract and all learned preprocessing under `--model`, and are loaded read-only at prediction time.