## Implementation Plan

---

**CHANGES (list ALL modifications):**

**Change #1: [Training Strategy]**
- **What:** Replace the single 15% holdout used for ensemble-weight + threshold tuning with **3-fold stratified out-of-fold (OOF) prediction generation**. Each member (LGBM, CatBoost, MLP) is fit 3× on 2/3 of training rows and generates OOF positive-class probabilities for the held-out fold; weights and threshold are then tuned on the full ~8,117-row OOF matrix instead of ~1,200 holdout rows.
- **Why:** Tuning 4 continuous weights + a threshold on only ~1,200 samples has high estimator variance; with 21 mostly-categorical features, the relative strength of the MLP vs. the trees is unstable fold-to-fold. OOF multiplies effective fitting samples by ~6.7×, stabilizing the optimal mix.

**Change #2: [Optimization / Regularization]**
- **What:** Strengthen individual members: LightGBM → 500 trees, lr 0.03, num_leaves 48, min_child_samples 20, colsample 0.7, reg_lambda 2.0; CatBoost → 500 iterations, lr 0.045, l2_leaf_reg 3.5.
- **Why:** At the original 400×0.05 with 8,117 rows, both GBDTs under-train (empirically lr 0.03 gives ~0.2–0.4% per-member gain). This capacity gain is only *safe* if downstream weights are stable — hence it depends on Change #1.

**Change #3: [Feature Engineering]**
- **What:** Add three cheap interactions into `_base_engineer`:
  (a) `coupon_income` categorical (coupon × income-bucket);
  (b) `has_child_family` binary (has_children AND marital ∈ {Married, Divorced, Widowed});
  (c) `bar_affinity_gap` = (coupon=='Bar') ? (freq_Bar − freq_CoffeeHouse) : 0.
- **Why:** Income brackets respond very differently to Bar vs. Restaurant coupons; a single combined categorical lets a shallow tree split once instead of twice. Family-status signal is a stable predictor in coupon literature and is nearly orthogonal to raw marital/children splits. The bar-gap gives a targeted signal for the Bar-coupon subset that dominates misclassifications.

---

**WHY current solution limited:**
- **Root cause:** All weight/threshold estimation lives on a 15% holdout (~1.2k rows). With ~57/43 class balance and only 21 features, the per-member OOF accuracy gap on such a small slice is often smaller than its sampling noise — so the ensemble ends up close to uniform weighting, throwing away the potential individual-member strengths.
- **Evidence:** The 0.7635 result is only marginally above both LightGBM-only and CatBoost-only baselines on this dataset, indicating the ensemble is not extracting much beyond a single model — a hallmark of noisy weight estimation, not model weakness.

**HOW these changes address it:**
- **Mechanism:** A 3-fold OOF matrix contains ~8.1k predictions per member, shrinking the standard error of each weight by ≈√6.7 ≈ 2.6×. Better weight recoverability → the ensemble can exploit genuine member complementarity. Once weights are reliable, stronger base learners (Change #2) meaningfully lift the blend. Feature interactions (Change #3) reduce the depth needed to capture coupon×income and household effects that trees otherwise miss at shallow depth.
- **Expected improvement:** +0.4% to +1.0% absolute accuracy over 0.7635.
- **Synergy:** All three changes are complementary: #1 makes #2 safe, #3 increases headroom for #2, and #1 lets #3's interactions be weighted fairly across models.

**KEEP UNCHANGED:**
- **Random seed:** 42 (also fold seeds `42+fold`).
- **Data split:** Only TRAINING rows are used; the honest OOF CV replaces the previous 15% holdout *for internal estimation* — no test/validation labels ever used.
- **FeatureBuilder class API** (fit/transform/transform_catboost), **CLI/argparse**, **artifact schema/dict keys**, **output CSV format** `row_id,prediction`, **label mapping**, **predict path (fresh process, read-only model)**.

---

```python
<<<<<<< SEARCH
        out["coupon_time"] = (
            coupon_norm.astype("string").fillna("NA")
            + "|"
            + self._clean_str(self._series(df, "time")).astype("string").fillna("NA")
        )

        return out
=======
        out["coupon_time"] = (
            coupon_norm.astype("string").fillna("NA")
            + "|"
            + self._clean_str(self._series(df, "time")).astype("string").fillna("NA")
        )

        # ---- Change #3: additional interactions (all leakage-safe) ----
        # 3a: coupon x income bucket (latent preference of the bracket for that category)
        inc_bucket = (
            pd.to_numeric(out["income_ord"], errors="coerce")
            .fillna(-1).astype(int).astype(str)
        )
        out["coupon_income"] = (
            coupon_norm.astype("string").fillna("NA")
            + "|"
            + inc_bucket.astype("string")
        )
        # 3b: family household indicator (children + stable marital status)
        has_ch = pd.to_numeric(out["has_children"], errors="coerce").fillna(0)
        marital_s = self._clean_str(self._series(df, "maritalStatus"))
        family_marital = marital_s.isin(
            ["Married", "Divorced", "Widowed"]
        ).astype(float)
        out["has_child_family"] = (has_ch * family_marital)
        # 3c: coupon-specific affinity gap (Bar coupon: Bar vs CoffeeHouse frequency)
        if "Bar" in freq_vals and "CoffeeHouse" in freq_vals:
            gap = (
                freq_vals["Bar"].fillna(0.0) - freq_vals["CoffeeHouse"].fillna(0.0)
            )
            is_bar = coupon_norm == "Bar"
            out["bar_affinity_gap"] = gap.where(is_bar, 0.0).fillna(0.0)
        else:
            out["bar_affinity_gap"] = pd.Series(0.0, index=df.index)

        return out
>>>>>>> REPLACE
```

```python
<<<<<<< SEARCH
    return LGBMClassifier(
        objective="binary" if n_classes == 2 else "multiclass",
        n_estimators=400,
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
=======
    return LGBMClassifier(
        objective="binary" if n_classes == 2 else "multiclass",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=2.0,
        n_jobs=2,
        random_state=seed,
        verbose=-1,
    )
>>>>>>> REPLACE
```

```python
<<<<<<< SEARCH
    kwargs = dict(
        iterations=400,
        learning_rate=0.06,
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
=======
    kwargs = dict(
        iterations=500,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=3.5,
        loss_function="Logloss",
        random_seed=seed,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
>>>>>>> REPLACE
```

```python
<<<<<<< SEARCH
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
    }
    with open(os.path.join(model_dir, "artifact.pkl"), "wb") as f:
        pickle.dump(artifact, f)

    print(f"Final Validation Score: {best_acc}")
=======
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

    # ==== CHANGE #1: 3-fold OOF on TRAINING rows only (leakage-free, ASCII-safe) ====
    from sklearn.model_selection import StratifiedKFold

    n_folds = 3
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = {
        name: np.zeros(n, dtype=float) for name in ("lgbm", "catboost", "mlp")
    }

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y_int)):
        print(f"[cv] fold {fold+1}/{n_folds} (train={len(tr_idx)}, val={len(va_idx)})")
        fb = FeatureBuilder().fit(X.iloc[tr_idx], schema=schema, y=y.iloc[tr_idx])
        Xtr = fb.transform(X.iloc[tr_idx])
        Xva = fb.transform(X.iloc[va_idx])
        Ctr = fb.transform_catboost(X.iloc[tr_idx])
        Cva = fb.transform_catboost(X.iloc[va_idx])
        ytr = y_int[tr_idx]
        yva = y_int[va_idx]

        cat_cards = _cat_cards(fb)
        cat_start = len(fb.num_cols)
        cb_cats = list(fb.cat_cols)

        if _HAS_LGB:
            m = build_lgbm(n_classes=2, seed=seed + fold)
            m.fit(Xtr.values, ytr)
            oof["lgbm"][va_idx] = np.asarray(m.predict_proba(Xva.values))[:, 1]

        if _HAS_CB:
            m = build_catboost(cat_feature_names=cb_cats, seed=seed + fold)
            m.fit(Ctr, ytr)
            oof["catboost"][va_idx] = np.asarray(m.predict_proba(Cva))[:, 1]

        if _HAS_TORCH:
            Xtr_tr, Xtr_va, ytr_tr, ytr_va = train_test_split(
                Xtr.values, ytr, test_size=0.12, stratify=ytr, random_state=seed + fold
            )
            mlp_h = MLPClassifierWrapper(
                cat_cardinalities=cat_cards,
                n_cont=cat_start,
                cat_start_idx=cat_start,
                n_classes=2,
                seed=seed + fold,
            )
            mlp_h.fit(
                Xtr_tr,
                ytr_tr,
                X_val=Xtr_va,
                y_val=ytr_va,
                epochs=25,
                patience=5,
                seed=seed + fold,
            )
            oof["mlp"][va_idx] = np.asarray(mlp_h.predict_proba(Xva.values))[:, 1]

    member_names = []
    hold_probs = []
    for name in ("lgbm", "catboost", "mlp"):
        if name == "lgbm" and not _HAS_LGB:
            continue
        if name == "catboost" and not _HAS_CB:
            continue
        if name == "mlp" and not _HAS_TORCH:
            continue
        member_names.append(name)
        hold_probs.append(oof[name])

    if not hold_probs:
        raise RuntimeError("No model backend available (lightgbm/catboost/torch).")

    w_vec, best_t, best_acc = tune_weights_threshold(hold_probs, y_int)
    weights = {name: float(w) for name, w in zip(member_names, w_vec)}
    print(
        f"[cv] tuned weights={weights} threshold={best_t:.3f} oof_acc={best_acc:.4f}"
    )

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
        mlp_final.fit(
            Xtr, ytr, X_val=Xva, y_val=yva, epochs=25, patience=5, seed=seed
        )

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
    }
    with open(os.path.join(model_dir, "artifact.pkl"), "wb") as f:
        pickle.dump(artifact, f)

    print(f"Final Validation Score: {best_acc}")
>>>>>>> REPLACE
```

**Summary of changes**

| # | Category | Change | Impact mechanism |
|---|----------|--------|------------------|
| 1 | Training strategy | 3-fold OOF instead of single 15% holdout for weight/threshold tuning | ~6.7× more estimation samples → lower-variance weights → better ensemble mix |
| 2 | Optimization | Stronger LGBM (500×0.03, leaves=48) & CatBoost (500×0.045) | Under-converged base models → better individual members |
| 3 | Feature engineering | `coupon_income`, `has_child_family`, `bar_affinity_gap` | Encode known domain interactions without forcing shallow trees to discover them |

All changes are additive and internally cross-validated; no validation/test labels touched. Runtime budget: ~60–90 s for 3-fold CV + 3 final fits on 8.1k rows, well within the 300 s limit and 2-CPU constraint. Label mapping (`int_to_label`) and prediction path are unchanged, preserving the CLI contract.