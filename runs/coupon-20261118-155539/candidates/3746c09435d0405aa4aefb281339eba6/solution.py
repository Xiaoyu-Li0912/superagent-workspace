## CHANGES

**Change #1: Model Architecture / Training Strategy — OOF stacker**
- **What:** Replace the hand-searched simplex blend weights + scalar threshold scan with a **logistic-regression stacking meta-learner** trained on the OOF probabilities of the three base models (`p_mlp, p_cb, p_gb`) *plus* three gate features (`direction_same`, `toCoupon_GEQ15min`, `toCoupon_GEQ25min`) and their interactions with `p_avg`. The `C` regularization is chosen via inner 5-fold CV on the OOF matrix. Saved as `stacker.joblib`, loaded read-only at predict.
- **Why:** The current approach assumes a *uniform* linear blend and a *single global* threshold. But on this dataset the ruling gate is *accessibility*: `toCoupon_GEQ15min` / `GEQ25min` / `direction_same` shift the optimal decision boundary drastically per segment (unreachable coupons are almost always rejected regardless of coupons affinity). A single shared weight/0.5-ish threshold cannot express that. A stacker with per-gate interactions lets the boundary bend by subgroup.

**Change #2: Regularization — inner-CV-tuned C**
- **What:** Inside the stacker, sweep `C ∈ {0.03, 0.1, 0.3, 1.0, 3.0}`, evaluate by 5-fold OOF accuracy, then refit on the whole OOF matrix with the best `C`.
- **Why:** With ~8k OOF rows and ~10 meta-features, small `C` prevents the linear meta-model from overfitting the noisy base-prob estimates, which is critical since base predictions are themselves fold-specific.

---

**WHY current solution limited:**
- Root cause: The final decision rule is a static linear blend + one global threshold, so segments with systematically different calibration curves (visible in `toCoupon_GEQ15min ∈ {0,1}`, `direction_same ∈ {0,1}`) share one boundary. The simplex grid search fits a scalar threshold to OOF, but that is effectively a single split of `w·p`, not a conditional split.
- Evidence: `distinct_counts` shows `toCoupon_GEQ15min`/`GEQ25min` are binary gates; and `toCoupon_GEQ5min` is constant (all ones) — so `direction`/`≥15`/`≥25` are precisely where the decision boundary should adapt, but the current pipeline only enters them as *features* of the base models, not as *decision-mode* gates.

**HOW these changes address it:**
- Mechanism: The stacker learns `P(y=1 | p_mlp, p_cb, p_gb, gate, p_avg·gate)`. The interaction terms give the meta-model per-segment slope corrections, implicitly learning different decision thresholds for reachable vs. unreachable coupons. It's calibrated (probabilistic) and trained with inner-CV to avoid overfitting.
- Expected improvement: +0.4%–1.0% absolute accuracy over the 0.7675 baseline (0.771–0.777 OOF), with lighter variance since reproducibility is deterministic (same seed=42 in base CV; stacker is convex).
- Synergy: The base models are unchanged — bagged CatBoost/LightGBM/MLP OOF still used, but consumed by a *conditional* rather than *linear* combiner.

**KEEP UNCHANGED (must explicitly list):**
- Random seed: **42** (base CV, seeds, splits identical).
- Data split: same `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and inner `test_size=0.12` early-stopping split.
- Feature engineering (`FeatureEngineer`), base model hyperparameters, epochs, bagging seeds `[42,143,244]`.
- Final refit on 100% of training rows for base models.
- Output format (`row_id,prediction`), argparse CLI, single-class guard.

---

## Implementation (SEARCH/REPLACE)

```python
<<<<<<< SEARCH
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
=======
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
>>>>>>> REPLACE
```

```python
<<<<<<< SEARCH
def _weight_simplex(step=0.1):
    n = int(round(1.0 / step))
    grid = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            grid.append((i * step, j * step, k * step))
    return grid


def _select_blend_and_threshold(p_mlp, p_cb, p_gb, y):
    best_acc, best_w, best_t = -1.0, (0.2, 0.5, 0.3), 0.5
    thresholds = np.arange(0.40, 0.6001, 0.01)
    for w in _weight_simplex(0.1):
        p = w[0] * p_mlp + w[1] * p_cb + w[2] * p_gb
        for t in thresholds:
            acc = accuracy_score(y, (p >= t).astype(int))
            if acc > best_acc + 1e-12:
                best_acc, best_w, best_t = acc, w, float(t)
    return np.array(best_w, dtype=np.float64), best_t, best_acc
=======
def _build_meta_features(p_mlp, p_cb, p_gb, df):
    """Stacking inputs: base probs + accessibility gates + gate-interactions.

    Gates are the accessibility indicators (which the base models only see
    as ordinary features).  Interactions let the meta-model learn a
    per-segment decision boundary (e.g. for unreachable coupons).
    """
    def _num(col):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(
                dtype=np.float64
            )
        return np.zeros(len(df), dtype=np.float64)

    ds = _num("direction_same")
    g15 = _num("toCoupon_GEQ15min")
    g25 = _num("toCoupon_GEQ25min")

    base = np.column_stack([p_mlp, p_cb, p_gb]).astype(np.float64)
    p_avg = base.mean(axis=1, keepdims=True)
    gates = np.column_stack([ds, g15, g25])
    inter = p_avg * gates
    return np.hstack([base, gates, inter])


def _fit_stacker(X_meta, y):
    """Fit an L2 logistic stacker; pick C by 5-fold OOF accuracy."""
    best_c, best_acc = 1.0, -1.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        oof = np.zeros(len(y), dtype=np.float64)
        for tr, va in skf.split(X_meta, y):
            lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
            lr.fit(X_meta[tr], y[tr])
            oof[va] = lr.predict_proba(X_meta[va])[:, 1]
        acc = accuracy_score(y, (oof >= 0.5).astype(int))
        if acc > best_acc + 1e-12:
            best_acc, best_c = acc, C
    lr = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs")
    lr.fit(X_meta, y)
    return lr, float(best_acc)
>>>>>>> REPLACE
```

```python
<<<<<<< SEARCH
    weights, thr, oof_acc = _select_blend_and_threshold(oof_mlp, oof_cb, oof_gb, y_enc)
    print(
        f"OOF ensemble weights={weights.tolist()} threshold={thr:.2f} "
        f"accuracy={oof_acc:.4f}"
    )
=======
    X_meta_oof = _build_meta_features(oof_mlp, oof_cb, oof_gb, features)
    stacker, oof_acc = _fit_stacker(X_meta_oof, y_enc)
    print(
        f"OOF stacker C={stacker.C} meta_accuracy={oof_acc:.4f}"
    )
    # Keep backward-compatible placeholders so downstream code still runs.
    weights = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    thr = 0.5
    joblib.dump(stacker, os.path.join(model_dir, "stacker.joblib"))
>>>>>>> REPLACE
```

```python
<<<<<<< SEARCH
    joblib.dump(
        {
            "classes": classes.tolist(),
            "pos_idx": int(POS_IDX),
            "weights": weights.tolist(),
            "threshold": float(thr),
            "input_dim": int(X_all.shape[1]),
            "cat_cols": list(fe_final.nominal_cols_),
            "n_catboost": len(cb_models),
            "n_mlp": len(mlp_models),
            "feature_names": list(fe_final.feature_names_),
        },
        os.path.join(model_dir, "meta.joblib"),
    )
=======
    joblib.dump(
        {
            "classes": classes.tolist(),
            "pos_idx": int(POS_IDX),
            "weights": weights.tolist(),
            "threshold": float(thr),
            "use_stacker": True,
            "input_dim": int(X_all.shape[1]),
            "cat_cols": list(fe_final.nominal_cols_),
            "n_catboost": len(cb_models),
            "n_mlp": len(mlp_models),
            "feature_names": list(fe_final.feature_names_),
        },
        os.path.join(model_dir, "meta.joblib"),
    )
>>>>>>> REPLACE
```

```python
<<<<<<< SEARCH
    wsum = max(w_mlp + w_cb + w_gb, 1e-12)
    proba = (w_mlp * p_mlp + w_cb * p_cb + w_gb * p_gb) / wsum
    is_pos = (proba >= thr).astype(int)
=======
    if meta.get("use_stacker"):
        stacker = joblib.load(os.path.join(model_dir, "stacker.joblib"))
        X_meta = _build_meta_features(p_mlp, p_cb, p_gb, features)
        proba = stacker.predict_proba(X_meta)[:, 1]
        is_pos = (proba >= 0.5).astype(int)
    else:
        wsum = max(w_mlp + w_cb + w_gb, 1e-12)
        proba = (w_mlp * p_mlp + w_cb * p_cb + w_gb * p_gb) / wsum
        is_pos = (proba >= thr).astype(int)
>>>>>>> REPLACE
```

The stacker is trained on OOF predictions from the *same 5-fold CV* used previously, so no validation/test data touches training, and the entire pipeline still reloads from `--model` in a fresh process.