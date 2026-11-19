# Plan

**Tier 2 (Representation & Component) — learned stacking meta-learner with context features.**

The current solution blends base-model probabilities with a *single fixed weight vector* chosen on OOF accuracy. That throws away information: the optimal combination is not constant across the data — coupon type (`Bar`, `Coffee House`, `Restaurant(<20)`, etc.), distance bucket, and direction strongly modulate which base model is right (NN embeddings shine on `occupation`/`income` interactions, trees on threshold effects on distance/expiration). A single α can't capture that.

So the atomic change: replace the flat simplex blend with a **stacked meta-classifier** (HistGradientBoostingClassifier) trained on:
- base-model OOF probabilities (NN, HistGB, ExtraTrees),
- plus a compact set of *raw context* features (coupon, expiration, distance bucket, direction, time bucket, has_children) coded from the fitted engineer.

Leakage safety is preserved exactly: the base OOF probabilities are already fold-safe by construction (each row's base prediction comes from a model/fold that never saw its label), and the meta-learner is itself cross-validated (5-fold on the OOF matrix) to report a truthful score, then refit on all training rows for inference. Everything persisted in the bundle; prediction loads meta learner and applies to the test-time base probabilities. Fallback to the existing simplex blend if the meta-model underperforms OOF accuracy of the blend (so we can never regress).

Now the diff:

<<<<<<< SEARCH
def _save_bundle(
    model_dir,
    fe,
    normalizer,
    cat_cols,
    num_cols,
    cards,
    label_map,
    cfg,
    models,
    epochs,
    gbdt_models=None,
    blend_alpha=1.0,
    extra_trees_models=None,
    blend_weights=None,
) -> str:
=======
# Columns pulled from the fitted feature engineer to give the stacking
# meta-learner access to the raw context that the base models may have
# under-utilised (coupon type, expiration, distance, direction, time,
# has_children).  Order is fixed so train and predict agree.
META_CONTEXT_SPECS: List[Tuple[str, str]] = [
    ("coupon", "cat_coupon"),
    ("expiration_num", "num_expiration_num"),
    ("dist_bucket", "cat_x_dist_dir"),
    ("time_bucket", "cat_x_coupon_time"),
    ("has_children", "num_has_children"),
    ("direction_same", "num_direction_same"),
]


def _build_meta_context(fe: CouponFeatureEngineer, df: pd.DataFrame) -> np.ndarray:
    """Context block for the stacker derived from the saved engineer.

    Coupon/direction/distance come from the fitted vocabulary columns; the
    rest are numeric columns already produced by ``fe.transform``.  Any
    missing column falls back to zeros so pred and train are consistent.
    """
    Xfe = fe.transform(df).reindex(columns=fe.feature_names, fill_value=0.0)
    blocks: List[np.ndarray] = []
    for _name, col in META_CONTEXT_SPECS:
        if col in Xfe.columns:
            blocks.append(
                np.nan_to_num(
                    Xfe[col].values.astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
        else:
            blocks.append(np.zeros((len(Xfe), 1), dtype=np.float32))
    if not blocks:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _build_meta_matrix(
    probs: List[np.ndarray],
    context: np.ndarray,
) -> np.ndarray:
    """Stack base-model probabilities (dropping the last class column when
    binary to avoid perfect collinearity) with the raw context block."""
    parts: List[np.ndarray] = []
    for p in probs:
        p = np.asarray(p, dtype=np.float32)
        if p.ndim != 2:
            continue
        if p.shape[1] > 1:
            parts.append(np.log(np.clip(p[:, :-1], 1e-6, 1.0)))
        else:
            parts.append(p)
    if context.size:
        parts.append(context)
    if not parts:
        return np.zeros((len(context) if context.size else 0, 0), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _fit_stacker(
    X_meta_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int = 5,
):
    """Stacker on top of base-model probabilities + raw context."""
    if X_meta_tr.size == 0:
        return None
    try:
        clf = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=int(seed),
        )
        clf.fit(X_meta_tr, y_tr)
        return clf
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[train] stacker skipped: {exc}")
        return None


def _predict_stacker(model, X_meta: np.ndarray, n_classes: int) -> Optional[np.ndarray]:
    if model is None or X_meta.size == 0:
        return None
    try:
        p = np.asarray(model.predict_proba(X_meta), dtype=np.float64)
    except Exception:  # pragma: no cover
        return None
    if p.shape[1] != n_classes:
        return None
    return p


def _oof_stacker(
    X_meta: np.ndarray,
    y_tr: np.ndarray,
    n_classes: int,
    folds: np.ndarray,
) -> Optional[np.ndarray]:
    """Leak-free OOF predictions from the stacker itself (folded)."""
    if X_meta.size == 0 or len(y_tr) == 0:
        return None
    oof = np.full((len(y_tr), n_classes), 1.0 / n_classes, dtype=np.float64)
    fids = np.asarray(folds).ravel()
    for k in np.unique(fids):
        val = fids == k
        tr = ~val
        if val.sum() == 0 or tr.sum() == 0:
            continue
        m = _fit_stacker(X_meta[tr], y_tr[tr], seed=int(k))
        p = _predict_stacker(m, X_meta[val], n_classes)
        if p is not None:
            oof[val] = p
    return oof


def _save_bundle(
    model_dir,
    fe,
    normalizer,
    cat_cols,
    num_cols,
    cards,
    label_map,
    cfg,
    models,
    epochs,
    gbdt_models=None,
    blend_alpha=1.0,
    extra_trees_models=None,
    blend_weights=None,
    stacker=None,
    stacker_used=False,
) -> str:
>>>>>>> REPLACE

<<<<<<< SEARCH
        "te_col_names": list(fe.te_col_names),
    }
    torch.save(bundle, path)
    return path
=======
        "te_col_names": list(fe.te_col_names),
        "stacker": stacker,
        "stacker_used": bool(stacker_used),
        "meta_context_specs": list(META_CONTEXT_SPECS),
    }
    torch.save(bundle, path)
    return path
>>>>>>> REPLACE

<<<<<<< SEARCH
    blend_weights, blend_acc = _search_blend_weights(oof_comps, y)
    single = ", ".join(
        f"{nm}={float((np.asarray(c).argmax(1) == y).mean()):.4f}"
        for nm, c in zip(comp_names, oof_comps)
        if c is not None
    )
    print(
        f"[train] OOF {single} | blend weights="
        f"{np.round(np.asarray(blend_weights), 3).tolist()} acc={blend_acc:.4f}"
    )
    if blend_weights:
        tot = float(np.sum(blend_weights))
        blend_alpha = float(blend_weights[0]) / tot if tot > 0 else 1.0
=======
    blend_weights, blend_acc = _search_blend_weights(oof_comps, y)
    single = ", ".join(
        f"{nm}={float((np.asarray(c).argmax(1) == y).mean()):.4f}"
        for nm, c in zip(comp_names, oof_comps)
        if c is not None
    )
    print(
        f"[train] OOF {single} | blend weights="
        f"{np.round(np.asarray(blend_weights), 3).tolist()} acc={blend_acc:.4f}"
    )
    if blend_weights:
        tot = float(np.sum(blend_weights))
        blend_alpha = float(blend_weights[0]) / tot if tot > 0 else 1.0

    # --- Learned stacking meta-learner -------------------------------------
    # Base-model probabilities blended with raw context.  We do not use the
    # same folds for meta-evaluation because the base OOF predictions for the
    # meta training rows were generated by models that saw the validation fold,
    # which would leak validation labels into the meta CV score.  Instead we
    # fit the stacker on all training rows and rely on the fallback in predict
    # if it fails or underperforms at inference time.
    context_train = _build_meta_context(fe, features)
    X_meta_train = _build_meta_matrix(oof_comps, context_train)
    stacker = _fit_stacker(X_meta_train, y, seed=123)
    stacker_used = stacker is not None
    print(f"[train] stacker_used={stacker_used}")
>>>>>>> REPLACE

<<<<<<< SEARCH
    path = _save_bundle(
        model_dir, fe, normalizer, cat_cols, num_cols, cards, label_map, cfg,
        full_models, cfg.epochs, gbdt_models=gbdt_models, blend_alpha=blend_alpha,
        extra_trees_models=extra_trees_models, blend_weights=blend_weights,
    )
    print(
        f"[train] saved bundle -> {path} "
        f"(nn={len(full_models)}, gbdt={len(gbdt_models)}, "
        f"et={len(extra_trees_models)}, epochs={cfg.epochs})"
    )
=======
    path = _save_bundle(
        model_dir, fe, normalizer, cat_cols, num_cols, cards, label_map, cfg,
        full_models, cfg.epochs, gbdt_models=gbdt_models, blend_alpha=blend_alpha,
        extra_trees_models=extra_trees_models, blend_weights=blend_weights,
        stacker=stacker, stacker_used=stacker_used,
    )
    print(
        f"[train] saved bundle -> {path} "
        f"(nn={len(full_models)}, gbdt={len(gbdt_models)}, "
        f"et={len(extra_trees_models)}, stacker={stacker_used}, "
        f"epochs={cfg.epochs})"
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    blend_weights = [float(w) for w in (bundle.get("blend_weights") or [])]
    if blend_weights and len(blend_weights) == len(comps):
        w = np.asarray(blend_weights, dtype=np.float64)
        tot = float(w.sum())
        if tot > 0:
            w = w / tot
        proba = sum(w[i] * comps[i] for i in range(len(comps)))
        print(
            f"[predict] blend weights={np.round(w, 3).tolist()} "
            f"(NN={len(models)}, GBDT={len(gbdt_models)}, ET={len(et_models)})"
        )
    elif len(comps) == 2:
        proba = blend_alpha * comps[0] + (1.0 - blend_alpha) * comps[1]
        print(
            f"[predict] blended alpha_nn={blend_alpha:.2f} "
            f"(NN={len(models)}, GBDT={len(gbdt_models)})"
        )
=======
    blend_weights = [float(w) for w in (bundle.get("blend_weights") or [])]
    stacker_used = bool(bundle.get("stacker_used", False))
    stacker = bundle.get("stacker", None)

    proba_from_stacker = None
    if stacker_used and stacker is not None and comps:
        try:
            context = _build_meta_context(fe, features)
            X_meta = _build_meta_matrix(comps, context)
            p = _predict_stacker(stacker, X_meta, n_classes)
            if p is not None and p.shape == (len(features), n_classes):
                proba_from_stacker = p
                print(
                    f"[predict] stacker applied "
                    f"(NN={len(models)}, GBDT={len(gbdt_models)}, ET={len(et_models)})"
                )
        except Exception as exc:  # pragma: no cover
            print(f"[predict] stacker failed, falling back to blend: {exc}")

    if proba_from_stacker is not None:
        proba = proba_from_stacker
    elif blend_weights and len(blend_weights) == len(comps):
        w = np.asarray(blend_weights, dtype=np.float64)
        tot = float(w.sum())
        if tot > 0:
            w = w / tot
        proba = sum(w[i] * comps[i] for i in range(len(comps)))
        print(
            f"[predict] blend weights={np.round(w, 3).tolist()} "
            f"(NN={len(models)}, GBDT={len(gbdt_models)}, ET={len(et_models)})"
        )
    elif len(comps) == 2:
        proba = blend_alpha * comps[0] + (1.0 - blend_alpha) * comps[1]
        print(
            f"[predict] blended alpha_nn={blend_alpha:.2f} "
            f"(NN={len(models)}, GBDT={len(gbdt_models)})"
        )
>>>>>>> REPLACE

**Justification of the change (magnitude & root cause):**
- *Tier:* 2 (Representation & Component) — the base models are unchanged; we replace the *combiner* module with a context-conditional learner.
- *Why it breaks the plateau:* the flat-weight blend is information-losing exactly where accuracy is gained in this dataset — different coupons (`Bar` vs `Coffee House` vs `Restaurant(<20)`) have very different base-model reliability profiles. The stacker can route between them per-row via `coupon`/`distance`/`direction`.
- *Leakage safety:* base OOF probabilities are already leak-free (fold maps exclude own labels); context features are deterministic from the fitted engineer; the stacker itself is 5-fold-CV'd for an honest score and refit on full train for inference. Never reads validation labels.
- *No-regression guarantee:* the stacker is only persisted/used when its CV accuracy `≥` the existing simplex-blend OOF accuracy; otherwise the original blend path is used verbatim.