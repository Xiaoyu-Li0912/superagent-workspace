import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(2)

BUNDLE_NAME = "bundle.pt"


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        GradientBoostingClassifier,
    )

    HAS_GBDT = True
except ImportError:
    HistGradientBoostingClassifier = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore
    HAS_GBDT = False

torch.set_num_threads(2)

BUNDLE_NAME = "bundle.pt"

# Diverse GBDT configs (different depth / leaf / reg / lr) for a small ensemble.
GBDT_CONFIGS: List[Dict[str, Any]] = [
    {
        "max_iter": 500,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "max_bins": 255,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 25,
        "random_state": 42,
    },
    {
        "max_iter": 500,
        "learning_rate": 0.03,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 10,
        "l2_regularization": 0.5,
        "max_bins": 255,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 25,
        "random_state": 43,
    },
    {
        "max_iter": 500,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
        "max_bins": 127,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 25,
        "random_state": 44,
    },
]


def _prepare_fold(fe: CouponFeatureEngineer, df: pd.DataFrame):
    Xfe = fe.transform(df).reindex(columns=fe.feature_names, fill_value=0.0)
    cat_cols, num_cols = split_feature_columns(fe.feature_names)
    cat = (
        np.nan_to_num(Xfe[cat_cols].values, nan=-1.0)
        if cat_cols
        else np.zeros((len(Xfe), 0))
    )
    num = (
        Xfe[num_cols].values.astype(np.float32)
        if num_cols
        else np.zeros((len(Xfe), 0), dtype=np.float32)
    )
    return cat, num, cat_cols, num_cols


def _prepare_fold(fe: CouponFeatureEngineer, df: pd.DataFrame):
    Xfe = fe.transform(df).reindex(columns=fe.feature_names, fill_value=0.0)
    cat_cols, num_cols = split_feature_columns(fe.feature_names)
    cat = (
        np.nan_to_num(Xfe[cat_cols].values, nan=-1.0)
        if cat_cols
        else np.zeros((len(Xfe), 0))
    )
    num = (
        Xfe[num_cols].values.astype(np.float32)
        if num_cols
        else np.zeros((len(Xfe), 0), dtype=np.float32)
    )
    return cat, num, cat_cols, num_cols


def _prepare_gbdt_matrix(
    fe: CouponFeatureEngineer,
    df: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Full engineered matrix (categoricals integer-encoded, numerics raw)."""
    cols = list(feature_names) if feature_names is not None else list(fe.feature_names)
    Xfe = fe.transform(df).reindex(columns=cols, fill_value=0.0)
    X = Xfe.values.astype(np.float32)
    X = np.nan_to_num(X, nan=-1.0, posinf=0.0, neginf=0.0)
    return X


def _train_gbdt(X_tr: np.ndarray, y_tr: np.ndarray) -> List[Any]:
    """Train a small diverse ensemble of gradient-boosted tree models."""
    if not HAS_GBDT:
        return []
    models: List[Any] = []
    for gc in GBDT_CONFIGS:
        try:
            clf = HistGradientBoostingClassifier(**gc)
        except Exception:
            clf = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                random_state=int(gc.get("random_state", 42)),
            )
        clf.fit(X_tr, y_tr)
        models.append(clf)
    return models


def _predict_gbdt(models: List[Any], X: np.ndarray) -> Optional[np.ndarray]:
    if not models:
        return None
    probas = [np.asarray(m.predict_proba(X), dtype=np.float64) for m in models]
    return np.mean(probas, axis=0)


def _predict_proba_single(
    model: nn.Module,
    cat_shifted: np.ndarray,
    num_mat: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    device = get_device()
    model.to(device).eval()
    chunks: List[np.ndarray] = []
    n = len(cat_shifted)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xc = torch.from_numpy(cat_shifted[start:end]).long().to(device)
        xn = torch.from_numpy(num_mat[start:end].astype(np.float32)).to(device)
        with torch.no_grad():
            chunks.append(torch.softmax(model(xc, xn), dim=1).cpu().numpy())
    if not chunks:
        return np.zeros((n, 2), dtype=np.float64)
    return np.concatenate(chunks, axis=0).astype(np.float64)


def _search_blend(
    p_nn: np.ndarray, p_gbdt: np.ndarray, y_idx: np.ndarray
) -> Tuple[float, float]:
    """Pick alpha (NN weight) maximizing accuracy of blended argmax on OOF."""
    best_alpha = 1.0
    best_acc = float((p_nn.argmax(1) == y_idx).mean())
    for alpha in np.linspace(0.0, 1.0, 21):
        pred = (float(alpha) * p_nn + (1.0 - float(alpha)) * p_gbdt).argmax(1)
        acc = float((pred == y_idx).mean())
        if acc > best_acc:
            best_acc, best_alpha = acc, float(alpha)
    return best_alpha, best_acc


def _save_bundle(
    model_dir, fe, normalizer, cat_cols, num_cols, cards, label_map, cfg, models, epochs
) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, BUNDLE_NAME)
    bundle = {
        "fe": fe,
        "feature_names": list(fe.feature_names),
        "normalizer_state": normalizer.state_dict(),
        "cat_cols": list(cat_cols),
        "num_cols": list(num_cols),
        "cards": list(cards),
        "label_map": list(label_map),
        "cfg": dict(cfg.__dict__),
        "state_dicts": [m.state_dict() for m in models],
        "epochs": int(epochs),
    }
    torch.save(bundle, path)
    return path


def _load_bundle(model_dir):
    path = os.path.join(model_dir, BUNDLE_NAME)
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    return bundle


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
) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, BUNDLE_NAME)
    bundle = {
        "fe": fe,
        "feature_names": list(fe.feature_names),
        "normalizer_state": normalizer.state_dict(),
        "cat_cols": list(cat_cols),
        "num_cols": list(num_cols),
        "cards": list(cards),
        "label_map": list(label_map),
        "cfg": dict(cfg.__dict__),
        "state_dicts": [m.state_dict() for m in models],
        "epochs": int(epochs),
        "gbdt_models": list(gbdt_models) if gbdt_models else [],
        "blend_alpha": float(blend_alpha),
    }
    torch.save(bundle, path)
    return path


def _load_bundle(model_dir):
    path = os.path.join(model_dir, BUNDLE_NAME)
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    return bundle


def run_predict(data_dir, model_dir, output_path) -> None:
    fe, normalizer, bundle, cfg, models = _load_bundle(model_dir)
    features, _schema = load_predict_data(data_dir)

    Xfe = fe.transform(features).reindex(
        columns=bundle["feature_names"], fill_value=0.0
    )
    cat_cols = bundle["cat_cols"]
    num_cols = bundle["num_cols"]

    cat = (
        np.nan_to_num(Xfe[cat_cols].values, nan=-1.0)
        if cat_cols
        else np.zeros((len(Xfe), 0))
    )
    num = (
        Xfe[num_cols].values.astype(np.float32)
        if num_cols
        else np.zeros((len(Xfe), 0), dtype=np.float32)
    )
    if num_cols:
        num = normalizer.transform(num)

    cat_shifted, _ = _shift_cat(cat, bundle["cards"])
    proba = _ensemble_proba(models, cat_shifted, num, batch_size=512)
    idx_pred = np.argmax(proba, axis=1)
    label_map = bundle["label_map"]
    preds = [label_map[int(i)] for i in idx_pred]

    out = pd.DataFrame(
        {
            "row_id": np.arange(len(features), dtype=np.int64),
            "prediction": preds,
        }
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[predict] wrote {len(out)} rows -> {output_path}")


def run_predict(data_dir, model_dir, output_path) -> None:
    fe, normalizer, bundle, cfg, models = _load_bundle(model_dir)
    features, _schema = load_predict_data(data_dir)

    Xfe = fe.transform(features).reindex(
        columns=bundle["feature_names"], fill_value=0.0
    )
    cat_cols = bundle["cat_cols"]
    num_cols = bundle["num_cols"]

    cat = (
        np.nan_to_num(Xfe[cat_cols].values, nan=-1.0)
        if cat_cols
        else np.zeros((len(Xfe), 0))
    )
    num = (
        Xfe[num_cols].values.astype(np.float32)
        if num_cols
        else np.zeros((len(Xfe), 0), dtype=np.float32)
    )
    if num_cols:
        num = normalizer.transform(num)

    cat_shifted, _ = _shift_cat(cat, bundle["cards"])
    proba_nn = _ensemble_proba(models, cat_shifted, num, batch_size=512)

    gbdt_models = bundle.get("gbdt_models", []) or []
    blend_alpha = float(bundle.get("blend_alpha", 1.0))
    proba = proba_nn
    if gbdt_models:
        X_gbdt = _prepare_gbdt_matrix(fe, features, bundle["feature_names"])
        proba_gbdt = _predict_gbdt(gbdt_models, X_gbdt)
        if proba_gbdt is not None:
            proba = blend_alpha * proba_nn + (1.0 - blend_alpha) * proba_gbdt
            print(
                f"[predict] blended alpha_nn={blend_alpha:.2f} "
                f"(NN={len(models)}, GBDT={len(gbdt_models)})"
            )

    idx_pred = np.argmax(proba, axis=1)
    label_map = bundle["label_map"]
    preds = [label_map[int(i)] for i in idx_pred]

    out = pd.DataFrame(
        {
            "row_id": np.arange(len(features), dtype=np.int64),
            "prediction": preds,
        }
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[predict] wrote {len(out)} rows -> {output_path}")