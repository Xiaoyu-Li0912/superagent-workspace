import os
import re
import copy
import json
import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(2)

BUNDLE_NAME = "bundle.pt"

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

# ---------------------------------------------------------------------------
# Core helpers / feature engineering / models
# ---------------------------------------------------------------------------
FREQ_MAP = {"never": 0.0, "less1": 1.0, "1~3": 2.0, "4~8": 3.0, "gt8": 4.0}
TIME_MAP = {"7AM": 7.0, "10AM": 10.0, "2PM": 14.0, "6PM": 18.0, "10PM": 22.0}
AGE_MAP = {
    "below21": 20.0, "21": 21.0, "26": 26.0, "31": 31.0, "36": 36.0,
    "41": 41.0, "46": 46.0, "50plus": 55.0,
}
CAT_COLS_RAW = ["gender", "maritalStatus", "education", "occupation", "coupon"]
FREQ_COLS_RAW = [
    "Bar", "CoffeeHouse", "CarryAway", "RestaurantLessThan20", "Restaurant20To50",
]
# Columns converted from arbitrary integer codes into smoothed acceptance stats.
TE_COLS_RAW = list(CAT_COLS_RAW) + list(FREQ_COLS_RAW)
BIN_COLS_RAW = [
    "toCoupon_GEQ5min", "toCoupon_GEQ15min", "toCoupon_GEQ25min",
    "direction_same", "direction_opp",
]
NUM_COLS_RAW = (
    ["temperature", "time_num", "expiration_num", "age_num", "income_num", "has_children"]
    + FREQ_COLS_RAW
    + BIN_COLS_RAW
)


def get_device() -> torch.device:
    return torch.device("cpu")


def _to_float(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _parse_age(v) -> float:
    if isinstance(v, str):
        key = v.strip().lower().replace(" ", "")
        if key in AGE_MAP:
            return AGE_MAP[key]
    return _to_float(v)


def parse_money(v) -> float:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if not isinstance(v, str):
        return float("nan")
    s = v.strip()
    low = "less" in s.lower()
    high = "more" in s.lower()
    nums = [float(x) for x in re.findall(r"\d+", s.replace(",", ""))]
    if not nums:
        return float("nan")
    if low:
        return nums[0] / 2.0
    if high:
        return nums[0] * 1.1
    return float(np.mean(nums))


def _cat_key(v) -> str:
    if v is None:
        return "__nan__"
    try:
        if pd.isna(v):
            return "__nan__"
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s else "__nan__"


class CouponFeatureEngineer:
    """Learned, pickle-safe preprocessing (fitted on TRAIN rows only)."""

    def __init__(self):
        self.feature_names: List[str] = []
        self.vocab: Dict[str, Dict[str, int]] = {}
        self.num_medians: Dict[str, float] = {}
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        # Smoothed target-encoding state (persisted with the engineer).
        self.te_col_names: List[str] = []
        self.te_maps: Dict[str, Dict[str, float]] = {}
        self.te_maps_fold: Dict[int, Tuple[Dict[str, Dict[str, float]], float]] = {}
        self.te_prior: float = 0.5
        self.te_m: int = 20

    def _raw(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        out = pd.DataFrame(index=pd.RangeIndex(n))

        def col(name):
            if name in df.columns:
                return pd.Series(pd.array(df[name].values, dtype=object), index=out.index)
            return pd.Series([None] * n, index=out.index, dtype=object)

        out["temperature"] = col("temperature").map(_to_float)
        out["time_num"] = col("time").map(
            lambda v: TIME_MAP.get(str(v).strip(), _to_float(v))
        )
        out["expiration_num"] = col("expiration").map(
            lambda v: 1.0 if str(v).strip() == "1d"
            else (0.0 if str(v).strip() == "2h" else float("nan"))
        )
        out["age_num"] = col("age").map(_parse_age)
        out["income_num"] = col("income").map(parse_money)
        out["has_children"] = col("has_children").map(_to_float)
        for c in FREQ_COLS_RAW:
            out[c] = col(c).map(lambda v: FREQ_MAP.get(str(v).strip(), float("nan")))
        for c in BIN_COLS_RAW:
            out[c] = col(c).map(_to_float)
        for c in CAT_COLS_RAW:
            out[c] = col(c)
        return out

    def fit(self, df: pd.DataFrame) -> "CouponFeatureEngineer":
        raw = self._raw(df)
        self.cat_cols = ["cat_" + c for c in CAT_COLS_RAW]
        self.num_cols = ["num_" + c for c in NUM_COLS_RAW]
        self.feature_names = list(self.num_cols) + list(self.cat_cols)
        for c in CAT_COLS_RAW:
            uniq = sorted(set(_cat_key(v) for v in raw[c].tolist()))
            self.vocab[c] = {v: i for i, v in enumerate(uniq)}
        for c in NUM_COLS_RAW:
            arr = pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=np.float64)
            med = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
            self.num_medians[c] = med
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self._raw(df)
        out = pd.DataFrame(index=pd.RangeIndex(len(df)))
        for c in self.num_cols:
            name = c[len("num_"):]
            arr = pd.to_numeric(raw[name], errors="coerce").to_numpy(dtype=np.float64)
            arr = np.where(np.isfinite(arr), arr, self.num_medians.get(name, 0.0))
            out[c] = arr
        for c in self.cat_cols:
            name = c[len("cat_"):]
            vocab = self.vocab.get(name, {})
            out[c] = np.asarray(
                [float(vocab.get(_cat_key(v), -1)) for v in raw[name].tolist()],
                dtype=np.float64,
            )
        return out[self.feature_names]

    # -- smoothed target encoding (leakage-free) ---------------------------
    def _te_keys(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        raw = self._raw(df)
        return {
            c: np.asarray([_cat_key(v) for v in raw[c].tolist()], dtype=object)
            for c in TE_COLS_RAW
        }

    def _te_build(self, keys, y, mask, prior, m):
        maps: Dict[str, Dict[str, float]] = {}
        yy = np.asarray(y, dtype=np.float64)[mask]
        if mask.any() and yy.size:
            prior = float(yy.mean())
        for c in TE_COLS_RAW:
            k = keys[c][mask]
            sums: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            for kk, vv in zip(k.tolist(), yy.tolist()):
                sums[kk] = sums.get(kk, 0.0) + float(vv)
                counts[kk] = counts.get(kk, 0) + 1
            maps[c] = {
                kk: (sums[kk] + prior * m) / (counts[kk] + m) for kk in sums
            }
        return maps, float(prior)

    def fit_target_encoding(
        self,
        df: pd.DataFrame,
        y,
        fold_ids=None,
        m: int = 20,
    ) -> "CouponFeatureEngineer":
        """Fit shrunk category->acceptance maps on TRAIN rows only.

        Always builds full-fit maps (used for the final models and for
        prediction). When ``fold_ids`` is given, also builds one map set per
        fold from the rows *outside* that fold, so every row can be encoded
        without ever seeing its own label.
        """
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n = len(y_arr)
        self.te_col_names = list(TE_COLS_RAW)
        self.te_m = int(m)
        if n == 0:
            self.te_prior = 0.5
            self.te_maps = {c: {} for c in self.te_col_names}
            self.te_maps_fold = {}
            return self
        keys = self._te_keys(df)
        full_mask = np.ones(n, dtype=bool)
        self.te_maps, self.te_prior = self._te_build(
            keys, y_arr, full_mask, float(y_arr.mean()), self.te_m
        )
        self.te_maps_fold = {}
        if fold_ids is not None:
            fids = np.asarray(fold_ids).ravel()
            if fids.shape[0] == n:
                for k in np.unique(fids):
                    self.te_maps_fold[int(k)] = self._te_build(
                        keys, y_arr, fids != k, self.te_prior, self.te_m
                    )
        return self

    def transform_target_encoding(
        self, df: pd.DataFrame, fold_id: Optional[int] = None
    ) -> np.ndarray:
        """Apply saved maps. Unseen / NaN categories fall back to the prior."""
        n = len(df)
        cols = list(self.te_col_names or [])
        if not cols or n == 0:
            return np.zeros((n, len(cols)), dtype=np.float32)
        maps, prior = self.te_maps, float(self.te_prior)
        if fold_id is not None:
            entry = self.te_maps_fold.get(int(fold_id))
            if entry is not None:
                maps, prior = entry
        keys = self._te_keys(df)
        out = np.empty((n, len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            mp = maps.get(c, {})
            out[:, j] = np.asarray(
                [mp.get(kk, prior) for kk in keys[c].tolist()],
                dtype=np.float64,
            )
        return out


def split_feature_columns(names: Sequence[str]):
    cat_cols = [n for n in names if n.startswith("cat_")]
    num_cols = [n for n in names if not n.startswith("cat_")]
    return cat_cols, num_cols


class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X) -> "Normalizer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] == 0:
            self.mean = np.zeros(0)
            self.std = np.ones(0)
            return self
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        self.mean = np.where(np.isfinite(mean), mean, 0.0)
        self.std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] == 0 or self.mean is None:
            return X.astype(np.float32)
        return ((X - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean = state["mean"]
        self.std = state["std"]


class CouponModel(nn.Module):
    def __init__(self, cards, n_num, n_classes=2, emb_dim=8, hidden=128, dropout=0.2):
        super().__init__()
        cards = [int(c) for c in cards]
        dims = [int(min(emb_dim, max(2, (c + 1) // 2))) for c in cards]
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards, dims)])
        in_dim = int(sum(dims)) + int(n_num)
        h2 = max(16, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )

    def forward(self, xc, xn):
        parts = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        if xn.shape[1] > 0:
            parts.append(xn)
        return self.net(torch.cat(parts, dim=1))


class TrainConfig:
    def __init__(self):
        self.epochs = 40
        self.batch_size = 256
        self.lr = 2e-3
        self.weight_decay = 1e-4
        self.emb_dim = 8
        self.hidden = 128
        self.dropout = 0.2
        self.n_seeds = 3
        self.patience = 8


def _shift_cat(cat, cards):
    cat = np.asarray(cat)
    cards = [int(c) for c in cards]
    if cat.size == 0:
        return np.zeros((cat.shape[0], 0), dtype=np.int64), cards
    shifted = np.nan_to_num(cat, nan=-1.0).astype(np.int64) + 1
    for j, c in enumerate(cards):
        if c > 0 and j < shifted.shape[1]:
            shifted[:, j] = np.clip(shifted[:, j], 0, c - 1)
    return shifted, cards


def _ensemble_proba(models, cat_shifted, num, batch_size=512):
    if not models:
        return None
    probs = [_predict_proba_single(m, cat_shifted, num, batch_size) for m in models]
    return np.mean(probs, axis=0)


def load_train_data(data_dir):
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    y = labels.iloc[:, 0].to_numpy()
    schema = {}
    sp = os.path.join(data_dir, "schema.json")
    if os.path.exists(sp):
        with open(sp, "r") as f:
            schema = json.load(f)
    return features, y, schema


def load_predict_data(data_dir):
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    schema = {}
    sp = os.path.join(data_dir, "schema.json")
    if os.path.exists(sp):
        with open(sp, "r") as f:
            schema = json.load(f)
    return features, schema


def _train_one_model(
    seed,
    Xc,
    Xn,
    y,
    cards,
    n_num,
    n_classes,
    cfg,
    Xc_val=None,
    Xn_val=None,
    y_val=None,
):
    """Train one NN. Validation tensors (if given) must come from the same
    fold split as the training arrays; they are NOT indexed by a full-length
    mask (that caused a length mismatch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CouponModel(cards, n_num, n_classes, cfg.emb_dim, cfg.hidden, cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    lossf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(np.asarray(Xc, dtype=np.int64)).long()
    Nt = torch.from_numpy(np.asarray(Xn, dtype=np.float32))
    Yt = torch.from_numpy(np.asarray(y, dtype=np.int64)).long()
    n = len(Yt)
    idx = np.arange(n)

    has_val = (
        Xc_val is not None
        and Xn_val is not None
        and y_val is not None
        and len(np.asarray(y_val)) > 0
    )
    if has_val:
        Xv = torch.from_numpy(np.asarray(Xc_val, dtype=np.int64)).long()
        Nv = torch.from_numpy(np.asarray(Xn_val, dtype=np.float32))
        Yv = np.asarray(y_val, dtype=np.int64)
        assert Xv.shape[0] == Nv.shape[0] == Yv.shape[0]

    best_state = copy.deepcopy(model.state_dict())
    best_acc, best_ep, since = -1.0, cfg.epochs, 0
    for ep in range(cfg.epochs):
        model.train()
        np.random.shuffle(idx)
        for s in range(0, n, cfg.batch_size):
            b = idx[s:s + cfg.batch_size]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xt[b], Nt[b]), Yt[b])
            loss.backward()
            opt.step()
        sched.step()
        if has_val:
            model.eval()
            with torch.no_grad():
                p = torch.softmax(model(Xv, Nv), dim=1).numpy()
            acc = float((p.argmax(1) == Yv).mean())
            if acc > best_acc:
                best_acc, best_ep, since = acc, ep + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since += 1
                if since >= cfg.patience:
                    break
    if has_val:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_ep, best_acc


def run_train(data_dir, model_dir) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    features, y_raw, _schema = load_train_data(data_dir)
    label_map = sorted(pd.unique(y_raw).tolist(), key=lambda v: str(v))
    cls_to_idx = {c: i for i, c in enumerate(label_map)}
    y = np.asarray([cls_to_idx[v] for v in y_raw], dtype=np.int64)
    n_classes = len(label_map)
    print(f"[train] rows={len(features)} classes={label_map}")

    fe = CouponFeatureEngineer().fit(features)

    cfg = TrainConfig()
    n = len(y)
    n_folds = 5
    rng = np.random.RandomState(7)
    folds = np.zeros(n, dtype=np.int64)
    for cls in np.unique(y):
        idx_c = np.where(y == cls)[0]
        rng.shuffle(idx_c)
        for k, i in enumerate(idx_c):
            folds[i] = k % n_folds

    # Leakage-free smoothed target encoding: full-fit maps (final models and
    # prediction) plus per-fold maps (CV, built without that fold's labels).
    fe.fit_target_encoding(features, y, folds, m=20)
    print(
        f"[train] target-encoding cols={len(fe.te_col_names)} "
        f"prior={fe.te_prior:.4f} m={fe.te_m}"
    )

    cat, num, cat_cols, num_cols = _prepare_fold(fe, features)
    normalizer = Normalizer().fit(num)
    num_n = normalizer.transform(num)
    cards = [len(fe.vocab[c]) + 1 for c in CAT_COLS_RAW]
    cat_shifted, cards = _shift_cat(cat, cards)

    X_gbdt_all = _prepare_gbdt_matrix(fe, features)
    oof_nn = np.zeros((n, n_classes), dtype=np.float64)
    oof_gb = np.zeros((n, n_classes), dtype=np.float64)
    best_epochs: List[int] = []

    for k in range(n_folds):
        val_mask = folds == k
        tr_mask = ~val_mask
        # Fold-k encodings come from this fold's TRAINING rows only, so both the
        # train and validation rows of the fold see leakage-free statistics.
        cat_shifted_k, _ = _shift_cat(
            _prepare_fold(fe, features, fold_id=k)[0], cards
        )
        num_n_k = normalizer.transform(_prepare_fold(fe, features, fold_id=k)[1])
        X_gbdt_k = _prepare_gbdt_matrix(fe, features, fold_id=k)
        seed_probs = []
        for s in range(2):
            m, bep, _ = _train_one_model(
                100 + s,
                cat_shifted_k[tr_mask],
                num_n_k[tr_mask],
                y[tr_mask],
                cards,
                num_n_k.shape[1],
                n_classes,
                cfg,
                Xc_val=cat_shifted_k[val_mask],
                Xn_val=num_n_k[val_mask],
                y_val=y[val_mask],
            )
            best_epochs.append(int(bep))
            seed_probs.append(
                _predict_proba_single(m, cat_shifted_k[val_mask], num_n_k[val_mask])
            )
        oof_nn[val_mask] = np.mean(seed_probs, axis=0)
        gb_models = _train_gbdt(X_gbdt_k[tr_mask], y[tr_mask])
        pg = _predict_gbdt(gb_models, X_gbdt_k[val_mask])
        if pg is not None and pg.shape == (int(val_mask.sum()), n_classes):
            oof_gb[val_mask] = pg
        print(f"[train] fold {k + 1}/{n_folds} done")

    acc_nn = float((oof_nn.argmax(1) == y).mean())
    print(f"[train] OOF nn acc={acc_nn:.4f}")

    blend_alpha = 1.0
    if HAS_GBDT and np.any(oof_gb):
        acc_gb = float((oof_gb.argmax(1) == y).mean())
        blend_alpha, blend_acc = _search_blend(oof_nn, oof_gb, y)
        print(
            f"[train] OOF gbdt acc={acc_gb:.4f} blend alpha_nn={blend_alpha:.2f} "
            f"acc={blend_acc:.4f}"
        )

    if best_epochs:
        cfg.epochs = int(max(10, min(cfg.epochs, int(np.median(best_epochs)) + 5)))
    full_models = []
    for s in range(cfg.n_seeds):
        m, _, _ = _train_one_model(
            200 + s, cat_shifted, num_n, y, cards, num_n.shape[1], n_classes, cfg
        )
        full_models.append(m)
    gbdt_models = _train_gbdt(X_gbdt_all, y) if HAS_GBDT else []

    path = _save_bundle(
        model_dir, fe, normalizer, cat_cols, num_cols, cards, label_map, cfg,
        full_models, cfg.epochs, gbdt_models=gbdt_models, blend_alpha=blend_alpha,
    )
    print(
        f"[train] saved bundle -> {path} "
        f"(nn={len(full_models)}, gbdt={len(gbdt_models)}, epochs={cfg.epochs})"
    )


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
        "n_iter_no_change": 15,
        "random_state": 42,
    },
    {
        "max_iter": 300,
        "learning_rate": 0.03,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 10,
        "l2_regularization": 0.5,
        "max_bins": 255,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
        "random_state": 43,
    },
    {
        "max_iter": 300,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
        "max_bins": 127,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
        "random_state": 44,
    },
]


def _prepare_fold(
    fe: CouponFeatureEngineer,
    df: pd.DataFrame,
    fold_id: Optional[int] = None,
):
    """NN inputs: integer codes (unchanged embedding cards) + numeric block.

    The numeric block is the raw numeric features followed by the smoothed
    target-encoding block (fold-specific maps when ``fold_id`` is given).
    """
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
    te = fe.transform_target_encoding(df, fold_id=fold_id)
    if te.shape[1] > 0:
        num = np.concatenate([num, te.astype(np.float32)], axis=1)
        num_cols = list(num_cols) + ["te_" + c for c in fe.te_col_names]
    return cat, num, cat_cols, num_cols


def _prepare_gbdt_matrix(
    fe: CouponFeatureEngineer,
    df: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
    fold_id: Optional[int] = None,
) -> np.ndarray:
    """GBDT matrix: numeric columns untouched (no false integer ordering left
    for the schema categoricals) plus the smoothed target-encoding block.

    ``feature_names`` is accepted for backwards compatibility; the column
    layout is derived from the fitted engineer so train and predict agree.
    """
    Xfe = fe.transform(df).reindex(columns=fe.feature_names, fill_value=0.0)
    _cat_cols, num_cols = split_feature_columns(fe.feature_names)
    blocks: List[np.ndarray] = []
    if num_cols:
        blocks.append(Xfe[num_cols].values.astype(np.float32))
    te = fe.transform_target_encoding(df, fold_id=fold_id)
    if te.shape[1] > 0:
        blocks.append(te.astype(np.float32))
    if blocks:
        X = np.concatenate(blocks, axis=1).astype(np.float32)
    else:
        X = np.zeros((len(Xfe), 0), dtype=np.float32)
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
        "nn_cfg": {
            "emb_dim": int(getattr(cfg, "emb_dim", 8)),
            "hidden": int(getattr(cfg, "hidden", 128)),
            "dropout": float(getattr(cfg, "dropout", 0.2)),
        },
        # Smoothed target-encoding state (also carried inside "fe"); persisted
        # explicitly so prediction can verify it never refits.
        "te_maps": fe.te_maps,
        "te_prior": float(fe.te_prior),
        "te_m": int(fe.te_m),
        "te_col_names": list(fe.te_col_names),
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
    bundle = _load_bundle(model_dir)
    fe = bundle["fe"]
    normalizer = Normalizer()
    normalizer.load_state_dict(bundle["normalizer_state"])
    cards = [int(c) for c in bundle["cards"]]
    label_map = bundle["label_map"]
    n_classes = len(label_map)
    num_cols = bundle["num_cols"]
    nn_cfg = bundle.get("nn_cfg") or {}

    models: List[nn.Module] = []
    for sd in bundle.get("state_dicts", []):
        m = CouponModel(
            cards,
            len(num_cols),
            n_classes,
            int(nn_cfg.get("emb_dim", 8)),
            int(nn_cfg.get("hidden", 128)),
            float(nn_cfg.get("dropout", 0.2)),
        )
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    features, _schema = load_predict_data(data_dir)
    # Apply the SAVED maps only (full-training fit); never refit here. Unseen /
    # missing categories fall back to the stored global prior.
    if not getattr(fe, "te_col_names", None) and bundle.get("te_col_names"):
        fe.te_col_names = list(bundle["te_col_names"])
        fe.te_maps = bundle.get("te_maps", {}) or {}
        fe.te_prior = float(bundle.get("te_prior", 0.5))
        fe.te_m = int(bundle.get("te_m", 20))
    cat, num, _cat_cols, _num_cols = _prepare_fold(fe, features)
    num_n = normalizer.transform(num)
    cat_shifted, cards = _shift_cat(cat, cards)

    proba = _ensemble_proba(models, cat_shifted, num_n, batch_size=512)
    if proba is None:
        proba = np.zeros((len(features), n_classes), dtype=np.float64)
        proba[:, 0] = 1.0

    gbdt_models = bundle.get("gbdt_models", []) or []
    blend_alpha = float(bundle.get("blend_alpha", 1.0))
    if gbdt_models:
        X_gbdt = _prepare_gbdt_matrix(fe, features, bundle["feature_names"])
        proba_gbdt = _predict_gbdt(gbdt_models, X_gbdt)
        if proba_gbdt is not None and proba_gbdt.shape == proba.shape:
            proba = blend_alpha * proba + (1.0 - blend_alpha) * proba_gbdt
            print(
                f"[predict] blended alpha_nn={blend_alpha:.2f} "
                f"(NN={len(models)}, GBDT={len(gbdt_models)})"
            )

    idx_pred = np.argmax(proba, axis=1)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        run_train(args.data, args.model)
    else:
        run_predict(args.data, args.model, args.output)


if __name__ == "__main__":
    main()