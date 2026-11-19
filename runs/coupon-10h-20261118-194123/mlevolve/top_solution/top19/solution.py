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

try:
    from sklearn.ensemble import ExtraTreesClassifier

    HAS_ET = True
except ImportError:
    ExtraTreesClassifier = None  # type: ignore
    HAS_ET = False

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

# Explicit domain interactions. Each entry maps an interaction name to the
# raw columns (see ``CouponFeatureEngineer._raw``) whose concatenated keys
# define the interaction level. They are treated as first-class categoricals:
# native GBDT codes, extra-trees one-hots, and smoothed target encoding.
INTERACTION_SPECS: List[Tuple[str, List[str]]] = [
    ("x_coupon_exp", ["coupon", "expiration"]),
    ("x_coupon_time", ["coupon", "time_bucket"]),
    ("x_coupon_income", ["coupon", "income_bucket"]),
    ("x_coupon_children", ["coupon", "has_children_raw"]),
    ("x_dist_dir", ["dist_bucket", "direction"]),
]
INTERACTION_NAMES: List[str] = [n for n, _ in INTERACTION_SPECS]
TE_KEY_COLS: List[str] = list(TE_COLS_RAW) + list(INTERACTION_NAMES)
# Columns with at most this many levels get an explicit one-hot block for the
# ExtraTrees member (one-hot matters there, native codes / TE for the others).
ET_MAX_CARD = 12
# Fixed NN seed list for fold models (full-data models use cfg.n_seeds).
NN_SEEDS: List[int] = [101, 202, 303, 404]
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
        # Raw key columns that receive smoothed target encoding (identity +
        # learned interactions). Persisted so prediction never re-derives them.
        self.te_key_cols: List[str] = list(TE_KEY_COLS)
        # One-hot specification for the ExtraTrees matrix (fitted vocabularies).
        self.et_onehot_cols: List[str] = []
        self.et_onehot_cards: Dict[str, int] = {}
        # Layout metadata for the GBDT matrix (True = native categorical col).
        self.gbdt_cat_flags: List[bool] = []

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
        # Raw categorical columns that back an explicit interaction but are
        # not themselves first-class categoricals (so they don't change the
        # feature layout / embedding cards). Needed by the x_coupon_exp spec.
        out["expiration"] = col("expiration")

        # --- bucket helpers backing the interaction columns -----------------
        t_num = out["time_num"].to_numpy(dtype=np.float64)
        out["time_bucket"] = pd.Series(
            [
                ("tb_%d" % int(v // 3)) if np.isfinite(v) else "__nan__"
                for v in t_num.tolist()
            ],
            index=out.index,
            dtype=object,
        )
        inc = out["income_num"].to_numpy(dtype=np.float64)
        out["income_bucket"] = pd.Series(
            [
                ("inc_low" if v < 25000 else ("inc_mid" if v < 75000 else "inc_high"))
                if np.isfinite(v) else "__nan__"
                for v in inc.tolist()
            ],
            index=out.index,
            dtype=object,
        )
        d15 = out["toCoupon_GEQ15min"].to_numpy(dtype=np.float64)
        d25 = out["toCoupon_GEQ25min"].to_numpy(dtype=np.float64)
        dist_bucket: List[str] = []
        for a, b in zip(d15.tolist(), d25.tolist()):
            if not np.isfinite(a) or not np.isfinite(b):
                dist_bucket.append("__nan__")
            elif a < 0.5:
                dist_bucket.append("dist_far")
            elif b < 0.5:
                dist_bucket.append("dist_mid")
            else:
                dist_bucket.append("dist_near")
        out["dist_bucket"] = pd.Series(dist_bucket, index=out.index, dtype=object)
        dsame = out["direction_same"].to_numpy(dtype=np.float64)
        out["direction"] = pd.Series(
            [
                "dir_same" if (np.isfinite(v) and v > 0.5) else
                ("dir_opp" if np.isfinite(v) else "__nan__")
                for v in dsame.tolist()
            ],
            index=out.index,
            dtype=object,
        )
        hc = out["has_children"].to_numpy(dtype=np.float64)
        out["has_children_raw"] = pd.Series(
            [
                ("hc_1" if v > 0.5 else "hc_0") if np.isfinite(v) else "__nan__"
                for v in hc.tolist()
            ],
            index=out.index,
            dtype=object,
        )

        # --- explicit interaction categorical levels ------------------------
        for name, parts in INTERACTION_SPECS:
            cols_list = [out[p].tolist() for p in parts]
            out[name] = pd.Series(
                ["|".join(_cat_key(v) for v in tup) for tup in zip(*cols_list)],
                index=out.index,
                dtype=object,
            )
        return out

    def fit(self, df: pd.DataFrame) -> "CouponFeatureEngineer":
        raw = self._raw(df)
        cat_sources = list(CAT_COLS_RAW) + list(INTERACTION_NAMES)
        self.cat_cols = ["cat_" + c for c in cat_sources]
        self.num_cols = ["num_" + c for c in NUM_COLS_RAW]
        self.feature_names = list(self.num_cols) + list(self.cat_cols)
        for c in cat_sources:
            uniq = sorted(set(_cat_key(v) for v in raw[c].tolist()))
            self.vocab[c] = {v: i for i, v in enumerate(uniq)}
        self.te_key_cols = list(TE_KEY_COLS)
        # Low-cardinality columns (identities and interactions) get one-hot
        # encoding for the ExtraTrees matrix.
        self.et_onehot_cols = [
            c for c in cat_sources if 1 <= len(self.vocab.get(c, {})) <= ET_MAX_CARD
        ]
        self.et_onehot_cards = {c: len(self.vocab[c]) for c in self.et_onehot_cols}
        self.gbdt_cat_flags = []
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
        cols = list(getattr(self, "te_key_cols", None) or TE_KEY_COLS)
        return {
            c: np.asarray([_cat_key(v) for v in raw[c].tolist()], dtype=object)
            for c in cols
        }

    def _te_build(self, keys, y, mask, prior, m):
        maps: Dict[str, Dict[str, float]] = {}
        yy = np.asarray(y, dtype=np.float64)[mask]
        if mask.any() and yy.size:
            prior = float(yy.mean())
        for c in list(getattr(self, "te_key_cols", None) or TE_KEY_COLS):
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
        if not getattr(self, "te_key_cols", None):
            self.te_key_cols = list(TE_KEY_COLS)
        self.te_col_names = list(self.te_key_cols)
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

    def transform_target_encoding_oof(
        self, df: pd.DataFrame, fold_ids
    ) -> np.ndarray:
        """Encode each row with the fold map that EXCLUDES its own fold.

        Leak-free training features for the final models: no row's encoding
        is built from a map that saw that row's own label.
        """
        fids = np.asarray(fold_ids).ravel()
        n = len(df)
        cols = list(self.te_col_names or [])
        if not cols or n == 0 or fids.shape[0] != n:
            return self.transform_target_encoding(df)
        keys = self._te_keys(df)
        out = np.empty((n, len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            vals = np.empty(n, dtype=np.float64)
            kk = np.asarray(keys[c], dtype=object)
            for k in np.unique(fids):
                mask = fids == k
                entry = self.te_maps_fold.get(int(k))
                if entry is None:
                    maps, prior = self.te_maps, float(self.te_prior)
                else:
                    maps, prior = entry
                mp = maps.get(c, {})
                vals[mask] = [mp.get(x, prior) for x in kk[mask].tolist()]
            out[:, j] = vals
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
        # Dropout on the concatenated embedding-vector block: prevents the
        # first linear layer from memorising rare category combinations.
        self.emb_drop = nn.Dropout(float(min(0.3, max(0.05, dropout * 0.5))))
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
        z = torch.cat(parts, dim=1)
        return self.net(self.emb_drop(z))


class TrainConfig:
    def __init__(self):
        self.epochs = 40
        self.batch_size = 256
        self.lr = 2e-3
        self.weight_decay = 1e-4
        self.emb_dim = 8
        self.hidden = 128
        self.dropout = 0.2
        self.n_seeds = 4
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
    try:
        lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    except TypeError:  # older torch without label smoothing
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
    best_loss = float("inf")
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
            # Model selection on validation log-loss (stabiliser); accuracy is
            # still the only metric used for reporting / blending.
            pv = np.clip(p.astype(np.float64), 1e-9, 1.0)
            vloss = float(-np.log(pv[np.arange(len(Yv)), Yv]).mean())
            if vloss < best_loss - 1e-6:
                best_loss, best_acc, best_ep, since = vloss, acc, ep + 1, 0
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
    cards = [len(fe.vocab[c]) + 1 for c in _cat_sources(fe)]
    cat_shifted, cards = _shift_cat(cat, cards)

    # Leak-free training features for the final models: encode each row with
    # the fold map that excludes its own fold so final models never train on
    # a target encoding derived from their own label.
    cat_tr, num_tr, _cat_cols_tr, _num_cols_tr = _prepare_fold(
        fe, features, oof_folds=folds
    )
    num_tr_n = normalizer.transform(num_tr)
    cat_shifted_tr, _ = _shift_cat(cat_tr, cards)

    X_gbdt_all = _prepare_gbdt_matrix(fe, features, oof_folds=folds)
    gbdt_cat_mask = np.asarray(getattr(fe, "gbdt_cat_flags", []) or [], dtype=bool)
    if gbdt_cat_mask.size != X_gbdt_all.shape[1]:
        gbdt_cat_mask = None
    X_et_all = _prepare_et_matrix(fe, features, oof_folds=folds)
    oof_nn = np.zeros((n, n_classes), dtype=np.float64)
    oof_gb = np.zeros((n, n_classes), dtype=np.float64)
    oof_et = np.zeros((n, n_classes), dtype=np.float64)
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
        for s in NN_SEEDS:
            m, bep, _ = _train_one_model(
                int(s),
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
        gb_models = _train_gbdt(X_gbdt_k[tr_mask], y[tr_mask], gbdt_cat_mask)
        pg = _predict_gbdt(gb_models, X_gbdt_k[val_mask])
        if pg is not None and pg.shape == (int(val_mask.sum()), n_classes):
            oof_gb[val_mask] = pg
        # ExtraTrees on fold-specific TE encodings (no label leakage into OOF).
        X_et_k = _prepare_et_matrix(fe, features, fold_id=k)
        et_fold = _fit_extra_trees(X_et_k[tr_mask], y[tr_mask], 7 + k)
        if et_fold is not None:
            pe = np.asarray(et_fold.predict_proba(X_et_k[val_mask]), dtype=np.float64)
            if pe.shape == (int(val_mask.sum()), n_classes):
                oof_et[val_mask] = pe
        print(f"[train] fold {k + 1}/{n_folds} done")

    acc_nn = float((oof_nn.argmax(1) == y).mean())
    print(f"[train] OOF nn acc={acc_nn:.4f}")

    # Non-negative 3-way probability blend (NN, HistGB, ExtraTrees) searched on
    # OOF accuracy, then normalised to sum to 1.
    blend_alpha = 1.0
    oof_comps: List[Optional[np.ndarray]] = [oof_nn]
    comp_names: List[str] = ["nn"]
    if np.any(oof_gb):
        oof_comps.append(oof_gb)
        comp_names.append("gbdt")
    if np.any(oof_et):
        oof_comps.append(oof_et)
        comp_names.append("et")
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

    if best_epochs:
        cfg.epochs = int(max(10, min(cfg.epochs, int(np.median(best_epochs)) + 5)))
    full_models = []
    for s in range(cfg.n_seeds):
        m, _, _ = _train_one_model(
            200 + s, cat_shifted_tr, num_tr_n, y, cards, num_tr_n.shape[1],
            n_classes, cfg,
        )
        full_models.append(m)
    gbdt_models = (
        _train_gbdt(X_gbdt_all, y, gbdt_cat_mask) if HAS_GBDT else []
    )
    et_full = _fit_extra_trees(X_et_all, y, 77)
    extra_trees_models = [et_full] if et_full is not None else []

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
    oof_folds=None,
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
    if oof_folds is not None:
        te = fe.transform_target_encoding_oof(df, oof_folds)
    else:
        te = fe.transform_target_encoding(df, fold_id=fold_id)
    if te.shape[1] > 0:
        num = np.concatenate([num, te.astype(np.float32)], axis=1)
        num_cols = list(num_cols) + ["te_" + c for c in fe.te_col_names]
    return cat, num, cat_cols, num_cols


def _cat_sources(fe: CouponFeatureEngineer) -> List[str]:
    """Raw column names backing the ``cat_*`` columns, in block order."""
    return [c[len("cat_"):] for c in list(getattr(fe, "cat_cols", []) or [])]


def _prepare_gbdt_matrix(
    fe: CouponFeatureEngineer,
    df: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
    fold_id: Optional[int] = None,
    oof_folds=None,
) -> np.ndarray:
    """GBDT matrix: NATIVE categorical block + numeric block + smoothed TE.

    The categorical block keeps integer-coded ``cat_*`` identity and
    interaction levels (missing -> ``NaN``) so HistGradientBoostingClassifier
    can use its native categorical splits instead of the false ordinal
    ordering the trees used to see. The per-column mask is stored on the
    fitted engineer (``gbdt_cat_flags``) so prediction rebuilds it identically.

    ``feature_names`` is accepted for backwards compatibility; the column
    layout is derived from the fitted engineer so train and predict agree.
    """
    Xfe = fe.transform(df).reindex(columns=fe.feature_names, fill_value=0.0)
    cat_cols, num_cols = split_feature_columns(fe.feature_names)
    blocks: List[np.ndarray] = []
    flags: List[bool] = []
    if cat_cols:
        codes = np.nan_to_num(Xfe[cat_cols].values, nan=-1.0).astype(np.float64)
        codes = np.where(codes < 0.0, np.nan, codes).astype(np.float32)
        blocks.append(codes)
        flags.extend([True] * len(cat_cols))
    if num_cols:
        vals = Xfe[num_cols].values.astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        blocks.append(vals)
        flags.extend([False] * len(num_cols))
    if oof_folds is not None:
        te = fe.transform_target_encoding_oof(df, oof_folds)
    else:
        te = fe.transform_target_encoding(df, fold_id=fold_id)
    if te.shape[1] > 0:
        te = np.nan_to_num(
            te.astype(np.float32), nan=0.5, posinf=0.5, neginf=0.5
        )
        blocks.append(te)
        flags.extend([False] * te.shape[1])
    if blocks:
        X = np.concatenate(blocks, axis=1).astype(np.float32)
    else:
        X = np.zeros((len(Xfe), 0), dtype=np.float32)
    fe.gbdt_cat_flags = flags
    return X


def _prepare_et_matrix(
    fe: CouponFeatureEngineer,
    df: pd.DataFrame,
    fold_id: Optional[int] = None,
    oof_folds=None,
) -> np.ndarray:
    """ExtraTrees matrix: numeric + TE + one-hot low-cardinality categoricals.

    ``fold_id`` selects the fold-specific TE maps so OOF rows are encoded with
    statistics that never saw their own label.
    """
    Xfe = fe.transform(df).reindex(columns=fe.feature_names, fill_value=0.0)
    _cat_cols, num_cols = split_feature_columns(fe.feature_names)
    blocks: List[np.ndarray] = []
    if num_cols:
        blocks.append(
            np.nan_to_num(
                Xfe[num_cols].values.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
    cards_by_name = dict(getattr(fe, "et_onehot_cards", {}) or {})
    oh_parts: List[np.ndarray] = []
    for c in list(getattr(fe, "et_onehot_cols", []) or []):
        cards = int(cards_by_name.get(c, 0))
        col = "cat_" + c
        if cards <= 0 or col not in Xfe.columns:
            continue
        codes = np.nan_to_num(Xfe[col].values, nan=-1.0).astype(np.int64)
        oh = np.zeros((len(Xfe), cards), dtype=np.float32)
        ok = (codes >= 0) & (codes < cards)
        rows = np.where(ok)[0]
        oh[rows, codes[ok]] = 1.0
        oh_parts.append(oh)
    if oh_parts:
        blocks.append(np.concatenate(oh_parts, axis=1))
    if oof_folds is not None:
        te = fe.transform_target_encoding_oof(df, oof_folds)
    else:
        te = fe.transform_target_encoding(df, fold_id=fold_id)
    if te.shape[1] > 0:
        blocks.append(
            np.nan_to_num(te.astype(np.float32), nan=0.5, posinf=0.5, neginf=0.5)
        )
    if not blocks:
        return np.zeros((len(Xfe), 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _fit_extra_trees(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 7):
    """Bagged, decorrelated tree member trained on one fold (or all rows)."""
    if not HAS_ET or ExtraTreesClassifier is None or X_tr.shape[1] == 0:
        return None
    try:
        clf = ExtraTreesClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=2,
            random_state=int(seed),
        )
        clf.fit(X_tr, y_tr)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[train] ExtraTrees skipped: {exc}")
        return None
    return clf


def _predict_et(models: List[Any], X: np.ndarray) -> Optional[np.ndarray]:
    if not models:
        return None
    probas = [np.asarray(m.predict_proba(X), dtype=np.float64) for m in models]
    return np.mean(probas, axis=0)


def _train_gbdt(
    X_tr: np.ndarray, y_tr: np.ndarray, cat_mask: Optional[np.ndarray] = None
) -> List[Any]:
    """Train a small diverse ensemble of gradient-boosted tree models.

    ``cat_mask`` (bool per column) enables HistGradientBoostingClassifier's
    native categorical splits for the interaction / identity block.
    """
    if not HAS_GBDT:
        return []
    mask = None
    if cat_mask is not None:
        cand = np.asarray(cat_mask, dtype=bool).ravel()
        if cand.size == X_tr.shape[1] and cand.any():
            mask = cand
    models: List[Any] = []
    for gc in GBDT_CONFIGS:
        clf = None
        if mask is not None:
            try:
                kwargs = dict(gc)
                kwargs["categorical_features"] = mask
                clf = HistGradientBoostingClassifier(**kwargs)
            except (TypeError, ValueError):
                clf = None
        if clf is None:
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


def _simplex_grid(k: int, step: float = 0.05):
    """Yield weight tuples on the (k-1)-simplex, coarse grid of size ``step``."""
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    if k <= 1:
        yield (1.0,)
        return
    stack: List[Tuple[float, ...]] = [tuple()]
    for _ in range(k - 1):
        nxt: List[Tuple[float, ...]] = []
        for pre in stack:
            s = float(sum(pre))
            for v in vals.tolist():
                if s + v <= 1.0 + 1e-9:
                    nxt.append(pre + (float(v),))
        stack = nxt
    for pre in stack:
        yield pre + (float(1.0 - sum(pre)),)


def _search_blend_weights(
    probs: List[Optional[np.ndarray]],
    y_idx: np.ndarray,
    step: float = 0.05,
) -> Tuple[List[float], float]:
    """Non-negative weights (normalised to 1) maximising OOF accuracy."""
    comps = [
        np.asarray(p, dtype=np.float64)
        for p in probs
        if p is not None and np.asarray(p).size > 0
    ]
    if not comps:
        return [], 0.0
    y_idx = np.asarray(y_idx).ravel()
    k = len(comps)
    if k == 1:
        return [1.0], float((comps[0].argmax(1) == y_idx).mean())
    best_w = np.full(k, 1.0 / k, dtype=np.float64)
    blended = sum(best_w[i] * comps[i] for i in range(k))
    best_acc = float((blended.argmax(1) == y_idx).mean())
    for combo in _simplex_grid(k, step):
        w = np.asarray(combo, dtype=np.float64)
        tot = float(w.sum())
        if tot <= 1e-9:
            continue
        w = w / tot
        blended = sum(w[i] * comps[i] for i in range(k))
        acc = float((blended.argmax(1) == y_idx).mean())
        if acc > best_acc + 1e-12:
            best_acc, best_w = acc, w
    tot = float(best_w.sum())
    best_w = best_w / tot if tot > 0 else np.full(k, 1.0 / k)
    return [float(x) for x in best_w.tolist()], best_acc


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
        "extra_trees_models": (
            list(extra_trees_models) if extra_trees_models else []
        ),
        "blend_alpha": float(blend_alpha),
        "blend_weights": [float(w) for w in (blend_weights or [])],
        "gbdt_cat_flags": list(getattr(fe, "gbdt_cat_flags", []) or []),
        "et_onehot_cols": list(getattr(fe, "et_onehot_cols", []) or []),
        "et_onehot_cards": dict(getattr(fe, "et_onehot_cards", {}) or {}),
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
    et_models = bundle.get("extra_trees_models", []) or []
    blend_alpha = float(bundle.get("blend_alpha", 1.0))
    comps: List[np.ndarray] = [np.asarray(proba, dtype=np.float64)]
    if gbdt_models:
        X_gbdt = _prepare_gbdt_matrix(fe, features, bundle["feature_names"])
        proba_gbdt = _predict_gbdt(gbdt_models, X_gbdt)
        if proba_gbdt is not None and proba_gbdt.shape == proba.shape:
            comps.append(np.asarray(proba_gbdt, dtype=np.float64))
    X_et = None
    if et_models:
        X_et = _prepare_et_matrix(fe, features)
        proba_et = _predict_et(et_models, X_et)
        if proba_et is not None and proba_et.shape == proba.shape:
            comps.append(np.asarray(proba_et, dtype=np.float64))

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