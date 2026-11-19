"""In-vehicle coupon acceptance — merged pipeline.

Leakage-safe target-encoded features + TabM batch-ensemble + K-fold OOF
training + fresh-process CLI predict.  Metric: accuracy.

CLI:
    python solution.py train   --data /data --model /model
    python solution.py predict --data /data --model /model --output /output/predictions.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

# =========================================================================== #
# Step 1: data processing & feature engineering
# =========================================================================== #
TARGET_COL = "label"
ROW_ID_COL = "row_id"

NOMINAL_CATS: List[str] = [
    "time",
    "coupon",
    "expiration",
    "gender",
    "maritalStatus",
    "education",
    "occupation",
    "income",
]

FREQ_COLS: List[str] = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]

BIN_INT_COLS: List[str] = [
    "has_children",
    "toCoupon_GEQ5min",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
    "direction_opp",
]

TE_SOURCES: List[str] = [
    "coupon_occupation",
    "coupon_income",
    "coupon_time",
]


def _norm(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v).strip().lower().replace(" ", "")


_AGE_N = {
    "under21": 0,
    "21": 1,
    "26": 2,
    "31": 3,
    "36": 4,
    "41": 5,
    "46": 6,
    "50plus": 7,
}
_INCOME_N = {
    "lessthan$12500": 0.5,
    "$12500-$24999": 18.75,
    "$25000-$37499": 31.25,
    "$37500-$49999": 43.75,
    "$50000-$62499": 56.25,
    "$62500-$74999": 68.75,
    "$75000-$87499": 81.25,
    "$87500-$99999": 93.75,
    "$100000ormore": 110.0,
}
_EDU_N = {
    "somehighschool": 0,
    "highschoolgraduate": 1,
    "somecollege-nodegree": 2,
    "associatesdegree": 3,
    "bachelorsdegree": 4,
    "graduatedegree": 5,
}
_TIME_N = {"7am": 7, "10am": 10, "2pm": 14, "6pm": 18, "10pm": 22}
_EXP_N = {"2h": 2.0, "1d": 24.0}
_FREQ_N = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}


def load_schema(data_dir: str) -> Dict[str, Any]:
    path = os.path.join(data_dir, "schema.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_features(data_dir: str, schema: Dict[str, Any]) -> pd.DataFrame:
    feats = pd.read_csv(os.path.join(data_dir, "features.csv"))
    cols = schema.get("feature_columns")
    if cols:
        missing = [c for c in cols if c not in feats.columns]
        if missing:
            raise ValueError(f"features.csv missing schema columns: {missing}")
        feats = feats[cols].copy()
    feats = feats.reset_index(drop=True)
    return feats


def load_train_data(data_dir: str) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    schema = load_schema(data_dir)
    feats = _read_features(data_dir, schema)

    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    if TARGET_COL in labels.columns:
        y_raw = labels[TARGET_COL]
    else:
        y_raw = labels.iloc[:, 0]

    n = min(len(feats), len(y_raw))
    feats = feats.iloc[:n].reset_index(drop=True)
    y_raw = y_raw.iloc[:n].reset_index(drop=True)

    classes = pd.unique(y_raw.astype(str))
    try:
        classes = sorted(classes, key=lambda s: float(s))
    except (TypeError, ValueError):
        classes = sorted(classes)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = y_raw.astype(str).map(label_to_idx).to_numpy(dtype=np.int64)

    schema = dict(schema)
    schema["label_classes"] = classes
    return feats, y, schema


def load_predict_data(data_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    schema = load_schema(data_dir)
    feats = _read_features(data_dir, schema)
    return feats, schema


def get_cv_splits(
    y: np.ndarray, n_splits: int = 5, random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y).ravel()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]


class FeatureEngineer:
    """Stateful, leakage-safe feature pipeline.

    `fit_transform(X, y)` returns out-of-fold target encodings (safe for
    training).  `transform(X)` applies full-training encodings (inference).
    """

    def __init__(
        self,
        te_sources: Optional[Sequence[str]] = None,
        smooth: float = 20.0,
        n_splits: int = 5,
        random_state: int = 42,
        scale: bool = True,
    ) -> None:
        self.te_sources = list(te_sources if te_sources is not None else TE_SOURCES)
        self.smooth = float(smooth)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.scale = bool(scale)

        self.prior_: float = 0.0
        self.base_cols_: List[str] = []
        self.te_maps_: Dict[str, Dict[str, float]] = {}
        self.feature_names_: List[str] = []
        self.base_medians_: pd.Series = pd.Series(dtype=float)
        self.mean_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.std_: np.ndarray = np.ones(0, dtype=np.float64)

    @staticmethod
    def _base(df: pd.DataFrame) -> pd.DataFrame:
        b = pd.DataFrame(index=np.arange(len(df)))
        b["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        for c in BIN_INT_COLS:
            if c in df.columns:
                b[c] = pd.to_numeric(df[c], errors="coerce")

        b["age_num"] = df["age"].map(lambda v: _AGE_N.get(_norm(v), np.nan))
        b["income_num"] = df["income"].map(lambda v: _INCOME_N.get(_norm(v), np.nan))
        b["education_num"] = df["education"].map(lambda v: _EDU_N.get(_norm(v), np.nan))
        b["time_hour"] = df["time"].map(lambda v: _TIME_N.get(_norm(v), np.nan))
        b["expiration_hours"] = df["expiration"].map(
            lambda v: _EXP_N.get(_norm(v), np.nan)
        )

        hour = b["time_hour"]
        b["is_morning"] = ((hour >= 7) & (hour < 12)).astype(float)
        b["is_lunch"] = (hour == 14).astype(float)
        b["is_evening"] = (hour >= 18).astype(float)

        freq_vals = {}
        for c in FREQ_COLS:
            if c in df.columns:
                vals = df[c].map(lambda v: _FREQ_N.get(_norm(v), np.nan))
                freq_vals[c] = vals
                b[c + "_ord"] = vals.fillna(-1.0)
        if freq_vals:
            stacked = pd.concat(freq_vals.values(), axis=1)
            b["freq_sum"] = stacked.sum(axis=1, skipna=True).fillna(0.0)
            b["freq_known"] = stacked.notna().sum(axis=1).astype(float)
            b["freq_missing"] = stacked.isna().sum(axis=1).astype(float)

        g15 = b.get("toCoupon_GEQ15min", pd.Series(0.0, index=b.index)).fillna(1.0)
        g25 = b.get("toCoupon_GEQ25min", pd.Series(0.0, index=b.index)).fillna(1.0)
        b["distance_est"] = np.where(g15 <= 0, 5.0, np.where(g25 <= 0, 15.0, 25.0))
        b["near_coupon"] = (b["distance_est"] <= 5.0).astype(float)

        cp = df["coupon"].map(_norm)
        b["is_bar"] = cp.eq("bar").astype(float)
        b["is_coffee"] = cp.str.contains("coffee", na=False).astype(float)
        b["is_restaurant"] = cp.str.contains("restaurant", na=False).astype(float)
        b["is_carryout"] = cp.str.contains("carry", na=False).astype(float)
        b["is_cheap_restaurant"] = cp.eq("restaurant(<20)").astype(float)

        b["income_x_children"] = b["income_num"] * b.get(
            "has_children", pd.Series(0.0, index=b.index)
        ).fillna(0.0)
        b["freq_per_distance"] = b["freq_sum"] / (1.0 + b["distance_est"])
        return b

    @staticmethod
    def _combos(df: pd.DataFrame) -> pd.DataFrame:
        def s(name: str) -> pd.Series:
            if name not in df.columns:
                return pd.Series(["__NA__"] * len(df), index=np.arange(len(df)))
            return df[name].astype(str).fillna("__NA__").reset_index(drop=True)

        coupon = s("coupon")
        out = pd.DataFrame(index=np.arange(len(df)))
        out["coupon"] = coupon
        out["occupation"] = s("occupation")
        out["time"] = s("time")
        out["coupon_time"] = coupon + "|" + s("time")
        out["coupon_expiration"] = coupon + "|" + s("expiration")
        out["coupon_marital"] = coupon + "|" + s("maritalStatus")
        out["coupon_children"] = coupon + "|" + s("has_children")
        out["coupon_income"] = coupon + "|" + s("income")
        out["coupon_age"] = coupon + "|" + s("age")
        out["coupon_occupation"] = coupon + "|" + s("occupation")

        # distance bucket derived from the already-computed binary flags
        if "toCoupon_GEQ15min" in df.columns:
            g15 = pd.to_numeric(df["toCoupon_GEQ15min"], errors="coerce")
        else:
            g15 = pd.Series([1.0] * len(df), index=np.arange(len(df)))
        if "toCoupon_GEQ25min" in df.columns:
            g25 = pd.to_numeric(df["toCoupon_GEQ25min"], errors="coerce")
        else:
            g25 = pd.Series([1.0] * len(df), index=np.arange(len(df)))
        g15 = g15.fillna(1.0).reset_index(drop=True).to_numpy()
        g25 = g25.fillna(1.0).reset_index(drop=True).to_numpy()
        dist = np.where(g15 <= 0, "5", np.where(g25 <= 0, "15", "25"))
        out["coupon_distance"] = coupon.to_numpy() + "|" + dist.astype(str)

        out["coupon_direction"] = coupon + "|" + s("direction_opp")
        return out

    def _te_map(self, keys: pd.Series, y: np.ndarray) -> Dict[str, float]:
        frame = pd.DataFrame({"k": keys.astype(str).to_numpy(), "y": np.asarray(y)})
        agg = frame.groupby("k")["y"].agg(["sum", "count"])
        smoothed = (agg["sum"] + self.prior_ * self.smooth) / (
            agg["count"] + self.smooth
        )
        return {str(k): float(v) for k, v in smoothed.items()}

    def _apply_te(
        self, combos: pd.DataFrame, maps: Dict[str, Dict[str, float]], prior: float
    ) -> pd.DataFrame:
        data = {}
        for name in self.te_sources:
            mapping = maps.get(name, {})
            data["te_" + name] = (
                combos[name].map(mapping).fillna(prior).to_numpy(dtype=np.float64)
            )
        return pd.DataFrame(data)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "FeatureEngineer":
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(df) != len(y):
            raise ValueError(f"X/y length mismatch: {len(df)} vs {len(y)}")
        self.prior_ = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))

        base = self._base(df)
        self.base_medians_ = base.median(numeric_only=True).fillna(0.0)
        base = self._fill_base(base)
        self.base_cols_ = list(base.columns)

        combos = self._combos(df)
        self.te_maps_ = {
            name: self._te_map(combos[name], y) for name in self.te_sources
        }
        self.feature_names_ = self.base_cols_ + ["te_" + n for n in self.te_sources]

        raw = self._transform_raw(df)
        arr = raw.to_numpy(dtype=np.float64)
        if self.scale:
            self.mean_ = arr.mean(axis=0)
            self.std_ = arr.std(axis=0)
            self.std_[self.std_ < 1e-8] = 1.0
        else:
            self.mean_ = np.zeros(arr.shape[1], dtype=np.float64)
            self.std_ = np.ones(arr.shape[1], dtype=np.float64)
        return self

    def fit_transform(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        self.fit(df, y_arr)

        raw = self._transform_raw(df)
        combos = self._combos(df)
        te_names = ["te_" + n for n in self.te_sources]

        oof = np.full((len(df), len(self.te_sources)), self.prior_, dtype=np.float64)
        if len(np.unique(y_arr)) > 1 and self.n_splits > 1:
            skf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            for tr, va in skf.split(np.zeros(len(y_arr)), y_arr):
                fold_maps = {
                    name: self._te_map(combos[name].iloc[tr], y_arr[tr])
                    for name in self.te_sources
                }
                for j, name in enumerate(self.te_sources):
                    mapped = combos[name].iloc[va].map(fold_maps[name])
                    oof[va, j] = mapped.fillna(self.prior_).to_numpy(dtype=np.float64)

        raw = raw.copy()
        raw[te_names] = oof
        return self._scale_frame(raw)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("FeatureEngineer.transform called before fit.")
        raw = self._transform_raw(df)
        return self._scale_frame(raw)

    def _fill_base(self, base: pd.DataFrame) -> pd.DataFrame:
        med = getattr(self, "base_medians_", None)
        if med is not None and len(med) > 0:
            base = base.fillna(med)
        return base.fillna(0.0)

    def _transform_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self._fill_base(self._base(df))
        combos = self._combos(df)
        te = self._apply_te(combos, self.te_maps_, self.prior_)
        raw = pd.concat(
            [base.reset_index(drop=True), te.reset_index(drop=True)], axis=1
        )
        cols = self.feature_names_ or list(raw.columns)
        return raw.reindex(columns=cols, fill_value=0.0)

    def _scale_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not self.scale:
            return raw.reset_index(drop=True)
        arr = raw.to_numpy(dtype=np.float64)
        arr = (arr - self.mean_[None, :]) / self.std_[None, :]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(arr, columns=list(self.feature_names_))

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "FeatureEngineer":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, FeatureEngineer):
            raise TypeError("Loaded object is not a FeatureEngineer")
        return obj


class GBDTFeatureEncoder:
    """Fold-fitted ordinal + NaN-tolerant encoder for the GBDT branch.

    Deterministic, picklable, and never fits inside `transform`.
    Slot 0 of each ordinal map is reserved for unknown categories.
    """

    _ENG_NUMERIC_NAMES: List[str] = [
        "age_num",
        "income_num",
        "education_num",
        "time_hour",
        "expiration_hours",
        "is_morning",
        "is_lunch",
        "is_evening",
        "freq_sum",
        "freq_known",
        "freq_missing",
        "distance_est",
        "near_coupon",
        "is_bar",
        "is_coffee",
        "is_restaurant",
        "is_carryout",
        "is_cheap_restaurant",
        "income_x_children",
        "freq_per_distance",
    ] + [c + "_ord" for c in FREQ_COLS]

    # Coupon-centric interaction categoricals built statelessly via _combos.
    _INTERACTION_CATS: List[str] = [
        "coupon_time",
        "coupon_expiration",
        "coupon_direction",
        "coupon_children",
        "coupon_income",
    ]

    def __init__(self) -> None:
        self.ordinal_maps_: Dict[str, Dict[str, int]] = {}
        self.medians_: Dict[str, float] = {}
        self.feature_names_: List[str] = []
        self.cat_mask_: np.ndarray = np.zeros(0, dtype=bool)

    @staticmethod
    def _cols() -> List[str]:
        cols = list(NOMINAL_CATS) + list(FREQ_COLS) + list(BIN_INT_COLS) + ["temperature"]
        cols += list(GBDTFeatureEncoder._ENG_NUMERIC_NAMES)
        cols += list(GBDTFeatureEncoder._INTERACTION_CATS)
        seen = set()
        out = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    @staticmethod
    def _engineered_numerics(df: pd.DataFrame) -> pd.DataFrame:
        """Stateless row-wise numerics mirroring FeatureEngineer._base.

        Uses only per-row transforms, so it is leakage-free and safe to call
        on any split of the data (train or inference).
        """
        base = FeatureEngineer._base(df)
        out = pd.DataFrame(index=np.arange(len(df)))
        for c in GBDTFeatureEncoder._ENG_NUMERIC_NAMES:
            if c in base.columns:
                out[c] = pd.to_numeric(base[c], errors="coerce").to_numpy(
                    dtype=np.float64
                )
            else:
                out[c] = np.nan
        return out

    def fit(self, df: pd.DataFrame) -> "GBDTFeatureEncoder":
        cols = self._cols()
        interaction_set = set(self._INTERACTION_CATS)
        cat_cols = set(NOMINAL_CATS) | set(FREQ_COLS) | interaction_set
        eng = self._engineered_numerics(df)
        eng_cols = set(eng.columns)
        combos = FeatureEngineer._combos(df)

        self.ordinal_maps_ = {}
        self.medians_ = {}
        self.feature_names_ = list(cols)
        mask = []
        for c in cols:
            if c in cat_cols:
                if c in interaction_set:
                    if c in combos.columns:
                        levels = sorted(
                            combos[c].astype(str).fillna("__NA__").unique().tolist()
                        )
                    else:
                        levels = []
                elif c in df.columns:
                    levels = sorted(df[c].astype(str).fillna("__NA__").unique().tolist())
                else:
                    levels = []
                mapping = {lvl: i + 1 for i, lvl in enumerate(levels)}
                self.ordinal_maps_[c] = mapping
                mask.append(True)
            elif c in eng_cols:
                num = pd.to_numeric(eng[c], errors="coerce")
                med = float(num.median()) if num.notna().any() else 0.0
                self.medians_[c] = med
                mask.append(False)
            else:
                if c in df.columns:
                    num = pd.to_numeric(df[c], errors="coerce")
                    med = float(num.median()) if num.notna().any() else 0.0
                else:
                    med = 0.0
                self.medians_[c] = med
                mask.append(False)
        self.cat_mask_ = np.asarray(mask, dtype=bool)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.feature_names_:
            raise RuntimeError("GBDTFeatureEncoder.transform called before fit.")
        n = len(df)
        eng = self._engineered_numerics(df)
        eng_cols = set(eng.columns)
        interaction_set = set(self._INTERACTION_CATS)
        combos = FeatureEngineer._combos(df)
        arr = np.zeros((n, len(self.feature_names_)), dtype=np.float64)
        for j, c in enumerate(self.feature_names_):
            if c in self.ordinal_maps_:
                mapping = self.ordinal_maps_[c]
                if c in interaction_set:
                    if c in combos.columns:
                        vals = combos[c].astype(str).fillna("__NA__")
                    else:
                        vals = pd.Series(["__NA__"] * n)
                elif c in df.columns:
                    vals = df[c].astype(str).fillna("__NA__")
                else:
                    vals = pd.Series(["__NA__"] * n)
                arr[:, j] = vals.map(mapping).fillna(0).to_numpy(dtype=np.float64)
            elif c in eng_cols:
                num = pd.to_numeric(eng[c], errors="coerce")
                num = num.fillna(self.medians_.get(c, 0.0))
                arr[:, j] = num.to_numpy(dtype=np.float64)
            else:
                if c in df.columns:
                    num = pd.to_numeric(df[c], errors="coerce")
                else:
                    num = pd.Series([np.nan] * n)
                num = num.fillna(self.medians_.get(c, 0.0))
                arr[:, j] = num.to_numpy(dtype=np.float64)
        return arr

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)


def build_train_matrix(
    data_dir: str,
    smooth: float = 20.0,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, "FeatureEngineer", Dict[str, Any]]:
    feats, y, schema = load_train_data(data_dir)
    fe = FeatureEngineer(
        smooth=smooth, n_splits=n_splits, random_state=random_state, scale=True
    )
    X = fe.fit_transform(feats, y)
    schema = dict(schema)
    schema["feature_names"] = list(fe.feature_names_)
    return X, y, fe, schema


def build_predict_matrix(
    data_dir: str, fe: "FeatureEngineer"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    feats, schema = load_predict_data(data_dir)
    X = fe.transform(feats)
    return X, schema


# =========================================================================== #
# Step 2: TabM model / loss / optimizer / inference helpers
# =========================================================================== #
class TabMRank1Block(nn.Module):
    """One MLP block shared across K ensemble heads (TabM formulation)."""

    def __init__(self, hidden: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)

        self.W = nn.Parameter(torch.empty(hidden, hidden))
        self.b = nn.Parameter(torch.zeros(hidden))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.zeros_(self.b)

        scale = 1.0 / math.sqrt(hidden)
        self.U = nn.Parameter(torch.randn(num_heads, hidden, 1) * scale)
        self.V = nn.Parameter(torch.randn(num_heads, 1, hidden) * scale)

        self.ln = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K, B, H = x.shape
        W_eff = self.W.unsqueeze(0) + torch.bmm(self.U, self.V)  # (K, H, H)
        h = torch.baddbmm(
            self.b.view(1, 1, -1).expand(K, B, -1),
            x,
            W_eff.transpose(-2, -1),
        )
        h = self.act(h)
        h = self.dropout(h)
        h = self.ln(h + x)
        return h


class TabM(nn.Module):
    """Batch-ensemble MLP with rank-1 head perturbations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_blocks = int(num_blocks)
        self.num_classes = int(num_classes)

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                TabMRank1Block(self.hidden_dim, self.num_heads, dropout)
                for _ in range(self.num_blocks)
            ]
        )

        self.W_out = nn.Parameter(
            torch.empty(self.num_heads, self.num_classes, self.hidden_dim)
        )
        self.b_out = nn.Parameter(torch.zeros(self.num_heads, self.num_classes))
        for k in range(self.num_heads):
            nn.init.xavier_uniform_(self.W_out[k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        h = h.unsqueeze(0).expand(self.num_heads, -1, -1)
        for block in self.blocks:
            h = block(h)
        logits = torch.baddbmm(
            self.b_out.unsqueeze(1),
            h,
            self.W_out.transpose(-2, -1),
        )
        return logits


def build_model(
    input_dim: int,
    hidden_dim: int = 128,
    num_heads: int = 8,
    num_blocks: int = 2,
    dropout: float = 0.1,
    num_classes: int = 2,
) -> TabM:
    return TabM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
        num_classes=num_classes,
    )


def build_criterion(label_smoothing: float = 0.05) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    betas: tuple = (0.9, 0.999),
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=betas,
    )


@torch.no_grad()
def predict_proba(
    model: TabM,
    X: torch.Tensor,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    if X.dim() == 1:
        X = X.unsqueeze(0)
    if batch_size is None or batch_size >= X.shape[0]:
        logits = model(X)
        return torch.softmax(logits, dim=-1).mean(dim=0)

    chunks = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size]
        logits = model(xb)
        chunks.append(torch.softmax(logits, dim=-1).mean(dim=0))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def predict_labels(
    model: TabM,
    X: torch.Tensor,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    probs = predict_proba(model, X, batch_size=batch_size)
    return probs.argmax(dim=-1)


# =========================================================================== #
# Step 3: training / evaluation / CLI
# =========================================================================== #
NUM_FOLDS = 5
NUM_EPOCHS = 60
BATCH_SIZE = 512
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
LABEL_SMOOTHING = 0.05
GRAD_CLIP = 1.0
DEVICE = torch.device("cpu")

MODEL_CONFIG = dict(
    hidden_dim=128,
    num_heads=4,
    num_blocks=2,
    dropout=0.15,
    num_classes=2,
)

FEATURE_ENGINEER_FILE = "feature_engineer.pkl"
ENSEMBLE_FILE = "tabm_ensemble.pt"
GBDT_FILE = "gbdt_ensemble.pkl"
NUM_EPOCHS = 40
PATIENCE = 8
HGB_CONFIG = dict(
    max_iter=600,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
)


def build_hgb(cat_mask: np.ndarray, seed: int = 0):
    """HistGradientBoosting classifier factory for the GBDT anchor branch."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    kwargs = dict(HGB_CONFIG)
    kwargs["random_state"] = int(seed)
    mask = np.asarray(cat_mask, dtype=bool)
    if mask.size and bool(mask.any()):
        kwargs["categorical_features"] = mask.copy()
    return HistGradientBoostingClassifier(**kwargs)


def _train_one_fold(X_tr, y_tr, X_va, y_va, seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**31 - 1))

    X_tr_t = torch.from_numpy(X_tr.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64))
    X_va_t = torch.from_numpy(X_va.astype(np.float32))

    model = build_model(input_dim=int(X_tr.shape[1]), **MODEL_CONFIG).to(DEVICE)
    criterion = build_criterion(label_smoothing=LABEL_SMOOTHING)
    optimizer = build_optimizer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    n = int(X_tr.shape[0])
    best_acc = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]

            logits = model(xb)
            K, B, C = logits.shape
            loss = criterion(logits.reshape(K * B, C), yb.repeat(K))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        model.eval()
        val_pred = predict_labels(model, X_va_t, batch_size=1024).numpy()
        val_acc = float((val_pred == y_va).mean())

        if val_acc > best_acc + 1e-6:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(
                f"      epoch {epoch:03d} loss={epoch_loss / max(n_batches, 1):.4f} "
                f"val_acc={val_acc:.4f} best={best_acc:.4f}"
            )

        if bad_epochs >= PATIENCE:
            print(f"      early-stop @ epoch {epoch} (best={best_acc:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def cmd_train(data_dir: str, model_dir: str) -> float:
    t0 = time.time()
    os.makedirs(model_dir, exist_ok=True)
    print(f"[train] data={data_dir} model={model_dir}")

    feats, y, schema = load_train_data(data_dir)
    y = np.asarray(y, dtype=np.int64)
    label_classes = list(schema["label_classes"])
    print(
        f"[train] n_rows={len(y)} n_classes={len(label_classes)} "
        f"classes={label_classes} pos_rate={float(y.mean()):.3f}"
    )

    splits = get_cv_splits(y, n_splits=NUM_FOLDS, random_state=42)
    oof_proba = np.zeros((len(y), len(label_classes)), dtype=np.float64)
    oof_gbdt = np.zeros((len(y), len(label_classes)), dtype=np.float64)
    fold_models = []
    fold_fes = []
    fold_accs = []
    fold_gbdt_encoders = []
    fold_gbdt_models = []
    feature_names = None
    input_dim = None

    for fi, (tr, va) in enumerate(splits):
        print(f"[train] fold {fi + 1}/{len(splits)} n_tr={len(tr)} n_va={len(va)}")
        fe_fold = FeatureEngineer(
            smooth=20.0, n_splits=NUM_FOLDS, random_state=42, scale=True
        )
        feats_tr = feats.iloc[tr].reset_index(drop=True)
        feats_va = feats.iloc[va].reset_index(drop=True)
        X_tr = fe_fold.fit_transform(feats_tr, y[tr]).to_numpy(dtype=np.float32)
        X_va = fe_fold.transform(feats_va).to_numpy(dtype=np.float32)
        feature_names = list(fe_fold.feature_names_)
        input_dim = int(X_tr.shape[1])
        model, best_acc = _train_one_fold(X_tr, y[tr], X_va, y[va], seed=1000 + fi)
        fold_models.append(model)
        fold_fes.append(fe_fold)
        fold_accs.append(best_acc)

        proba = predict_proba(model, torch.from_numpy(X_va), batch_size=1024).numpy()
        oof_proba[va] = proba

        # GBDT anchor branch — encoder fit on outer-train rows only;
        # deterministic 2-seed bagging per fold for variance reduction.
        gb_enc = GBDTFeatureEncoder().fit(feats_tr)
        gb_tr = gb_enc.transform(feats_tr)
        gb_va = gb_enc.transform(feats_va)
        gb_proba = np.zeros((len(va), len(label_classes)), dtype=np.float64)
        gb_seed_models = []
        for gb_seed in (2000 + fi, 3000 + fi):
            gb_model = build_hgb(gb_enc.cat_mask_, seed=gb_seed)
            gb_model.fit(gb_tr, y[tr])
            gb_raw = gb_model.predict_proba(gb_va)
            gb_classes = gb_model.classes_.astype(int)
            gb_p = np.zeros((len(va), len(label_classes)), dtype=np.float64)
            for ci, cls in enumerate(gb_classes):
                if 0 <= cls < len(label_classes):
                    gb_p[:, cls] = gb_raw[:, ci]
            gb_proba += gb_p
            gb_seed_models.append(gb_model)
        gb_proba /= float(len(gb_seed_models))
        oof_gbdt[va] = gb_proba
        fold_gbdt_encoders.append(gb_enc)
        fold_gbdt_models.append(gb_seed_models)
        gb_acc = float((gb_proba.argmax(axis=1) == y[va]).mean())
        print(f"[train]   fold {fi + 1} best_val_acc={best_acc:.4f} gbdt_acc={gb_acc:.4f}")

    oof_pred = oof_proba.argmax(axis=1)
    oof_acc = float((oof_pred == y).mean())
    gbdt_oof_acc = float((oof_gbdt.argmax(axis=1) == y).mean())

    # Constrain the MLP blend weight so GBDT keeps >= 0.7 weight and the
    # blend choice cannot be overfit on OOF.
    best_w = 0.15
    best_blend_acc = -1.0
    for w in np.linspace(0.0, 0.3, 16):
        blended = w * oof_proba + (1.0 - w) * oof_gbdt
        acc = float((blended.argmax(axis=1) == y).mean())
        if acc > best_blend_acc + 1e-9:
            best_blend_acc = acc
            best_w = float(w)
    print(
        f"[train] OOF accuracy: mlp={oof_acc:.5f} gbdt={gbdt_oof_acc:.5f} "
        f"blend={best_blend_acc:.5f} (w_mlp={best_w:.3f}) "
        f"(mean_fold={float(np.mean(fold_accs)):.5f} "
        f"std={float(np.std(fold_accs)):.5f})"
    )
    oof_acc = best_blend_acc

    with open(os.path.join(model_dir, FEATURE_ENGINEER_FILE), "wb") as fh:
        pickle.dump(fold_fes, fh, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(
        {
            "state_dicts": [m.state_dict() for m in fold_models],
            "input_dim": int(input_dim),
            "feature_names": feature_names,
            "label_classes": label_classes,
            "config": dict(MODEL_CONFIG),
            "oof_acc": float(oof_acc),
        },
        os.path.join(model_dir, ENSEMBLE_FILE),
    )
    with open(os.path.join(model_dir, GBDT_FILE), "wb") as fh:
        pickle.dump(
            {
                "encoders": fold_gbdt_encoders,
                "models": fold_gbdt_models,
                "blend_weight": float(best_w),
                "label_classes": label_classes,
                "oof_mlp_acc": float(oof_acc),
                "oof_gbdt_acc": float(gbdt_oof_acc),
                "oof_blend_acc": float(best_blend_acc),
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[train] artifacts saved -> {model_dir} ({time.time() - t0:.1f}s)")
    return oof_acc


def _restore_label(s: Any) -> Any:
    """Restore an original (typed) label from its stored string form."""
    try:
        return int(s)
    except (TypeError, ValueError):
        pass
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _ckpt_load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def cmd_predict(data_dir: str, model_dir: str, output_path: str) -> None:
    t0 = time.time()
    fe_path = os.path.join(model_dir, FEATURE_ENGINEER_FILE)
    ens_path = os.path.join(model_dir, ENSEMBLE_FILE)
    gb_path = os.path.join(model_dir, GBDT_FILE)
    if not os.path.exists(fe_path) or not os.path.exists(ens_path):
        raise FileNotFoundError(
            f"Missing saved artifacts in {model_dir}: need "
            f"{FEATURE_ENGINEER_FILE} and {ENSEMBLE_FILE}"
        )

    with open(fe_path, "rb") as fh:
        fold_fes = pickle.load(fh)
    if not isinstance(fold_fes, list) or len(fold_fes) == 0:
        raise TypeError("Loaded feature engineers are not a non-empty list")

    feats, _schema = load_predict_data(data_dir)
    n_rows = len(feats)

    ckpt = _ckpt_load(ens_path)
    label_classes = list(ckpt["label_classes"])
    input_dim = int(ckpt["input_dim"])
    config = dict(ckpt["config"])
    state_dicts = list(ckpt["state_dicts"])

    C = int(config["num_classes"])
    proba_sum = np.zeros((n_rows, C), dtype=np.float64)
    n_models = 0
    for fe_fold, sd in zip(fold_fes, state_dicts):
        X = fe_fold.transform(feats).to_numpy(dtype=np.float32)
        X_t = torch.from_numpy(X)
        m = build_model(input_dim=input_dim, **config)
        m.load_state_dict(sd)
        m.eval()
        proba_sum += predict_proba(m, X_t, batch_size=1024).numpy()
        n_models += 1
    proba_sum /= max(n_models, 1)

    blend_w = 1.0
    gbdt_proba = None
    if os.path.exists(gb_path):
        with open(gb_path, "rb") as fh:
            gb_bundle = pickle.load(fh)
        gb_encoders = list(gb_bundle["encoders"])
        gb_models = list(gb_bundle["models"])
        blend_w = float(gb_bundle.get("blend_weight", 1.0))
        C = len(label_classes)
        gb_sum = np.zeros((n_rows, C), dtype=np.float64)
        n_gb = 0
        for enc, gm_list in zip(gb_encoders, gb_models):
            gb_X = enc.transform(feats)
            gm_iter = gm_list if isinstance(gm_list, (list, tuple)) else [gm_list]
            for gm in gm_iter:
                raw = gm.predict_proba(gb_X)
                cls = gm.classes_.astype(int)
                for ci, c in enumerate(cls):
                    if 0 <= c < C:
                        gb_sum[:, c] += raw[:, ci]
                n_gb += 1
        gb_sum /= max(n_gb, 1)
        gbdt_proba = gb_sum
    else:
        print("[predict] WARNING: gbdt_ensemble.pkl not found; MLP-only inference")

    if gbdt_proba is not None:
        proba_sum = blend_w * proba_sum + (1.0 - blend_w) * gbdt_proba

    print(f"[predict] n_rows={n_rows} n_models={n_models} blend_w={blend_w:.3f}")

    pred_idx = proba_sum.argmax(axis=1)
    pred_labels = [_restore_label(label_classes[int(i)]) for i in pred_idx]

    out = pd.DataFrame(
        {
            "row_id": np.arange(len(pred_labels), dtype=np.int64),
            "prediction": pred_labels,
        }
    )
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[predict] wrote {len(out)} rows -> {output_path} ({time.time() - t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="In-vehicle coupon acceptance")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        score = cmd_train(args.data, args.model)
        print(f"Final Validation Score: {score}")
    elif args.cmd == "predict":
        cmd_predict(args.data, args.model, args.output)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()