"""
solution.py

End-to-end leakage-safe pipeline for coupon-acceptance prediction
(binary classification, official metric = accuracy).

CLI:
  python solution.py train   --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv
"""

import os
import json
import random
import argparse

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

# =========================================================================== #
# STEP 1 : DATA PROCESSING & FEATURE ENGINEERING
# =========================================================================== #
REQUIRED_COLUMNS = [
    "temperature",
    "time",
    "coupon",
    "expiration",
    "gender",
    "age",
    "maritalStatus",
    "has_children",
    "education",
    "occupation",
    "income",
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
    "toCoupon_GEQ5min",
    "toCoupon_GEQ15min",
    "toCoupon_GEQ25min",
    "direction_same",
    "direction_opp",
]

FREQ_MAP = {"never": 0.0, "less1": 1.0, "1~3": 2.0, "4~8": 3.0, "gt8": 4.0}
TIME_MAP = {"7am": 0.0, "10am": 1.0, "2pm": 2.0, "6pm": 3.0, "10pm": 4.0}
EXP_MAP = {"2h": 0.0, "1d": 1.0}
AGE_MAP = {
    "below21": 0.0,
    "21": 1.0,
    "26": 2.0,
    "31": 3.0,
    "36": 4.0,
    "41": 5.0,
    "46": 6.0,
    "50plus": 7.0,
}
INCOME_MAP = {
    "less than $12500": 0.0,
    "$12500 - $24999": 1.0,
    "$25000 - $37499": 2.0,
    "$37500 - $49999": 3.0,
    "$50000 - $62499": 4.0,
    "$62500 - $74999": 5.0,
    "$75000 - $87499": 6.0,
    "$87500 - $99999": 7.0,
    "$100000 or more": 8.0,
}
EDU_MAP = {
    "some high school": 0.0,
    "high school graduate": 1.0,
    "some college - no degree": 2.0,
    "associates degree": 3.0,
    "bachelors degree": 4.0,
    "graduate degree": 5.0,
}

FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]

COUPON_FREQ_COL = {
    "bar": "Bar",
    "coffee house": "CoffeeHouse",
    "restaurant(<20)": "RestaurantLessThan20",
    "restaurant(20-50)": "Restaurant20To50",
    "carry out & take away": "CarryAway",
}


def _norm(s):
    return s.astype(str).str.strip().str.lower()


def _map_series(s, mapping):
    return _norm(s).map(mapping).astype(float)


def _to_num(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def load_features(data_dir):
    return pd.read_csv(os.path.join(data_dir, "features.csv"))


def load_labels(data_dir):
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    return labels.iloc[:, 0].to_numpy()


class FeatureEngineer:
    """Leakage-safe tabular feature builder (fit on train, transform anywhere).

    Produces three consistent views:
      * transform()      -> standardized numeric+one-hot matrix (MLP / LightGBM)
      * transform_cat()  -> numeric + raw string categoricals (CatBoost native)
    """

    def __init__(self):
        self.temperature_median_ = 0.0
        self.ohe_ = None
        self.numeric_cols_ = None
        self.nominal_cols_ = [
            "gender",
            "maritalStatus",
            "occupation",
            "time",
            "coupon",
            "coupon_time",
            "coupon_exp",
            "coupon_dir",
            "coupon_kids",
            "coupon_income",
        ]
        self.feature_names_ = None
        self.mean_ = None
        self.std_ = None

    # ------------------------------------------------------------------ #
    def _engineer(self, df):
        df = df.copy()
        for c in REQUIRED_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan

        X = pd.DataFrame(index=df.index)

        temp = _to_num(df["temperature"])
        X["temperature"] = temp.fillna(self.temperature_median_)
        X["temperature_na"] = temp.isna().astype(np.int8)

        X["time_ord"] = _map_series(df["time"], TIME_MAP).fillna(-1.0)
        X["age_ord"] = _map_series(df["age"], AGE_MAP).fillna(-1.0)
        X["income_ord"] = _map_series(df["income"], INCOME_MAP).fillna(-1.0)
        X["education_ord"] = _map_series(df["education"], EDU_MAP).fillna(-1.0)
        X["expiration_ord"] = _map_series(df["expiration"], EXP_MAP).fillna(-1.0)

        freq_vals = {}
        for c in FREQ_COLS:
            v = _map_series(df[c], FREQ_MAP)
            X[c] = v.fillna(-1.0)
            X[f"{c}_missing"] = v.isna().astype(np.int8)
            freq_vals[c] = v.fillna(0.0)

        X["social_activity"] = freq_vals["Bar"] + freq_vals["CoffeeHouse"]
        X["restaurant_activity"] = (
            freq_vals["RestaurantLessThan20"]
            + freq_vals["Restaurant20To50"]
            + freq_vals["CarryAway"]
        )
        X["total_activity"] = X["social_activity"] + X["restaurant_activity"]

        coupon_lower = _norm(df["coupon"])
        affinity = np.full(len(df), -1.0, dtype=float)
        for key, col in COUPON_FREQ_COL.items():
            mask = (coupon_lower == key).to_numpy()
            if mask.any():
                affinity[mask] = freq_vals[col].to_numpy()[mask]
        X["coupon_affinity"] = affinity
        X["coupon_affinity_missing"] = (affinity < 0).astype(np.int8)

        g5 = _to_num(df["toCoupon_GEQ5min"]).fillna(0.0)
        g15 = _to_num(df["toCoupon_GEQ15min"]).fillna(0.0)
        g25 = _to_num(df["toCoupon_GEQ25min"]).fillna(0.0)
        X["toCoupon_GEQ5min"] = g5
        X["toCoupon_GEQ15min"] = g15
        X["toCoupon_GEQ25min"] = g25
        X["distance_rank"] = g5 + g15 + g25
        X["distance_proxy"] = 5.0 * g5 + 10.0 * g15 + 10.0 * g25

        X["direction_same"] = _to_num(df["direction_same"]).fillna(0.0)
        X["direction_opp"] = _to_num(df["direction_opp"]).fillna(0.0)
        X["is_on_the_way"] = X["direction_same"]

        kids = _to_num(df["has_children"]).fillna(0.0)
        X["has_children"] = kids

        X["affinity_x_onway"] = X["coupon_affinity"] * X["is_on_the_way"]
        X["affinity_x_distance"] = X["coupon_affinity"] * X["distance_rank"]
        X["same_x_distance"] = X["is_on_the_way"] * X["distance_rank"]
        X["evening"] = (X["time_ord"] >= 3.0).astype(np.float32)

        # --- raw string categoricals (fed natively to CatBoost) --------- #
        gender_s = _norm(df["gender"].fillna("missing"))
        marital_s = _norm(df["maritalStatus"].fillna("missing"))
        occ_s = _norm(df["occupation"].fillna("missing"))
        time_s = _norm(df["time"].fillna("missing"))
        exp_s = _norm(df["expiration"].fillna("missing"))
        dir_s = _to_num(df["direction_same"]).fillna(0.0).astype(int).astype(str)
        kids_s = kids.astype(int).astype(str)
        income_s = _norm(df["income"].fillna("missing"))

        X["gender"] = gender_s
        X["maritalStatus"] = marital_s
        X["occupation"] = occ_s
        X["time"] = time_s
        X["coupon"] = coupon_lower
        X["coupon_time"] = coupon_lower + "|" + time_s
        X["coupon_exp"] = coupon_lower + "|" + exp_s
        X["coupon_dir"] = coupon_lower + "|" + dir_s
        X["coupon_kids"] = coupon_lower + "|" + kids_s
        X["coupon_income"] = coupon_lower + "|" + income_s
        return X

    # ------------------------------------------------------------------ #
    def fit(self, df):
        temp = _to_num(df["temperature"])
        self.temperature_median_ = float(temp.median()) if temp.notna().any() else 0.0

        X = self._engineer(df)
        self.numeric_cols_ = [c for c in X.columns if c not in self.nominal_cols_]

        try:
            self.ohe_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float32
            )
        except TypeError:
            self.ohe_ = OneHotEncoder(
                handle_unknown="ignore", sparse=False, dtype=np.float32
            )
        self.ohe_.fit(X[self.nominal_cols_].astype(str))

        ohe_names = list(self.ohe_.get_feature_names_out(self.nominal_cols_))
        self.feature_names_ = list(self.numeric_cols_) + ohe_names

        raw = self._raw_matrix(X)
        self.mean_ = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std < 1e-8] = 1.0
        self.std_ = std
        return self

    def _raw_matrix(self, X):
        ohe_arr = self.ohe_.transform(X[self.nominal_cols_].astype(str))
        return np.hstack(
            [X[self.numeric_cols_].to_numpy(dtype=np.float32), ohe_arr]
        ).astype(np.float32)

    def transform(self, df):
        if self.ohe_ is None:
            raise RuntimeError("FeatureEngineer must be fitted before transform().")
        X = self._engineer(df)
        raw = self._raw_matrix(X)
        return ((raw - self.mean_) / self.std_).astype(np.float32)

    def transform_cat(self, df):
        X = self._engineer(df)
        cols = self.numeric_cols_ + self.nominal_cols_
        out = X[cols].copy()
        for c in self.nominal_cols_:
            out[c] = out[c].astype(str)
        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def save_feature_engineer(fe, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(fe, os.path.join(model_dir, "feature_engineer.joblib"))


def load_feature_engineer(model_dir):
    return joblib.load(os.path.join(model_dir, "feature_engineer.joblib"))


# =========================================================================== #
# STEP 2 : MODEL DESIGN
# =========================================================================== #
class CouponMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.30):
        super().__init__()
        blocks = []
        prev = input_dim
        for h in hidden_dims:
            blocks.append(nn.Linear(prev, h))
            blocks.append(nn.BatchNorm1d(h))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.02, pos_weight=None):
        super().__init__()
        self.smoothing = float(smoothing)
        pw = None if pos_weight is None else torch.tensor([float(pos_weight)])
        self.register_buffer("pos_weight_t", pw if pw is not None else torch.empty(0))
        self._base = nn.BCEWithLogitsLoss(pos_weight=None if pw is None else pw)

    def forward(self, logits, targets):
        targets = targets.float()
        if self.smoothing > 0.0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self._base(logits, targets)


def build_criterion(pos_weight=None, smoothing=0.02):
    return LabelSmoothingBCE(smoothing=smoothing, pos_weight=pos_weight)


def build_optimizer(model, lr=1e-3, weight_decay=1e-4):
    return torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )


def build_scheduler(optimizer, epochs, warmup_frac=0.1):
    warmup = max(1, int(epochs * warmup_frac))
    total = max(epochs, warmup + 1)

    def lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / float(total - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _catboost_params(seed, iterations, eval_set_present):
    kwargs = dict(
        iterations=int(iterations),
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        bootstrap_type="Bernoulli",
        subsample=0.85,
        rsm=0.9,
        random_seed=seed,
        verbose=False,
        thread_count=2,
        allow_writing_files=False,
    )
    if eval_set_present:
        kwargs.update(od_type="Iter", od_wait=100)
    return kwargs


def _fit_catboost(X_tr, y_tr, X_es, y_es, cat_cols, seed=42, iterations=2000):
    has_es = X_es is not None and len(X_es) > 0
    model = CatBoostClassifier(**_catboost_params(seed, iterations, has_es))
    if has_es:
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_es, y_es),
            use_best_model=True,
            cat_features=cat_cols,
        )
    else:
        model.fit(X_tr, y_tr, cat_features=cat_cols)
    return model


def _fit_lightgbm(X_tr, y_tr, X_es, y_es, seed=42, n_estimators=1200):
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=int(n_estimators),
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbose=-1,
    )
    if X_es is not None and len(X_es) > 0:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_es, y_es)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    else:
        model.fit(X_tr, y_tr)
    return model


# =========================================================================== #
# STEP 3 : TRAINING & EVALUATION
# =========================================================================== #
def _set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def _to_tensor(x, device="cpu"):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def _fit_mlp_once(
    X_tr,
    y_tr,
    X_es,
    y_es,
    input_dim,
    seed=42,
    epochs=40,
    batch_size=256,
    lr=2e-3,
    weight_decay=1e-4,
    device="cpu",
):
    _set_seeds(seed)
    mlp = CouponMLP(input_dim=input_dim).to(device)
    criterion = build_criterion(pos_weight=None, smoothing=0.02)
    opt = build_optimizer(mlp, lr=lr, weight_decay=weight_decay)
    sched = build_scheduler(opt, epochs=epochs, warmup_frac=0.1)

    Xtr_t = _to_tensor(X_tr, device)
    ytr_t = torch.as_tensor(np.asarray(y_tr, dtype=np.float32), device=device)
    Xes_t = _to_tensor(X_es, device) if (X_es is not None and len(X_es) > 0) else None

    n = len(Xtr_t)
    best_acc, best_state, patience, bad = -1.0, None, 8, 0
    for _ep in range(epochs):
        mlp.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = criterion(mlp(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 5.0)
            opt.step()
        sched.step()

        if Xes_t is not None:
            mlp.eval()
            with torch.no_grad():
                p = torch.sigmoid(mlp(Xes_t)).cpu().numpy()
            acc = accuracy_score(y_es, (p >= 0.5).astype(int))
            if acc > best_acc + 1e-6:
                best_acc = acc
                best_state = {
                    k: v.detach().clone() for k, v in mlp.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state is not None:
        mlp.load_state_dict(best_state)
    mlp.eval()
    return mlp


@torch.no_grad()
def _mlp_proba(mlp, X, batch_size=1024, device="cpu"):
    mlp.eval()
    Xt = _to_tensor(X, device)
    out = []
    for i in range(0, len(Xt), batch_size):
        out.append(torch.sigmoid(mlp(Xt[i : i + batch_size])).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


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


def train_and_evaluate(data_dir, model_dir, seed=42, n_splits=5):
    _set_seeds(seed)
    os.makedirs(model_dir, exist_ok=True)

    features = load_features(data_dir)
    y = load_labels(data_dir)
    if len(features) != len(y):
        raise ValueError("features / labels length mismatch")

    classes = np.unique(y)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_bin = np.array([label_to_idx[c] for c in y], dtype=np.int64)
    POS_IDX = 1 if len(classes) > 1 else 0
    y_enc = (y_bin == POS_IDX).astype(np.int64)

    # Degenerate single-class guard ------------------------------------- #
    if len(classes) < 2:
        fe = FeatureEngineer().fit(features)
        save_feature_engineer(fe, model_dir)
        joblib.dump(
            {
                "classes": classes.tolist(),
                "pos_idx": 0,
                "weights": [0.0, 0.0, 0.0],
                "threshold": 0.5,
                "input_dim": int(len(fe.feature_names_)),
                "cat_cols": list(fe.nominal_cols_),
            },
            os.path.join(model_dir, "meta.joblib"),
        )
        return 1.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_mlp = np.zeros(len(y_enc), dtype=np.float64)
    oof_cb = np.zeros(len(y_enc), dtype=np.float64)
    oof_gb = np.zeros(len(y_enc), dtype=np.float64)
    cb_iters, gb_iters = [], []

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_enc)), y_enc)):
        X_tr = features.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_enc[tr_idx]
        X_va = features.iloc[va_idx].reset_index(drop=True)
        y_va = y_enc[va_idx]

        # 88 / 12 split inside the fold (was: next(5-fold) => 20% used)
        sub_tr, sub_es = train_test_split(
            np.arange(len(X_tr)),
            test_size=0.12,
            random_state=seed + fold_i,
            stratify=y_tr,
        )

        fe = FeatureEngineer().fit(X_tr.iloc[sub_tr])

        Xa = fe.transform(X_tr.iloc[sub_tr])
        Xe = fe.transform(X_tr.iloc[sub_es])
        Xv = fe.transform(X_va)
        Ca = fe.transform_cat(X_tr.iloc[sub_tr])
        Ce = fe.transform_cat(X_tr.iloc[sub_es])
        Cv = fe.transform_cat(X_va)
        ya, ye = y_tr[sub_tr], y_tr[sub_es]

        cb = _fit_catboost(
            Ca, ya, Ce, ye, fe.nominal_cols_, seed=seed + fold_i, iterations=2000
        )
        oof_cb[va_idx] = cb.predict_proba(Cv)[:, 1]
        bi = int(cb.get_best_iteration() or 800)
        cb_iters.append(max(bi, 100))

        gb = _fit_lightgbm(Xa, ya, Xe, ye, seed=seed + fold_i, n_estimators=1500)
        oof_gb[va_idx] = gb.predict_proba(Xv)[:, 1]
        gbi = getattr(gb, "best_iteration_", 0) or 600
        gb_iters.append(max(int(gbi), 100))

        mlp = _fit_mlp_once(
            Xa,
            ya,
            Xe,
            ye,
            input_dim=Xa.shape[1],
            seed=seed + fold_i,
            epochs=40,
            batch_size=256,
        )
        oof_mlp[va_idx] = _mlp_proba(mlp, Xv)

        a_cb = accuracy_score(y_va, (oof_cb[va_idx] >= 0.5).astype(int))
        a_gb = accuracy_score(y_va, (oof_gb[va_idx] >= 0.5).astype(int))
        a_ml = accuracy_score(y_va, (oof_mlp[va_idx] >= 0.5).astype(int))
        print(
            f"[fold {fold_i}] cb={a_cb:.4f} lgbm={a_gb:.4f} mlp={a_ml:.4f} "
            f"(cb_it={cb_iters[-1]}, gb_it={gb_iters[-1]})"
        )

    print(
        f"OOF singles: catboost="
        f"{accuracy_score(y_enc, (oof_cb >= 0.5).astype(int)):.4f} "
        f"lightgbm={accuracy_score(y_enc, (oof_gb >= 0.5).astype(int)):.4f} "
        f"mlp={accuracy_score(y_enc, (oof_mlp >= 0.5).astype(int)):.4f}"
    )

    weights, thr, oof_acc = _select_blend_and_threshold(oof_mlp, oof_cb, oof_gb, y_enc)
    print(
        f"OOF ensemble weights={weights.tolist()} threshold={thr:.2f} "
        f"accuracy={oof_acc:.4f}"
    )

    # ------------------------------------------------------------------ #
    # FINAL FIT ON 100% OF TRAINING ROWS, multi-seed bagging
    # ------------------------------------------------------------------ #
    fe_final = FeatureEngineer().fit(features)
    X_all = fe_final.transform(features)
    C_all = fe_final.transform_cat(features)
    y_all = y_enc

    n_cb = int(np.clip(np.mean(cb_iters) * 1.15, 200, 2500))
    n_gb = int(np.clip(np.mean(gb_iters) * 1.15, 150, 2500))
    print(f"Final iterations: catboost={n_cb} lightgbm={n_gb}")

    FINAL_SEEDS = [seed, seed + 101, seed + 202]

    cb_models, gb_models, mlp_models = [], [], []
    for s in FINAL_SEEDS:
        cb_models.append(
            _fit_catboost(
                C_all,
                y_all,
                None,
                None,
                fe_final.nominal_cols_,
                seed=s,
                iterations=n_cb,
            )
        )
        gb_models.append(
            _fit_lightgbm(X_all, y_all, None, None, seed=s, n_estimators=n_gb)
        )
        mlp_models.append(
            _fit_mlp_once(
                X_all,
                y_all,
                None,
                None,
                input_dim=X_all.shape[1],
                seed=s,
                epochs=45,
                batch_size=256,
            )
        )

    save_feature_engineer(fe_final, model_dir)
    for i, m in enumerate(cb_models):
        m.save_model(os.path.join(model_dir, f"catboost_{i}.cbm"))
    joblib.dump(gb_models, os.path.join(model_dir, "lightgbm.joblib"))
    torch.save([m.state_dict() for m in mlp_models], os.path.join(model_dir, "mlp.pt"))

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

    print(f"Saved model artifacts to {model_dir}")
    return float(oof_acc)


@torch.no_grad()
def predict_split(data_dir, model_dir, output_path):
    _set_seeds(0)
    meta = joblib.load(os.path.join(model_dir, "meta.joblib"))
    classes = meta["classes"]
    pos_idx = int(meta["pos_idx"])
    w_mlp, w_cb, w_gb = [float(v) for v in meta["weights"]]
    thr = float(meta.get("threshold", 0.5))
    input_dim = int(meta["input_dim"])

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    n_rows = len(features)

    if len(classes) < 2 or (w_mlp + w_cb + w_gb) <= 0.0:
        preds = np.array([classes[pos_idx]] * n_rows, dtype=object)
        out = pd.DataFrame(
            {"row_id": np.arange(n_rows, dtype=np.int64), "prediction": preds}
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        out.to_csv(output_path, index=False)
        return

    fe = load_feature_engineer(model_dir)
    X = fe.transform(features)
    C = fe.transform_cat(features)

    p_cb_list = []
    for i in range(int(meta.get("n_catboost", 1))):
        cb = CatBoostClassifier()
        cb.load_model(os.path.join(model_dir, f"catboost_{i}.cbm"))
        p_cb_list.append(cb.predict_proba(C)[:, 1])
    p_cb = np.mean(np.column_stack(p_cb_list), axis=1)

    gb_models = joblib.load(os.path.join(model_dir, "lightgbm.joblib"))
    if not isinstance(gb_models, list):
        gb_models = [gb_models]
    p_gb = np.mean(
        np.column_stack([g.predict_proba(X)[:, 1] for g in gb_models]), axis=1
    )

    states = torch.load(os.path.join(model_dir, "mlp.pt"), map_location="cpu")
    if isinstance(states, dict):
        states = [states]
    p_mlp_list = []
    for st in states:
        mlp = CouponMLP(input_dim=input_dim)
        mlp.load_state_dict(st)
        mlp.eval()
        p_mlp_list.append(_mlp_proba(mlp, X))
    p_mlp = np.mean(np.column_stack(p_mlp_list), axis=1)

    wsum = max(w_mlp + w_cb + w_gb, 1e-12)
    proba = (w_mlp * p_mlp + w_cb * p_cb + w_gb * p_gb) / wsum
    is_pos = (proba >= thr).astype(int)

    classes_arr = np.array(classes, dtype=object)
    neg_idx = 0 if pos_idx != 0 else (1 if len(classes_arr) > 1 else 0)
    preds = np.where(is_pos == 1, classes_arr[pos_idx], classes_arr[neg_idx])

    out = pd.DataFrame(
        {"row_id": np.arange(n_rows, dtype=np.int64), "prediction": preds}
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} predictions to {output_path}")


# =========================================================================== #
# CLI
# =========================================================================== #
def _build_argparser():
    ap = argparse.ArgumentParser(description="Coupon acceptance - train/predict")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True)
    tr.add_argument("--model", required=True)
    tr.add_argument("--seed", type=int, default=42)

    pr = sub.add_parser("predict")
    pr.add_argument("--data", required=True)
    pr.add_argument("--model", required=True)
    pr.add_argument("--output", required=True)
    return ap


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.cmd == "train":
        score = train_and_evaluate(
            args.data, args.model, seed=getattr(args, "seed", 42)
        )
        print(f"Final Validation Score: {score:.4f}")
    elif args.cmd == "predict":
        predict_split(args.data, args.model, args.output)
