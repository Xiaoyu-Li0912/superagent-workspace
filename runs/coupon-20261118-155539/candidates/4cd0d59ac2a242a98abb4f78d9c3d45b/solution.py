"""
Coupon acceptance prediction (accuracy metric) with an FT-Transformer.

Pipeline
--------
* train : load features+labels (train-only), fit leakage-safe FeatureEngineer,
          5-fold stratified CV + OOF threshold search, refit on all rows,
          persist config.json + model.pt under --model.
* predict : NEW process, load FE state + model read-only, run real forward
            inference per row, threshold, map back to ORIGINAL labels,
            write row_id,prediction (row order authoritative).

Resource limits: 2 CPU threads, 4096 MB, 600 s per process.
"""

import os
import json
import math
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # stay within the 2-CPU-thread budget if the host allows it
    torch.set_num_threads(2)
except Exception:
    pass


# ===========================================================================
# STEP 1 : Data loading + leakage-safe feature engineering
# ===========================================================================
INCOME_MAP = {
    "less than $12500": 0,
    "$12500 - $24999": 1,
    "$25000 - $37499": 2,
    "$37500 - $49999": 3,
    "$50000 - $62499": 4,
    "$62500 - $74999": 5,
    "$75000 - $87499": 6,
    "$87500 - $99999": 7,
    "$100000+": 8,
}

EDUCATION_MAP = {
    "some high school": 0,
    "high school graduate": 1,
    "some college - no degree": 2,
    "associates degree": 3,
    "bachelors degree": 4,
    "graduate degree (masters, doctorate)": 5,
    "graduate degree(masters, doctorate)": 5,
}

AGE_MAP = {
    "below21": 0,
    "21": 1,
    "26": 2,
    "31": 3,
    "36": 4,
    "41": 5,
    "46": 6,
    "50plus": 7,
}

FREQ_MAP = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}

TIME_MAP = {"7am": 7, "10am": 10, "2pm": 14, "6pm": 18, "10pm": 22}

EXPIRATION_MAP = {"2h": 0, "1d": 1}

GENDER_MAP = {"female": 0, "male": 1}

COUPON_TYPE_KEYWORDS = [
    ("carry out", 4),
    ("take away", 4),
    ("restaurant(20-50)", 3),
    ("restaurant(20", 3),
    ("restaurant(<20)", 2),
    ("restaurant<", 2),
    ("coffee", 1),
    ("bar", 0),
]

FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
]

NOMINAL_COLS = ["coupon", "occupation", "maritalStatus"]


def _norm_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.lower()
    out = out.replace({"nan": "__nan__", "none": "__nan__", "": "__nan__"})
    return out


def _safe_int_series(s: pd.Series, default: int = -1) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(default).astype(np.int32)


def load_training_data(data_dir: str):
    feat = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    schema_path = os.path.join(data_dir, "schema.json")
    schema = {}
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema = json.load(f)
    if "label" not in labels.columns:
        cand = [c for c in labels.columns if c not in {"row_id", "id"}]
        if len(cand) != 1:
            raise ValueError(
                f"labels.csv must contain a 'label' column; got {list(labels.columns)}"
            )
        labels = labels.rename(columns={cand[0]: "label"})
    y = labels["label"].reset_index(drop=True)
    return feat.reset_index(drop=True), y, schema


def load_prediction_features(data_dir: str):
    feat = pd.read_csv(os.path.join(data_dir, "features.csv"))
    schema_path = os.path.join(data_dir, "schema.json")
    schema = {}
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema = json.load(f)
    return feat.reset_index(drop=True), schema


class FeatureEngineer:
    VERSION = 1

    def __init__(self):
        self.state_ = {
            "version": self.VERSION,
            "nominal_vocab": {},
            "interaction_vocab": {},
            "freq_medians": {},
            "feature_columns": [],
        }
        self._fitted = False

    @staticmethod
    def _map_ordinal(series: pd.Series, mapping: dict, missing: int = -1) -> pd.Series:
        norm = _norm_series(series)
        return norm.map(lambda v: mapping.get(v, missing)).astype(np.int32)

    @staticmethod
    def _coupon_type(series: pd.Series) -> pd.Series:
        norm = _norm_series(series)

        def _tag(v):
            for kw, tag in COUPON_TYPE_KEYWORDS:
                if v is not None and kw in v:
                    return tag
            return -1

        return norm.map(_tag).astype(np.int32)

    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        if "temperature" in df.columns:
            out["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        else:
            out["temperature"] = np.nan

        out["time_hour"] = (
            self._map_ordinal(df["time"], TIME_MAP) if "time" in df.columns else -1
        )
        out["expiration_ord"] = (
            self._map_ordinal(df["expiration"], EXPIRATION_MAP)
            if "expiration" in df.columns
            else -1
        )

        if "age" in df.columns:
            norm = _norm_series(df["age"])
            age_num = pd.to_numeric(norm, errors="coerce")
            age_ord = norm.map(lambda v: AGE_MAP.get(v, -1)).astype(np.int32)
            age_ord = age_ord.where(age_ord >= 0, age_num.fillna(-1).astype(np.int32))
            out["age_ord"] = age_ord
        else:
            out["age_ord"] = -1

        out["has_children"] = (
            _safe_int_series(df["has_children"]) if "has_children" in df.columns else -1
        )
        out["income_ord"] = (
            self._map_ordinal(df["income"], INCOME_MAP)
            if "income" in df.columns
            else -1
        )
        out["education_ord"] = (
            self._map_ordinal(df["education"], EDUCATION_MAP)
            if "education" in df.columns
            else -1
        )

        freq_ord_cols = []
        miss_cols = []
        for c in FREQ_COLS:
            if c in df.columns:
                ord_c = self._map_ordinal(df[c], FREQ_MAP, missing=-1)
                miss_c = (ord_c < 0).astype(np.int32)
            else:
                ord_c = pd.Series(-1, index=df.index, dtype=np.int32)
                miss_c = pd.Series(1, index=df.index, dtype=np.int32)
            out[f"freq_{c}"] = ord_c
            out[f"freq_{c}_missing"] = miss_c
            freq_ord_cols.append(f"freq_{c}")
            miss_cols.append(f"freq_{c}_missing")

        freq_mat = out[freq_ord_cols].values.astype(np.float32)
        freq_mat_clean = np.where(freq_mat < 0, 0.0, freq_mat)
        out["total_visits"] = freq_mat_clean.sum(axis=1)
        out["bar_cafe_visits"] = freq_mat_clean[:, 0] + freq_mat_clean[:, 1]
        out["restaurant_visits"] = freq_mat_clean[:, 2:].sum(axis=1)
        out["visit_entropy"] = (freq_mat_clean > 0).sum(axis=1).astype(np.int32)
        out["num_freq_missing"] = out[miss_cols].sum(axis=1).astype(np.int32)

        for c in ["toCoupon_GEQ5min", "toCoupon_GEQ15min", "toCoupon_GEQ25min"]:
            out[c] = _safe_int_series(df[c]) if c in df.columns else 0
        out["distance_score"] = (
            out["toCoupon_GEQ5min"]
            + out["toCoupon_GEQ15min"]
            + out["toCoupon_GEQ25min"]
        ).astype(np.int32)
        out["direction_same"] = (
            _safe_int_series(df["direction_same"])
            if "direction_same" in df.columns
            else 0
        )
        out["direction_opp"] = (
            _safe_int_series(df["direction_opp"])
            if "direction_opp" in df.columns
            else 0
        )

        out["gender_enc"] = (
            self._map_ordinal(df["gender"], GENDER_MAP)
            if "gender" in df.columns
            else -1
        )
        out["coupon_type"] = (
            self._coupon_type(df["coupon"]) if "coupon" in df.columns else -1
        )
        return out

    def _make_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        inter = pd.DataFrame(index=df.index)
        coupon = (
            _norm_series(df["coupon"])
            if "coupon" in df.columns
            else pd.Series("__nan__", index=df.index)
        )
        time_ = (
            _norm_series(df["time"])
            if "time" in df.columns
            else pd.Series("__nan__", index=df.index)
        )
        exp_ = (
            _norm_series(df["expiration"])
            if "expiration" in df.columns
            else pd.Series("__nan__", index=df.index)
        )
        inter["coupon_x_time"] = coupon + "|" + time_
        inter["coupon_x_expiration"] = coupon + "|" + exp_
        return inter

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        if df is None or len(df) == 0:
            raise ValueError("FeatureEngineer.fit received an empty dataframe")

        for c in NOMINAL_COLS:
            if c in df.columns:
                vals = _norm_series(df[c]).value_counts().index.tolist()
                self.state_["nominal_vocab"][c] = {v: i for i, v in enumerate(vals)}

        inter = self._make_interactions(df)
        for c in inter.columns:
            vals = inter[c].value_counts().index.tolist()
            self.state_["interaction_vocab"][c] = {v: i for i, v in enumerate(vals)}

        base = self._base_transform(df)
        for c in FREQ_COLS:
            col = f"freq_{c}"
            if col in base.columns:
                valid = base.loc[base[col] >= 0, col]
                self.state_["freq_medians"][c] = (
                    float(valid.median()) if len(valid) else 0.0
                )

        # Mark fitted BEFORE calling transform so the pre-fit guard inside
        # transform() does not fire on this internal, legitimate call.
        self._fitted = True
        self.state_["feature_columns"] = list(self.transform(df).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.transform called before fit")

        out = self._base_transform(df)

        for c in NOMINAL_COLS:
            vocab = self.state_["nominal_vocab"].get(c, {})
            if c in df.columns:
                enc = _norm_series(df[c]).map(lambda v, v_=vocab: v_.get(v, -1))
                out[f"{c}_enc"] = enc.astype(np.int32)
            else:
                out[f"{c}_enc"] = -1

        inter = self._make_interactions(df)
        for c in inter.columns:
            vocab = self.state_["interaction_vocab"].get(c, {})
            out[f"{c}_enc"] = (
                inter[c].map(lambda v, v_=vocab: v_.get(v, -1)).astype(np.int32)
            )

        out = out.replace([np.inf, -np.inf], np.nan)
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        cols = self.state_["feature_columns"]
        if cols:
            for col in cols:
                if col not in out.columns:
                    out[col] = 0
            out = out[cols].copy()
        # Coerce to numeric and remove any lingering NaN/Inf so no NaN can
        # propagate into the tokenizer / continuous-feature path at inference.
        out = out.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def state_dict(self) -> dict:
        return json.loads(json.dumps(self.state_))

    def load_state_dict(self, state: dict):
        self.state_ = state
        self._fitted = True
        return self


def make_stratified_folds(y: pd.Series, n_splits: int = 5, seed: int = 42):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_arr = np.asarray(y)
    return list(skf.split(np.zeros(len(y_arr)), y_arr))


# ===========================================================================
# STEP 2 : FT-Transformer architecture / loss / optimizers
# ===========================================================================
class FeatureTokenizer(nn.Module):
    def __init__(self, cat_cardinalities, n_cont, d_token, dropout=0.1):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_cont = int(n_cont)
        self.d_token = int(d_token)

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(int(c) + 1, self.d_token) for c in cat_cardinalities]
        )

        if self.n_cont > 0:
            self.cont_weight = nn.Parameter(torch.empty(self.n_cont, self.d_token))
            self.cont_bias = nn.Parameter(torch.empty(self.n_cont, self.d_token))
            nn.init.normal_(self.cont_weight, std=1.0 / math.sqrt(self.d_token))
            nn.init.zeros_(self.cont_bias)
        else:
            self.cont_weight = None
            self.cont_bias = None

        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_cat, x_num):
        tokens = []
        if self.n_cat > 0:
            for i, emb in enumerate(self.cat_embeddings):
                col = x_cat[:, i].long()
                col = col.clamp(min=-1) + 1
                tokens.append(emb(col).unsqueeze(1))
        if self.n_cont > 0:
            cont = x_num.unsqueeze(-1) * self.cont_weight.unsqueeze(
                0
            ) + self.cont_bias.unsqueeze(0)
            tokens.append(cont)
        if not tokens:
            raise ValueError("FeatureTokenizer received no features.")
        out = torch.cat(tokens, dim=1)
        return self.dropout(out)


class FTTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        n_cont,
        d_token=64,
        n_layers=3,
        n_heads=8,
        ffn_mult=2.0,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        residual_dropout=0.1,
        head_dropout=0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            cat_cardinalities, n_cont, d_token, dropout=residual_dropout
        )
        n_field_tokens = len(cat_cardinalities) + int(n_cont)
        self.n_tokens = 1 + n_field_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        self.field_pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_token))
        nn.init.normal_(self.field_pos, std=0.02)

        ffn_hidden = int(round(d_token * ffn_mult))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ffn_hidden,
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_token)

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x_cat, x_num):
        tok = self.tokenizer(x_cat, x_num)
        b = tok.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        seq = seq + self.field_pos
        enc = self.encoder(seq)
        cls_out = self.final_norm(enc[:, 0])
        logit = self.head(cls_out).squeeze(-1)
        return logit

    @torch.no_grad()
    def predict_proba(self, x_cat, x_num):
        self.eval()
        return torch.sigmoid(self.forward(x_cat, x_num))


class LabelSmoothingBCEWithLogits(nn.Module):
    def __init__(self, smoothing=0.0, pos_weight=None):
        super().__init__()
        self.smoothing = float(smoothing)
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32)
            )
        else:
            self.register_buffer("pos_weight", torch.tensor(1.0))

    def forward(self, logits, targets):
        targets = targets.float()
        if self.smoothing > 0.0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


def build_model(cat_cardinalities, n_cont, **kwargs):
    return FTTransformer(cat_cardinalities, n_cont, **kwargs)


def build_criterion(smoothing=0.03, pos_weight=None):
    return LabelSmoothingBCEWithLogits(smoothing=smoothing, pos_weight=pos_weight)


def build_optimizer(model, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )


def build_scheduler(
    optimizer, epochs, steps_per_epoch, warmup_frac=0.06, min_lr_ratio=0.01
):
    total_steps = max(1, int(epochs) * int(steps_per_epoch))
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


GRAD_CLIP_NORM = 1.0


def mixup_batch(x_cat, x_num, y, alpha=0.2, rng=None):
    if rng is None:
        rng = torch.Generator()
        rng.manual_seed(0)
    b = x_num.size(0) if x_num is not None else x_cat.size(0)
    lam = (
        float(torch.distributions.Beta(alpha, alpha).sample().item())
        if alpha > 0
        else 1.0
    )
    perm = torch.randperm(b, generator=rng)
    if x_num is not None:
        x_num = lam * x_num + (1.0 - lam) * x_num[perm]
    if x_cat is not None:
        mask = (torch.rand(b, generator=rng) < (1.0 - lam)).unsqueeze(1)
        x_cat = torch.where(mask, x_cat[perm], x_cat)
    return x_cat, x_num, y, y[perm], lam


# ===========================================================================
# STEP 3 : Training / evaluation + predict pipeline
# ===========================================================================
def _split_feature_columns(feature_columns):
    cat_cols = []
    for c in NOMINAL_COLS:
        name = f"{c}_enc"
        if name in feature_columns:
            cat_cols.append(name)
    for c in ("coupon_x_time", "coupon_x_expiration"):
        name = f"{c}_enc"
        if name in feature_columns:
            cat_cols.append(name)
    if "gender_enc" in feature_columns:
        cat_cols.append("gender_enc")
    cat_set = set(cat_cols)
    cont_cols = [c for c in feature_columns if c not in cat_set]
    return cat_cols, cont_cols


def _compute_cards(X, cat_cols):
    cards = []
    for c in cat_cols:
        v = X[c].values
        try:
            m = int(np.nanmax(v)) if len(v) else -1
        except (TypeError, ValueError):
            m = -1
        cards.append(max(1, m + 1))
    return cards


def _fit_cont_stats(X, cont_cols):
    if not cont_cols:
        z = np.zeros(0, np.float32)
        return z, z.copy(), np.ones(0, np.float32)
    arr = X[cont_cols].astype(np.float64).values
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    filled = np.where(np.isnan(arr), med[None, :], arr)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return med.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _make_tensors(X, cat_cols, cont_cols, med, mean, std):
    if cat_cols:
        xc = X[cat_cols].astype(np.int64).values
    else:
        xc = np.zeros((len(X), 0), dtype=np.int64)
    if cont_cols:
        xn = X[cont_cols].astype(np.float64).values
        xn = np.where(np.isnan(xn), med[None, :], xn)
        xn = (xn - mean[None, :]) / std[None, :]
    else:
        xn = np.zeros((len(X), 0), dtype=np.float64)
    return torch.from_numpy(xc), torch.from_numpy(xn.astype(np.float32))


def _accuracy(prob, y, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype(np.int64)
    return float((pred == np.asarray(y).astype(np.int64)).mean())


def _encode_labels(y):
    arr = np.asarray(y.values if hasattr(y, "values") else y)
    uniq = list(pd.unique(arr))
    try:
        uniq = sorted(uniq)
    except TypeError:
        pass
    mapping = {v: i for i, v in enumerate(uniq)}
    y_enc = np.array([mapping[v] for v in arr], dtype=np.int64)
    label_values = []
    for v in uniq:
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        label_values.append(v)
    return label_values, y_enc


def _predict_proba(model, X, cat_cols, cont_cols, med, mean, std):
    model.eval()
    xc, xn = _make_tensors(X, cat_cols, cont_cols, med, mean, std)
    out = []
    with torch.no_grad():
        bs = 512
        for i in range(0, len(xc), bs):
            out.append(
                model.predict_proba(xc[i : i + bs], xn[i : i + bs]).cpu().numpy()
            )
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def _train_one(
    X_tr,
    y_tr,
    X_va,
    y_va,
    cat_cols,
    cont_cols,
    cards,
    med,
    mean,
    std,
    seed=42,
    epochs=30,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    patience=6,
    model_kwargs=None,
    verbose=True,
):
    model_kwargs = dict(model_kwargs or {})
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(cards, len(cont_cols), **model_kwargs)
    criterion = build_criterion(smoothing=0.03)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)

    n = len(X_tr)
    steps = max(1, int(math.ceil(n / batch_size)))
    scheduler = build_scheduler(optimizer, epochs, steps)

    xc_all, xn_all = _make_tensors(X_tr, cat_cols, cont_cols, med, mean, std)
    yt = torch.from_numpy(np.asarray(y_tr, dtype=np.float32))

    if X_va is not None:
        yva = np.asarray(y_va).astype(np.int64)

    best = {"acc": -1.0, "state": None, "epoch": 0}
    rng = np.random.RandomState(seed)
    mix_gen = torch.Generator()
    mix_gen.manual_seed(seed + 7)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total, nbatches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:
                continue
            xc = xc_all[idx]
            xn = xn_all[idx]
            yb = yt[idx]
            if rng.rand() < 0.5 and len(idx) >= 4:
                xcm, xnm, ya, yb2, lam = mixup_batch(xc, xn, yb, alpha=0.2, rng=mix_gen)
                logits = model(xcm, xnm)
                loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(
                    logits, yb2
                )
            else:
                logits = model(xc, xn)
                loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            total += float(loss.item())
            nbatches += 1

        avg_loss = total / max(1, nbatches)
        msg = f"  epoch {ep:02d} loss {avg_loss:.4f}"

        if X_va is not None:
            prob = _predict_proba(model, X_va, cat_cols, cont_cols, med, mean, std)
            acc = _accuracy(prob, yva, 0.5)
            msg += f" val_acc {acc:.4f}"
            if acc > best["acc"]:
                best = {
                    "acc": acc,
                    "state": copy.deepcopy(model.state_dict()),
                    "epoch": ep,
                }
            elif ep - best["epoch"] >= patience:
                if verbose:
                    print(msg)
                break
        if verbose:
            print(msg)

    if X_va is not None and best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best


def train_pipeline(data_dir, model_dir, seed=42, n_splits=5):
    print(f"[train] loading training data from {data_dir}")
    df, y, schema = load_training_data(data_dir)
    print(f"[train] rows={len(df)} raw_features={df.shape[1]}")

    label_values, y_enc = _encode_labels(y)
    print(f"[train] labels={label_values} pos_rate={float(y_enc.mean()):.4f}")

    fe = FeatureEngineer().fit(df)
    X = fe.transform(df)
    feature_columns = list(X.columns)
    cat_cols, cont_cols = _split_feature_columns(feature_columns)
    cards = _compute_cards(X, cat_cols)
    med, mean, std = _fit_cont_stats(X, cont_cols)
    print(
        f"[train] design matrix={X.shape} | cat={len(cat_cols)} cont={len(cont_cols)} "
        f"| cards={cards}"
    )

    folds = make_stratified_folds(pd.Series(y_enc), n_splits=n_splits, seed=seed)
    oof = np.zeros(len(y_enc), dtype=np.float64)
    best_epochs = []
    fold_accs = []

    for k, (tr, va) in enumerate(folds):
        print(f"[fold {k}] train={len(tr)} val={len(va)}")
        model, best = _train_one(
            X.iloc[tr],
            y_enc[tr],
            X.iloc[va],
            y_enc[va],
            cat_cols,
            cont_cols,
            cards,
            med,
            mean,
            std,
            seed=seed + k,
            epochs=30,
            batch_size=128,
            lr=1e-3,
            weight_decay=1e-4,
            patience=6,
        )
        prob = _predict_proba(model, X.iloc[va], cat_cols, cont_cols, med, mean, std)
        oof[va] = prob
        best_epochs.append(max(1, int(best["epoch"])))
        fold_accs.append(float(best["acc"]))
        print(f"[fold {k}] best_epoch={best['epoch']} val_acc={best['acc']:.4f}")

    oof_acc = _accuracy(oof, y_enc, 0.5)

    best_thr, best_thr_acc = 0.5, oof_acc
    for thr in np.arange(0.30, 0.701, 0.02):
        acc = _accuracy(oof, y_enc, float(thr))
        if acc > best_thr_acc + 1e-9:
            best_thr, best_thr_acc = float(thr), acc

    print(
        f"[cv] oof_acc@0.5={oof_acc:.4f} | best_thr={best_thr:.2f} "
        f"oof_acc={best_thr_acc:.4f} | fold_acc mean={np.mean(fold_accs):.4f} "
        f"std={np.std(fold_accs):.4f}"
    )

    final_epochs = int(round(float(np.median(best_epochs)))) + 1
    print(f"[final] refitting on all rows for epochs={final_epochs}")
    final_model, _ = _train_one(
        X,
        y_enc,
        None,
        None,
        cat_cols,
        cont_cols,
        cards,
        med,
        mean,
        std,
        seed=seed,
        epochs=final_epochs,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        patience=final_epochs + 1,
    )

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(final_model.state_dict(), model_path)

    config = {
        "fe_state": fe.state_dict(),
        "feature_columns": feature_columns,
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "cards": [int(c) for c in cards],
        "cont_med": [float(v) for v in med],
        "cont_mean": [float(v) for v in mean],
        "cont_std": [float(v) for v in std],
        "threshold": float(best_thr),
        "label_values": label_values,
        "model_kwargs": {},
        "cv": {
            "oof_acc_at_0_5": float(oof_acc),
            "best_threshold": float(best_thr),
            "oof_acc_best_thr": float(best_thr_acc),
            "fold_accs": [float(a) for a in fold_accs],
            "final_epochs": int(final_epochs),
        },
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)

    print(f"[train] saved model.pt + config.json to {model_dir}")
    print(f"Internal OOF accuracy (train-only, thr={best_thr:.2f}): {best_thr_acc:.4f}")
    return best_thr_acc


def predict_pipeline(data_dir, model_dir, output_path):
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    fe = FeatureEngineer()
    fe.load_state_dict(cfg["fe_state"])

    df, schema = load_prediction_features(data_dir)
    X = fe.transform(df)

    feature_columns = cfg["feature_columns"]
    for c in feature_columns:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_columns]

    cat_cols = cfg["cat_cols"]
    cont_cols = cfg["cont_cols"]
    med = np.asarray(cfg["cont_med"], dtype=np.float32)
    mean = np.asarray(cfg["cont_mean"], dtype=np.float32)
    std = np.asarray(cfg["cont_std"], dtype=np.float32)

    model = build_model(cfg["cards"], len(cont_cols), **cfg.get("model_kwargs", {}))
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    prob = _predict_proba(model, X, cat_cols, cont_cols, med, mean, std)
    threshold = float(cfg.get("threshold", 0.5))
    class_idx = (prob >= threshold).astype(np.int64)

    label_values = cfg["label_values"]
    preds = [label_values[i] for i in class_idx]

    out = pd.DataFrame(
        {
            "row_id": np.arange(len(df), dtype=np.int64),
            "prediction": preds,
        }
    )
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[predict] wrote {len(out)} rows -> {output_path}")
    return out


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Coupon acceptance: train / predict")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="fit pipeline and persist to --model")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)

    p_pred = sub.add_parser("predict", help="load --model and write predictions")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        score = train_pipeline(args.data, args.model)
        print(f"Final Validation Score: {score}")
    else:
        predict_pipeline(args.data, args.model, args.output)


if __name__ == "__main__":
    main()
