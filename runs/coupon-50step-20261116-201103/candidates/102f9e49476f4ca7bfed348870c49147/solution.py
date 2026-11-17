import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import lightgbm as lgb

ORDINAL_MAPS = {
    "age": {
        "below21": 0,
        "21": 1,
        "26": 2,
        "31": 3,
        "36": 4,
        "41": 5,
        "46": 6,
        "50plus": 7,
    },
    "income": {
        "Less than $12500": 0,
        "$12500 - $24999": 1,
        "$25000 - $37499": 2,
        "$37500 - $49999": 3,
        "$50000 - $62499": 4,
        "$62500 - $74999": 5,
        "$75000 - $87499": 6,
        "$87500 - $99999": 7,
        "$100000 or More": 8,
    },
    "education": {
        "Some High School": 0,
        "High School Graduate": 1,
        "Some college - no degree": 2,
        "Associates degree": 3,
        "Bachelors degree": 4,
        "Graduate degree (Masters, Doctorate)": 5,
    },
    "Bar": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "CoffeeHouse": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "CarryAway": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "RestaurantLessThan20": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "Restaurant20To50": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
}

FREQ_COMBO_COLS = [
    ("coupon", "occupation"),
    ("coupon", "income"),
    ("coupon", "age"),
    ("coupon", "maritalStatus"),
    ("coupon", "education"),
    ("coupon", "has_children"),
    ("coupon", "time"),
    ("coupon", "expiration"),
]


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def get_feature_columns(schema, df):
    cols = schema.get("feature_columns")
    if cols:
        cols = [c for c in cols if c in df.columns]
    else:
        cols = [c for c in df.columns if c != "row_id"]
    return cols


def _to_str_series(s):
    return s.where(s.notna(), "__NA__").astype(str)


def build_features(df, feature_columns, cat_vocab=None, freq_maps=None, fit=False):
    X = df[feature_columns].copy()

    for col, mapping in ORDINAL_MAPS.items():
        if col in X.columns:
            X[col + "_ord"] = _to_str_series(X[col]).map(mapping).fillna(-1).astype(int)

    def _s(col):
        if col in X.columns:
            return _to_str_series(X[col])
        return pd.Series(["__NA__"] * len(X), index=X.index)

    X["coupon_x_time"] = _s("coupon") + "|" + _s("time")
    X["coupon_x_expiration"] = _s("coupon") + "|" + _s("expiration")
    X["children_x_marital"] = _s("has_children") + "|" + _s("maritalStatus")

    if "toCoupon_GEQ15min" in X.columns and "toCoupon_GEQ25min" in X.columns:
        g15 = (
            pd.to_numeric(X["toCoupon_GEQ15min"], errors="coerce").fillna(0).astype(int)
        )
        g25 = (
            pd.to_numeric(X["toCoupon_GEQ25min"], errors="coerce").fillna(0).astype(int)
        )
        X["distance_bucket"] = g15 + g25
    if "direction_same" in X.columns and "direction_opp" in X.columns:
        same = pd.to_numeric(X["direction_same"], errors="coerce").fillna(0).astype(int)
        opp = pd.to_numeric(X["direction_opp"], errors="coerce").fillna(0).astype(int)
        X["direction"] = same - opp
    if "time" in X.columns:
        X["is_weekend"] = (
            _to_str_series(X["time"]).isin(["10AM", "2PM", "6PM"]).astype(int)
        )

    X["coupon_x_direction"] = (
        _s("coupon") + "|" + _s("direction_same") + "_" + _s("direction_opp")
    )
    X["coupon_x_children"] = _s("coupon") + "|" + _s("has_children")
    if "income_ord" in X.columns:
        X["coupon_x_income_ord"] = _s("coupon") + "|" + X["income_ord"].astype(str)
    if "distance_bucket" in X.columns:
        X["coupon_x_distance_bucket"] = (
            _s("coupon") + "|" + X["distance_bucket"].astype(str)
        )

    act_cols = [
        "Bar_ord",
        "CoffeeHouse_ord",
        "CarryAway_ord",
        "RestaurantLessThan20_ord",
        "Restaurant20To50_ord",
    ]
    present = [c for c in act_cols if c in X.columns]
    if present:
        X["user_activity_score"] = X[present].clip(lower=0).sum(axis=1)

    if fit:
        freq_maps = {}
        for a, b in FREQ_COMBO_COLS:
            if a in X.columns and b in X.columns:
                key = f"freq_{a}_{b}"
                combo = _s(a) + "|" + _s(b)
                vc = combo.value_counts(normalize=True)
                freq_maps[key] = vc.to_dict()
                X[key] = combo.map(vc).fillna(0.0).astype(float)
    else:
        for a, b in FREQ_COMBO_COLS:
            if a in X.columns and b in X.columns:
                key = f"freq_{a}_{b}"
                combo = _s(a) + "|" + _s(b)
                m = freq_maps.get(key, {}) if freq_maps else {}
                X[key] = combo.map(m).fillna(0.0).astype(float)

    cat_cols = []
    for c in X.columns:
        if (
            X[c].dtype == object
            or str(X[c].dtype).startswith("str")
            or X[c].dtype.name == "category"
        ):
            X[c] = _to_str_series(X[c])
            cat_cols.append(c)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-1)

    if fit:
        cat_vocab = {}
        for c in cat_cols:
            uniq = sorted(X[c].unique().tolist())
            cat_vocab[c] = {v: i for i, v in enumerate(uniq)}
    for c in cat_cols:
        vocab = cat_vocab.get(c, {}) if cat_vocab else {}
        X[c] = X[c].map(vocab).fillna(-1).astype(int)

    return X, cat_cols, cat_vocab, freq_maps


def train(args):
    data_dir = args.data
    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)

    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    label_col = "label" if "label" in labels.columns else labels.columns[0]
    y_raw = labels[label_col].values

    feature_columns = get_feature_columns(schema, features)
    X, cat_cols, cat_vocab, freq_maps = build_features(
        features, feature_columns, fit=True
    )

    classes = np.unique(y_raw)
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y = np.array([class_to_idx[str(v)] for v in y_raw], dtype=int)

    n_classes = len(classes)
    cat_indices = [X.columns.get_loc(c) for c in cat_cols]
    feature_names = list(X.columns)

    def make_cat(seed=42):
        return CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            l2_leaf_reg=3.0,
            bagging_temperature=0.5,
            random_strength=1.0,
            verbose=False,
            thread_count=2,
            random_seed=seed,
            allow_writing_files=False,
            auto_class_weights="Balanced" if n_classes == 2 else None,
        )

    def make_lgb(seed=42):
        return lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=30,
            reg_lambda=1.0,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=2,
            random_state=seed,
            verbose=-1,
            class_weight="balanced" if n_classes == 2 else None,
        )

    # OOF stacking with CatBoost + LGBM only (drop HGB for speed/memory)
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    n_models = 2
    oof = np.zeros((len(X), n_models * n_classes), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]

        cb = make_cat(seed=42 + fold)
        cb.fit(X_tr, y_tr, cat_features=cat_indices)
        oof[va_idx, 0:n_classes] = cb.predict_proba(X_va)

        X_tr_l = X_tr.copy()
        X_va_l = X_va.copy()
        for c in cat_cols:
            X_tr_l[c] = X_tr_l[c].astype("category")
            X_va_l[c] = X_va_l[c].astype("category")
        lg = make_lgb(seed=42 + fold)
        lg.fit(X_tr_l, y_tr, categorical_feature=cat_cols)
        oof[va_idx, n_classes : 2 * n_classes] = lg.predict_proba(X_va_l)

    meta = LogisticRegression(max_iter=1000, C=1.0)
    meta.fit(oof, y)

    # Final base models trained on FULL training data (2 seeds each)
    cb_models, lg_models = [], []
    for seed in (1000, 1001):
        cb = make_cat(seed=seed)
        cb.fit(X, y, cat_features=cat_indices)
        cb_models.append(cb)

        X_l = X.copy()
        for c in cat_cols:
            X_l[c] = X_l[c].astype("category")
        lg = make_lgb(seed=seed)
        lg.fit(X_l, y, categorical_feature=cat_cols)
        lg_models.append(lg)

    for i, cb in enumerate(cb_models):
        cb.save_model(os.path.join(model_dir, f"cb_{i}.cbm"))
    with open(os.path.join(model_dir, "lgb_models.pkl"), "wb") as f:
        pickle.dump(lg_models, f)
    with open(os.path.join(model_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    meta_info = {
        "feature_columns": feature_columns,
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "cat_vocab": cat_vocab,
        "freq_maps": {k: v for k, v in (freq_maps or {}).items()},
        "idx_to_class": {
            str(k): (v.item() if hasattr(v, "item") else v)
            for k, v in idx_to_class.items()
        },
        "n_classes": int(n_classes),
        "n_cb": len(cb_models),
        "n_lgb": len(lg_models),
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta_info, f)


def predict(args):
    data_dir = args.data
    model_dir = args.model
    output_path = args.output

    with open(os.path.join(model_dir, "meta.json"), "r") as f:
        meta_info = json.load(f)

    feature_columns = meta_info["feature_columns"]
    cat_cols = meta_info["cat_cols"]
    cat_vocab = meta_info["cat_vocab"]
    freq_maps = meta_info.get("freq_maps", {})
    idx_to_class = {int(k): v for k, v in meta_info["idx_to_class"].items()}
    n_cb = meta_info["n_cb"]
    n_lgb = meta_info["n_lgb"]
    n_classes = meta_info["n_classes"]

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X, _, _, _ = build_features(
        features, feature_columns, cat_vocab=cat_vocab, freq_maps=freq_maps, fit=False
    )

    X_l = X.copy()
    for c in cat_cols:
        X_l[c] = X_l[c].astype("category")

    p_cb = np.zeros((len(X), n_classes))
    for i in range(n_cb):
        cb = CatBoostClassifier()
        cb.load_model(os.path.join(model_dir, f"cb_{i}.cbm"))
        p_cb += cb.predict_proba(X)
    p_cb /= n_cb

    with open(os.path.join(model_dir, "lgb_models.pkl"), "rb") as f:
        lg_models = pickle.load(f)
    p_lg = np.zeros((len(X), n_classes))
    for lg in lg_models:
        p_lg += lg.predict_proba(X_l)
    p_lg /= n_lgb

    with open(os.path.join(model_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    stacked = np.hstack([p_cb, p_lg])
    final_proba = meta.predict_proba(stacked)
    preds = np.argmax(final_proba, axis=1)

    original = [idx_to_class[int(p)] for p in preds]

    out_df = pd.DataFrame({"row_id": np.arange(len(features)), "prediction": original})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--model", required=True)
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)
    p_pred.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
