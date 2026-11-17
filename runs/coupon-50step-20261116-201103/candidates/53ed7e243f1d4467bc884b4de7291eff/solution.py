import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
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

FREQ_COMBOS = [
    ("coupon", "occupation"),
    ("coupon", "income"),
    ("coupon", "time"),
    ("coupon", "direction_same", "direction_opp"),
    ("coupon", "has_children"),
    ("occupation", "income"),
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
        for combo in FREQ_COMBOS:
            key = "__".join(combo)
            parts = [_s(c) for c in combo]
            combo_series = parts[0]
            for p in parts[1:]:
                combo_series = combo_series + "|" + p
            vc = combo_series.value_counts(normalize=True)
            freq_maps[key] = vc.to_dict()
            X["freq_" + key] = combo_series.map(vc).fillna(0.0).astype(float)
    else:
        for combo in FREQ_COMBOS:
            key = "__".join(combo)
            parts = [_s(c) for c in combo]
            combo_series = parts[0]
            for p in parts[1:]:
                combo_series = combo_series + "|" + p
            m = freq_maps.get(key, {}) if freq_maps else {}
            X["freq_" + key] = combo_series.map(m).fillna(0.0).astype(float)

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


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


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

    classes = np.unique(y_raw)
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y = np.array([class_to_idx[str(v)] for v in y_raw], dtype=int)
    n_classes = len(classes)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_cb = np.zeros((len(features), n_classes))
    oof_lg = np.zeros((len(features), n_classes))
    oof_hg = np.zeros((len(features), n_classes))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(features, y)):
        df_tr = features.iloc[tr_idx].reset_index(drop=True)
        df_va = features.iloc[va_idx].reset_index(drop=True)
        y_tr = y[tr_idx]

        X_tr, cat_cols, cat_vocab, freq_maps = build_features(
            df_tr, feature_columns, fit=True
        )
        X_va, _, _, _ = build_features(
            df_va, feature_columns, cat_vocab=cat_vocab, freq_maps=freq_maps, fit=False
        )

        cat_indices = [X_tr.columns.get_loc(c) for c in cat_cols]

        cb = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            l2_leaf_reg=3.0,
            bagging_temperature=0.5,
            verbose=False,
            thread_count=2,
            random_seed=1000 + fold,
            allow_writing_files=False,
            auto_class_weights="Balanced" if n_classes == 2 else None,
        )
        cb.fit(X_tr, y_tr, cat_features=cat_indices)
        oof_cb[va_idx] = cb.predict_proba(X_va)

        lgb_cat_vocab = {c: sorted(X_tr[c].unique().tolist()) for c in cat_cols}
        X_tr_l = X_tr.copy()
        X_va_l = X_va.copy()
        for c in cat_cols:
            X_tr_l[c] = pd.Categorical(X_tr_l[c], categories=lgb_cat_vocab[c])
            X_va_l[c] = pd.Categorical(X_va_l[c], categories=lgb_cat_vocab[c])

        lg = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=30,
            reg_lambda=1.0,
            subsample=0.9,
            colsample_bytree=0.8,
            n_jobs=2,
            random_state=1000 + fold,
            verbose=-1,
            class_weight="balanced" if n_classes == 2 else None,
        )
        lg.fit(X_tr_l, y_tr, categorical_feature=cat_cols)
        oof_lg[va_idx] = lg.predict_proba(X_va_l)

        hg = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=1000 + fold,
            categorical_features=cat_indices,
        )
        hg.fit(X_tr, y_tr)
        oof_hg[va_idx] = hg.predict_proba(X_va)

    oof_stack = np.hstack([oof_cb, oof_lg, oof_hg])
    stacker = LogisticRegression(
        C=1.0, max_iter=1000, multi_class="multinomial", solver="lbfgs"
    )
    stacker.fit(oof_stack, y)

    # Refit full pipeline on ALL training rows for persistence
    X_full, cat_cols, cat_vocab, freq_maps = build_features(
        features, feature_columns, fit=True
    )
    cat_indices = [X_full.columns.get_loc(c) for c in cat_cols]
    feature_names = list(X_full.columns)
    lgb_cat_vocab = {c: sorted(X_full[c].unique().tolist()) for c in cat_cols}

    cb_full = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
        l2_leaf_reg=3.0,
        bagging_temperature=0.5,
        verbose=False,
        thread_count=2,
        random_seed=42,
        allow_writing_files=False,
        auto_class_weights="Balanced" if n_classes == 2 else None,
    )
    cb_full.fit(X_full, y, cat_features=cat_indices)
    cb_full.save_model(os.path.join(model_dir, "cb_full.cbm"))

    X_full_l = X_full.copy()
    for c in cat_cols:
        X_full_l[c] = pd.Categorical(X_full_l[c], categories=lgb_cat_vocab[c])
    lg_full = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=30,
        reg_lambda=1.0,
        subsample=0.9,
        colsample_bytree=0.8,
        n_jobs=2,
        random_state=42,
        verbose=-1,
        class_weight="balanced" if n_classes == 2 else None,
    )
    lg_full.fit(X_full_l, y, categorical_feature=cat_cols)
    with open(os.path.join(model_dir, "lgb_full.pkl"), "wb") as f:
        pickle.dump(lg_full, f)

    hg_full = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
        categorical_features=cat_indices,
    )
    hg_full.fit(X_full, y)
    with open(os.path.join(model_dir, "hg_full.pkl"), "wb") as f:
        pickle.dump(hg_full, f)

    np.savez(
        os.path.join(model_dir, "stacker.npz"),
        coef=stacker.coef_,
        intercept=stacker.intercept_,
    )

    meta_info = {
        "feature_columns": feature_columns,
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "cat_vocab": cat_vocab,
        "lgb_cat_vocab": lgb_cat_vocab,
        "freq_maps": freq_maps,
        "idx_to_class": {
            str(k): (v.item() if hasattr(v, "item") else v)
            for k, v in idx_to_class.items()
        },
        "n_classes": int(n_classes),
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
    lgb_cat_vocab = meta_info["lgb_cat_vocab"]
    freq_maps = meta_info["freq_maps"]
    idx_to_class = {int(k): v for k, v in meta_info["idx_to_class"].items()}
    n_classes = meta_info["n_classes"]

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X, _, _, _ = build_features(
        features, feature_columns, cat_vocab=cat_vocab, freq_maps=freq_maps, fit=False
    )

    X_l = X.copy()
    for c in cat_cols:
        X_l[c] = pd.Categorical(X_l[c], categories=lgb_cat_vocab[c])

    cb = CatBoostClassifier()
    cb.load_model(os.path.join(model_dir, "cb_full.cbm"))
    p_cb = cb.predict_proba(X)

    with open(os.path.join(model_dir, "lgb_full.pkl"), "rb") as f:
        lg = pickle.load(f)
    p_lg = lg.predict_proba(X_l)

    with open(os.path.join(model_dir, "hg_full.pkl"), "rb") as f:
        hg = pickle.load(f)
    p_hg = hg.predict_proba(X)

    stack_in = np.hstack([p_cb, p_lg, p_hg])
    st = np.load(os.path.join(model_dir, "stacker.npz"))
    coef, intercept = st["coef"], st["intercept"]
    if n_classes == 2 and coef.shape[0] == 1:
        z = stack_in @ coef[0] + intercept[0]
        p1 = 1.0 / (1.0 + np.exp(-z))
        final_proba = np.stack([1 - p1, p1], axis=1)
    else:
        z = stack_in @ coef.T + intercept
        final_proba = _softmax(z)

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