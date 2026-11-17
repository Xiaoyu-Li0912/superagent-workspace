import argparse
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.model_selection import StratifiedKFold

SEED = 42

ORDINAL_MAPS = {
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
    "education": {
        "Some High School": 0,
        "High School Graduate": 1,
        "Some college - no degree": 2,
        "Associates degree": 3,
        "Bachelors degree": 4,
        "Graduate degree (Masters, Doctorate)": 5,
    },
    "freq": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
}
FREQ_COLS = [
    "Bar",
    "CoffeeHouse",
    "CarryAway",
    "RestaurantLessThan20",
    "Restaurant20To50",
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


def engineer(df):
    df = df.copy()
    for col, mp in [
        ("income", ORDINAL_MAPS["income"]),
        ("age", ORDINAL_MAPS["age"]),
        ("education", ORDINAL_MAPS["education"]),
    ]:
        if col in df.columns:
            df[col + "_ord"] = df[col].astype(str).map(mp)
    for c in FREQ_COLS:
        if c in df.columns:
            df[c + "_ord"] = df[c].astype(str).map(ORDINAL_MAPS["freq"])

    if all(c in df.columns for c in ["toCoupon_GEQ15min", "toCoupon_GEQ25min"]):
        g15 = pd.to_numeric(df["toCoupon_GEQ15min"], errors="coerce").fillna(0)
        g25 = pd.to_numeric(df["toCoupon_GEQ25min"], errors="coerce").fillna(0)
        df["distance_bucket"] = (g15.astype(int) + g25.astype(int)).astype(str)

    if "time" in df.columns:
        t = df["time"].astype(str)
        df["is_weekend"] = t.isin(["Sat", "Sun"]).astype(int)
        df["is_evening"] = t.isin(["6PM", "7PM", "10PM"]).astype(int)
        df["is_lunch"] = t.isin(["11AM", "2PM"]).astype(int)

    if "coupon" in df.columns and "time" in df.columns:
        df["coupon_x_time"] = df["coupon"].astype(str) + "|" + df["time"].astype(str)
    if "coupon" in df.columns and "expiration" in df.columns:
        df["coupon_x_exp"] = (
            df["coupon"].astype(str) + "|" + df["expiration"].astype(str)
        )
    if "has_children" in df.columns and "maritalStatus" in df.columns:
        df["child_marital"] = (
            df["has_children"].astype(str) + "|" + df["maritalStatus"].astype(str)
        )
    if "coupon" in df.columns and "direction_same" in df.columns:
        df["coupon_x_dir"] = (
            df["coupon"].astype(str) + "|" + df["direction_same"].astype(str)
        )
    return df


EXTRA_COLS = [c + "_ord" for c in ["income", "age", "education"] + FREQ_COLS] + [
    "distance_bucket",
    "is_weekend",
    "is_evening",
    "is_lunch",
    "coupon_x_time",
    "coupon_x_exp",
    "child_marital",
    "coupon_x_dir",
]


def build_matrix(df, feature_columns, extra_cols, final_columns=None):
    """Build the feature matrix. If final_columns is given, align exactly to it."""
    df = engineer(df)
    cols = list(feature_columns)
    if extra_cols:
        for c in extra_cols:
            if c in df.columns and c not in cols:
                cols.append(c)

    if final_columns is not None:
        for c in final_columns:
            if c not in df.columns:
                df[c] = np.nan
        cols = list(final_columns)

    X = df[cols].copy()
    cat_cols = []
    for c in X.columns:
        if (
            X[c].dtype == object
            or str(X[c].dtype).startswith("str")
            or X[c].dtype.name == "category"
        ):
            X[c] = X[c].astype(str).fillna("__NA__")
            cat_cols.append(c)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, cat_cols


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
    X, cat_cols = build_matrix(features, feature_columns, EXTRA_COLS)
    final_columns = list(X.columns)

    classes = np.unique(y_raw)
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y = np.array([class_to_idx[str(v)] for v in y_raw], dtype=int)

    cat_indices = [X.columns.get_loc(c) for c in cat_cols]
    n_classes = len(classes)
    is_binary = n_classes == 2

    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * counts)
    sample_w = weights[y]

    cat_categories = {}
    for c in cat_cols:
        cat_categories[c] = sorted(X[c].astype(str).unique().tolist())

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    cat_models = []
    lgb_models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = sample_w[tr_idx]

        cb = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="Logloss" if is_binary else "MultiClass",
            verbose=False,
            thread_count=2,
            random_seed=SEED + fold,
            allow_writing_files=False,
            early_stopping_rounds=100,
            od_type="Iter",
        )
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices, weight=w_tr)
        va_pool = Pool(X_va, y_va, cat_features=cat_indices)
        cb.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        cat_models.append(cb)

        if HAS_LGB:
            X_tr_l = X_tr.copy()
            X_va_l = X_va.copy()
            for c in cat_cols:
                cats = cat_categories[c]
                X_tr_l[c] = pd.Categorical(X_tr_l[c].astype(str), categories=cats)
                X_va_l[c] = pd.Categorical(X_va_l[c].astype(str), categories=cats)
            lgb_params = dict(
                objective="binary" if is_binary else "multiclass",
                num_class=1 if is_binary else n_classes,
                learning_rate=0.03,
                num_leaves=31,
                min_data_in_leaf=20,
                feature_fraction=0.85,
                bagging_fraction=0.85,
                bagging_freq=5,
                lambda_l2=1.0,
                verbose=-1,
                num_threads=2,
                seed=SEED + fold,
            )
            dtrain = lgb.Dataset(
                X_tr_l, label=y_tr, weight=w_tr, categorical_feature=cat_cols
            )
            dval = lgb.Dataset(X_va_l, label=y_va, reference=dtrain)
            lgbm = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            lgb_models.append(lgbm)

    for i, m in enumerate(cat_models):
        m.save_model(os.path.join(model_dir, f"cat_{i}.cbm"))
    for i, m in enumerate(lgb_models):
        m.save_model(os.path.join(model_dir, f"lgb_{i}.txt"))

    meta = {
        "feature_columns": feature_columns,
        "extra_cols": EXTRA_COLS,
        "final_columns": final_columns,
        "cat_cols": cat_cols,
        "cat_indices": cat_indices,
        "cat_categories": cat_categories,
        "idx_to_class": {
            str(k): (v.item() if hasattr(v, "item") else v)
            for k, v in idx_to_class.items()
        },
        "n_classes": int(n_classes),
        "n_cat_models": len(cat_models),
        "n_lgb_models": len(lgb_models),
        "has_lgb": HAS_LGB,
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def predict(args):
    data_dir = args.data
    model_dir = args.model
    output_path = args.output

    with open(os.path.join(model_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    feature_columns = meta["feature_columns"]
    extra_cols = meta.get("extra_cols", [])
    final_columns = meta.get("final_columns")
    cat_cols = meta["cat_cols"]
    cat_indices = meta.get("cat_indices")
    cat_categories = meta.get("cat_categories", {})
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    n_classes = meta["n_classes"]
    is_binary = n_classes == 2

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X, _ = build_matrix(features, feature_columns, extra_cols, final_columns)

    if cat_indices is None:
        cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

    proba_sum = np.zeros((len(X), n_classes), dtype=float)
    n_models = 0

    for i in range(meta["n_cat_models"]):
        cb = CatBoostClassifier()
        cb.load_model(os.path.join(model_dir, f"cat_{i}.cbm"))
        p = cb.predict_proba(X)
        proba_sum += p
        n_models += 1

    if meta.get("has_lgb") and meta.get("n_lgb_models", 0) > 0:
        X_l = X.copy()
        for c in cat_cols:
            cats = cat_categories.get(c, sorted(X_l[c].astype(str).unique().tolist()))
            X_l[c] = pd.Categorical(X_l[c].astype(str), categories=cats)
        for i in range(meta["n_lgb_models"]):
            lgbm = lgb.Booster(model_file=os.path.join(model_dir, f"lgb_{i}.txt"))
            p = lgbm.predict(X_l)
            if is_binary:
                p2 = np.column_stack([1 - p, p])
                proba_sum += p2
            else:
                proba_sum += p
            n_models += 1

    proba = proba_sum / max(n_models, 1)
    preds = np.argmax(proba, axis=1)

    original = [idx_to_class[int(p)] for p in preds]

    out_df = pd.DataFrame({"row_id": np.arange(len(features)), "prediction": original})
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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