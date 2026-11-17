import argparse
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

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

    freq_ord_cols = [c + "_ord" for c in FREQ_COLS if c + "_ord" in df.columns]
    if freq_ord_cols:
        df["total_freq"] = df[freq_ord_cols].sum(axis=1, min_count=1)

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

    if "coupon" in df.columns:
        cp = df["coupon"].astype(str)
        if "income_ord" in df.columns:
            df["coupon_x_income"] = cp + "|" + df["income_ord"].astype(str)
        if "age_ord" in df.columns:
            df["coupon_x_age"] = cp + "|" + df["age_ord"].astype(str)
        if "has_children" in df.columns:
            df["coupon_x_children"] = cp + "|" + df["has_children"].astype(str)
        if "occupation" in df.columns:
            df["coupon_x_occ"] = cp + "|" + df["occupation"].astype(str)
        if "gender" in df.columns:
            df["coupon_x_gender"] = cp + "|" + df["gender"].astype(str)
        if "maritalStatus" in df.columns:
            df["coupon_x_marital"] = cp + "|" + df["maritalStatus"].astype(str)
        if "direction_opp" in df.columns:
            df["coupon_x_dir_opp"] = cp + "|" + df["direction_opp"].astype(str)
        if "toCoupon_GEQ15min" in df.columns:
            df["coupon_x_g15"] = cp + "|" + df["toCoupon_GEQ15min"].astype(str)
        if "toCoupon_GEQ25min" in df.columns:
            df["coupon_x_g25"] = cp + "|" + df["toCoupon_GEQ25min"].astype(str)

    if "time" in df.columns and "expiration" in df.columns:
        df["time_x_exp"] = df["time"].astype(str) + "|" + df["expiration"].astype(str)

    return df


EXTRA_COLS = [c + "_ord" for c in ["income", "age", "education"] + FREQ_COLS] + [
    "total_freq",
    "distance_bucket",
    "is_weekend",
    "is_evening",
    "is_lunch",
    "coupon_x_time",
    "coupon_x_exp",
    "child_marital",
    "coupon_x_dir",
    "coupon_x_income",
    "coupon_x_age",
    "coupon_x_children",
    "coupon_x_occ",
    "coupon_x_gender",
    "coupon_x_marital",
    "coupon_x_dir_opp",
    "coupon_x_g15",
    "coupon_x_g25",
    "time_x_exp",
]


def build_matrix(df, feature_columns, extra_cols, final_columns=None):
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

    cb = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss" if is_binary else "MultiClass",
        verbose=False,
        thread_count=2,
        random_seed=SEED,
        allow_writing_files=False,
        bagging_temperature=0.5,
    )
    tr_pool = Pool(X, y, cat_features=cat_indices, weight=sample_w)
    cb.fit(tr_pool)
    cb.save_model(os.path.join(model_dir, "model.cbm"))

    meta = {
        "feature_columns": feature_columns,
        "extra_cols": EXTRA_COLS,
        "final_columns": final_columns,
        "cat_cols": cat_cols,
        "cat_indices": cat_indices,
        "idx_to_class": {
            str(k): (v.item() if hasattr(v, "item") else v)
            for k, v in idx_to_class.items()
        },
        "n_classes": int(n_classes),
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
    cat_indices = meta.get("cat_indices", [])
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X, _ = build_matrix(features, feature_columns, extra_cols, final_columns)

    cb = CatBoostClassifier()
    cb.load_model(os.path.join(model_dir, "model.cbm"))
    test_pool = Pool(X, cat_features=cat_indices)
    proba = cb.predict_proba(test_pool)
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