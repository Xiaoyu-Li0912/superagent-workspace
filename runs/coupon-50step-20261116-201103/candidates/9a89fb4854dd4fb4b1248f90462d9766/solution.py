import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd


def load_schema(data_dir):
    with open(os.path.join(data_dir, "schema.json"), "r") as f:
        return json.load(f)


def get_feature_columns(schema, df):
    cols = schema.get("feature_columns")
    if cols:
        return [c for c in cols if c in df.columns]
    return [c for c in df.columns if c != "row_id"]


FREQ_ORDER = ["never", "less1", "1~3", "4~8", "gt8"]
AGE_ORDER = ["below21", "21", "26", "31", "36", "41", "46", "50plus"]
INCOME_ORDER = [
    "Less than $12500",
    "$12500 - $24999",
    "$25000 - $37499",
    "$37500 - $49999",
    "$50000 - $62499",
    "$62500 - $74999",
    "$75000 - $87499",
    "$87500 - $99999",
    "$100000 or More",
]
TIME_ORDER = ["7AM", "10AM", "2PM", "6PM", "10PM"]
EDU_ORDER = [
    "Some High School",
    "High School Graduate",
    "Some college - no degree",
    "Associates degree",
    "Bachelors degree",
    "Graduate degree (Masters or Doctorate)",
]


def safe_ordinal(series, order):
    mapping = {v: i for i, v in enumerate(order)}
    return series.astype(str).map(mapping).astype(float)


def build_features(df, feature_cols, cat_cols, num_cols, freq_maps=None, fit=False):
    out = pd.DataFrame(index=df.index)

    for c in feature_cols:
        if c in cat_cols:
            out[c] = df[c].astype("object").where(df[c].notna(), "__NA__").astype(str)
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    if "age" in df.columns:
        out["age_ord"] = safe_ordinal(df["age"], AGE_ORDER)
    if "income" in df.columns:
        out["income_ord"] = safe_ordinal(df["income"], INCOME_ORDER)
    if "time" in df.columns:
        out["time_ord"] = safe_ordinal(df["time"], TIME_ORDER)
    if "education" in df.columns:
        out["edu_ord"] = safe_ordinal(df["education"], EDU_ORDER)
    for fcol in [
        "Bar",
        "CoffeeHouse",
        "CarryAway",
        "RestaurantLessThan20",
        "Restaurant20To50",
    ]:
        if fcol in df.columns:
            out[fcol + "_ord"] = safe_ordinal(df[fcol], FREQ_ORDER)

    if all(
        c in df.columns
        for c in ["toCoupon_GEQ5min", "toCoupon_GEQ15min", "toCoupon_GEQ25min"]
    ):
        g5 = pd.to_numeric(df["toCoupon_GEQ5min"], errors="coerce").fillna(0)
        g15 = pd.to_numeric(df["toCoupon_GEQ15min"], errors="coerce").fillna(0)
        g25 = pd.to_numeric(df["toCoupon_GEQ25min"], errors="coerce").fillna(0)
        out["dist_bucket"] = g5 + g15 + g25
        out["dist_15_25"] = g15 - g25

    if "direction_same" in df.columns and "direction_opp" in df.columns:
        ds = pd.to_numeric(df["direction_same"], errors="coerce").fillna(0)
        do = pd.to_numeric(df["direction_opp"], errors="coerce").fillna(0)
        out["dir_same"] = ds
        out["dir_opp"] = do

    if "coupon" in df.columns and "time" in df.columns:
        out["coupon_time"] = df["coupon"].astype(str) + "|" + df["time"].astype(str)
    if "coupon" in df.columns and "expiration" in df.columns:
        out["coupon_exp"] = (
            df["coupon"].astype(str) + "|" + df["expiration"].astype(str)
        )
    if "coupon" in df.columns and "income" in df.columns:
        out["coupon_income"] = df["coupon"].astype(str) + "|" + df["income"].astype(str)

    freq_cols = [
        c
        for c in [
            "Bar",
            "CoffeeHouse",
            "CarryAway",
            "RestaurantLessThan20",
            "Restaurant20To50",
        ]
        if c in df.columns
    ]
    if freq_cols:
        never_count = (df[freq_cols].astype(str) == "never").sum(axis=1)
        out["never_count"] = never_count.astype(float)
        out["freq_mean"] = out[[c + "_ord" for c in freq_cols]].mean(axis=1)

    if "occupation" in df.columns:
        if fit:
            freq_maps = {
                "occupation": df["occupation"]
                .astype(str)
                .value_counts(normalize=True)
                .to_dict()
            }
        occ_map = (freq_maps or {}).get("occupation", {})
        out["occupation_freq"] = df["occupation"].astype(str).map(occ_map).fillna(0.0)

    return out, freq_maps


def _prep_catboost(Xdf, all_cat_cols):
    Xc = Xdf.copy()
    for c in all_cat_cols:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype(str)
    return Xc


def _prep_lgbm(Xdf, all_cat_cols, cat_vocab):
    Xc = Xdf.copy()
    for c in all_cat_cols:
        if c in Xc.columns:
            Xc[c] = pd.Categorical(
                Xc[c].astype(str), categories=cat_vocab.get(c, [])
            )
    return Xc


def _prep_hgb(Xdf, all_cat_cols, cat_vocab):
    Xc = Xdf.copy()
    for c in all_cat_cols:
        if c in Xc.columns:
            mapping = {v: i for i, v in enumerate(cat_vocab.get(c, []))}
            Xc[c] = Xc[c].astype(str).map(mapping).fillna(-1).astype(float)
    return Xc


def train(args):
    data_dir = args.data
    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)

    schema = load_schema(data_dir)
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    label_col = "label" if "label" in labels.columns else labels.columns[0]
    y_raw = labels[label_col].values

    feature_cols = get_feature_columns(schema, features)
    dtypes = schema.get("dtypes", {})
    cat_cols, num_cols = [], []
    for c in feature_cols:
        dt = str(dtypes.get(c, "")).lower()
        if dt in ("object", "str", "string", "category") or features[c].dtype == object:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    X, freq_maps = build_features(features, feature_cols, cat_cols, num_cols, fit=True)

    classes = np.unique(y_raw)
    label_to_idx = {str(v): i for i, v in enumerate(classes)}
    idx_to_label = {i: v for v, i in label_to_idx.items()}
    y = np.array([label_to_idx[str(v)] for v in y_raw], dtype=np.int64)
    n_classes = len(classes)

    eng_cat = [
        c for c in ["coupon_time", "coupon_exp", "coupon_income"] if c in X.columns
    ]
    all_cat_cols = [c for c in cat_cols if c in X.columns] + eng_cat

    cat_vocab = {}
    for c in all_cat_cols:
        if c in X.columns:
            cat_vocab[c] = sorted(X[c].astype(str).unique().tolist())

    cat_idx = [X.columns.get_loc(c) for c in all_cat_cols if c in X.columns]

    base_models = {}

    # CatBoost
    try:
        from catboost import CatBoostClassifier

        cb = CatBoostClassifier(
            iterations=800,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
            thread_count=2,
            allow_writing_files=False,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
        )
        cb.fit(_prep_catboost(X, all_cat_cols), y, cat_features=cat_idx)
        cb.save_model(os.path.join(model_dir, "catboost.cbm"))
        base_models["catboost"] = True
    except Exception as e:
        print("catboost failed:", e)

    # LightGBM
    try:
        import lightgbm as lgb

        lg = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2,
            verbose=-1,
        )
        lg.fit(_prep_lgbm(X, all_cat_cols, cat_vocab), y)
        with open(os.path.join(model_dir, "lgbm.pkl"), "wb") as f:
            pickle.dump(lg, f)
        base_models["lgbm"] = True
    except Exception as e:
        print("lgbm failed:", e)

    # HistGradientBoosting
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        hg = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.06,
            max_depth=None,
            l2_regularization=1.0,
            random_state=42,
        )
        hg.fit(_prep_hgb(X, all_cat_cols, cat_vocab), y)
        with open(os.path.join(model_dir, "hgb.pkl"), "wb") as f:
            pickle.dump(hg, f)
        base_models["hgb"] = True
    except Exception as e:
        print("hgb failed:", e)

    # Always persist a majority-class fallback so prediction is never empty.
    from collections import Counter

    majority = Counter(y.tolist()).most_common(1)[0][0]
    with open(os.path.join(model_dir, "fallback.json"), "w") as f:
        json.dump({"majority_idx": int(majority)}, f)

    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "cat_cols": cat_cols,
                "num_cols": num_cols,
                "all_cat_cols": all_cat_cols,
                "eng_cat": eng_cat,
                "cat_vocab": cat_vocab,
                "idx_to_label": {
                    str(k): (v.item() if hasattr(v, "item") else v)
                    for k, v in idx_to_label.items()
                },
                "freq_maps": freq_maps,
                "n_classes": n_classes,
                "base_models": list(base_models.keys()),
            },
            f,
        )


def predict(args):
    data_dir = args.data
    model_dir = args.model
    output_path = args.output

    with open(os.path.join(model_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    all_cat_cols = meta["all_cat_cols"]
    cat_vocab = meta.get("cat_vocab", {})
    idx_to_label = {int(k): v for k, v in meta["idx_to_label"].items()}
    freq_maps = meta.get("freq_maps", {})
    n_classes = meta["n_classes"]
    base_names = meta["base_models"]

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X, _ = build_features(
        features, feature_cols, cat_cols, num_cols, freq_maps=freq_maps, fit=False
    )

    probs_sum = np.zeros((len(X), n_classes), dtype=np.float64)
    n_used = 0

    if "catboost" in base_names:
        try:
            from catboost import CatBoostClassifier

            cb = CatBoostClassifier()
            cb.load_model(os.path.join(model_dir, "catboost.cbm"))
            p = cb.predict_proba(_prep_catboost(X, all_cat_cols))
            probs_sum += p
            n_used += 1
        except Exception as e:
            print("catboost predict failed:", e)

    if "lgbm" in base_names:
        try:
            with open(os.path.join(model_dir, "lgbm.pkl"), "rb") as f:
                lg = pickle.load(f)
            p = lg.predict_proba(_prep_lgbm(X, all_cat_cols, cat_vocab))
            probs_sum += p
            n_used += 1
        except Exception as e:
            print("lgbm predict failed:", e)

    if "hgb" in base_names:
        try:
            with open(os.path.join(model_dir, "hgb.pkl"), "rb") as f:
                hg = pickle.load(f)
            p = hg.predict_proba(_prep_hgb(X, all_cat_cols, cat_vocab))
            probs_sum += p
            n_used += 1
        except Exception as e:
            print("hgb predict failed:", e)

    if n_used == 0:
        fb_path = os.path.join(model_dir, "fallback.json")
        if os.path.exists(fb_path):
            with open(fb_path, "r") as f:
                majority = json.load(f)["majority_idx"]
            preds = np.full(len(X), majority, dtype=np.int64)
        else:
            preds = np.zeros(len(X), dtype=np.int64)
    else:
        avg = probs_sum / n_used
        preds = np.argmax(avg, axis=1)

    labels_out = [idx_to_label[int(p)] for p in preds]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df = pd.DataFrame(
        {"row_id": np.arange(len(features)), "prediction": labels_out}
    )
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