```python
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
        out["dist_any"] = ((g15 + g25) > 0).astype(float)

    if "direction_same" in df.columns and "direction_opp" in df.columns:
        ds = pd.to_numeric(df["direction_same"], errors="coerce").fillna(0)
        do = pd.to_numeric(df["direction_opp"], errors="coerce").fillna(0)
        out["dir_same"] = ds
        out["dir_opp"] = do

    if "has_children" in df.columns:
        out["has_children_num"] = pd.to_numeric(
            df["has_children"], errors="coerce"
        ).fillna(0)

    if "expiration" in df.columns:
        out["exp_is_2h"] = (df["expiration"].astype(str) == "2h").astype(float)

    if "coupon" in df.columns and "time" in df.columns:
        out["coupon_time"] = df["coupon"].astype(str) + "|" + df["time"].astype(str)
    if "coupon" in df.columns and "expiration" in df.columns:
        out["coupon_exp"] = (
            df["coupon"].astype(str) + "|" + df["expiration"].astype(str)
        )
    if "coupon" in df.columns and "income" in df.columns:
        out["coupon_income"] = df["coupon"].astype(str) + "|" + df["income"].astype(str)
    if "coupon" in df.columns and "has_children" in df.columns:
        out["coupon_children"] = (
            df["coupon"].astype(str)
            + "|"
            + pd.to_numeric(df["has_children"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
        )
    if "coupon" in df.columns and "age" in df.columns:
        out["coupon_age"] = df["coupon"].astype(str) + "|" + df["age"].astype(str)
    if "coupon" in df.columns and "direction_opp" in df.columns:
        do = pd.to_numeric(df["direction_opp"], errors="coerce").fillna(0).astype(int)
        out["coupon_diropp"] = df["coupon"].astype(str) + "|" + do.astype(str)
    if "time" in df.columns and "expiration" in df.columns:
        out["time_exp"] = df["time"].astype(str) + "|" + df["expiration"].astype(str)

    if all(c in df.columns for c in ["coupon", "expiration", "time"]):
        out["coupon_exp_time"] = (
            df["coupon"].astype(str)
            + "|"
            + df["expiration"].astype(str)
            + "|"
            + df["time"].astype(str)
        )
    if all(
        c in df.columns
        for c in ["coupon", "direction_opp", "toCoupon_GEQ15min", "toCoupon_GEQ25min"]
    ):
        g15 = (
            pd.to_numeric(df["toCoupon_GEQ15min"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        g25 = (
            pd.to_numeric(df["toCoupon_GEQ25min"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        do = pd.to_numeric(df["direction_opp"], errors="coerce").fillna(0).astype(int)
        out["coupon_dir_dist"] = (
            df["coupon"].astype(str)
            + "|"
            + do.astype(str)
            + "|"
            + (g15 + g25).astype(str)
        )

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
        ord_cols = [c + "_ord" for c in freq_cols if c + "_ord" in out.columns]
        if ord_cols:
            out["freq_mean"] = out[ord_cols].mean(axis=1)
            out["freq_sum"] = out[ord_cols].sum(axis=1)
            out["freq_any"] = (out[ord_cols].fillna(0) > 0).any(axis=1).astype(float)

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
            Xc[c] = pd.Categorical(Xc[c].astype(str), categories=cat_vocab.get(c, []))
    return Xc


def _fit_catboost(X, y, cat_idx, n_classes):
    from catboost import CatBoostClassifier

    cb = CatBoostClassifier(
        iterations=1200,
        depth=7,
        learning_rate=0.04,
        l2_leaf_reg=4.0,
        random_seed=42,
        verbose=False,
        thread_count=2,
        allow_writing_files=False,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        one_hot_max_size=10,
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
    )
    cb.fit(_prep_catboost(X, [X.columns[i] for i in cat_idx]), y, cat_features=cat_idx)
    return cb


def _fit_lgbm(X, y, all_cat_cols, cat_vocab):
    import lightgbm as lgb

    lg = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.04,
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
    return lg


def _predict_proba(model, name, X, all_cat_cols, cat_vocab):
    if name == "catboost":
        return model.predict_proba(_prep_catboost(X, all_cat_cols))
    if name == "lgbm":
        return model.predict_proba(_prep_lgbm(X, all_cat_cols, cat_vocab))
    raise ValueError(name)


def _align_probs(p, n_classes, n_rows):
    if p is None:
        return np.full((n_rows, n_classes), 1.0 / n_classes)
    if p.shape[1] != n_classes:
        p2 = np.zeros((p.shape[0], n_classes))
        for j in range(min(p.shape[1], n_classes)):
            p2[:, j] = p[:, j]
        p = p2
    return p


def _log_probs(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def _search_weights(prob_list, y, n_classes):
    n_models = len(prob_list)
    if n_models == 1:
        return np.array([1.0])
    best_w = np.ones(n_models) / n_models
    best_ll = float("inf")
    if n_models == 2:
        grid = [np.array([a, 1 - a]) for a in np.linspace(0, 1, 21)]
    elif n_models == 3:
        grid = []
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1 - a, 11):
                c = 1 - a - b
                if c >= -1e-9:
                    grid.append(np.array([a, b, max(c, 0.0)]))
    else:
        grid = [np.ones(n_models) / n_models]
    logs = [_log_probs(p) for p in prob_list]
    for w in grid:
        w = w / w.sum()
        combined = sum(w[i] * logs[i] for i in range(n_models))
        combined = combined - combined.max(axis=1, keepdims=True)
        e = np.exp(combined)
        p = e / e.sum(axis=1, keepdims=True)
        ll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))
        if ll < best_ll:
            best_ll = ll
            best_w = w
    return best_w


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
        c
        for c in [
            "coupon_time",
            "coupon_exp",
            "coupon_income",
            "coupon_children",
            "coupon_age",
            "coupon_diropp",
            "time_exp",
            "coupon_exp_time",
            "coupon_dir_dist",
        ]
        if c in X.columns
    ]
    all_cat_cols = [c for c in cat_cols if c in X.columns] + eng_cat

    cat_vocab = {}
    for c in all_cat_cols:
        if c in X.columns:
            cat_vocab[c] = sorted(X[c].astype(str).unique().tolist())

    cat_idx = [X.columns.get_loc(c) for c in all_cat_cols if c in X.columns]

    available = []
    try:
        import catboost  # noqa

        available.append("catboost")
    except Exception:
        pass
    try:
        import lightgbm  # noqa

        available.append("lgbm")
    except Exception:
        pass

    if not available:
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
                    "base_models": [],
                    "stacker_models": [],
                    "weights": [],
                    "stacker": False,
                },
                f,
            )
        return

    from sklearn.model_selection import StratifiedKFold

    n_splits = 4
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = {name: np.zeros((len(X), n_classes), dtype=np.float64) for name in available}
    oof_valid = {name: np.zeros(len(X), dtype=bool) for name in available}

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_tr = y[tr_idx]

        fold_vocab = {}
        for c in all_cat_cols:
            if c in X_tr.columns:
                fold_vocab[c] = sorted(X_tr[c].astype(str).unique().tolist())

        for name in available:
            try:
                if name == "catboost":
                    m = _fit_catboost(X_tr, y_tr, cat_idx, n_classes)
                elif name == "lgbm":
                    m = _fit_lgbm(X_tr, y_tr, all_cat_cols, fold_vocab)
                else:
                    continue
                p = _predict_proba(m, name, X_va, all_cat_cols, fold_vocab)
                p = _align_probs(p, n_classes, len(X_va))
                oof[name][va_idx] = p
                oof_valid[name][va_idx] = True
            except Exception as e:
                print(f"fold {fold_i} {name} failed: {e}")

    stacker_models = [name for name in available if oof_valid[name].all()]
    if not stacker_models:
        stacker_models = list(available)

    prob_list = [oof[name] for name in stacker_models]
    weights = _search_weights(prob_list, y, n_classes)

    base_models = {}
    for name in available:
        try:
            if name == "catboost":
                m = _fit_catboost(X, y, cat_idx, n_classes)
                m.save_model(os.path.join(model_dir, "catboost.cbm"))
            elif name == "lgbm":
                m = _fit_lgbm(X, y, all_cat_cols, cat_vocab)
                with open(os.path.join(model_dir, "lgbm.pkl"), "wb") as f:
                    pickle.dump(m, f)
            base_models[name] = True
        except Exception as e:
            print(f"final fit {name} failed: {e}")

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
                "stacker_models": stacker_models,
                "weights": [float(w) for w in weights],
                "stacker": False,
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
    base_names = meta.get("base_models", [])
    stacker_models = meta.get("stacker_models", base_names)
    weights = meta.get("weights", [])

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X, _ = build_features(
        features, feature_cols, cat_cols, num_cols, freq_maps=freq_maps, fit=False
    )

    n_rows = len(features)

    if not base_names:
        fb_path = os.path.join(model_dir, "fallback.json")
        if os.path.exists(fb_path):
            with open(fb_path, "r") as f:
                majority = json.load(f)["majority_idx"]
            preds = np.full(n_rows, majority, dtype=np.int64)
        else:
            preds = np.zeros(n_rows, dtype=np.int64)
        labels_out = [idx_to_label[int(p)] for p in preds]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame({"row_id": np.arange(n_rows), "prediction": labels_out}).to_csv(
            output_path, index=False
        )
        return

    loaded = {}
    for name in base_names:
        try:
            if name == "catboost":
                from catboost import CatBoostClassifier

                m = CatBoostClassifier()
                m.load_model(os.path.join(model_dir, "catboost.cbm"))
                loaded[name] = m
            elif name == "lgbm":
                with open(os.path.join(model_dir, "lgbm.pkl"), "rb") as f:
                    loaded[name] = pickle.load(f)
        except Exception as e:
            print(f"load {name} failed: {e}")

    if not loaded:
        fb_path = os.path.join(model_dir, "fallback.json")
        with open(fb_path, "r") as f:
            majority = json.load(f)["majority_idx"]
        preds = np.full(n_rows, majority, dtype=np.int64)
    else:
        prob_map = {}
        for name in base_names:
            if name not in loaded:
                prob_map[name] = np.full((n_rows, n_classes), 1.0 / n_classes)
                continue
            try:
                p = _predict_proba(loaded[name], name, X, all_cat_cols, cat_vocab)
                prob_map[name] = _align_probs(p, n_classes, n_rows)
            except Exception as e:
                print(f"predict {name} failed: {e}")
                prob_map[name] = np.full((n_rows, n_classes), 1.0 / n_classes)

        use_names = [n for n in stacker_models if n in prob_map]
        if use_names:
            if len(weights) == len(stacker_models):
                w_map = {n: weights[i] for i, n in enumerate(stacker_models)}
            else:
                w_map = {n: 1.0 / len(use_names) for n in use_names}
            w_arr = np.array([w_map.get(n, 0.0) for n in use_names], dtype=np.float64)
            if w_arr.sum() <= 0:
                w_arr = np.ones(len(use_names)) / len(use_names)
            else:
                w_arr = w_arr / w_arr.sum()
            logs = [_log_probs(prob_map[n]) for n in use_names]
            combined = sum(w_arr[i] * logs[i] for i in range(len(use_names)))
            preds = np.argmax(combined, axis=1).astype(np.int64)
        else:
            avg = np.mean([prob_map[n] for n in base_names if n in prob_map], axis=0)
            preds = np.argmax(avg, axis=1).astype(np.int64)

    labels_out = [idx_to_label[int(p)] for p in preds]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df = pd.DataFrame({"row_id": np.arange(n_rows), "prediction": labels_out})
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
```