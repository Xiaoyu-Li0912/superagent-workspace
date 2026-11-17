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

    if "coupon" in df.columns and "time" in df.columns:
        out["coupon_time"] = df["coupon"].astype(str) + "|" + df["time"].astype(str)
    if "coupon" in df.columns and "expiration" in df.columns:
        out["coupon_exp"] = (
            df["coupon"].astype(str) + "|" + df["expiration"].astype(str)
        )
    if "coupon" in df.columns and "income" in df.columns:
        out["coupon_income"] = df["coupon"].astype(str) + "|" + df["income"].astype(str)

    # New interaction features
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
            Xc[c] = pd.Categorical(Xc[c].astype(str), categories=cat_vocab.get(c, []))
    return Xc


def _prep_hgb(Xdf, all_cat_cols, cat_vocab):
    Xc = Xdf.copy()
    for c in all_cat_cols:
        if c in Xc.columns:
            mapping = {v: i for i, v in enumerate(cat_vocab.get(c, []))}
            Xc[c] = Xc[c].astype(str).map(mapping).fillna(-1).astype(float)
    return Xc


def _prep_xgb(Xdf, all_cat_cols, cat_vocab):
    Xc = Xdf.copy()
    for c in all_cat_cols:
        if c in Xc.columns:
            mapping = {v: i for i, v in enumerate(cat_vocab.get(c, []))}
            Xc[c] = Xc[c].astype(str).map(mapping).fillna(-1).astype(float)
    return Xc


def _fit_catboost(X, y, cat_idx, n_classes):
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
    cb.fit(_prep_catboost(X, [X.columns[i] for i in cat_idx]), y, cat_features=cat_idx)
    return cb


def _fit_lgbm(X, y, all_cat_cols, cat_vocab):
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
    return lg


def _fit_hgb(X, y, all_cat_cols, cat_vocab):
    from sklearn.ensemble import HistGradientBoostingClassifier

    hg = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.06,
        max_depth=None,
        l2_regularization=1.0,
        random_state=42,
    )
    hg.fit(_prep_hgb(X, all_cat_cols, cat_vocab), y)
    return hg


def _fit_xgb(X, y, all_cat_cols, cat_vocab, n_classes):
    import xgboost as xgb

    Xp = _prep_xgb(X, all_cat_cols, cat_vocab)
    if n_classes > 2:
        xg = xgb.XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2,
            tree_method="hist",
            verbosity=0,
            objective="multi:softprob",
            num_class=n_classes,
        )
    else:
        xg = xgb.XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2,
            tree_method="hist",
            verbosity=0,
            eval_metric="logloss",
        )
    xg.fit(Xp, y)
    return xg


def _predict_proba(model, name, X, all_cat_cols, cat_vocab):
    if name == "catboost":
        return model.predict_proba(_prep_catboost(X, all_cat_cols))
    if name == "lgbm":
        return model.predict_proba(_prep_lgbm(X, all_cat_cols, cat_vocab))
    if name == "hgb":
        return model.predict_proba(_prep_hgb(X, all_cat_cols, cat_vocab))
    if name == "xgb":
        return model.predict_proba(_prep_xgb(X, all_cat_cols, cat_vocab))
    raise ValueError(name)


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
        ]
        if c in X.columns
    ]
    all_cat_cols = [c for c in cat_cols if c in X.columns] + eng_cat

    cat_vocab = {}
    for c in all_cat_cols:
        if c in X.columns:
            cat_vocab[c] = sorted(X[c].astype(str).unique().tolist())

    cat_idx = [X.columns.get_loc(c) for c in all_cat_cols if c in X.columns]

    # Determine which base models are available
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
    try:
        import sklearn  # noqa

        available.append("hgb")
    except Exception:
        pass
    try:
        import xgboost  # noqa

        available.append("xgb")
    except Exception:
        pass

    if not available:
        # fallback only
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
                    "stacker": False,
                },
                f,
            )
        return

    # ---- OOF stacking ----
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = {name: np.zeros((len(X), n_classes), dtype=np.float64) for name in available}

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Recompute cat_vocab on fold-train (safe: no leakage)
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
                elif name == "hgb":
                    m = _fit_hgb(X_tr, y_tr, all_cat_cols, fold_vocab)
                elif name == "xgb":
                    m = _fit_xgb(X_tr, y_tr, all_cat_cols, fold_vocab, n_classes)
                else:
                    continue
                p = _predict_proba(m, name, X_va, all_cat_cols, fold_vocab)
                # align columns to n_classes
                if p.shape[1] != n_classes:
                    p2 = np.zeros((p.shape[0], n_classes))
                    for j in range(min(p.shape[1], n_classes)):
                        p2[:, j] = p[:, j]
                    p = p2
                oof[name][va_idx] = p
            except Exception as e:
                print(f"fold {fold_i} {name} failed: {e}")

    # Build stacker features: concatenate OOF probs from each model
    stack_X = np.concatenate([oof[name] for name in available], axis=1)
    stacker = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=1)
    stacker.fit(stack_X, y)

    # ---- Refit base models on full training data ----
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
            elif name == "hgb":
                m = _fit_hgb(X, y, all_cat_cols, cat_vocab)
                with open(os.path.join(model_dir, "hgb.pkl"), "wb") as f:
                    pickle.dump(m, f)
            elif name == "xgb":
                m = _fit_xgb(X, y, all_cat_cols, cat_vocab, n_classes)
                with open(os.path.join(model_dir, "xgb.pkl"), "wb") as f:
                    pickle.dump(m, f)
            base_models[name] = True
        except Exception as e:
            print(f"final fit {name} failed: {e}")

    with open(os.path.join(model_dir, "stacker.pkl"), "wb") as f:
        pickle.dump(stacker, f)

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
                "stacker": True,
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
    use_stacker = meta.get("stacker", False)

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    X, _ = build_features(
        features, feature_cols, cat_cols, num_cols, freq_maps=freq_maps, fit=False
    )

    if not base_names:
        fb_path = os.path.join(model_dir, "fallback.json")
        if os.path.exists(fb_path):
            with open(fb_path, "r") as f:
                majority = json.load(f)["majority_idx"]
            preds = np.full(len(X), majority, dtype=np.int64)
        else:
            preds = np.zeros(len(X), dtype=np.int64)
        labels_out = [idx_to_label[int(p)] for p in preds]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame(
            {"row_id": np.arange(len(features)), "prediction": labels_out}
        ).to_csv(output_path, index=False)
        return

    # Load base models
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
            elif name == "hgb":
                with open(os.path.join(model_dir, "hgb.pkl"), "rb") as f:
                    loaded[name] = pickle.load(f)
            elif name == "xgb":
                with open(os.path.join(model_dir, "xgb.pkl"), "rb") as f:
                    loaded[name] = pickle.load(f)
        except Exception as e:
            print(f"load {name} failed: {e}")

    if not loaded:
        fb_path = os.path.join(model_dir, "fallback.json")
        with open(fb_path, "r") as f:
            majority = json.load(f)["majority_idx"]
        preds = np.full(len(X), majority, dtype=np.int64)
    else:
        # Get base probabilities
        probs_list = []
        for name in base_names:
            if name not in loaded:
                # pad with uniform
                probs_list.append(np.full((len(X), n_classes), 1.0 / n_classes))
                continue
            try:
                p = _predict_proba(loaded[name], name, X, all_cat_cols, cat_vocab)
                if p.shape[1] != n_classes:
                    p2 = np.zeros((len(X), n_classes))
                    for j in range(min(p.shape[1], n_classes)):
                        p2[:, j] = p[:, j]
                    p = p2
                probs_list.append(p)
            except Exception as e:
                print(f"predict {name} failed: {e}")
                probs_list.append(np.full((len(X), n_classes), 1.0 / n_classes))

        if use_stacker and os.path.exists(os.path.join(model_dir, "stacker.pkl")):
            with open(os.path.join(model_dir, "stacker.pkl"), "rb") as f:
                stacker = pickle.load(f)
            stack_X = np.concatenate(probs_list, axis=1)
            preds = stacker.predict(stack_X).astype(np.int64)
        else:
            avg = np.mean(probs_list, axis=0)
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
