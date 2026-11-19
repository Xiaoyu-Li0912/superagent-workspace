# Summary

I've produced a complete, corrected `solution.py`. Key fixes over my draft:

**Critical bug fixed:** The HGB ordinal mapping was per-fold at train time but not persisted — I now fit a **single global ordinal mapping** on training rows and persist it in the bundle, applied identically at predict.

**Other correctness fixes:**
- Removed the placeholder `venue_nonpref` and gave it a meaningful definition: `(4 - matched_venue_freq) * social_index` — penalizes when habitual venue use doesn't match coupon venue while social activity is high.
- LightGBM category dtype is aligned via sorted categories at train and `booster_.pandas_categorical` at predict, so codes are consistent.
- Cleaned meta-JSON serialization.

**Stacking ensemble:** CatBoost (native cats) + LightGBM (native cats) + HistGradientBoosting (ordinal cats), each 5-fold OOF, blended by a LogisticRegression meta-learner, threshold tuned over OOF (161-point grid). Diversity across three GBM families captures complementary structure on this 8.1k-row mixed-cardinality dataset.

**Feature additions (all stateless, leakage-safe):** `coupon_children`, `coupon_marital`, `income_per_child`, `age_bucket_x_children`, `venue_nonpref`.

**Preserved:** seed 42, identical StratifiedKFold(5, shuffle, seed 42), all stateful fits on train rows only, argparse CLI contract, complete bundle persistence and fresh-process predict.