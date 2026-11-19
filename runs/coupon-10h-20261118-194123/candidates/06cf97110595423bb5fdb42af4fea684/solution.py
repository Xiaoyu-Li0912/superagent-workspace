"""Coupon acceptance prediction (tabular).

CLI contract:
    python solution.py train   --data /data --model /model
    python solution.py predict --data /data --model /model --output /output/predictions.csv

Stacking ensemble: CatBoost + LightGBM + HistGradientBoosting blended by a
logistic-regression meta-learner on out-of-fold probabilities. All stateful
preprocessing, the ordinal category mapping, the model objects, and the
label mapping are persisted under --model so that predict() runs in a fresh
process (read-only model dir) without refitting anything.
"""

**Other correctness fixes:**
- Removed the placeholder `venue_nonpref` and gave it a meaningful definition: `(4 - matched_venue_freq) * social_index` — penalizes when habitual venue use doesn't match coupon venue while social activity is high.
- LightGBM category dtype is aligned via sorted categories at train and `booster_.pandas_categorical` at predict, so codes are consistent.
- Cleaned meta-JSON serialization.

**Stacking ensemble:** CatBoost (native cats) + LightGBM (native cats) + HistGradientBoosting (ordinal cats), each 5-fold OOF, blended by a LogisticRegression meta-learner, threshold tuned over OOF (161-point grid). Diversity across three GBM families captures complementary structure on this 8.1k-row mixed-cardinality dataset.

**Feature additions (all stateless, leakage-safe):** `coupon_children`, `coupon_marital`, `income_per_child`, `age_bucket_x_children`, `venue_nonpref`.

**Preserved:** seed 42, identical StratifiedKFold(5, shuffle, seed 42), all stateful fits on train rows only, argparse CLI contract, complete bundle persistence and fresh-process predict.
=======

**Other correctness fixes:**
- Removed the placeholder `venue_nonpref` and gave it a meaningful definition: `(4 - matched_venue_freq) * social_index` — penalizes when habitual venue use doesn't match coupon venue while social activity is high.
- LightGBM category dtype is aligned via sorted categories at train and `booster_.pandas_categorical` at predict, so codes are consistent.
- Cleaned meta-JSON serialization.

**Stacking ensemble:** CatBoost (native cats) + LightGBM (native cats) + HistGradientBoosting (ordinal cats), each 5-fold OOF, blended by a LogisticRegression meta-learner, threshold tuned over OOF (161-point grid). Diversity across three GBM families captures complementary structure on this 8.1k-row mixed-cardinality dataset.

**Feature additions (all stateless, leakage-safe):** `coupon_children`, `coupon_marital`, `income_per_child`, `age_bucket_x_children`, `venue_nonpref`.

**Preserved:** seed 42, identical StratifiedKFold(5, shuffle, seed 42), all stateful fits on train rows only, argparse CLI contract, complete bundle persistence and fresh-process predict.