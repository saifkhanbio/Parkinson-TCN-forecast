# Stage 2 implementation specification

Prepared 29 September 2026 before inspecting non-neural development scores. Implements [locked design v1.0](locked_v1/protocol.md); the primary design and stage-1 outputs are unchanged.

## Fixed scope

Saudi prevalence, both sexes, eleven ages; candidate origins 2003–2013, outer development origins 2009–2013, verification through 2018 only. Six families: pooled ridge/boosting, donor-only ridge/boosting without adaptation, and those donor models with the locked two-parameter correction. The primary pool uses all six available non-Saudi countries. Alternative donor pools and other GCC targets remain later analyses.

Pooled models include Saudi and six donor histories; donor-only models exclude every Saudi age/sex row from base fitting and preprocessing. One base model learns both sexes and all ages. Sex-specific settings and adaptation are selected through the same completed-window chronology as stage 1. Candidate base checkpoints are shared between adapted and unadapted versions.

## Inputs, weighting and algorithms

Each example has eight annual log rates centered on the last input, that last log rate, a male indicator, and eleven age indicators: 21 numerical columns, including the identically zero final centered lag. Labels are five future log changes. At origin `t`, examples end at `u <= t−5`; labels never extend beyond `t`. Prediction uses the target history ending at `t`. Require complete country–sex–age histories; do not interpolate missing outcomes.

Balance countries equally, then sexes within country, ages within country/sex, and windows within each stratum. Rescale weights to sum to the number of examples (mean weight one), preserving the standard weighted-SSE interpretation of the configured ridge penalties. In the complete panel all examples consequently have equal weight. The weight function is tested with unequal artificial window counts.

Fit `StandardScaler` with those weights on eligible base-training features only. Use the fitted transformation unchanged for target history and prediction; do not standardize over future inputs, adaptation cases, or a full-series panel. Do not scale the outcome or introduce extra covariates.

- Ridge: scikit-learn 1.6.1, intercept included, SVD solver, the five locked penalties. Fit all five horizon outputs together; this is equivalent to five independent ridge fits sharing inputs and penalty.
- Boosting: five independent `GradientBoostingRegressor` models, absolute-error loss, learning rate 0.03, the eight locked depth/tree/leaf combinations; full-sample fitting, all features eligible, `random_state=11`, no early stopping. Fit every configured combination; do not modify the tree grid after seeing scores.
- Parameter-count ties: ridge coefficients plus intercepts across horizons; boosting initial values plus tree leaf values and split thresholds; add two coefficients for an adapted target-sex prediction. Preprocessing moments are logged separately.

The configured search creates 78 family/setting combinations: 13 pooled, 13 unadapted donor, and 52 adapted donor combinations. There are 26 base fits per origin, reused across sexes/adaptation penalties. Preserve compressed checkpoints and fingerprints before/after adaptation. [Ridge documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Ridge.html), [boosting documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).

## Exact adaptation objective

For each sex, minimize `mean(abs(observed_log_change − donor_prediction − b0 − b1*h/5)) + lambda*(b0²+b1²)` over all eligible Saudi age/window/horizon cells. The mean gives each complete age/window/horizon equal weight. Donor models may use donor data through the current fit origin when predicting historical Saudi adaptation examples; all Saudi labels used for correction must also end by that origin. Each earlier validation origin reconstructs its own base fit and correction.

Use a deterministic convex profile solver: for a fixed `b1`, find the exact regularized absolute-error location `b0` from sorted residuals; minimize the resulting one-dimensional convex function with bounded Brent optimization. The bound `sqrt(unadapted_MAE/lambda)` follows from the objective at zero correction. Use `xatol=1e-12`, at most 500 iterations, and compare the returned solution with zero and the two boundary slopes. This preserves the locked absolute-error objective without replacing it with squared error. Validate against an independent slack-variable constrained optimization and a known analytic solution.

Adaptation consumes predictions, never the mutable base estimator. Verify the serialized object fingerprint remains unchanged. A failed base fit or correction emits a labeled persistence fallback with its reason; no forecast cells are dropped. This is training calibration, not a claim of improved out-of-sample performance.

## Evaluation and outputs

Save all candidate predictions before loading scoring truth. Use the unchanged stage-1 scoring functions and chronological selector. For each of the six families and each outer sex/origin, select base settings and, where applicable, the adaptation penalty jointly on completed inner horizon-five errors. Select the non-neural comparator family for each later reliability origin from completed development forecasts only. Preserve 2019–2023 final errors for stage 5.

For adaptation-harm diagnostics, compare each selected adapted setting with the **same base checkpoint and settings** without correction. An independently tuned unadapted family is a different comparison and cannot isolate adaptation.

Save predictions, separate scores, model/weight/scaling audits, calibration coefficients and labels, checkpoints, tuning choices, comparator selections, timings, source/code hashes and validation results. Checks include future-row invariance, target exclusion, fit-only scaling, balanced weights, completed labels, frozen-base adaptation, complete grids, preserved stage-1/design hashes, and forecast-before-score ordering.

```bash
/home/saif/agpu_env/bin/python tests/test_nonneural.py
/home/saif/agpu_env/bin/python scripts/run_nonneural.py
```

Run statistical models on CPU within `agpu`, with one BLAS thread per process and at most four processes. A passing, current test report is required; an existing result directory is never overwritten.
