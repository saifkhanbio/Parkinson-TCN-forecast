# Bounded exploratory improvement plan

The specification in `bounded_extension_2026-10-01.json` was hashed at 08:40 UTC on 1 October 2026, before fitting the added model. Earlier study outcomes had already been inspected. This is a locally prespecified extension, not external preregistration or restoration of an untouched historical test set.

## One additional comparator

Use a target-only Bayesian hierarchical Gaussian age–time model, fitted separately by country, outcome and sex. Correlated age-specific level/slope innovations permit information sharing over neighbouring age groups, with damped local trends. A fixed 162-component discrete prior averages observation noise, level/slope innovation scales, age correlation length and damping using training-data likelihoods. Exact Kalman filtering and mixture quantiles avoid MCMC convergence ambiguity. Initialisation conditions on the first observation; subsequent observations enter once.

This is a Bayesian age–time structural model. It is **not** a conventional Poisson Bayesian age–period–cohort implementation: there is no explicit birth-cohort effect, and model-derived GBD counts are not treated as observed independent Poisson events. It addresses the added hierarchical-comparator objective, while performance against conventional BAPC remains untested. The general local-trend model is documented in [statsmodels](https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_local_linear_trend.html); related age-dependent Gaussian forecasting approaches appear in [Pavone and colleagues](https://arxiv.org/abs/2209.12047). This implementation is not a replication of either package or paper.

Preserve all eleven ≥45 age groups, two sexes, prevalence/incidence, six GCC targets and the original five-year endpoint and rolling origins. Save predictions before scoring. Compare native model intervals separately from the original matched residual calibration. Use the same operational population forecasts for derived burden diagnostics. Model posterior intervals are conditional on assumptions and require empirical checking.

## Population and uncertainty work

Audit reference date, geography, nationality/residence, units, age boundaries, algebraic denominator consistency and source uncertainty products. Correct demonstrable processing errors if identified, documenting the correction. Do not choose a population source because it improves a forecast score. Do not reconstruct joint uncertainty by summing marginal bounds.

## Stopping rule

After this single additional model family and denominator audit, stop model/grid/calibration searches on the inspected evaluation years. Report all country–sex outcomes, including deterioration. Permitted reruns are documented implementation/data-integrity corrections, preserving prior artifacts. New external evidence may justify a separately specified validation; the new WHO 2024 extract remains unscored. A younger-age extension requires its own specification and cannot replace the original primary population or dilute ≥80 failures.

Original scientific outputs and the current manuscript package remain preserved. New results are supplied as an explicitly exploratory addendum for the next manuscript revision.
