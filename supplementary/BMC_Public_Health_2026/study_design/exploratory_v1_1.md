# Exploratory amendment v1.1: age correction and residual bias

Prepared on 29 September 2026 after the Saudi primary and reliability results were viewed. This is a new, explicitly exploratory analysis; it does not amend or replace the [locked v1.0 protocol](locked_v1/protocol.md), its issued forecasts, or its primary verdict. The original joint success criterion was not met: the adapted TCN improved both male comparisons but had higher female error than adapted donor ridge. Primary and reliability interval undercoverage was also observed. These findings motivated the questions below. Chronological reconstruction reduces direct leakage within this new analysis, but cannot make the post-result choices confirmatory or restore an untouched final test.

## Fixed scope and preserved models

Study Saudi prevalence in the original eleven age bands, both sexes, and horizons one through five. Reuse the already selected five-seed TCN at every origin 2003–2018. Preserve its source models, standardization, donor countries, original two-coefficient sex correction, hyperparameters, seeds, and failure policy. No source training or original correction refitting is allowed. Save a crosswalk to every frozen payload, original correction, and original prediction used.

Evaluate two point procedures: the unchanged v1 adapted TCN and a new correction with three broad-age offsets per sex. Comparisons with original frozen champions may be reported for context, but cannot change their choices or the original primary conclusion. Do not search additional architectures, donor pools, interval widths, or penalty values after exploratory errors are viewed.

## Strongly regularized broad-age correction

Use the prespecified broad ages 45–64, 65–79, and 80+. For each sex and fit origin, obtain historical target predictions by averaging the original five seeds' log changes and adding that origin's saved two-coefficient correction. Every target training window's fifth label must be available at the fit origin. Fit additional offsets `delta_g`, constant over horizons and shared across ages within group `g`, to residuals `e` after the original correction:

```text
objective = mean_all_target_age_window_horizon_cells(abs(e_i − delta_group(i)))
            + lambda * sum_g(delta_g^2)
```

The finite candidates are no additional correction and `lambda` in `{100, 10, 1}`, in that tie-breaking order. With `p_g` the fraction of training cells in group `g`, solve each offset by the existing exact regularized absolute-error location solver using penalty `lambda/p_g`. This retains equal-age/window/horizon loss while shrinking every group offset toward zero; `abs(delta_g) <= p_g/(2*lambda)`. The original `b0,b1` remain fixed. The additional correction introduces three coefficients per sex, not new biological or demographic predictors.

At each origin, choose the candidate separately by sex using mean equal-age absolute log error at horizon five over completed inner origins 2003 through `t−5`. Use `lambda=100` when no completed inner block exists; exact ties follow the declared candidate order. Produce all candidates before scoring them, then save the chronologically selected historical forecasts. At evaluation origins 2014–2018, only candidate scores verified by the respective origin may choose the correction. The selected age offsets are added to the unchanged point forecast in log space. Failures retain labeled forecasts without deleting cells.

## Residual-bias retention

For each point procedure, reconstruct its own genuinely prequential selected forecasts at origins 2003–2013. The age-corrected procedure must not use residuals of retrospectively applied final settings. Form complete historical residual blocks with sign `log(observed) − log(point_forecast)`, retaining paired origin, sex, age, and horizon coordinates. At fit origin `t`, use only residual origins 2003 through `t−5` and preserve the minimum of five blocks.

For fixed bias-retention fraction `r` in `{0, 0.5, 1}`, use log residual adjustments `e_o − (1−r)*mean(e)` coordinate by coordinate. Thus `r=0` reproduces the original mean-centered interval procedure, `r=1` retains the historical residual mean, and `r=0.5` retains half. Every variant uses the same complete blocks and their dependence; this changes the predictive distribution's location, not its empirical log spread. Rate widths may change after exponentiation. No resampling, tail extrapolation, or coverage-targeted widening is introduced.

Retain all three fixed variants and a chronologically selected variant for each point procedure. Select `r` separately by sex and procedure using equal-age horizon-five **log-rate 50/80 WIS** from completed earlier development interval origins 2012 and 2013. The minimum-block rule means no such evidence is complete at origins 2014–2016: use `r=0`. At origin 2017 only 2012 is eligible, and at origin 2018 both 2012 and 2013 are eligible. Exact ties favor smaller `r`. Current evaluation outcomes never select retention. Use shared historical-origin indices even when the two sexes select different fractions.

Compute linear-interpolated empirical quantiles separately on the log-rate draws and the exponentiated rate draws, following the original scale convention. Keep the original point forecast distinct from the predictive median. Report central 50%, 80%, and exploratory 95% intervals, principal 50/80 WIS, supplementary 50/80/95 WIS, and all retained variants. In particular, preserve 80% coverage, width, and WIS together; increased coverage alone is not evidence of improvement.

## Execution, diagnostics, and interpretation

Save this amendment, its machine configuration, and hashes of original manifests before producing new results. Freeze all evaluation correction and retention choices, point forecasts, interval forecasts, and joint draws before scoring origins 2014–2018. Report the five origins separately and their equal-origin mean, with origin 2018/horizon five shown separately as an exploratory revisit. It is not a new primary test. Record failures, correction coefficients, shrinkage bounds, selected penalties, block counts, and source references.

Diagnose signed error using `log(prediction) − log(observed)` and original-rate signed error by age, sex, horizon, and origin; positive values mean overprediction. Keep these diagnostic signs explicit because interval residuals use the opposite sign. Separate the observed historical residual mean from the correction and interval retention fraction. Source lower/upper bounds remain source-estimate uncertainty summaries and are not forecast intervals or calibration targets. Biological or causal interpretation of age offsets is not justified.

Required checks include future-row invariance, complete chronology, analytic group shrinkage, preservation of original two-coefficient terms and source states, paired joint blocks, exact reproduction of the unchanged method, scale-specific quantiles/WIS, fixed-grid selection, and forecast commitment before evaluation scoring. All old artifacts remain immutable. Report any negative or mixed exploratory result and retain all finite candidates. Overlapping origins, dependent cells, sparse tails, and only one or two eligible interval-tuning origins preclude exact coverage or independent-sample significance claims. Further validation on genuinely new data remains necessary.
