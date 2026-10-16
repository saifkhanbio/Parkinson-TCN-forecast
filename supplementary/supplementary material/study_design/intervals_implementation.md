# Historical residual banks and interval implementation

Prepared on 29 September 2026 before inspecting interval development results. This specification implements stage 4 of the [locked protocol](locked_v1/protocol.md), particularly Section 7. Stage-1 to stage-3 point-forecast results have already been viewed; their numerical procedures and artifacts remain unchanged. The scope here is Saudi prevalence, ages 45+, with all six non-Saudi countries retained as the primary donor pool.

## Historical point forecasts

Construct one prequential forecast per family, origin, sex, age, and horizon for origins 2003–2013. Include all five local families, six non-neural families, and the adapted, unadapted, and intercept-only TCN arms: 14 families and 16,940 forecast cells. Every forecast must reconstruct the procedure available at its own origin. Historical training, tuning, and residual windows become eligible only when their fifth-horizon label is available.

For origins 2003–2007 use the locked cold-start settings. At origin 2008, only inner origin 2003 is eligible. At later origins `t`, settings use inner origins 2003 through `t−5`. Persistence and damped ETS have no external hyperparameter choice; ARIMA retains its original training-only AICc order selection. Reuse existing local/non-neural candidate forecasts at the setting selected by this chronology. For origins 2009–2013, require agreement with the prior stages' selected development ledgers.

TCN forecasts use the fixed five-seed log ensemble and the documented shared-base/sex-specific-penalty selection. Reuse the selected seed-11 checkpoint at each of origins 2003–2008 and fit the four remaining seeds, totaling 24 new fits. Average historical target predictions across seeds before fitting one correction per sex. The matched unadapted and intercept-only arms retain that same base ensemble and selected penalty. Preserve all seed identities, fit/calibration failures, and persistence fallbacks under the stage-3 policy. Later historical TCN forecasts reuse the stage-3 development ledger.

Commit the complete prequential prediction ledger before computing residuals. Keep verification values in separate scoring/residual tables, preserve provenance, and verify source/code/prior-result hashes. A historical verification outcome cannot choose the forecast that generates its own residual.

## Residual banks and centering

For family `m` and historical origin `o`, define the complete error block by `e[m,o,sex,age,h] = log(observed_rate) − log(predicted_rate)`. Each block has 2 sexes × 11 ages × 5 horizons. Keep its residual-origin identifier and all coordinate labels; never sort or resample individual coordinates independently. Retain labeled fallback forecasts as part of the forecasting procedure rather than dropping their errors.

At fit origin `t`, include only historical origins 2003 through `t−5`. Compute the arithmetic mean error separately for each family/sex/age/horizon over exactly those eligible origins, then subtract that mean from each corresponding historical error. Save raw errors, coordinate means, and centered blocks. Do not subtract a mean estimated using later residual origins, and do not add the removed mean to the current point forecast.

The banks for fit origins 2014–2018 contain respectively 7, 8, 9, 10, and 11 complete blocks. These are overlapping historical forecast blocks, not independent draws or 110 independent observations per origin. Never increase the stated block count by replication or random resampling. Before current-origin models are fitted, saved bank quantiles describe centered log offsets and rate multipliers only; label them as calibration quantities without forecast years or disease point predictions.

## Joint predictive draws and quantiles

For each eligible block, add its centered log residuals to the corresponding current log point forecasts. Exponentiate every complete resulting block to obtain rate draws. Preserve the same historical-origin index across sexes, ages, and horizons. The point forecast remains stored separately from the predictive median; zero-mean log deviations need not have zero median or preserve the rate mean.

Use deterministic linear-interpolated empirical quantiles, equivalent to NumPy `quantile(..., method="linear")`. Compute the median and central 50%, 80%, and 95% intervals separately on each reported scale: quantile the log draws for log-rate summaries, and quantile the exponentiated draws for rate summaries. Linear interpolation and exponentiation do not commute, so the rate endpoints and median are not defined by exponentiating the interpolated log endpoints or median. This is the explicit numerical interpretation of the protocol's transform-then-quantile procedure. Each scale's scores must use its own quantiles.

Require at least five complete blocks to emit intervals. With fewer blocks, emit an unavailable status, the actual count and reason, and missing interval values. Do not substitute seed spread, source lower/upper bounds, a normal approximation, or artificial tail extrapolation. Label 95% intervals as a sparse-tail sensitivity.

## Sex-specific comparator champions

At a later fit origin, choose the male and female comparator families using only the already specified completed family-selection evidence. To form a joint champion bank, take the male coordinates from the currently selected male family's prequential bank and the female coordinates from the currently selected female family's bank, matched on the same eligible residual origins. Preserve each family's own historically selected settings; do not apply the current winning setting retrospectively to all historical origins.

Center the resulting coordinates over the current eligible origin set. This preserves the historical cross-sex dependence between the two selected methods. It does not remove the reuse of historical information for family selection and interval calibration; disclose that reuse and make no exact or distribution-free coverage claim.

## Scoring and limited development check

For nominal central coverage `c`, set `alpha = 1−c`. With bounds `l,u`, observation `y`, and median `m`, compute:

```text
IS_alpha = (u−l) + (2/alpha)*(l−y)*I(y<l) + (2/alpha)*(y−u)*I(y>u)
WIS = [0.5*abs(y−m) + sum_k((alpha_k/2)*IS_alpha_k)] / (K+0.5)
```

The principal WIS uses the 50% and 80% intervals (`K=2`); supplementary WIS also includes 95% (`K=3`). Coverage includes equality at either endpoint. Report coverage, width, and WIS separately by sex, horizon, and scale, averaging ages equally within an origin and origins equally when summarized. Keep unavailable/fallback counts visible. These definitions follow the [WIS methods paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618) and its [correction](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010592).

A descriptive implementation check may evaluate intervals issued at origins 2012 and 2013, using respectively five and six completed calibration blocks and outcomes through 2018. Save those interval predictions before scoring. These two development origins do not establish reliability or choose interval variants, model settings, or donor pools. Do not inspect or score 2019–2023 in this stage. Final-origin and nested reliability coverage remain stage 5.

## Acceptance checks and later transformations

Check chronological setting selection, complete block keys, exclusion of future residuals, mean-zero coordinates, unchanged paired-origin dependence, minimum-block handling, hand-calculated quantiles/WIS, and exact reuse of previous selected forecasts. Confirm that nonlinear rate quantiles differ from exponentiated log quantiles when expected, and that predictive medians remain distinct from point forecasts.

Later sex ratios, age shares, and conditional counts must transform entire matched rate draws before computing quantiles. Multiple outcomes must share residual-origin indices across components. Population uncertainty requires paired historical population/rate errors; prevalence residual banks alone cannot supply it. Keep coherent point sums separate from predictive medians, and do not treat these intervals as full uncertainty about true disease burden or propagated GBD estimation uncertainty.
