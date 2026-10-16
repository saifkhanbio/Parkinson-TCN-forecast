# Exploratory amendment v1.2: bounded interval calibration

Authorized on 29 September 2026 after the primary, secondary, donor and demographic results were inspected. This amendment does not replace the locked v1.0 design, its 45+ primary population, issued point forecasts or failed joint primary criterion. Chronological fitting within this experiment does not make its post-result design confirmatory. New results are written to a separate directory.

## Scope and fixed inputs

Use both prevalence and incidence in all six GCC countries; Saudi prevalence is the main exploratory target and Saudi incidence the key secondary target. Preserve all eleven age bands 45–49 through 95+, both sexes, horizons 1–5 and evaluation origins 2014–2018. Report the 2018-origin/horizon-five endpoint separately and once within the five-origin mean. Evaluate the frozen adapted TCN, local champion and non-neural champion, including each origin's frozen sex-specific underlying family. No point model, source model, adaptation, population forecast or donor pool is refitted.

Reuse the CPU primary/secondary forecast ledgers, original target-specific residual banks and demographic ledgers. Calibration development uses the fourteen underlying families' saved prequential forecasts at origins 2012 and 2013. Champion roles did not exist at those early origins: construct family-specific calibration losses, then map the current origin's already frozen champion family to its own past loss curve. Do not invent historical champion winners. Family selection and calibration reuse some development evidence; no exact coverage guarantee is claimed.

## Joint width transformation and finite candidates

For each original bank, retain the same complete historical-origin blocks and coordinate arithmetic mean. If its centered log residual is `z[o,sex,age,h]`, the new draw is:

```text
log_rate_draw = frozen_log_point + c[sex] * z[o,sex,age,h]
```

The width factor is shared over all ages and horizons for that sex, outcome, current underlying family and fit origin. Candidates are exactly `{1, 1.25, 1.5, 2, 3}`; no extension after viewing results. The prior favors 1. All factors retain zero coordinate mean in log residuals and historical block pairing. They change dispersion without adding a new bias correction. Predictive medians may change; original point forecasts do not. Because sparse empirical quantiles can lie on one side of zero, larger factors do not guarantee nested bounds, higher coverage or monotone rate-scale width. Do not independently clip or force interval endpoints around the point.

Report seven rate variants: the five fixed factors applied equally to both sexes, plus a target-only selected factor and a GCC-assisted selected factor (sex-specific). Factor 1 is the exact original-method control. Fixed variants are sensitivity analyses and cannot be promoted by evaluation performance. Derived burdens are evaluated for the three main procedures only: original, target-only selected and GCC-assisted selected.

## Strict historical selection

At development issuance `u` in `{2012,2013}`, reconstruct the underlying family's bank from residual origins 2003 through `u−5`, with at least five complete blocks. Use the saved family's genuinely historical settings and its saved forecast at `u`. Calculate log-rate WIS using the median and central 50% and 80% intervals at horizon five, averaging equally across the eleven ages separately by sex. Normalize this loss by `s = max(mean(abs(raw_log_residual_h5)), 1e−6)` over that development bank's historical origins and eleven ages for the same sex. This fixed, past-only scalar makes countries' losses comparable; it does not reweight ages within a country or increase the number of independent temporal blocks.

At evaluation issuance `t`, use a development loss only if `u+5 <= t`. Thus 2014–2016 select factor 1 by default, 2017 can use 2012, and 2018 can use 2012 and 2013. For each current sex-specific underlying family:

- Target-only: average that target's eligible normalized development losses.
- GCC-assisted: 0.5 times the target mean plus 0.5 times the equally weighted mean of the other five GCC country means, for the same underlying family, outcome and sex. The target is excluded from this donor component. Jordan is not a calibration donor. All six countries must have the same complete eligible development grid.
- Add the fixed penalty `0.05 * log(c)^2` to either loss. Select the exact minimum; ties choose the smaller factor. The grid, penalty, normalization floor and 50/50 borrowing weight are not tuned.

Only the scalar width choice transfers between countries; rate residual shapes and population residuals remain those of the forecast target. Donor point models may have used the target's earlier observations in their original regional training, but all such inputs and donor calibration labels respect the current cutoff. Country pooling supplies no independent-time sample-size or conformal guarantee. No latest evaluation result or age-80+ diagnostic selects a factor.

## Derived burdens and age-specific checks

For each main procedure, multiply transformed target rate draws by the unchanged target population draws from the matching historical origin. Reuse both original population methods (eight-year log trend and persistence), including their fit-origin-specific centered residuals. Do not rescale population residuals, independently resample ages/sexes, add GBD source bounds, or add neural seed spread. This isolates rate-interval calibration and may leave demographic uncertainty failures unresolved.

Recompute the 31-node count hierarchy, same-sex 65+ and 80+ burden shares within 45+, and age-specific male/female rate ratios. Count sums are coherent by construction; ratio/share distributions come from whole transformed blocks before quantiles. Preserve each derived point forecast and retain the original units. Compare factor-1 rate and derived intervals with the frozen outputs on both stated scales, using explicit CSV-round-trip tolerances.

Report all horizons, sexes and countries, with main emphasis on horizon five. Rate diagnostics include ages 45+ equally weighted, 45–64, 65–79, 80+, and each age band separately, retaining 80–84, 85–89, 90–94 and 95+. Show 50/80 coverage, mean width, lower/upper misses and WIS; 95% is supplementary. For burdens, retain sex totals, grand totals, broad-age counts, both share thresholds and all rate ratios. Show original-versus-selected changes in coverage, width and WIS, including adverse changes, at every origin and their equal-origin average. No pooling of incompatible WIS units or independent age-cell significance tests.

## Execution and acceptance

Freeze this document, machine configuration and input manifests before running the new analysis. Keep source bounds and future truth out of interval issuance functions. Commit all selected factors, point references, intervals and paired draws for every case before beginning final evaluation scoring. Saved development loss curves and factor-choice audits record underlying families, eligible origins, country weights, normalization, unpenalized losses and penalties.

Use `agpu` Python and up to twelve independent CPU workers with one numerical thread each. No neural fitting is required; GPU execution adds no necessary work to these small tabular transformations. Preserve original manifests, code and outputs. Required checks cover future-data perturbation, unavailable development blocks, family mapping, target exclusion, fixed-grid selection, factor-1 reproduction, unchanged point/population forecasts, joint block identities, age groups, direct transformed quantiles/WIS and forecast-before-score commitments. An independent audit reconstructs selections and both rate and burden intervals from saved inputs. Report negative findings without further grid search. Stronger confirmation requires untouched future validation data.
