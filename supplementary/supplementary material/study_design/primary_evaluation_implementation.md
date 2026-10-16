# Saudi primary and nested reliability evaluation

Prepared on 29 September 2026 before evaluating the study forecasts against 2019–2023 outcomes. This specification implements stage 5 of the [locked protocol](locked_v1/protocol.md). Earlier point-forecast development and interval development diagnostics have been viewed. Their models, numerical procedures, and residual-bank rules remain unchanged. As disclosed in the protocol, descriptive 2023 estimates were inspected before the design lock; this is a retrospective forecast evaluation, not a blinded confirmatory test.

## Scope and frozen choices

Evaluate Saudi Parkinson's prevalence rates for males and females separately in the eleven age bands from 45–49 through 95+. Use origins 2014–2018 and horizons one through five. Include all five local families, six non-neural families, and the adapted, unadapted, and intercept-only TCN arms. Add two reporting roles, `local_champion` and `nonneural_champion`, as exact copies of their selected sex-specific source-family forecasts, not additional fitted models.

Before any final-period scoring, save all origin/sex/family setting decisions, comparator-family mappings, the prescribed TCN seed set, donor lists/hashes, and residual-bank references. At origin `t`, settings can use only completed inner origins 2003 through `t−5`; comparator families can use only completed development origins 2009 through `min(2013,t−5)`. Recompute these selections from frozen historical artifacts and require agreement with the saved later-evaluation choices and stage-4 champion mappings.

Use the original settings grids, tie rules, and numerical implementations. The TCN retains one shared base selected with equal-sex loss and separately selected correction penalties, as documented before stage 3. The donor pool remains the six other regional countries. Development performance and interval undercoverage do not authorize a different primary model, donor pool, correction, centering rule, or interval width.

## Fitting and point-forecast ledgers

Fit each model using outcomes available through its origin only. Learned supervised windows require the fifth label by the fit origin; source preprocessing excludes future rows. Donor-only models exclude Saudi rows of both sexes from base fitting and preprocessing. Pooled controls include eligible Saudi history. Local models retain their original training rules, including training-only ARIMA AICc selection.

Fit only the selected local settings. For non-neural models, reuse the unchanged fitting routine with the union of the selected source-base configurations at each origin. That routine may emit auxiliary pooled/donor versions and correction penalties; preserve these auxiliary predictions without scoring them, and select the evaluation rows solely by the previously frozen setting identifiers. This reuse does not introduce a final-period search.

Fit all five prescribed TCN seeds at each origin: 25 new source fits in total. Use the same CPU device, `agpu` environment, software versions, deterministic settings, and fixed epoch counts as stages 3–4. Average current and historical target predictions in log-change space before fitting one two-coefficient correction per sex. The unadapted and intercept-only ablations use exactly the same ensemble and selected penalty. Preserve every seed and the established full-ensemble persistence fallback if a required seed fails; keep all fit and correction failures in the audit.

Construct one unique point-forecast ledger with 16 families/roles × 5 origins × 2 sexes × 11 ages × 5 horizons = 8,800 cells. The underlying 14 families contribute 7,700 cells; champion roles contribute 1,100 copied cells. Retain selected settings, source-family identities, checkpoint references, seed identities, statuses, fallback reasons, and provenance. Never refit a champion separately from its source forecast.

Origin 2018 is fitted once. Its existing rows supply both the final-origin analysis and the last nested reliability origin; there is no second independently generated 2018 forecast.

## Frozen interval banks

Apply the 80 already committed stage-4 banks: 14 families plus two champion roles at five fit origins. Verify each bank's hash, coordinate ordering, origin cutoff, and source-family mapping. The available block counts are 7, 8, 9, 10, and 11. Champion banks already pair the currently selected male/female source methods by the same historical origin; do not reconstruct a different family mixture after seeing outcomes.

Preserve the arithmetic-mean centering, joint block identities, separate log/rate quantiles, and predictive medians defined in [the interval specification](intervals_implementation.md). The expected outputs are 52,800 interval cells across two scales and three central levels, and 79,200 joint draw-coordinate rows. Point forecasts remain distinct from predictive medians. Neither source uncertainty bounds nor seed dispersion are forecast intervals.

Save the complete point, interval, and joint-draw ledgers, their hashes, and chronological execution events before any scoring against outcomes from 2019 onward. Verification data enter separate scoring tables only after that commitment. No stage-5 error or coverage result can change an issued forecast or selected setting.

## Primary endpoint and comparisons

The primary endpoint is origin 2018, horizon five, verifying year 2023. For each sex and method, compute the unweighted mean over eleven ages of `abs(log(predicted_rate) − log(observed_GBD_rate))`. Use the issued point forecast for this endpoint, not the predictive median or an interval-derived point.

Compare the prespecified adapted TCN with both sex-specific champions. Report `TCN_loss − comparator_loss` and relative improvement `(comparator_loss − TCN_loss)/comparator_loss`. Relative improvement is undefined when the comparator loss equals zero; retain the absolute contrast. The joint primary success criterion is strictly lower TCN loss in all four sex-by-comparator contrasts, evaluated at full numerical precision. Ties do not satisfy success. Publish the age-specific errors and the results of all baseline families regardless of the primary outcome.

## Reliability and secondary rate metrics

Report each of the five reliability origins separately, then average their origin-level results equally. Keep the primary endpoint table separate: it is a subset of this same forecast ledger, not an additional independent observation. Report each horizon, along with its equally weighted horizon mean, for absolute log error and original-rate absolute error. Do not combine development-origin errors with these evaluation results.

If origin-population-weighted rate errors are produced, use each sex/age cell's implied GBD population from the Saudi prevalence Number/Rate pair at the fit origin. Normalize these eleven positive populations within sex to sum to one, and apply the same weights to every method and horizon at that origin. Save the weights before scoring. This is a secondary weighting sensitivity using a modeled denominator; it does not replace the equal-age primary estimand or use realized future populations.

Evaluate central 50% and 80% coverage, width, and principal 50/80 WIS separately by sex, horizon, and scale. Report 95% coverage/width and 50/80/95 WIS as sparse-tail sensitivities. Use each scale's predictive median and quantiles with the unchanged WIS formula. Report block counts, missing/fallback counts, and the number of origins and age cells behind summaries.

For TCN adaptation harm, compare corrected and uncorrected predictions from the same fitted five-seed ensemble. An independently tuned unadapted non-neural family is not a matched adaptation control; any such diagnostic requires the corresponding adapted model's exact source checkpoint. Seed dispersion may describe optimization stability but cannot supply epidemiological confidence intervals.

## Acceptance, interpretation, and deferred analyses

Require current passing tests, exact frozen-artifact hashes, complete unique grids, completed-label checks, saved predictions before scores, and independent metric/interval reconstruction after the run. Verify champion point forecasts and intervals coincide with their source-family coordinates. Preserve failed cells through the specified fallbacks. Any implementation defect discovered after scores are viewed requires a disclosed correction with preserved original artifacts and a separately identified rerun; do not silently replace results.

These errors compare forecasts with GBD-modeled estimates, not observed patient diagnoses. The five reliability origins overlap, their age/sex cells are dependent, and selection and interval calibration reuse historical information. Make no independent-cell significance test, exact coverage guarantee, or clinical/biological significance claim from forecast improvement. Negative or mixed transfer remains a valid primary result.

This stage evaluates prevalence rates only. Incidence, sex-rate ratios, age composition, conditional counts, population-source scenarios, reconciliation, alternative donor pools, and other GCC targets remain stage 6. Supporting outcomes, release diagnostics, limited-history experiments, and 2024–2028 projections remain later analyses.
