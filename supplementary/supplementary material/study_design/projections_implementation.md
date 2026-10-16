# Projections from the 2023 data cutoff

Specified on 30 September 2026 after the completed historical evaluations were inspected, before fitting this projection package. This implements the locked 2023-origin/2024–2028 projection stage without changing earlier models, decisions or results. The supplied GBD 2023 vintage ends in 2023; the actual computation date is recorded separately. These are retrospective-vintage scenario projections, not forecasts known to have been issued in calendar 2023. In 2026, the supplied file still provides no verification outcomes for 2024–2026.

## Scope and unchanged procedures

Project age-specific prevalence and incidence for all six GCC countries, both sexes and eleven ages from 45–49 through 95+, at horizons 1–5. Jordan remains an additional donor. Retain all fourteen original model families and the two comparator champion views. Use the same CPU numerical environment, model grids, eight input years, equal hierarchical training weights, five TCN seeds and limited target adaptation as the original experiments. Actual outcomes and years 1990 through the fitting cutoff are selected before the internal prevalence-channel alias. All public outputs retain their actual outcome. Do not forecast age-standardized counts or use the global ASR data in this package.

## Settings, family selection and historical evidence

At the 2023 cutoff, the frozen settings-selection rule permits completed inner origins **2003–2018**, with their horizon-five labels available by 2023. Reuse verified candidate predictions at 2003–2013; fit the original complete candidate grids at 2014–2018, retaining tuning seed 11. Commit those candidate ledgers before joining their historical verification values. Select settings without any 2024–2028 values, then refit the selected models at cutoff 2023 and form the original five-seed TCN ensemble.

Comparator **family** selection deliberately retains the locked development-origin list **2009–2013**. It does not extend to the evaluated 2014–2018 origins and does not select a winner from the inspected 2023 endpoint. Family selection uses the original, historically selected forecasts and completed horizon-five losses. Save settings and family decisions separately, including all eligible origin lists and latest label years. A selected family's 2023 settings may differ from its 2018 settings under the unchanged chronological rule.

## Sixteen complete residual blocks

For each underlying family, concatenate its verified original prequential forecasts at 2003–2013 and original issued evaluation forecasts at 2014–2018. Do not replace them with newly fitted candidate predictions or with 2023 settings. These forecasts provide sixteen complete 110-coordinate blocks: two sexes × eleven ages × five horizons. Join only GBD point estimates available through 2023 and center log errors by their coordinate-wise arithmetic means.

For each champion view, stitch the current 2023 sex-specific selected families across the entire historical ledger, preserving the settings originally selected at each past origin. Apply the unchanged centered joint-block interval procedure to 2023-cutoff point forecasts. Retain central 50% and 80% intervals and supplementary 95% intervals on rate and log-rate scales; quantiles are computed separately after transforming to each scale. Save all joint draw identities and point forecasts separately from predictive medians. Sixteen overlapping blocks do not establish nominal coverage; previous undercoverage and age-specific failures remain relevant limitations. No exploratory calibration modification is substituted.

## Population scenarios and derived quantities

Verify and reuse the prepared population handoff for the two locked scenarios: unaligned UN WPP 2024 medium populations, and `GBD-implied N(2023) × UN N(year) / UN N(2023)`. Verify positive persons, exact country/sex/age/year/scenario grids, source baseline formulas and 95+ preservation. Keep the historical GBD/UN reference-date caveat explicit.

For adapted TCN and the local/non-neural champion views, transform every complete residual block to conditional counts using `rate × scenario population / 100000`. Report all 31 count nodes, sex-specific 65+/80+ shares within 45+, and age-specific male/female rate ratios. Preserve common residual origins across sexes, ages and horizons before nonlinear transformations; derive quantiles from transformed blocks rather than summing marginal interval endpoints. Population is fixed within each scenario, so burden intervals describe rate uncertainty conditional on that population path. They do not include full demographic or GBD estimation uncertainty.

Retain native GBD 2023 rates/counts and separate scenario-consistent 2023 baselines: UN baseline populations for the unaligned scenario and GBD-implied baseline populations for the aligned scenario. Report projected changes against the appropriate scenario baseline and show the native baseline alongside it; population-source differences must not be misread as disease growth.

## Execution, validation and reporting

Write only a new `results/projections_v1` run and corresponding report. Use sixteen CPU fit workers and a bounded four-worker case phase by default, with one numerical thread each. Record all consumed input hashes, prior manifests, code/specification/test hashes, numerical versions, settings, seeds, device, actual computation timestamps and cutoff/vintage metadata. Refuse existing outputs without explicit verified resume; changed identities or corrupted checkpoints fail closed. No fitting begins until synthetic checks pass. Commit the complete twelve-case projection ledgers and distributions before reporting them. There is no future-truth join, future coverage computation or new success verdict.

Synthetic checks cover actual-outcome/1990/cutoff filtering and future-row invariance, the distinction between settings and family origin sets, reuse of historical issued forecasts, complete sixteen-block banks, champion mapping, deterministic population alignment, joint ratios/count sums/shares, quantile transformations, duplicate/missing-cell rejection, cached fits and corruption guards. A separate `--verify` mode performs no fitting and reconstructs saved decisions, banks, interval arithmetic, scenario transformations and representative neural/non-neural checkpoints. Reports retain all family rate outputs and emphasize Saudi 2028 adapted-TCN/comparator scenarios and 2023 baselines, with no claim that previously poor interval reliability has been resolved.
