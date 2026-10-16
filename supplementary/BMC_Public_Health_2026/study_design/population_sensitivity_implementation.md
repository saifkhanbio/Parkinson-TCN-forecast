# Population-source sensitivity implementation

This implements the remaining demographic sensitivity in locked design v1.0. It is specified on 30 September 2026 after the primary, secondary and exploratory reliability results have been viewed. It does not revise those analyses, select a population source by its evaluated error, or fit a disease model. The historical UN comparisons below are retrospective diagnostics, not forecasts available at the historical origin.

## Fixed scope and comparisons

Use prevalence and incidence in all six GCC countries, both sexes and all eleven ages 45–49 through 95+. Reuse the original saved `tcn_adapted`, `local_champion` and `nonneural_champion` rates and joint rate-error blocks at origins 2014–2018 and horizons 1–5. Retain the 2018/horizon-five endpoint separately from the five overlapping-origin summaries.

Compare five population inputs, with no parameter selection:

1. Saved operational implied-GBD log trend over the last eight years.
2. Saved operational implied-GBD population persistence.
3. Realized implied-GBD population at the verification year: an oracle diagnostic.
4. Unaligned UN WPP 2024 population at that year: a later-vintage source scenario.
5. Origin-aligned UN growth: `N_GBD(origin) × N_UN(year) / N_UN(origin)`, separately in each age–sex cell. This removes the origin-level difference but still uses later-vintage future population information.

Implied GBD population is Number/Rate × 100,000, separately by outcome. It is not an independently measured population. Never use a 2023 alignment baseline for a 2014–2018 diagnostic. UN values are persons with 95–99 and 100+ combined into 95+. Population reference-date comparability with the supplied GBD extract remains unverified; source differences are not necessarily demographic estimation errors.

## Counts, age composition and uncertainty

Multiply each frozen age-specific rate by each specified population and divide by 100,000. Report the existing 31 count nodes and sex-specific 65+/80+ burden shares within 45+. Retain the complete paired rate-error blocks when transforming distributions. For all five population inputs, hold population deterministic: resulting intervals are **rate-conditional scenario intervals**, excluding population forecast uncertainty. Report their coverage, width, directional misses and 50/80 weighted interval score only as descriptive conditional diagnostics against native GBD modeled counts/shares.

Also retain the original operational joint rate/population interval scores as a separately labeled reference. They have a different uncertainty scope and must not be presented as interchangeable with conditional intervals. No source bounds or neural seed spread are propagated as predictive draws. No interval recalibration, new thresholds or post-result winner selection is allowed.

At each age–sex cell, report exact log-count error as log-rate error plus log-population error. Attribute signed count error symmetrically to rate and population differences: `rate_effect = (r_hat-r) × (N_hat+N)/2/100000`, `population_effect = (N_hat-N) × (r_hat+r)/2/100000`. Sum these accounting contributions at the existing count nodes. These are algebraic attributions, not causes or biological mechanisms.

## Source checks and projection handoff

Tabulate UN versus implied-GBD population differences for 1990–2023 and explicit oldest-age bands. Compare Saudi 2023 GASTAT resident totals at its actual age resolution: aggregate GBD/UN to 80+, without splitting the national 80+ group or inferring nationality-specific disease risk. GASTAT/GCC-Stat concordance is not independent replication.

Prepare both locked 2024–2028 population scenarios: unaligned UN medium and `N_GBD(2023) × N_UN(year)/N_UN(2023)`. These population tables are a handoff for later disease projections; no future disease estimates are invented or substituted from observed rates.

## Integrity and validation

Use `agpu`, up to twelve CPU workers with one numerical thread each; no GPU fitting is needed. Refuse existing output directories. Verify locked inputs and saved source manifests before use and again after execution. Save hashes of consumed artifacts, code, specification and tests. Save transformed points and conditional intervals before joining verification values for scoring, while explicitly disclosing that oracle/UN diagnostics already use future population information and are not a fresh held-out evaluation.

Test origin alignment, units, missing/duplicate-cell rejection, 95+ preservation, complete joint-block transformations, conditional-versus-joint interval labels and exact error identities. Independently reconstruct population formulas, aggregate points and interval summaries from preserved source files. Preserve every primary/model/calibration artifact and `AGENTS.md`.
