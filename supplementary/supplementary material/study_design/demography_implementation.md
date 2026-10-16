# Operational burden and count consistency implementation

This implements the locked v1 secondary analyses for prevalence and incidence in all six GCC countries. It introduces no primary rate-model changes. The Saudi primary result has already been inspected. Numerical conventions below are fixed before inspecting new demographic or reconciliation results.

## Operational populations and joint transformations

Infer denominators as native GBD Number/Rate × 100,000, separately by outcome to preserve the source's exact accounting. Forecast each age–sex population using ordinary least squares on its last eight annual log values, with an intercept at the origin and a slope per year. Population persistence is the fixed sensitivity. Neither method uses UN 2024 data or realized future denominators at historical issuance.

For every rate family, origin and population method, multiply rates by forecast populations and divide by 100,000. Sum the same 22 age–sex cells to six broad-age/sex groups, two sex totals and one 45+ grand total. Calculate 65+/80+ count shares within ages 45+, separately by sex. Calculate age-specific male/female **rate** ratios; unequal sex populations must not enter these ratios. No age-standardized rate is multiplied by a population.

Reconstruct population errors for precisely the historical origins retained in each rate bank; all fifth-horizon labels must be available at issuance. Center population log errors by age, sex and horizon. Pair each population residual with the rate draw from the same historical origin. Transform whole paired blocks into counts, shares and ratios; retain all blocks and calculate linear-interpolated quantiles directly on each transformed scale. These intervals concern GBD point-estimate forecast errors, not full disease-burden uncertainty or independent participants. Point sums and predictive medians remain separate.

Report count MAE, percentage-point share error, and absolute log-ratio error, with coverage, width and 50/80 WIS (95% supplementary). Count WIS is in modeled-number units, share WIS in percentage points, and ratio WIS on the ratio scale. Do not pool these scales. Report origins and horizons separately. All fourteen model families and the two fixed comparator-role views are retained. Symmetric six-order Shapley replacement accounts for 2018–2023 count change through population size, age composition and age-specific rates, separately by sex. Its rate component is descriptive, not biological attribution.

## Native-count reconciliation

Independently forecast native GBD modeled counts at the same 31 nodes using the unchanged damped ETS log-series implementation and its persistence failure policy. Fit at origins 2003–2018 without altering local-model source code. Historical forecasts through origin 2013 provide genuinely prequential node errors. No method or ETS setting is selected using final outcomes.

At each evaluation origin and horizon, compute each node's mean squared count residual after centering across eligible complete historical blocks (denominator `n`, equivalent to `ddof=0`). Require at least five blocks. Floor variances at `1e-8 × mean(training_node_count)^2`, using history from 1990 through the current origin. Otherwise use unit weights and record the equal-weight fallback. Compare independent forecasts, bottom-up sums of the 22 native cell forecasts, and nonnegative diagonal WLS. WLS solves `min_{b>=0} ||diag(variance^-1/2) (S b − independent)||²` with SciPy NNLS; `S` is the fixed 31×22 summing matrix. Do not clip an unconstrained reconciliation after fitting. Zero reconciled counts are permitted and scored with count MAE.

Report errors at each hierarchy level and accounting discrepancies separately. Coherence alone does not demonstrate greater accuracy. These are native-count experiments; they do not replace population-conditioned counts from the rate models.

## Reproduction and validation

Use `/home/saif/agpu_env/bin/python`. Run `tests/test_demography.py`, `tests/test_count_coherence.py` and `tests/test_demography_runner.py`, then record their result and current source/specification hashes in `work/demography-validation/tests.json` before execution. `scripts/run_count_coherence.py --workers 12` fits the independent experiment. `scripts/run_demography.py --workers 4` requires completed, verified primary and secondary manifests. Each saves unscored forecast distributions before joining evaluation truth, refuses existing result directories, and records hashes. Tests cover future-row invariance, complete joint cells, ratio definitions, transformation quantiles/WIS, Shapley accounting, variance cutoffs/floors and constrained reconciliation. Preserve all original input and model artifacts.
