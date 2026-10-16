# Exploratory amendment v1.3: structured calibration and joint dynamic burden forecasts

Authorized after inspection of the original and v1.1/v1.2 results. This is a bounded post-result experiment, not a replacement primary analysis or an untouched validation. Saudi Arabia remains primary; prevalence and incidence are evaluated identically in all six GCC countries. Include all eleven ages 45–49 through 95+, both sexes, horizons 1–5 and origins 2014–2018. The 2018→2023 endpoint is shown separately but counted once in rolling summaries.

## Comparisons and fixed scope

Retain the original, Saudi/target-only calibrated and GCC-assisted calibrated distributions for each frozen adapted TCN, local champion and non-neural champion. Add one structured distribution postprocessor for each of those three roles and one distinct target-only robust dynamic model: thirteen procedures per country/outcome. Original point forecasts, champion mappings, historical models, raw inputs, frozen protocols and previous results remain unchanged. The new dynamic model has its own point and population forecasts and is identified separately throughout.

No candidate search, outcome-informed donor selection or further adjustment follows evaluation. Existing earlier forecasts are reused; no additional neural fits or new calibration donors are needed. Available partially matured forecasts add historical information for shorter horizons, without being counted as five-year errors. Earlier GBD-release and external population-source analyses remain separately defined work; source bounds are not converted into forecast draws here.

## Structured age/horizon distribution calibration

Use the current role's frozen underlying family separately for each sex. Error sign is `log(observed rate) − log(predicted rate)`. At issuance t, only historical origins u≥2003 with u<t and u+h≤t can contribute. Join the saved underlying-family forecasts at origins 2014 onward to verification data already available by t; exclude future labels before numeric validation. Do not invent earlier champion selections.

For each sex and horizon, calculate empirical 20th, 50th and 80th percentiles of raw errors within ages 45–64, 65–79 and 80+, and across all eleven ages. Calculate corresponding percentiles from the original coordinate-mean-centered joint bank. Smooth each percentile over the current and adjacent horizons with normalized weights 2 at the centre and 1 at each available neighbour. Then form left/right semispreads around the smoothed median. The historical-to-bank semispread ratios are clipped to [0.5,3]; a bank semispread≤1e−6 gives ratio 1 with an explicit degeneracy flag.

Let n be the normalized kernel-weighted mean of distinct origin counts at the contributing horizons, a=n/(n+20), and b=n/(n+8). Neither n nor an age-row count is an independent effective sample size. For group and sex-pool medians m_g,m_p and tail ratios r_g,r_p:

```text
location = a²*m_g + a*(1−a)*m_p
scale_tail = exp(b*[a*log(r_g) + (1−a)*log(r_p)])
new_offset(z) = location + scale_left*min(z,0) + scale_right*max(z,0)
```

Apply this monotone map within each sex/broad-age/horizon to the original complete historical residual blocks. The original 7–11 block IDs, age/sex/horizon pairing and population residual pairing remain intact. Partially matured rows inform the marginal transformation only; they do not enter the joint bank. Original point values remain fixed, but predictive means, medians and covariance may change. This is a conservative location/asymmetric-scale heuristic, not exact quantile matching or conformal prediction. Original controls, rather than an identity claim for the new heuristic, establish reproduction.

## Robust dynamic model

Use only the current country's outcome-specific 1990…t rate/count panel. For each sex represent the series by log total native count, ten identifiable orthonormal Helmert log-ratio coordinates of the eleven age proportions, and eleven log implied populations. There are 44 observed coordinates and an 88-dimensional level/slope state. Counts are reconstructed as positive sex totals times softmax proportions, and rates as 100000×count/population. No redundant rate state is fitted.

The fixed local-linear transition is F=[[I,I],[0,0.95*S]]. S smooths neighbouring-age slopes using (I+D′D)⁻¹, projected into the composition basis; total-count slopes are not spatially smoothed. Estimate a regularized innovation covariance Sigma from historical second differences using MAD scales (floor 1e−4), radial clipping at standardized RMS 2.5, and 75% shrinkage toward its diagonal. This shares cross-coordinate information while limiting estimation with few annual vectors.

Set level process covariance to 0.2*Sigma, slope process covariance to 0.1*Sigma and observation discrepancy covariance R=Sigma/12. Initialize the first observed level with zero slope and covariance blockdiag(R,25*Sigma); begin updates in the next year. Use one whole-vector robust weight min(1,(5+44)/(5+Mahalanobis²)) per annual innovation and Joseph-form covariance updates with R divided by that weight. These fixed allocations are modeling assumptions, not estimated GBD source uncertainty.

Generate 1,024 deterministic Monte Carlo paths as 512 antithetic pairs. Draw the initial state from its full filtered Gaussian distribution, then recursively add joint standardized Student-t innovations with five degrees of freedom for process and future observation discrepancy. Derive every age, sex, horizon, count, population, share and rate from the same path. No empirical residual bank is added. The point forecast follows the zero-innovation state path and remains distinct from predictive marginal medians.

The filtered covariance represents uncertainty in estimated levels/slopes, conditional on fitted noise matrices and fixed hyperparameters. Covariance-parameter and hyperparameter uncertainty are not fully propagated; there is no claim of an exact Bayesian posterior or calibrated coverage. Monte Carlo paths improve numerical integration of this assumed model, not the number of observed years. Store the fitted states, covariance inputs, transformation basis, random seed, annual robust weights and all fixed settings for replay. Any invalid covariance or nonfinite reconstruction fails visibly rather than silently dropping a case.

## Derived outcomes and evaluation

Structured calibration retains both original population procedures: eight-year log trend and persistence. The dynamic model jointly forecasts implied populations and counts. Report its population errors separately, and do not attribute differences between these bundled procedures to one component causally. For every draw construct the 31-node count hierarchy, sex-specific 65+/80+ shares within 45+, and age-specific male/female rate ratios before taking quantiles.

Report coverage, mean width, lower/upper miss rates, interval score and WIS using 50% and 80% intervals, with 95% supplementary. Keep log-rate and rate-scale scores separate and never average incompatible burden units. Add point absolute/signed log errors because the new dynamic model changes point forecasts. Report all ages individually and groups 45–64, 65–79 and 80+; retain ages 80–84, 85–89, 90–94 and 95+ even if they perform poorly. Give each rolling origin equal weight. Repeated ages, overlapping origins and GCC targets are dependent; avoid independent-cell tests or exact subgroup-coverage claims.

Success requires a defensible joint assessment of coverage, width, WIS, directional bias and stability across origins, sexes and endpoints. A higher coverage percentage alone does not establish improvement. No new winner is selected for the original primary hypothesis. Preserve mixed or negative results; do not tune this experiment against its own evaluation.

## Execution, integrity and tests

Use agpu Python with up to twelve CPU workers and one numerical thread each. Freeze this amendment, machine-readable settings, implementation and synthetic test evidence before production scoring. Verify prior input/code/output hashes at start and end. Commit every case's intervals, points, calibration parameters and joint draws before any of the twelve cases is scored.

Tests cover unavailable-label perturbation, shorter-horizon maturation, source-family mapping, normalized pooling counts, asymmetric-map monotonicity, immutable originals, state/innovation covariance, deterministic draws, positivity, coherent totals/shares, direct nonlinear quantiles, score agreement and the global issuance gate. Independently reconstruct structured parameters and transformed draws, replay the dynamic fit/simulation, recompute derived endpoints and interval scores, and inspect figures. Report any implementation-only repair transparently; do not alter scientific settings after viewing new outcomes.
