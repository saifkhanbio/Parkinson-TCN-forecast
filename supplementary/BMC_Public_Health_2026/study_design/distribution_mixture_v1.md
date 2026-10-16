# Exploratory equal-weight distribution mixture

Specified 30 September 2026 after the source audit, hierarchy verification and native regional ASR comparison. Earlier model/calibration evaluations have been inspected. This is one bounded exploratory extension, not a fresh confirmatory test or a revision of the original primary hypothesis.

## Candidate and unchanged inputs

Evaluate prevalence and incidence in all six GCC countries, Saudi Arabia primary, using origins 2014–2018 and horizons 1–5. Retain male/female and all eleven age bands 45–49 through 95+. Use only the original frozen local statistical champion, pooled non-neural champion and adapted TCN predictive trajectories. Champion identities remain origin/sex specific under the original chronological selection. Structured/scaled/dynamic distributions are not additional candidate components. There is no refitting, weight search, sex-specific model switch or candidate expansion.

The sole candidate assigns probability 1/3 to each complete role trajectory distribution. Enumerate every role × historical residual-origin trajectory deterministically: 21–33 atoms, based on only 7–11 distinct historical origins. A role applies jointly to both sexes, all ages and all five horizons. This is distribution pooling, not coordinatewise averaging of draws or interval endpoints. Duplicate roles/families remain separate fixed roles; their total probability is not reassigned after inspecting overlap.

## Quantiles and controls

The finite mixture is the uniform discrete distribution over those complete trajectories. Its quantile is the generalized inverse of its empirical CDF, using rank `max(1, ceil(q*N - 1e-12))`. The small fixed rank tolerance handles floating-point representations of integer ranks. Unlike linear interpolation over a concatenated sample, this rule is invariant to duplicating an identical component distribution.

Produce three original component controls with the archived linear quantiles, three additional controls using the same inverse-CDF rule as the mixture, and the one mixture. **The primary pooling comparison uses the quantile-matched component controls.** Report the archived original controls alongside them and quantify the quantile-convention effect. No inverse-CDF control is presented as a new calibrated model. The 95% interval is a sparse-tail sensitivity; no coverage guarantee follows from either rule.

The mixture's deterministic point rate is the arithmetic mean of the three frozen role point rates. It remains distinct from its predictive median and empirical draw mean. Compute point counts and ratios/shares from this common point-rate vector and unchanged point populations, preserving coherence. Do not average already-computed ratios or shares.

## Joint demographic transformations

Reuse both saved population procedures: eight-year log trend (primary) and persistence (sensitivity). Within every component, pair population errors with the same residual-origin identifier as the rate errors. Keep whole age–sex–horizon blocks intact. Derive all 31 count nodes, male/female 65+ and 80+ shares within 45+, and eleven male/female rate ratios before calculating quantiles. Source uncertainty bounds, UN marginal quantiles, scenario spread and seed spread are not added to these conditional distributions. The new full-age ASR data are verification/supporting data, not substitutes for age-specific trajectories.

## Scoring and fixed decision criteria

Issue and hash all twelve country–outcome cases before scoring. Report every origin and horizon; the main assessment averages ages equally within origin and then the five origins equally at horizon five. Evaluate rate and log-rate WIS from central 50%/80% intervals, coverage, width, directional misses, and point absolute log error. Report 45+, 45–64, 65–79, 80+ and each of the eleven age bands. Keep burden WIS in each endpoint's native unit; never pool count/share/ratio WIS into one number.

The numerical guardrails below are prespecified editorial decision thresholds, not established clinical margins, statistical significance criteria or a definition of adequate calibration. Assess the same rule in all cases; only Saudi prevalence is the primary decision, with incidence and GCC results supporting.

For **each sex**, mixture rate WIS at 45+ must be at least 5% lower than quantile-matched TCN and no more than 5% worse than either quantile-matched comparator. Log-rate WIS and point absolute log error may be at most 5% worse than matched TCN. At ages 80+, rate WIS may be at most 5% worse. For both 45+ and 80+, absolute deviation of 80% coverage from 0.8 must not increase, and width may be at most twice matched TCN width.

For both-sex and sex-specific 45+ count totals, each sex's 80+ share, and the equal-age summary of sex-rate ratios, require WIS no more than 5% worse than matched TCN, no increased absolute coverage deviation from 0.8, and width at most twice matched TCN. These guards use the primary population procedure; report persistence independently. A zero reference score passes a ratio limit only if the candidate score is also zero. All primary guards must pass to call the candidate promising under this rule. Report failed guards and do not tune another mixture. Even a pass does not establish reliable forecasting or overturn the original primary result.

## Validation and execution

Freeze this document, JSON specification, tested code and input identities before issuing any candidate forecast. Use up to twelve independent CPU workers, one numerical thread each, in `agpu_env`; no GPU training is needed. Check complete historical-origin grids and cutoff years, original point/interval replay, no mutation of sources, replication/permutation invariance, whole-trajectory component selection, paired population errors, nonlinear burden coherence, independent score arithmetic, and future-truth perturbation. Save role/origin membership and mixture mass for every atom. Preserve original failures/fallbacks in the component audit.

The prior source questions are documented in [native ASR verification](../reports/source_asr_verification_v1/report.md). Population comparability and cross-release harmonization remain limitations. Overlapping origins, ages, sexes and GCC countries do not constitute independent replications; inspected historical years remain development evidence for this new method. New source bounds do not validate prediction intervals.
