# Locked study design and statistical analysis plan — version 1.0

**Title:** Reliable Forecasting of Parkinson’s Burden in Saudi Arabia: Age–Sex Transfer Learning, Demographic Sensitivity, and Gulf Benchmarking.

**Locked locally on 29 September 2026 following the user’s agreement.** This is the controlling design for subsequent implementation. It supersedes conflicting recommendations in earlier proposals and the earlier standardized-rate protocol. It is not an external registration. Descriptive 2023 outcomes were inspected before this lock; the final block is a forecast evaluation with disclosed prior inspection, not a blinded confirmatory test. No study forecast models have yet been fitted.

The accompanying [machine-readable specification](design.json) fixes scope, grids, sources and cutoffs. `lock_manifest.json` records SHA-256 hashes. Amendments require a new version, a dated reason, and disclosure of which results had been viewed; never silently edit the locked files and retain the same version.

## 1. Research question, target and hierarchy

**Primary objective:** Determine whether regional pretraining with limited Saudi adaptation improves five-year forecasts of male and female age-specific Parkinson’s prevalence among Saudi residents aged 45+, compared with Saudi-only statistical models and pooled non-neural models.

The outcome is GBD-estimated prevalence, not directly observed patient diagnoses. Use eleven bands from 45–49 through 90–94 and 95+, annual 1990–2023 data, and separate male/female reporting. Saudi Arabia is primary. Bahrain, Kuwait, Oman, Qatar and UAE repeat the procedure as secondary targets; Jordan is an additional regional donor.

| Priority | Fixed analysis | Principal outputs |
|---|---|---|
| Primary | Age–sex prevalence transfer experiment | Saudi horizon-five error, both comparator contrasts, every age-specific error |
| Key secondary | Incidence; sex-rate gaps; age composition | Incidence errors, ratio errors, 65+/80+ shares and counts |
| Key secondary | Donor harm and identical GCC replications | Borrowing/adaptation harm matrices; country-specific errors |
| Key secondary | Count aggregation and population assumptions | Consistency-versus-accuracy comparisons; conditional counts and source sensitivity |
| Supporting | Mortality/disability, limited-history experiments, global ASR benchmark, available release revisions | Clearly separated secondary tables |
| Exploratory only | Global-to-regional pretraining, multiple historical outcome channels, service-pressure and equal-rate nationality scenarios | No substitution for a failed primary result |

Survival analysis and modeled survival are excluded. Individual progression, causal biological effects, province-specific PD, nationality-specific PD risk and validated staffing requirements are not study endpoints. A 2050 disease forecast is not part of this design. Additional raw data are not required to start the core experiment.

## 2. Data architecture and readiness

Preserve all originals. The regional panel holds Number and Rate values with their separate source bounds, outcome, age, sex, geography, year and release. Keep 1980–1989 deaths/YLLs for a labeled sensitivity; primary and shared-outcome work start in 1990. Number values are fractional modeled estimates, not independent observed Poisson counts.

Join countries using the explicit GBD–UN crosswalk, not numeric IDs shared across providers. Prepare matching UN ages for all seven countries, including Jordan, from the existing compressed raw file. Combine 95–99 and 100+ into 95+ and convert thousands to persons. Keep GASTAT/GCC-Stat’s 80+ resolution intact; comparing it with finer GBD cells requires an explicit aggregation/allocation assumption.

The 203-location file remains a separate full-age standardized-rate experiment. Its age-standardized rates cannot be multiplied by real populations or treated as missing international age-specific rows. Older GBD releases remain separate. No raw file is modified.

Metadata still requiring confirmation: native extraction dates, the supplied GBD release citation’s publication year, nominal coverage of the source bounds, and population reference-date comparability. These do not prevent point-forecast implementation; until verified, call the bounds “source lower/upper bounds” and population combinations “scenarios.” Do not claim full GBD uncertainty propagation without joint draws. Source references remain in the [reference register](../data_source_references.md).

## 3. Time-based evaluation

Use eight input years and five direct forecast horizons. At fit origin `t`, a training window ending at `u` is eligible only when `u + 5 <= t`. Restrict every donor and every preprocessing calculation to the fit origin. A chronological evaluation compares multistep forecasts with later outcomes while keeping those outcomes out of fitting. [Methodological guidance](https://otexts.com/fpp3/tscv.html)

| Stage | Origins | Rule |
|---|---|---|
| Inner tuning | 2003 through `t−5` | Only fully completed five-year blocks; refit all learned steps at each inner origin |
| Historical family selection | 2009–2013 | Choose comparator families using only scores completed by the outer origin |
| Nested reliability | 2014–2018 | At `t`, family selection may use 2009 through `min(2013,t−5)` |
| Primary final-origin evaluation | 2018 | Forecast 2019–2023; primary endpoint is 2023, horizon five |
| Projections | 2023 | Refit frozen procedures and forecast 2024–2028 |

The 2018 forecast appears once, also serving as the last reliability origin. At that origin there are 17 overlapping training windows per age–sex series: 374 Saudi windows across 22 strata and 2,244 donor windows across six countries. These are dependent training examples, not independent participants. Eleven ages do not become eleven countries.

Use fixed cold-start defaults where no completed inner block exists. The existing calendar’s 2009–2013 family-selection stage is development evidence, not pooled with final errors as independent validation. Projections from this 2023-ending PD file remain projections for 2024–2026 despite the current calendar date. Using revised GBD histories means the study is retrospective/pseudo-out-of-time.

## 4. Models and fair comparisons

Required local candidates: persistence; log-linear trend over 8, 12 or all years; additive damped ETS on log rates without seasonality; nonseasonal ARIMA selected by training-only AICc; and a Saudi-only age-smoothed trend. Local models are fitted separately by sex.

For the age-smoothed model, fit `log rate(a,y) = intercept_a + slope_a × (y−t)/10`, minimizing mean squared residuals plus a second-difference penalty on ordered age slopes. Include a separate 95+ slope without forcing a numeric midpoint or monotonically increasing rates. Apply the smoothness penalty only to the ten closed bands. Use the windows and penalties in the configuration.

Required non-neural controls: pooled ridge and shallow boosting, plus donor-only ridge/boosting before and after the same target calibration as the neural model. Pooled fits include eligible target history; donor-only fits exclude it. Select local and non-neural champion families separately by target sex using eligible historical selection scores. Publish all baseline results. Exact ties use fewer fitted parameters, then declared family/grid order.

The primary transfer model is a compact causal TCN pretrained on the other six regional countries, followed by two-parameter target adaptation. Supply the same information to learned controls: eight origin-centered log lags, the last log level, sex and eleven age indicators. Predict five future log changes. No country embedding or extra demographic/outcome channels enter the primary model. Weight countries, sexes, ages, eligible windows and horizons equally within their hierarchy.

Use three kernel-3 convolution layers with dilations 1/2/4, dropout 0.1, and a final-time representation concatenated with the static inputs. The head outputs five changes. Enforce fewer than 10,000 parameters. Finite grids, AdamW settings and seeds are in `design.json`; no large architecture search is authorized by this design. Tune TCN settings with seed 11 and equal-sex/age horizon-five loss. Use the fixed five-seed log-prediction ensemble for reported outer forecasts, including historical selection forecasts. All seeds remain in the audit; never select the best final seed.

Freeze encoder and head during adaptation. For each sex estimate `b0 + b1 × h/5`, shared over ages, by mean absolute log error plus the configured squared-coefficient penalty. Apply the identical correction to donor-only non-neural controls. Include zero-adaptation and intercept-only ablations; head-only or full-network tuning is exploratory. Select settings only with completed inner blocks. Every failure emits its reason and a persistence forecast; difficult cells cannot disappear from comparisons.

## 5. Donor comparisons and history efficiency

The fixed primary pool is all six other countries. Secondary pools are the other five GCC countries, the three most similar of six, and three random donors under five fixed seeds. Similarity is sex-specific: combine a trajectory block (mean age log level, slope, variability of annual changes, recent three-year change) and the centered eleven-age log profile. Use only the last eight available years. Scale components by donor median/IQR, omit zero-IQR components, and give the two blocks equal mean-squared-distance weight. Break distance ties by GBD location ID. A selected donor contributes both sexes; calibration uses the target sex only. Save every donor list and distance.

Do not select donors using final accuracy. Compare borrowing harm against local/non-neural champions and adaptation harm against the same unadapted checkpoint. The all-donor primary arm remains primary even if a selected pool wins.

At origin 2018, compare the latest 15, 20 and all 29 years of Saudi history, giving 3, 8 and 17 windows per stratum. Ten years cannot supply an eight-year input plus five labels and is explicitly excluded. Use fixed cold-start hyperparameters for every learning-curve budget and report individual families, avoiding family/settings selection from excluded older target history. Retain full pre-origin donor histories, fixed all-other-country pools, the same feature definitions, and bounded target inputs/adaptation. This is a sensitivity to target information, not an independent primary test.

## 6. Endpoints and interpretation rules

For each sex, primary loss is `mean_age(abs(log(predicted rate) − log(GBD rate)))` at origin 2018, horizon five. Report absolute loss differences and relative improvement against both champions. If comparator loss is zero, relative improvement is undefined; retain the absolute contrast. Claim lower error for both sexes only when each sex improves against both champions. Do not label this statistical or clinical significance.

Secondary scores: all horizons separately and their mean; original-rate MAE; age-specific errors; origin-population-weighted loss; count MAE; absolute log-ratio error; and percentage-point errors for 65+/80+ shares among ages 45+. Use one common origin-population definition within each comparison. Report all five reliability endpoints and their mean separately from the primary endpoint. Any cell-level harm fraction describes dependent cells.

No t-tests over ages/windows, bootstrap of neural seeds as epidemiological samples, or nominal significance claims from six GCC countries. No formal power calculation treats the 68,992 source rows as participants. A negative or mixed transfer result remains publishable evidence if the evaluation is complete.

## 7. Prediction intervals suited to the regional panel

Replace the earlier broad-international pseudo-target calibration with **target-history prequential residual blocks** for the core regional experiment. For each method family, make historical target forecasts at origins 2003 through `t−5`, always reconstructing fitting/tuning at that earlier origin. Calibration residual outcomes must never choose the forecast that generated them. Comparator-family selection is still based on historical errors, so disclose its reuse of development information; no exact conformal guarantee is claimed.

At origin 2018 there are eleven eligible residual origins, not hundreds of independent age/sex residuals. For each origin retain the complete 22-stratum × five-horizon log-error block. Center each coordinate across eligible blocks, preserving joint deviations; add these blocks to the current log point forecast and exponentiate. Use deterministic empirical quantiles with linear interpolation. Do not treat replicated/resampled blocks as new information. Require at least five blocks for reported intervals; otherwise emit unavailable status. Five-year residual blocks overlap and have limited tail information.

Report central 50% and 80% intervals routinely and 95% as a sparse-tail sensitivity, together with residual-block count. Assess coverage, width and weighted interval score separately by sex/horizon, using the predictive median and standard weights. Provide 50/80 WIS as the principal interval score, with 50/80/95 WIS supplementary. [WIS methods](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618), [correction](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010592)

For sex ratios, age shares and count totals, transform entire joint blocks. For multiple-outcome sums align residual blocks by origin and use shared draws across components. Store coherent point sums separately from predictive medians: the sum of marginal medians need not equal the median of the sum. These intervals describe forecast errors of GBD point estimates under the method, not the complete uncertainty of true disease burden. Source bounds and seed spread are not interchangeable with predictive intervals.

## 8. Counts, demography and forecast consistency

Convert rates to conditional counts using `sum_age(rate × population / 100000)`. Final scenarios are unadjusted UN medium populations and baseline-aligned populations `N_GBD_2023 × N_UN_t / N_UN_2023`. The inferred GBD denominator is derived from Number/Rate, not an independently observed population series. Report assumptions, source vintage and aligned baseline explicitly.

For historical operational count tests, forecast implied populations independently from the last eight pre-origin log-population values using a linear trend, with persistence as a sensitivity. Match this population procedure across rate models. These are simple demographic forecasting controls, not cohort-component demographic models. Realized future populations provide an oracle diagnostic only. For joint operational count intervals, retain paired historical population and rate errors from the same origin; otherwise report point count scores and rate-conditional scenario intervals with that limitation. UN 2024 information cannot masquerade as available at a 2018 issuance.

Report broad age counts, 65+/80+ shares, and sex ratios. Use symmetric/Shapley replacement of population size, age composition and rates for accounting contributions. Do not attribute its rate component to a biological cause.

The fixed coherence experiment independently forecasts **native GBD counts** at 31 nodes: 22 age–sex cells, six broad-age/sex totals, two sex totals, one 45+ grand total. Use damped ETS log-count forecasts at every node. Compare independent forecasts, bottom-up forecasts, and nonnegative weighted least-squares reconciliation. This isolates aggregation effects within one base family and does not change the primary rate model. Fit diagonal weights from pre-origin squared count residuals after centering, requiring five blocks; floor variance at `1e-8 × training_node_mean²`. If unavailable, emit an equal-weight fallback flag. Evaluate errors at every level, not merely zero accounting discrepancy. [Reconciliation method](https://robjhyndman.com/publications/mint/)

Forecast YLDs and YLLs secondarily and derive DALYs on the same scale. Compare direct DALY forecasts and a prevalence-times-training-ratio YLD baseline. These are modeled accounting relationships, not independent mechanistic validation. Never reconcile the 45+ total to all-age totals without the missing age groups.

## 9. Execution stages and acceptance gates

| Stage | Deliverable | Acceptance gate |
|---|---|---|
| 0 — this step | Locked protocol/configuration, source hashes, tidy regional panel, seven-country population crosswalk/panel, temporal manifests | Complete expected keys; bounds/order/positivity checks; count/rate consistency; time-exclusion assertions |
| 1 | Local prevalence baselines, scoring code and forecast ledger | Hand-checkable metrics; future-row perturbation cannot alter past-origin forecasts; all required cells present |
| 2 | Pooled and donor-only non-neural controls | Fit-only scaling; identical information/weights; correct target exclusions and calibration |
| 3 | TCN and limited adaptation | GPU smoke test in `agpu`; parameter cap; frozen-weight check; reproducible seeds; no future labels |
| 4 | Historical selection and regional interval banks | Save eligible predictions, donor lists and settings before scores; show cold-start/failure records |
| 5 | Saudi final-origin and nested reliability evaluation | Complete male/female comparison matrix; final block does not choose models or settings |
| 6 | Incidence, demography, consistency and GCC repeats | Identical evaluation rules; correct age/sex sums and explicitly named population assumptions |
| 7 | Supporting outcomes, projections and reporting | Separate estimands/uncertainty types; export point/interval tables, figures and limitations |

Stages are ordered implementation gates, not repeated permission requests. Fix implementation defects without changing the estimand; disclose defects that affect viewed scores. Actual model runtime must be measured during the stage-3 smoke test; earlier synthetic hardware benchmarks establish feasibility, not a study completion time.

The run ledger must include protocol/config/source hashes, target, sex, age, outcome, origin, horizon, forecast year, method, donor strategy/hash, adaptation, seed/ensemble, point prediction, predictive median/intervals, population scenario, residual-block count, fit status and fallback reason. Keep verification values in a separate scoring table. Save software versions and training time.

## 10. Manuscript outputs and next implementation step

Main outputs: study/sample table; chronological evaluation figure; Saudi sex-specific primary error table with every age shown; incidence and age-composition forecasts; donor-harm and GCC matrices; interval calibration plots; population-source sensitivity and reconciliation comparisons. Mortality/disability, learning curves, global ASR and available revision diagnostics are supporting tables. No claim of being the first study follows from this design lock.

**The next implementation step is stage 1: local prevalence baselines and the evaluation ledger on the frozen panel.** Data assembly and split validation are executed as part of this design handoff; final model errors are not calculated during the design lock.
