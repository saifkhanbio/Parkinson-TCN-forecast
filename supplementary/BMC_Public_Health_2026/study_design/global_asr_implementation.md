# Separate global age-standardized-rate benchmark

Implementation specification, 30 September 2026. This implements the locked design's supporting global ASR benchmark after the regional results were inspected. It neither replaces the Saudi age-specific primary experiment nor transfers weights into it.

## Scope and sources

Use the preserved `age_standard/parkinsons_ml_ready_wide.csv` for prevalence and incidence, male and female separately, all 203 supplied countries/territories, 1990–2023. The workbook's README and Data_Dictionary identify Parkinson's disease (cause 544), GBD 2023, and full-age ASRs per 100,000 using the GBD world standard. Replay the CSV against the workbook's Raw_Data ASR fields. This verifies internal consistency, not an independently acquired native IHME export. Exact standard weights, extraction settings and source authenticity remain unverified. Do not infer complete global coverage from the supplied 203 locations.

Match every supplied location to preserved UN WPP country/area metadata using exact names and explicit aliases; require unique ISO3 and `Country/Area` type, with no aggregate or unresolved match. Record the source labels, UN labels, IDs and aliases. The UN hierarchy is a geographic identity check, not confirmation of identical GBD boundaries. Retain territories as separately supplied GBD locations; do not add regional or overlapping subnational aggregates. The omitted GBD location is not imputed.

Saudi Arabia is the primary supporting target; Bahrain, Kuwait, Oman, Qatar and UAE repeat the identical procedure. At origin 2018 issue horizons 1–5 (2019–2023); the five-year endpoint is descriptive and previously inspected. No new rolling-origin interval calibration, population conversion or inferential significance test is included.

## Fixed comparisons

Fit persistence, eight-year log trend, damped ETS and training-AICc ARIMA locally. An age-smoothing model has no age axis in this ASR experiment and is omitted. Fit pooled ridge/boosting, donor-only ridge/boosting before and after two-parameter adaptation, and TCN unadapted, two-parameter and intercept-only variants. For each learned method compare all other supplied countries/territories (202 donors) with the six other countries in the original seven-country regional panel. Pooled controls add the target's eligible history; donor-only fits exclude all target records.

Use locked cold-start settings throughout: ridge penalty 1; boosting depth 1, 100 trees, minimum leaf 5, learning rate .03; TCN 16 channels, weight decay .001, 50 epochs, five seeds 11/23/37/53/71; adaptation penalty 1. No hyperparameter, donor or champion selection uses these endpoint errors. Inputs are eight origin-centered log lags, last log level, sex, and one constant ASR indicator. Reuse the immutable architecture with this explicitly separate one-category representation (1,732 parameters). Country/sex/window weighting is equal. Training windows must have all five labels available by 2018; each sex has 17 target windows. Adaptation uses the seed-mean log predictions and frozen source weights.

Both donor scopes use the same device, seeds and numerical settings. A short real-data timing benchmark may choose CPU versus GPU before production; timing results never select predictive settings. Retain failed-seed records and apply the existing whole-ensemble persistence fallback. Save source-only preprocessing, window metadata, all seeds/checkpoints, adaptation coefficients and immutable fingerprints.

## Issuance, verification and interpretation

There are 12 country/outcome cases, 22 forecast views per case (four local plus nine learned methods in each scope), and 2,640 sex/horizon point forecasts. Commit every case and a global issuance ledger before joining any 2019–2023 verification rates. Hash sources, code, settings, tests and output artifacts; reject incompatible resumes and altered issuance artifacts.

Report absolute log error and absolute rate error by country, sex, horizon, outcome and family; report H5 and five-horizon mean separately. Compare global versus regional using matched learned families, and adaptation against its matching unadapted source. Keep all local/non-neural results visible without selecting a post-result champion. Male/female ASRs cannot reconstruct both-sex ASR. No ASR-to-count conversion, interval-reliability claim, individual prognosis, causal biological conclusion, or replacement of the regional primary verdict follows from this bounded benchmark.
