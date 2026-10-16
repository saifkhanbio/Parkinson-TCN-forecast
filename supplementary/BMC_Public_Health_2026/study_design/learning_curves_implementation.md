# Saudi target-history learning curves

Implementation dated 30 September 2026, after the preceding study results were inspected. The information budgets, model settings and comparisons below follow the previously locked learning-curve design; none is selected using the new learning-curve evaluation scores.

## Scope and interpretation

This supporting sensitivity implements the locked `learning_curve` specification without replacing the primary analysis. Saudi prevalence and incidence are evaluated separately, for both sexes and all eleven age bands from 45–49 through 95+. Forecast origin is 2018; horizons are 1–5, with horizon 5 emphasized. The three Saudi information budgets are 15 years (2004–2018), 20 years (1999–2018), and 29 years (1990–2018). This is a single-origin information-budget comparison, not an independent replication or a new interval-calibration experiment.

## Information boundaries

Every donor retains 1990–2018 history: Bahrain, Kuwait, Oman, Qatar, United Arab Emirates, and Jordan. Saudi rows before the budget start are excluded before feature construction, local fitting, preprocessing, or adaptation. Future rows and unrelated actual outcomes are likewise excluded before fitting. Donor-only models exclude Saudi Arabia completely. All supervised windows require eight input years and all five labels observed by 2018. The Saudi budgets yield 3, 8, and 17 windows per sex–age stratum, respectively; donor strata each retain 17. No full-history Saudi tuning, champion selection, historical residual-bank fitting, or prediction intervals are used.

## Models and fixed settings

All fourteen original families are retained: persistence, log trend, damped ETS, ARIMA, age-smoothed trend, pooled ridge/boosting, donor ridge/boosting before and after two-coefficient adaptation, and TCN before adaptation, intercept-only adaptation, and two-coefficient adaptation. Fixed locked cold-start settings are used for every budget: trend window 8; age-smoothed full available history with penalty 1; ridge penalty 1; boosting depth 1, 100 trees and minimum leaf 5; TCN 16 channels, weight decay 0.001, 50 epochs; adaptation penalty 1. ARIMA order selection uses only the arm's training AICc, as in the existing implementation.

The five TCN seeds are 11, 23, 37, 53, and 71. Each outcome has one shared donor ridge model, one shared donor boosting model, and five shared TCN source checkpoints, reused exactly across all budgets. TCN seed log changes are averaged before sex-specific adaptation. The frozen source predictor is never updated during adaptation. Pooled models are refitted for each budget. Training and scaler weights give equal mass to country, then sex, then age, then eligible windows; additional donor windows cannot increase a country's aggregate weight. CPU is used consistently for all learning-curve TCN fits to match the main experimental device. There is no cross-device accuracy comparison.

## Issuance, validation, and reporting

The complete six-arm ledger contains 9,240 forecasts: 2 outcomes × 3 budgets × 14 families × 2 sexes × 11 ages × 5 horizons. Its predictions, adaptation records, source/checkpoint references, and fit metadata are committed and hashed before any 2019–2023 scoring. Failed fits remain as labeled persistence fallbacks. Resume requires identical source, config, test gate, code, software versions, device, and worker settings; cached jobs are hash verified. Existing completed runs, raw data, locked design, and their scientific code are verified and preserved.

Report equal-age absolute log error and absolute rate error for 45+ and 80+, separately by sex, horizon, family, outcome, and budget. Compare each family against its own 29-year arm, report matched adapted/unadapted effects, and show all family comparisons without declaring a retrospectively selected champion. Numerical improvements are descriptive; age cells and horizons are not independent inferential replicates. Meaningful tests cover excluded-target-history and future-label invariance, eligible-window counts, equal hierarchy weights, fit-only scalers, actual small model fits, shared checkpoints, complete issuance before scoring, and cache/ledger tamper rejection.
