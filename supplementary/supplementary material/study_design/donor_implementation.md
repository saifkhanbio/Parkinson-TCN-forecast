# Donor-selection implementation specification

This document fixes numerical details of section 5 of the [locked protocol](locked_v1/protocol.md) before the donor comparisons are fitted. It does not change the primary all-six-country model or use final accuracy to choose donors.

## Prespecified pools and ordering

`src/gbd_park/donors.py::select_donors` supports `all_six_other_regional_countries`, `other_five_gcc`, `three_similar`, and `three_random`. The target is excluded from every pool, irrespective of sex or outcome. A GCC-only pool requires a GCC target. Country order is ascending GBD location ID, including before random sampling; the returned country list uses that same order. Tied similarity distances use location ID.

Random pools sample three countries without replacement using NumPy `default_rng(seed).choice` and each of the five configured seeds: 101, 211, 307, 401, and 503. Every random-pool result is retained. Fixed and random pools do not inspect outcome values.

## Similarity features at each historical origin

Use the requested outcome and target sex, all eleven age bands, and exactly the eight years ending at the origin. Require a complete unique grid of positive finite rates for all seven countries. Exclude other sexes, outcomes, and years before validating or computing features.

Compute four trajectory components: final-year mean age log rate; mean age ordinary-least-squares log slope over eight years; mean age sample standard deviation (`ddof=1`) of the seven annual log changes; and mean age cumulative log change from `t−3` to `t`. The profile block is the final-year eleven-age log-rate vector minus its own age mean.

For each component, fit the median and IQR only across the six candidate donors, using linearly interpolated quartiles. Omit components whose IQR is numerically zero: `IQR <= 32 × machine_epsilon × max(1, abs(median))`. This tolerance addresses floating-point cancellation in constant features. Within each block, average squared standardized differences over retained components. The final distance is half the trajectory mean squared difference plus half the profile mean squared difference. A wholly constant block contributes zero; its declared half-weight remains fixed.

## Fitting and audit requirements

Similarity is selected separately for each target sex, but each selected country supplies both donor sexes to source fitting. Target calibration uses only its corresponding sex. Recompute historical donor decisions at each historical fitting origin, including inner tuning; never substitute an outer-origin donor list into earlier fits.

Save all six similarity distances and ranks, raw features, donor-only scaling values, retained-component masks, cutoffs, selected countries, random seeds, and SHA-256 decision/list hashes. Incomplete similarity inputs raise an explicit error; a runner must report the failure and preserve forecast cells under the study fallback policy. Source data and locked files remain unchanged.
