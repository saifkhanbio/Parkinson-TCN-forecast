# Published GBD release-point comparison — implementation v1

Specified 30 September 2026 before computing the new comparison tables. This implements the supporting `release_point_revisions` item in the [locked protocol](locked_v1/protocol.md); it does not amend the primary analysis. Saudi 2019 discrepancies had already been described in the original proposal, and forecast/calibration results have been inspected. This is a descriptive source comparison, not prospective or cross-release predictive validation.

## Scope and compatibility gate

Use exactly Saudi Arabia, Bahrain, Kuwait, Oman, Qatar and United Arab Emirates; both sexes together; calendar years 1990 and 2019; prevalence, deaths and DALYs. Compare 36 full-age age-standardized-rate (ASR) cells and, separately, 36 all-age Number cells. Take supplied both-sex estimates directly, without averaging sex-specific ASRs or substituting the regional 45+ panel. No incidence, male/female revision, oldest-age revision, annual vintage series, or forecast-coverage comparison is available from these published older tables.

The older source is Safiri et al., GBD 2019, Tables S2–S4 and main Table 1, DOI [10.1186/s12889-023-15018-x](https://doi.org/10.1186/s12889-023-15018-x). The newer source is the user-supplied GBD 2023 workbook/CSV, whose dictionary labels the standard as the GBD world standard. The [GBD 2023 methods paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12535840/) names a GBD 2023 world standard. Identical age weights across releases are **not verified**. Exact cross-release case/cause mapping, denominator definitions, extraction settings and the newer extract's original acquisition date are also not fully established. Therefore, report conditional same-label published-release differences, not fully harmonized revisions or biological changes. Counts avoid the age-standard-weight problem but remain subject to other release/provenance differences. Do not infer why estimates differ.

## Reproducible extraction and contrasts

1. Verify the existing design-lock and primary-output hashes before and after execution. Preserve sources and existing outputs.
2. Reparse the three original DOCX tables and verify all 330 extracted published cells against the prepared MENA CSV, and the GCC subset against its CSV. Independently check matching 2019 cells in the article XML.
3. Parse workbook XML without installing dependencies, retaining README/dictionary metadata and exact source cell locations. Verify the 72 matched newer points against both `Raw_Data` and `ML_Ready_Wide` and the analysis-ready CSV. Use the source's combined values directly.
4. Report same-year differences `new − old`, percentages `100 × (new / old − 1)` and log ratios; these are changes between estimates for the **same calendar year**, not growth from 2019 to 2023. Preserve the older published bounds and raw text. Do not use interval overlap as a significance or coverage test.
5. Separately calculate 1990–2019 endpoint percentage changes within each release from its displayed points. Retain published GBD 2019 percentage changes separately: a ratio of displayed means need not reproduce a draw-derived published change. Do not invent intermediate annual observations or revision uncertainty.

## Precision, reporting and validation

Retain full supplied numerical values in machine-readable output and source text for auditing. The selected older ASRs lie on a 0.1 grid, newer ASRs on 0.01, and counts on integers. Verify those grids. A conditional nearest-grid rounding envelope may be recorded, explicitly distinguished from uncertainty intervals and from proof of the original rounding algorithm. Source bounds are source uncertainty, not forecast intervals; ASR bounds are absent from the newer workbook.

Produce a compatibility matrix, matched values, endpoint changes, Saudi and GCC tables, source/metadata manifest and read-only validation. Tests must catch wrong metrics, sex/age scope, duplicate/missing matches and incorrect contrast arithmetic. Run using `/home/saif/agpu_env/bin/python scripts/run_release_sensitivity.py`; use `--verify-only` to verify the saved artifacts without rewriting them. No model fitting, donor selection, interval tuning or new clinical inference is part of this step.
