# Data-source references for the Saudi Parkinson's study

**Analytic integration update (3 October 2026):** The MOH/GASTAT population and CHI insurance files now contribute to the [Saudi raw-data integration analysis](../reports/saudi_raw_integration_v1/raw_data_integration_report.docx), including conditional population scenarios and insurance composition. Its [derived-file citation crosswalk](../reports/saudi_raw_integration_v1/derived_file_citations.csv) extends the 1 October source inventory. The [revised manuscript](../manuscript/ije_draft_v2_raw_integration/manuscript.docx) reports this work as exploratory; no additional Parkinson outcome observations or new interval-calibration evidence are claimed.

**Current citation supplement (1 October 2026):** The [dataset citation and reporting guide](dataset_citations_2026-10-01/dataset_citations_and_reporting.docx), [source register](dataset_citations_2026-10-01/source_register.csv), and [complete input-file crosswalk](dataset_citations_2026-10-01/file_to_citation_crosswalk.csv) supersede the proposal-stage source inventory below for manuscript reporting. The package includes verified GBD methods papers, population methods, WHO mortality, CHI insurance/claims documentation and NHRSP project citations. Import the [RIS](dataset_citations_2026-10-01/dataset_references.ris) or [BibTeX](dataset_citations_2026-10-01/dataset_references.bib) bibliography; incomplete export metadata are listed [separately](dataset_citations_2026-10-01/missing_reference_metadata.docx). Newly audited material is not represented as a fitted-model input.

Prepared 26 September 2026. These source references accompany the study proposal and the supporting-data package. Labels D1–D9 are stable working labels; renumber them with the manuscript's complete bibliography at submission. Use the target journal's citation style then. An importable [BibTeX bibliography](data_source_references.bib) is provided.

Updated 27 September 2026 to register the user-supplied age-specific GBD export and its native citation. Existing 26 September supporting-data retrieval dates remain unchanged; the new file's inspection date is not asserted to be its original download date.

Updated 29 September 2026: the [modeling-extension review](saudi_parkinsons_modeling_extensions.md) and three descriptive CSVs reuse D1, D5 and D6; their citation mappings were added without changing source retrieval dates. The raw MOH rehabilitation rows (`43.`/row 7 in the 2023 workbook; `4-55`/row 7 in the 2024 workbook) contain overlapping all-condition series covering 2017–2023 and 2017–2024, respectively. These are not Parkinson's-specific utilization data.

Retrieval dates for the initial supporting-data package below are **26 September 2026**; later acquisitions have their own dates in the current crosswalk. Retrieval is distinct from a dataset's reference year, release year or publication date. The original extraction date of the user-supplied GBD outcome files has not been established and should not be replaced with the supporting-data download date.

## References

**D1 — Primary GBD outcomes.** Institute for Health Metrics and Evaluation. *GBD Results Tool: Global Burden of Disease Study 2023 results* [Internet]. Seattle (WA): IHME; [cited 2026 Sep 26]. Available from: [GBD Results Tool](https://vizhub.healthdata.org/gbd-results/).

Use for the existing GBD 2023 Parkinson's incidence, prevalence, mortality, YLD, YLL and DALY series. Record the cause, location, sex, age, measure, metric, years and release in Methods. This is a tool/source citation, not an invented dataset DOI. Before submission, retain any release-specific recommended citation and methodology references supplied with the original native export; the prepared local workbook alone does not establish that bibliographic metadata or the original extraction date.

The additional export in `More data/IHME-GBD_2023_DATA-e22a15d9-1/`, inspected 27 September 2026, contains seven countries, male/female values, ages 45–49 through 95+, Number/Rate metrics, and lower/upper bounds. Its `citation.txt` supplies: Global Burden of Disease Collaborative Network. *Global Burden of Disease Study 2023 (GBD 2023) Results*. Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024. Available from the GBD Results tool. The year 2024 is preserved from that file, not independently inferred; confirm final release citation details before submission. BibTeX key `IHME_GBD2023_AgeSpecific_Export` preserves this supplied citation under D1. The [new-data audit](../work/new-data-review-2026-09-27/audit_summary.json) records its source hash and exact scope. It is another extract of D1, not an independent disease-data source.

**D2 — UN population data.** United Nations, Department of Economic and Social Affairs, Population Division. *World Population Prospects 2024: Online Edition* [dataset on the Internet]. New York: United Nations; 2024 [cited 2026 Sep 26]. Available from: [World Population Prospects](https://population.un.org/wpp/).

Specific products used: `WPP2024_PopulationByAge5GroupSex_Medium.csv.gz`, `WPP2024_CSV_files_update.zip`, `WPP2024_Demographic_Indicators_notes.csv` and `WPP2024_Locations_notes.csv`, as listed in the [official file index](https://population.un.org/wpp/assets/downloads.json). Cite D2 for all derived UN/GCC demographic features. State that raw populations are in thousands, the medium variant was selected, and the official Togo correction was applied to the global donor extract. No Gulf values were affected by that correction.

**D3 — GCC population data.** GCC Statistical Centre (GCC-Stat). *Population: DF_PSS_DEM_POP, version 1.0* [dataset on the Internet]. Muscat: GCC-Stat; [date not stated; cited 2026 Sep 26]. Available from: [GCC-Stat population dataset](https://dp.marsa.gccstat.org/dataset/population).

Downloaded resource: `DF_PSS_DEM_POP.csv`, from the dataset's public SDMX export. The selected dataflow version is specified in the saved source URL. Cite for 2010–2024 population, age, sex and nationality data and derived extracts. Do not substitute the access year for an unverified publication year. The landing page did not specify a licence.

**D4 — Saudi national population data.** General Authority for Statistics (GASTAT). *Population Estimates Publication 2024*, with accompanying *Population estimates EN* workbook [Internet]. Riyadh: GASTAT; [publication date not stated; cited 2026 Sep 26]. Available from: [official report](https://www.stats.gov.sa/documents/20117/1400941/Population%2BEstimates%2BStatistics%2B2024%2BEN.pdf/c4b02cf7-decf-d948-a5c1-c55f0a4376f3?t=1738828756797) and [linked data workbook](https://www.stats.gov.sa/documents/d/guest/population-estimates-en).

Use for the 2023/2024 national population estimates, their age/sex/nationality breakdown, and the national methodology description. The workbook is cited with its parent publication because the official PDF links directly to it. The workbook supplies 80+ as the oldest band; its parent report's population pyramid displays a finer oldest-age split. "2024" here identifies the publication/data edition, not a verified release date. Matching GCC-Stat values are not independent observations.

**D5 — Saudi workforce, 2023.** Ministry of Health, Kingdom of Saudi Arabia. *Statistical Yearbook 2023* [Internet]. Riyadh: Ministry of Health; [publication date not stated; cited 2026 Sep 26]. Available from: [original Excel workbook](https://www.moh.gov.sa/Ministry/Statistics/book/Documents/Statistical-Yearbook-2023.xlsx).

Neurology extraction: Table 2-5, worksheet `5.`, row 23, "Physicians and Dentists in KSA Health Sectors by Speciality and Professional Categories, 2023G." The [official archive](https://www.moh.gov.sa/ministry/statistics/book/pages/default.aspx) displays a file date of 25 February 2025; a hosting/file date is not automatically the original publication date.

**D6 — Saudi workforce, 2024.** Ministry of Health, Kingdom of Saudi Arabia. *Statistical Yearbook 2024* [Internet]. Riyadh: Ministry of Health; [publication date not stated; cited 2026 Sep 26]. Available from: [original Excel workbook](https://www.moh.gov.sa/Ministry/Statistics/book/Documents/Statistical-Yearbook-2024.xlsx).

Neurology extraction: Table 2-19, worksheet `2-19`, row 23, "Physicians and Dentists in KSA Health Sectors by Speciality and Professional Categories, 2024G." The official archive displays a file date of 20 October 2025. Cite D5 and D6 together when comparing the two years, and retain the professional-category/sector definitions.

**D7 — Published Parkinson's comparison tables.** Safiri S, Noori M, Nejadghaderi SA, Mousavi SE, Sullman MJM, Araj-Khodaei M, et al. The burden of Parkinson's disease in the Middle East and North Africa region, 1990–2019: results from the global burden of disease study 2019. *BMC Public Health*. 2023;23:107. doi: [10.1186/s12889-023-15018-x](https://doi.org/10.1186/s12889-023-15018-x).

Historical source inventory: main Table 1 and supplementary Tables S2–S4 (additional files 5–7). The article XML and all seven supplements were downloaded on 26 September 2026. The article remains available for literature background; earlier numerical comparisons across releases are retired and are not part of the current reporting plan. Article metadata identify CC BY; preserve attribution when adapting published material.

**D8 — Registry documentation only.** Khalil H, Shraim M, Jaradat B, Ibrahim RH, Kamel WA, Khan S, et al. Parkinson's disease database in the Middle East, North Africa, and South Asia countries. *Int J Public Health*. 2025;70:1608016. doi: [10.3389/ijph.2025.1608016](https://doi.org/10.3389/ijph.2025.1608016).

Specific material: supplementary questionnaire `DataSheet1.docx`, downloaded on 26 September 2026. Use this citation for registry design/background if discussed in the manuscript. Do not describe it as patient data used for training or validation. If the final manuscript does not use or discuss the questionnaire, omit D8 from its analytic data-source list.

**D9 — Conditional: SDI not yet acquired.** Global Burden of Disease Collaborative Network. *Global Burden of Disease Study 2023 (GBD 2023) Socio-Demographic Index (SDI) 1950–2023* [dataset on the Internet]. Seattle (WA): Institute for Health Metrics and Evaluation; 2025. doi: [10.6069/BCX2-VV69](https://doi.org/10.6069/BCX2-VV69). [Official dataset record](https://ghdx.healthdata.org/record/gbd-2023-socio-demographic-index-sdi) inspected 26 September 2026.

The record's suggested citation and DOI were verified in the browser. Download remains pending sign-in. Add D9 to analytic Methods only after acquisition and actual inclusion; record the eventual file retrieval date separately. Include its codebook and release information sheet when available.

## Citation placement in the study

| Statement, result or artifact | Reference |
|---|---|
| GBD 2023 PD outcomes and their release | D1, plus relevant official GBD 2023 methods paper(s) confirmed from the native export before submission |
| UN population estimates/projections and derived ageing features | D2 |
| GCC demographic/nationality comparisons | D3 |
| GASTAT population checks and Saudi citizen/non-citizen composition | D4; also D3 when discussing their concordance |
| Saudi neurology workforce figures | D5–D6 |
| Regional literature background | D7; earlier numerical comparisons are retired |
| MENASA questionnaire or registry discussion | D8, as documentation/background only |
| SDI covariates or socioeconomic donor matching | D9, only if subsequently acquired and used |

The [file-to-reference crosswalk](data_file_citations.csv) maps all 18 raw supporting files and all 14 prepared CSVs to these references, including files derived from two sources. This avoids treating our processed files as independently published datasets. Keep the original-source citations in table/figure captions when derived results are exported.

## Current reporting guidance

Use the section-specific passages in the [current reporting guide](dataset_citations_2026-10-01/dataset_citations_and_reporting.docx). They cite the verified source-methods papers alongside the datasets, separate primary outcomes from demographic scenarios and audits, and identify documentation-only acquisitions.

Limit analytic Methods to data actually used. Report original extraction dates where established, precise query filters and transformations. Registry and survey codebooks are documentation, and catalog inspection does not establish data acquisition. Citation alone does not establish permission to redistribute a source file; retain the source terms documented in the data guide.
