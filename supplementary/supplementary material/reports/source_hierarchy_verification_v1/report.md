# Verification of the downloaded GBD hierarchy

Completed 30 September 2026 using the two files supplied in `More data/`. **The hierarchy comparison is complete. The new Saudi ASR checkpoint export was not found in the inspected repository inputs.** No disease values or forecasts were changed.

## Main finding: the 203-location roster has an official classification

The supplied ASR workbook's 203 location IDs exactly match both of these sets in the downloaded hierarchy:

| Official classification | Location set version | Selected level | Locations | Match to ASR workbook |
|---|---:|---:|---:|---|
| World Bank Regions | 1300 | 2 | 203 | Exact ID-set match |
| World Bank Income Levels | 1305 | 2 | 203 | Exact ID-set match |
| Main GBD hierarchy, national locations | 1511 | 3 | 204 | 201 locations in common |

This explains the observed roster and is consistent with the workbook's mention of World Bank mapping. It does not independently establish the tool that produced its disease values. The earlier description as “203 of the 204 GBD national countries/territories, with one exclusion” remains inaccurate.

The main national set additionally contains **Cook Islands (320), Niue (374), and Tokelau (413)**. The World Bank sets additionally contain **Hong Kong (354) and Macao (361)**. Both classifications include China (6).

In the World Bank regional set, China, Hong Kong and Macao share parent **239, East Asia & Pacific – WB**. In the income set, China belongs to **44576, Upper Middle Income**; Hong Kong and Macao belong to **44575, High Income**. Hong Kong and Macao are absent from the main GBD set in this file. The hierarchy does not place them beneath China in these World Bank sets, and it does not establish the population boundaries or overlap of the underlying disease estimates. Do not infer double counting from these identifiers alone.

Use this wording for the supporting benchmark: **“203 locations represented in the GBD World Bank region and income classifications, comprising 201 of the 204 main GBD national locations plus Hong Kong and Macao.”** A national-only sensitivity would use the available 201 national locations; a complete 204-location panel would require the three missing outcome series. Either changed donor experiment requires separate results.

## Regional and cause identifiers verified

- Saudi Arabia, Bahrain, Kuwait, Oman, Qatar, United Arab Emirates and Jordan are all level-3 locations in set 1511.
- The separate **Gulf Cooperation Council** set, version **1277**, contains exactly the six intended GCC countries.
- **Parkinson's disease = cause 544**, level 3, with parent **542, Neurological disorders**. These identifiers match the existing regional outcome CSV.

The regional primary analysis needs no location-membership change from this verification. The supporting benchmark's numerical results are preserved; its location description is now clarified.

## Source integrity and limitations

The files are `IHME_GBD_2023_HIERARCHIES_Y2025M10D23.XLSX` (98,044 bytes) and `IHME_GBD_2023_HIERARCHIES_INFO_SHEET_Y2025M10D23.pdf` (193,475 bytes). Exact copies and hashes are preserved in the [supporting package](../../supporting_data/2026-09-30_hierarchy_verification/README.md). The information sheet identifies GBD 2023 and release date 23 October 2025, consistent with the [official record](https://ghdx.healthdata.org/record/gbd-2023-cause-rei-and-location-hierarchies), DOI 10.6069/KMAH-ET96.

Its summary paragraph nevertheless says “GBD 2021”; that official wording inconsistency persists in the downloaded PDF. We retain it as a metadata caveat. The workbook contains 1,510 location rows across 14 set versions, 381 cause rows and 227 REI rows. Location IDs repeat across classification sets, so keys must include the set version. Two country names differ from the outcome workbook through apparent character-encoding errors; ID matching avoids changing either source.

The hierarchy confirms classification membership and cause identifiers. It does not validate the source ASR values, supply their uncertainty bounds, reconcile population denominators, or resolve the 124.61% same-year prevalence ASR difference. Those findings in the [source audit](../source_compatibility_v1/report.md) remain open. The existing 162-point local checkpoint is still available from the [inventory recheck](../source_inventory_recheck_v1/report.md).

## Remaining step

Locate the original ASR export/query record or the new **54-row Saudi ASR CSV/ZIP**, with native point/lower/upper values. The two newly supplied files satisfy the hierarchy request; no repeat hierarchy download is needed. Model expansion remains deferred until the source-value comparison is completed or the remaining uncertainty is explicitly incorporated into the analysis scope.

This report updates the geography interpretation in the earlier [review notice](../global_asr_source_review_notice.md); the archived notice and numerical reports remain unchanged.

## Reproduction

`python3 scripts/verify_downloaded_gbd_hierarchy.py` creates a fresh verification package and refuses to overwrite existing output. [Location crosswalk](../../results/source_hierarchy_verification_v1/location_crosswalk.csv), [name-encoding differences](../../results/source_hierarchy_verification_v1/source_name_encoding_differences.csv), and [validation record](../../results/source_hierarchy_verification_v1/validation.json) retain the comparisons, source hashes and protected-file identities. No model was fitted.
