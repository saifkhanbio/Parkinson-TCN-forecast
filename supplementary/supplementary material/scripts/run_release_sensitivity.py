#!/usr/bin/env python3
"""Read-only source comparison; no models, selection, or interval calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "supporting_data/2026-09-26/raw"
PROCESSED = RAW.parent / "processed"
OUT = ROOT / "results/release_sensitivity_v1"
REPORT = ROOT / "reports/release_sensitivity_v1"
COUNTRIES = {"Saudi Arabia": "SAU", "Bahrain": "BHR", "Kuwait": "KWT",
             "Oman": "OMN", "Qatar": "QAT", "United Arab Emirates": "ARE"}
MEASURES = {"Prevalence": "prevalence", "Deaths": "deaths", "DALYs": "dalys"}
METRICS = {"Age-standardized rate": "rate_age_std_combined", "Number": "count_combined"}
KEY = ["location", "measure", "year", "metric"]
TRIPLE = re.compile(r"^\s*(-?[\d.]+)\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s*$")
W = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
S = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
WORKBOOK = ROOT / "age_standard/parkinsons_gbd2023_dataset.xlsx"
WIDE = ROOT / "age_standard/parkinsons_ml_ready_wide.csv"
PLAN = ROOT / "study_design/release_sensitivity_implementation.md"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def verify_map(mapping):
    for name, expected in mapping.items():
        check(sha(ROOT / name) == expected, f"Hash mismatch: {name}")


def protected_hashes():
    lock_path = ROOT / "study_design/locked_v1/lock_manifest.json"
    lock = json.loads(lock_path.read_text())
    expected = {relative(lock_path): sha(lock_path)}
    for key in ["source_sha256", "design_sha256", "output_sha256"]:
        expected.update(lock[key])
    primary_path = ROOT / "results/primary_v1/run_manifest.json"
    primary = json.loads(primary_path.read_text())
    expected[relative(primary_path)] = sha(primary_path)
    expected.update(primary["code_sha256"])
    expected.update({f"results/primary_v1/{p}": h for p, h in primary["output_sha256"].items()})
    verify_map(expected)
    return expected


def parse_published():
    rows = []
    for i, measure in [(5, "Prevalence"), (6, "Deaths"), (7, "DALYs")]:
        path = RAW / f"12889_2023_15018_MOESM{i}_ESM.docx"
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("word/document.xml"))
        tables = root.findall(".//w:tbl", W)
        check(len(tables) == 1, "Unexpected DOCX table layout")
        for row_number, row in enumerate(tables[0].findall("w:tr", W), 1):
            cells = ["".join(t.text or "" for t in c.findall(".//w:t", W)).strip()
                     for c in row.findall("w:tc", W)]
            if len(cells) != 6 or not TRIPLE.fullmatch(cells[1]):
                continue
            specs = [("1990", "Number"), ("1990", "Age-standardized rate"),
                     ("2019", "Number"), ("2019", "Age-standardized rate"),
                     ("1990-2019", "Percent change in ASR")]
            for col, (period, metric) in enumerate(specs, 1):
                m = TRIPLE.fullmatch(cells[col])
                check(m is not None, "Unparseable published cell")
                point, lower, upper = map(float, m.groups())
                check(lower <= point <= upper, "Invalid published source bounds")
                rows.append(dict(location=cells[0], measure=measure, sex="Both",
                                 year_or_period=period, metric=metric, value=point,
                                 lower_95ui=lower, upper_95ui=upper,
                                 original_cell_text=cells[col], source_file=path.name,
                                 source_table_row=row_number, source_table_column=col + 1))
    frame = pd.DataFrame(rows)
    check(len(frame) == 330, "Published source coverage changed")
    keys = ["location", "measure", "sex", "year_or_period", "metric"]
    for name, extracted in [("published_mena_gbd2019_tables_1990_2019.csv", frame),
                            ("published_gcc_gbd2019_tables_1990_2019.csv", frame[frame.location.isin(COUNTRIES)])]:
        prepared = pd.read_csv(PROCESSED / name, dtype={"year_or_period": str})
        check(set(prepared.gbd_release) == {"GBD 2019"}, "Wrong older release")
        joined = extracted.merge(prepared, on=keys, suffixes=("_raw", "_prepared"),
                                 how="outer", validate="one_to_one", indicator=True)
        check(joined._merge.eq("both").all(), "Published prepared keys differ")
        for c in ["value", "lower_95ui", "upper_95ui", "original_cell_text", "source_file", "source_table_row"]:
            check(joined[c + "_raw"].eq(joined[c + "_prepared"]).all(), f"Prepared mismatch: {c}")
        units = np.where(prepared.metric.eq("Age-standardized rate"), "per 100000",
                         np.where(prepared.metric.eq("Percent change in ASR"), "percent",
                                  np.where(prepared.measure.eq("DALYs"), "DALYs", "persons")))
        check((prepared.unit == units).all(), "Published metric/unit mismatch")
    # Independent main-article cross-check (2019 only, all 22 locations).
    article = ET.parse(RAW / "mena_gbd2019_article.xml").getroot()
    checked = 0
    for row in article.findall(".//table-wrap/table/tbody/tr"):
        cells = ["".join(c.itertext()).strip() for c in row]
        if len(cells) != 10:
            continue
        for offset, measure in [(1, "Prevalence"), (4, "Deaths"), (7, "DALYs")]:
            for delta, metric in [(0, "Number"), (1, "Age-standardized rate")]:
                cell = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", cells[offset + delta])
                values = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", cell)))
                selected = frame[(frame.location == cells[0]) & (frame.measure == measure)
                                 & (frame.year_or_period == "2019") & (frame.metric == metric)]
                check(len(selected) == 1 and len(values) == 3, "Article match missing")
                check(np.array_equal(selected[["value", "lower_95ui", "upper_95ui"]].iloc[0], values),
                      "Article / supplement values disagree")
                checked += 1
    check(checked == 132, "Article cross-check coverage changed")
    return frame, checked


def workbook_rows(z, sheet):
    """Read literal numeric/inline-string cells, preserving coordinates and blanks."""
    with z.open(f"xl/worksheets/sheet{sheet}.xml") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "{" + S["s"] + "}row":
                continue
            out = {}
            for c in elem:
                check(c.attrib.get("t") != "s", "Unexpected shared-string workbook; update explicit parser")
                check(c.find("s:f", S) is None, "Unexpected formula in source")
                value = c.find("s:v", S)
                text = "".join(t.text or "" for t in c.findall(".//s:t", S))
                out[c.attrib["r"]] = value.text if value is not None else text
            yield out
            elem.clear()


def parse_new():
    metadata = {}
    raw = {}
    workbook_wide = {}
    with zipfile.ZipFile(WORKBOOK) as z:
        for sheet, label in [(1, "README"), (2, "Data_Dictionary")]:
            metadata[label] = [list(row.values()) for row in workbook_rows(z, sheet)]
        for sheet, target in [(5, raw), (4, workbook_wide)]:
            iterator = workbook_rows(z, sheet)
            header = {re.sub(r"\d", "", cell): name for cell, name in next(iterator).items()}
            for row in iterator:
                values = {header[re.sub(r"\d", "", cell)]: (value, cell) for cell, value in row.items()}
                location = values["location_name"][0]
                year = int(values["year"][0])
                if location not in COUNTRIES or year not in [1990, 2019]:
                    continue
                if sheet == 5:
                    measure = values["measure"][0]
                    if measure not in MEASURES.values():
                        continue
                    check(values["cause_id"][0] == "544" and values["age_group"][0] == "all_ages",
                          "Wrong source cause or age scope")
                    key = (location, year, measure)
                else:
                    key = (location, year)
                check(key not in target, "Duplicate source workbook key")
                target[key] = values
    check(len(raw) == 36 and len(workbook_wide) == 12, "Incomplete source workbook coverage")
    wide = pd.read_csv(WIDE, dtype=str)
    check(not wide.duplicated(["location_name", "year"]).any(), "Duplicate wide CSV keys")
    rows = []
    for location in COUNTRIES:
        for year in [1990, 2019]:
            source = wide[(wide.location_name == location) & (wide.year == str(year))]
            check(len(source) == 1, "Missing wide CSV match")
            for measure, code in MEASURES.items():
                values = raw[(location, year, code)]
                for metric, suffix in METRICS.items():
                    column = f"{code}_{suffix}"
                    point = float(source.iloc[0][column])
                    rawpoint, rawcell = values[suffix]
                    widepoint, widecell = workbook_wide[(location, year)][column]
                    check(math.isclose(point, float(rawpoint), rel_tol=1e-12, abs_tol=1e-12)
                          and math.isclose(point, float(widepoint), rel_tol=1e-12, abs_tol=1e-12),
                          "Workbook / CSV point mismatch")
                    rows.append(dict(location=location, measure=measure, year=year, metric=metric,
                                     new_value=point, new_source_csv_column=column,
                                     new_source_csv_text=source.iloc[0][column],
                                     new_source_raw_cell=rawcell, new_source_wide_cell=widecell,
                                     new_source_raw_text=rawpoint,
                                     new_count_lower=(float(values["count_lower_combined"][0]) if metric == "Number" else np.nan),
                                     new_count_upper=(float(values["count_upper_combined"][0]) if metric == "Number" else np.nan)))
    return pd.DataFrame(rows), metadata


def compare_points(old, new):
    selected = old[old.location.isin(COUNTRIES) & old.year_or_period.isin(["1990", "2019"])].copy()
    check(set(selected.sex) == {"Both"}, "Only both-sex older values are comparable")
    check(set(selected.metric) == set(METRICS), "Unexpected older metric")
    check(set(new.metric) == set(METRICS), "Unexpected newer metric")
    selected["year"] = selected.year_or_period.astype(int)
    selected = selected.rename(columns={"value": "old_value"})
    joined = selected.merge(new, on=KEY, how="outer", validate="one_to_one", indicator=True)
    check(joined._merge.eq("both").all() and len(joined) == 72, "Incomplete comparison keys")
    expected = {(l, m, y, k) for l in COUNTRIES for m in MEASURES for y in [1990, 2019] for k in METRICS}
    check(set(map(tuple, joined[KEY].to_numpy())) == expected, "Wrong comparison scope")
    check(np.isfinite(joined[["old_value", "new_value"]]).all().all()
          and (joined[["old_value", "new_value"]] > 0).all().all(), "Nonpositive/nonfinite comparison")
    joined["iso3"] = joined.location.map(COUNTRIES)
    joined["old_release"] = "GBD 2019"
    joined["new_release"] = "GBD 2023 as labeled in supplied extract"
    joined["age_scope"] = np.where(joined.metric.eq("Number"), "All ages", "Full-age age-standardized")
    joined["unit"] = np.where(joined.metric.eq("Age-standardized rate"), "per 100000",
                              np.where(joined.measure.eq("DALYs"), "DALYs", np.where(joined.measure.eq("Deaths"), "deaths", "persons")))
    joined["absolute_difference"] = joined.new_value - joined.old_value
    joined["relative_difference_percent"] = 100 * (joined.new_value / joined.old_value - 1)
    joined["log_ratio"] = np.log(joined.new_value / joined.old_value)
    joined["old_display_grid"] = np.where(joined.metric.eq("Number"), 1., .1)
    joined["new_display_grid"] = np.where(joined.metric.eq("Number"), 1., .01)
    for label in ["old", "new"]:
        scaled = joined[f"{label}_value"] / joined[f"{label}_display_grid"]
        check(np.allclose(scaled, np.round(scaled), rtol=0, atol=1e-7), "Display precision changed")
    joined["conditional_rounding_difference_halfwidth"] = (joined.old_display_grid + joined.new_display_grid) / 2
    joined["conditional_rounding_sign_stable"] = joined.absolute_difference.abs() > joined.conditional_rounding_difference_halfwidth
    joined["compatibility_status"] = np.where(joined.metric.eq("Number"),
        "same_labels_all_age_counts; source_definitions_not_fully_verified",
        "same_labels_ASR; identical_standard_weights_not_verified")
    check(joined[joined.metric.eq("Age-standardized rate")].new_count_lower.isna().all(), "Crude/count bounds assigned to ASR")
    numbers = joined[joined.metric.eq("Number")]
    check((numbers.new_count_lower <= numbers.new_value).all() and (numbers.new_value <= numbers.new_count_upper).all(), "New count bounds invalid")
    return joined.drop(columns="_merge").sort_values(KEY).reset_index(drop=True)


def endpoint_changes(points, published):
    rows = []
    for (location, measure, metric), g in points.groupby(["location", "measure", "metric"]):
        g = g.set_index("year")
        check(set(g.index) == {1990, 2019}, "Endpoint years missing")
        old_change = 100 * (g.loc[2019, "old_value"] / g.loc[1990, "old_value"] - 1)
        new_change = 100 * (g.loc[2019, "new_value"] / g.loc[1990, "new_value"] - 1)
        p = published[(published.location == location) & (published.measure == measure)
                      & (published.metric == "Percent change in ASR")]
        rows.append(dict(location=location, iso3=COUNTRIES[location], measure=measure, metric=metric,
                         sex="Both", start_year=1990, end_year=2019,
                         old_point_endpoint_change_percent=old_change, new_point_endpoint_change_percent=new_change,
                         endpoint_change_difference_pp=new_change - old_change,
                         published_old_asr_change_percent=(p.iloc[0].value if metric == "Age-standardized rate" else np.nan),
                         old_asr_change_source_text=(p.iloc[0].original_cell_text if metric == "Age-standardized rate" else "not applicable"),
                         direction_reversal=bool(old_change * new_change < 0)))
    return pd.DataFrame(rows)


def compatibility():
    return pd.DataFrame([
        ["Geography", "Six named GCC countries", "matched names; national coverage details not independently reconstructed"],
        ["Years", "1990 and 2019", "matched calendar years; publication/acquisition years are different concepts"],
        ["Sex", "Both only", "direct published combined values; no male/female revision inference"],
        ["Outcomes", "Prevalence, deaths, DALYs", "labels matched; exact cross-release case/cause mapping unverified"],
        ["ASR scope and unit", "Full-age standard; per 100000", "same-label comparison only; not the 45+ primary estimand"],
        ["ASR weights", "GBD standard named in sources", "identical numerical weights NOT verified; cannot isolate rate revision from standardization"],
        ["Number scope", "All ages; persons/deaths/DALYs", "no 45+ substitution; population and estimation revisions remain possible"],
        ["GBD2023 provenance", "Supplied workbook README/dictionary", "not an independently reproduced native export; original acquisition date unknown"],
        ["Bounds", "Older published 95% UI; newer count bounds only", "preserve source values; no cross-release difference UI or predictive-coverage inference"],
        ["Incidence", "No older numerical cells available", "not evaluated"],
        ["Age and sex revisions", "No older numeric age-by-sex series", "45+, 80+, and sex-specific revision robustness not evaluated"],
        ["Annual vintage backtesting", "Only two older endpoints", "forecast ranking/calibration stability across releases not evaluated"],
        ["Source uncertainty draws", "No compatible joint cross-release draws", "no significance testing or uncertainty propagation for differences"],
    ], columns=["dimension", "available_scope", "interpretation_or_limit"])


def markdown_table(frame, columns):
    rows = ["| " + " | ".join(label for _, label in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for key, _ in columns:
            value = row[key]
            cells.append(f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def report_text(points, endpoints):
    asr = points[points.metric.eq("Age-standardized rate")]
    saudi = asr[asr.location.eq("Saudi Arabia")]
    counts = points[points.metric.eq("Number") & points.location.eq("Saudi Arabia")]
    columns = [("measure", "Outcome"), ("year", "Year"), ("old_value", "GBD2019"),
               ("new_value", "Supplied GBD2023"), ("relative_difference_percent", "Same-year difference (%)")]
    summary = asr.groupby(["measure", "year"]).relative_difference_percent.agg(["min", "median", "max"]).reset_index()
    trends = endpoints[endpoints.location.eq("Saudi Arabia") & endpoints.metric.eq("Age-standardized rate")]
    return f"""# Published GBD release-point comparison

This supporting analysis compares **72 matched both-sex cells** across all six GCC countries: 36 full-age age-standardized rates (ASRs) and 36 all-age Numbers, for prevalence, deaths and DALYs in 1990 and 2019. Saudi Arabia remains the primary descriptive focus. No models were fitted or selected, and no forecasting result was replaced.

**These are conditional differences between published/supplied release estimates, not fully harmonized revision estimates or validation of forecast stability.** Exact common age-standard weights and cross-release case/cause definitions have not been verified. The older numeric tables cannot assess male/female, ages 45+, ages 80+, incidence, annual vintage forecasts or prediction-interval reliability.

## Saudi full-age standardized rates

All values are per 100,000. Each row compares the same calendar year in two releases, not disease growth between release years.

{markdown_table(saudi, columns)}

## Saudi all-age Numbers

Prevalence is persons, deaths are events, and DALYs are burden-years. These totals include ages below 45 and are not interchangeable with the primary study's 45+ totals.

{markdown_table(counts, columns)}

## Identical GCC comparison

Across six countries, the descriptive range and median of same-year ASR percentage differences are:

{markdown_table(summary, [("measure", "Outcome"), ("year", "Year"), ("min", "Minimum (%)"), ("median", "Median (%)"), ("max", "Maximum (%)")])}

These six-country summaries are not confidence intervals or independent replication samples. Every cell, source reference and country-specific contrast is retained in [matched_points.csv](../../results/release_sensitivity_v1/matched_points.csv).

## Historical endpoint changes are a different quantity

The following changes compare 1990 with 2019 **within each release**, computed as ratios of displayed point estimates. The full table preserves the older publication's separately reported percentage change; it need not equal a ratio of displayed means because of rounding and/or draw-level calculation.

{markdown_table(trends, [("measure", "Outcome"), ("old_point_endpoint_change_percent", "GBD2019 endpoint change (%)"), ("new_point_endpoint_change_percent", "GBD2023 endpoint change (%)"), ("endpoint_change_difference_pp", "Difference (percentage points)")])}

No intermediate annual history was inferred. We cannot identify which source, denominator, model or standardization changes produced these differences. A changed historical estimate is not a biological change in the same historical population, and does not establish that either release is closer to clinical truth.

## Compatibility and source integrity

The [compatibility matrix](compatibility_matrix.csv) identifies both matched dimensions and unavailable capabilities. The source workbook dictionary names a GBD world standard; the [GBD2023 methods paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12535840/) specifically names a GBD2023 world standard. Matching these labels does not establish identical numerical weights. Without older full-age age-specific data and verified standard weights, we cannot restandardize both releases to a common population or decompose the ASR differences.

The older data come from [Safiri et al., Tables S2–S4 and Table 1](https://doi.org/10.1186/s12889-023-15018-x). All 330 prepared published cells were reparsed from the three preserved DOCX files; 132 matching 2019 main-table cells were independently checked in the article XML. All 72 newer points were checked against the workbook's Raw_Data, ML_Ready_Wide and analysis-ready CSV. The workbook metadata and exact cell references are preserved. This verifies local consistency, not the accuracy of the original extraction or source estimates.

Selected older ASRs lie on a 0.1 display grid and newer ASRs on 0.01; Number values are integers. Source text is retained. Conditional nearest-grid rounding envelopes in the CSV are arithmetic sensitivity bounds, not uncertainty intervals or verified original rounding rules. Newer count bounds are preserved, but no newer ASR bounds exist in this workbook; crude-rate or count bounds are never assigned to ASRs. No bound-overlap test or significance claim is made.

## Implication for the study

Keep the primary GBD2023 analysis unchanged. Report these source differences as a limitation and supporting sensitivity result. The objective of validating sex-specific forecast rankings or calibration across compatible releases remains **unachieved with the available older data**. The completed comparison motivates release-specific interpretation, not another calibration adjustment or a rescaling of the primary forecasts.

## Reproduction and audit

Implementation: [prespecified supporting plan](../../study_design/release_sensitivity_implementation.md). Run `/home/saif/agpu_env/bin/python scripts/run_release_sensitivity.py` to create outputs in an empty destination; reruns refuse to overwrite. Run the same command with `--verify-only` for a read-only hash and numerical replay audit. [Validation](validation.json) records source, locked design, primary output and artifact integrity; [run manifest](../../results/release_sensitivity_v1/run_manifest.json) records execution and provenance. This source-table comparison is CPU-only and requires no training or GPU allocation.
"""


def sources():
    paths = [WORKBOOK, WIDE, PLAN, Path(__file__), ROOT / "tests/test_release_sensitivity.py",
             PROCESSED / "published_mena_gbd2019_tables_1990_2019.csv",
             PROCESSED / "published_gcc_gbd2019_tables_1990_2019.csv", RAW / "mena_gbd2019_article.xml",
             RAW.parent / "manifest.json", ROOT / "study_design/data_source_references.md"]
    paths += [RAW / f"12889_2023_15018_MOESM{i}_ESM.docx" for i in [5, 6, 7]]
    return {relative(p): sha(p) for p in paths}


def verify_saved():
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    verify_map(manifest["source_sha256"])
    verify_map(manifest["protected_sha256"])
    verify_map(manifest["artifact_sha256"])
    old, _ = parse_published()
    new, _ = parse_new()
    replay = compare_points(old, new)
    saved = pd.read_csv(OUT / "matched_points.csv")
    for c in KEY:
        check(saved[c].astype(str).eq(replay[c].astype(str)).all(), f"Saved key mismatch: {c}")
    for c in ["old_value", "new_value", "absolute_difference", "relative_difference_percent", "log_ratio"]:
        check(np.allclose(saved[c], replay[c], rtol=1e-12, atol=1e-12), f"Replay mismatch: {c}")
    return {"passed": True, "read_only": True, "matched_points_replayed": len(replay),
            "protected_files_verified": len(manifest["protected_sha256"]),
            "artifacts_verified": len(manifest["artifact_sha256"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_saved(), indent=2))
        return
    check(not OUT.exists() and not REPORT.exists(), "Existing release diagnostic: use --verify-only; refuse overwrite")
    started = time.perf_counter()
    protected = protected_hashes()
    source_hashes = sources()
    old, article_checks = parse_published()
    new, metadata = parse_new()
    points = compare_points(old, new)
    endpoints = endpoint_changes(points, old)
    OUT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    points.to_csv(OUT / "matched_points.csv", index=False)
    endpoints.to_csv(OUT / "endpoint_changes.csv", index=False)
    points[points.location.eq("Saudi Arabia")].to_csv(REPORT / "saudi_points.csv", index=False)
    compatibility().to_csv(REPORT / "compatibility_matrix.csv", index=False)
    write_json(OUT / "source_metadata.json", {
        "workbook_metadata": metadata, "older_publication_doi": "10.1186/s12889-023-15018-x",
        "older_publication_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9841703/",
        "gbd2023_method_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12535840/",
        "metadata_web_checked_date": "2026-09-30",
        "gbd2023_method_note": "Methods name GBD2023 world standard; identical numerical cross-release weights not verified.",
        "newer_original_acquisition_date": "not established; workbook creation date is not an extraction date",
        "bounds_use": "preserved source metadata only; no difference UI or forecast interval calculated",
        "regional_primary_panel_use": "none; ages45+ male/female panel cannot supply older both-sex full-age matches"})
    (REPORT / "report.md").write_text(report_text(points, endpoints))
    verify_map(source_hashes)
    verify_map(protected)
    artifact_paths = [p for folder in [OUT, REPORT] for p in folder.iterdir() if p.is_file()]
    artifact_hashes = {relative(p): sha(p) for p in artifact_paths}
    validation = dict(passed=True, older_docx_cells_reparsed=330, article_2019_cells_checked=article_checks,
                      gcc_matched_points=72, newer_points_checked_in_three_representations=72,
                      asr_points=36, all_age_number_points=36, endpoint_comparisons=36,
                      countries=list(COUNTRIES), years=[1990, 2019], sex="Both",
                      duplicated_or_missing_keys=0, protected_files_verified_before_and_after=len(protected),
                      source_files_unchanged=True, primary_outputs_unchanged=True,
                      identical_standard_weights_verified=False, forecast_stability_evaluated=False,
                      artifact_sha256=artifact_hashes)
    write_json(REPORT / "validation.json", validation)
    artifact_hashes[relative(REPORT / "validation.json")] = sha(REPORT / "validation.json")
    write_json(OUT / "run_manifest.json", dict(
        status="complete", role="descriptive conditional published-release point comparison",
        created_utc=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.perf_counter() - started,
        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, device="cpu",
        fits=0, source_sha256=source_hashes, protected_sha256=protected, artifact_sha256=artifact_hashes))
    print(json.dumps(verify_saved(), indent=2))


if __name__ == "__main__":
    main()
