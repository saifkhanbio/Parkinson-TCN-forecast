"""Export frozen Saudi forecasts and verified published context without rounding.

Run from any directory: python3 scripts/export_forecast_comparison.py
No model fitting, interpolation, or cross-study accuracy ranking is performed.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "reports/saudi_raw_integration_v1/conditional_projection_summary_2024_2028.csv"
SOURCES = ROOT / "study_design/forecast_comparison_2026-10-03.json"
OUTPUT = ROOT / "forecast.csv"
VALIDATION = ROOT / "reports/forecast_export_v1/validation.json"
FIELDS = [
    "record_type", "study_id", "geography", "outcome", "sex", "age_scope",
    "forecast_year", "forecast_origin", "forecast_horizon_years", "model",
    "population_scenario", "within_80plus_allocation", "scenario_role",
    "metric", "unit", "forecast_value", "count_80plus", "share_80plus_percent",
    "lower_bound", "upper_bound", "interval_level", "interval_type",
    "reported_value", "comparison_status", "comparison_reference_ids",
    "direct_difference", "direct_percent_difference", "data_source",
    "citation", "doi", "source_url", "source_file", "source_locator",
    "source_checked_date", "comments",
]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_register = json.loads(SOURCES.read_text(encoding="utf-8"))
    rate_paths = [ROOT / f"results/projections_v1/trials/SAU_{outcome}/predictions.csv"
                  for outcome in ("prevalence", "incidence")]
    input_paths = [COUNTS, SOURCES, *rate_paths]
    before_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in input_paths}
    selected = [r for r in read_csv(COUNTS) if r["family"] == "tcn_adapted"]
    assert len(selected) == 180
    rows = []
    scenario_order = {
        "gbd_2023_aligned_un_growth": 0, "un_medium_unaligned": 1,
        "national_2024_un_growth": 2, "national_2022_un_growth": 3,
    }
    sex_order = {"Male": 0, "Female": 1, "Both": 2}
    selected.sort(key=lambda r: (
        scenario_order[r["scenario"]], r["within_80plus_allocation"],
        -int(r["outcome"] == "prevalence"), int(r["year"]), sex_order[r["sex"]]))
    base = {
        "study_id": "current_study", "geography": "Saudi Arabia",
        "forecast_origin": "2023", "model": "tcn_adapted",
        "comparison_status": "no_matched_published_numeric_forecast_verified",
        "data_source": "GBD 2023 disease-rate forecasts",
        "citation": "Current Saudi Parkinson forecasting study; retained adapted TCN",
        "source_checked_date": "2026-10-03",
        "interval_type": "point estimate only in this export",
    }
    for r in selected:
        scenario = r["scenario"]
        national = scenario.startswith("national_")
        role = "original_reference" if scenario == "gbd_2023_aligned_un_growth" else "demographic_sensitivity"
        if national and r["within_80plus_allocation"] == "un_within_80plus":
            role = "demographic_and_within_80plus_allocation_sensitivity"
        comments = (
            "Exact stored point estimate; conditional on population scenario. "
            "Prevalence is existing cases; incidence is new cases during the specified year. "
            "Stored decimal precision is computational, not clinical precision. "
            "Published context rows are not matched Saudi benchmarks."
        )
        if national:
            comments += " National open-age 80+ total allocated using the stated weights; no new disease-model fitting."
        rows.append({
            **base, "record_type": "current_study_count_forecast",
            "outcome": r["outcome"], "sex": r["sex"], "age_scope": "45+",
            "forecast_year": r["year"], "forecast_horizon_years": str(int(r["year"]) - 2023),
            "population_scenario": scenario, "within_80plus_allocation": r["within_80plus_allocation"],
            "scenario_role": role, "metric": "count",
            "unit": "people living with Parkinson's disease" if r["outcome"] == "prevalence" else "new Parkinson's cases during year",
            "forecast_value": r["count_45plus"], "count_80plus": r["count_80plus"],
            "share_80plus_percent": r["share_80plus"],
            "comparison_reference_ids": "su_2025_bmj;marras_2018_npj;wu_2025_frontiers" if r["outcome"] == "prevalence" else "wu_2025_frontiers",
            "source_file": str(COUNTS.relative_to(ROOT)), "comments": comments,
        })
    for path in rate_paths:
        rates = [r for r in read_csv(path) if r["family"] == "tcn_adapted"]
        assert len(rates) == 110 and all(r["status"] == "ok" for r in rates)
        rates.sort(key=lambda r: (int(r["forecast_year"]), sex_order[r["sex"]], int(r["age"][:2])))
        for r in rates:
            rows.append({
                **base, "record_type": "current_study_age_specific_rate_forecast",
                "outcome": r["outcome"], "sex": r["sex"], "age_scope": r["age"],
                "forecast_year": r["forecast_year"], "forecast_horizon_years": r["horizon"],
                "scenario_role": "shared_disease_rate_forecast", "metric": "age_specific_rate",
                "unit": "per 100000 population", "forecast_value": r["prediction"],
                "comparison_reference_ids": "su_2025_bmj;wu_2025_frontiers" if r["outcome"] == "prevalence" else "wu_2025_frontiers",
                "source_file": str(path.relative_to(ROOT)),
                "comments": "Exact stored age-specific rate, not an age-standardised rate. Shared across the population scenarios. Historical GBD-modelled rates through 2023 supplied the disease data; no future outcome verification is available in this export.",
            })
    for published in source_register["published_forecasts"]:
        source = source_register["sources"][published["study_id"]]
        rows.append({
            **published, "record_type": "published_context_forecast",
            "forecast_origin": source["origin"],
            "forecast_horizon_years": str(int(published["forecast_year"]) - int(source["origin"])),
            "model": source["model"], "data_source": source["data_source"],
            "citation": source["citation"], "doi": source["doi"], "source_url": source["url"],
            "source_checked_date": source_register["checked_date"],
            "comparison_reference_ids": "current_study",
        })

    assert len(rows) == 406
    assert all(r.get("comments") for r in rows)
    assert all(not r.get("direct_difference") and not r.get("direct_percent_difference") for r in rows)
    assert all(2024 <= int(r["forecast_year"]) <= 2028 for r in rows if r["study_id"] == "current_study")
    keys = [(r["record_type"], r["study_id"], r["geography"], r["outcome"], r["sex"],
             r["age_scope"], r["forecast_year"], r.get("population_scenario", ""),
             r.get("within_80plus_allocation", "")) for r in rows]
    assert len(keys) == len(set(keys))
    groups = defaultdict(dict)
    for r in selected:
        key = (r["outcome"], r["year"], r["scenario"], r["within_80plus_allocation"])
        groups[key][r["sex"]] = r
    max_sex_sum_discrepancy = Decimal(0)
    for strata in groups.values():
        assert set(strata) == {"Male", "Female", "Both"}
        for col in ("count_45plus", "count_80plus"):
            error = abs(Decimal(strata["Male"][col]) + Decimal(strata["Female"][col]) - Decimal(strata["Both"][col]))
            max_sex_sum_discrepancy = max(max_sex_sum_discrepancy, error)
            assert error < Decimal("1e-8")
        for r in strata.values():
            expected = Decimal(100) * Decimal(r["count_80plus"]) / Decimal(r["count_45plus"])
            assert abs(expected - Decimal(r["share_80plus"])) < Decimal("1e-10")
    for r in rows:
        assert Decimal(r["forecast_value"]) > 0
        if r.get("lower_bound"):
            assert Decimal(r["lower_bound"]) <= Decimal(r["forecast_value"]) <= Decimal(r["upper_bound"])
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    exported = read_csv(OUTPUT)
    # String equality preserves every stored digit, avoiding binary-float reformatting.
    for intended, actual in zip(rows, exported):
        assert all(str(value) == actual[key] for key, value in intended.items())
    assert len(exported) == len(rows)
    assert before_hashes == {str(p.relative_to(ROOT)): sha256(p) for p in input_paths}
    report = {
        "output": "forecast.csv", "sha256": sha256(OUTPUT), "rows": len(rows),
        "row_types": dict(Counter(r["record_type"] for r in rows)),
        "current_study_years": [2024, 2025, 2026, 2027, 2028],
        "stored_numeric_strings_preserved": True, "source_files_unchanged": True,
        "unique_keys": True, "comments_complete": True,
        "max_sex_sum_discrepancy_from_stored_precision": str(max_sex_sum_discrepancy),
        "direct_cross_study_differences_calculated": False,
        "published_values_precision": "As reported; units converted exactly where stated",
        "published_context_years": [2026, 2030, 2050],
        "interval_note": "Current-study rows export points only; historical coverage is not assigned to future years. Published bounds retain their published interval type.",
        "comparison_limit": source_register["sources"]["su_2025_bmj"]["verification"],
        "source_hashes": before_hashes,
    }
    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
