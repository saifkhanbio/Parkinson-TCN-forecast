"""Audit the supplied age-specific GBD export; preserve every source file."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = next((REPO / "More data").rglob("*.csv"))
df = pd.read_csv(SOURCE)
MEASURES = {
    "Prevalence": "prevalence", "Incidence": "incidence", "Deaths": "deaths",
    "YLDs (Years Lived with Disability)": "ylds", "YLLs (Years of Life Lost)": "ylls",
    "DALYs (Disability-Adjusted Life Years)": "dalys",
}
keys = ["population_group_id", "measure_id", "location_id", "sex_id", "age_id", "cause_id", "metric_id", "year"]
coverage = df.groupby(["measure_name", "metric_name"]).agg(
    rows=("val", "size"), first_year=("year", "min"), last_year=("year", "max"),
    locations=("location_id", "nunique"), ages=("age_id", "nunique"), sexes=("sex_id", "nunique")
).reset_index()
coverage["expected_rows"] = (coverage.last_year - coverage.first_year + 1) * coverage.locations * coverage.ages * coverage.sexes
assert (coverage.rows == coverage.expected_rows).all()
coverage.to_csv(ROOT / "coverage.csv", index=False)
df["outcome"] = df.measure_name.map(MEASURES)
df["age_start"] = df.age_name.str.extract(r"^(\d+)").astype(int)
df["age_group"] = df.age_name.str.replace(" years", "", regex=False)
pair_keys = ["location_id", "location_name", "sex_name", "age_id", "age_group", "age_start", "year", "outcome"]
paired = df.pivot(index=pair_keys, columns="metric_name", values="val").reset_index()
paired["implied_population"] = paired.Number / paired.Rate * 100000
pop_keys = ["location_id", "location_name", "sex_name", "age_id", "age_group", "age_start", "year"]
pop_check = paired.groupby(pop_keys).implied_population.agg(["min", "max", "median"]).reset_index()
pop_check["relative_range"] = (pop_check["max"] - pop_check["min"]) / pop_check["median"]
pop_check.to_csv(ROOT / "implied_population_consistency.csv", index=False)
prev = paired[paired.outcome.eq("prevalence")].copy()
un = pd.read_csv(REPO / "supporting_data/2026-09-26/processed/un_wpp2024_gcc_45plus_gbd_age_bands.csv")
joined = prev.merge(un, left_on=["location_name", "sex_name", "age_group", "year"],
                    right_on=["location", "sex", "age_group", "year"], how="left", validate="one_to_one")
matched = joined[joined.population_persons.notna()].copy()
matched["un_over_implied_gbd_population"] = matched.population_persons / matched.implied_population
matched["un_based_prevalent_persons"] = matched.Rate * matched.population_persons / 100000
pop_comparison = matched.groupby(["location_name", "sex_name", "year"]).agg(
    gbd_prevalent_persons_45plus=("Number", "sum"),
    un_based_prevalent_persons_45plus=("un_based_prevalent_persons", "sum"),
    implied_gbd_population_45plus=("implied_population", "sum"),
    un_population_45plus=("population_persons", "sum"),
    min_age_band_population_ratio=("un_over_implied_gbd_population", "min"),
    max_age_band_population_ratio=("un_over_implied_gbd_population", "max"),
).reset_index()
pop_comparison["un_based_to_native_gbd_case_ratio"] = pop_comparison.un_based_prevalent_persons_45plus / pop_comparison.gbd_prevalent_persons_45plus
pop_comparison.to_csv(ROOT / "un_vs_gbd_population_comparison.csv", index=False)
rates = df[df.metric_name.eq("Rate")].pivot(index=["location_name", "year", "age_start", "age_group", "outcome"], columns="sex_name", values="val").reset_index()
rates["male_female_rate_ratio"] = rates.Male / rates.Female
rates.to_csv(ROOT / "age_specific_sex_rate_ratios.csv", index=False)
counts = df[df.metric_name.eq("Number")].copy()
counts["age_band"] = np.select([counts.age_start < 65, counts.age_start < 80], ["45-64", "65-79"], default="80+")
summaries = counts.groupby(["location_name", "year", "sex_name", "outcome", "age_band"], as_index=False).val.sum()
summaries.to_csv(ROOT / "age_group_counts.csv", index=False)
totals = counts.groupby(["location_name", "year", "sex_name", "outcome"], as_index=False).val.sum()
old = pd.read_csv(REPO / "age_standard/parkinsons_ml_ready_wide.csv")
comparisons = []
for (outcome, sex), sub in totals.groupby(["outcome", "sex_name"]):
    column = f"{outcome}_count_{sex.lower()}"
    merged = sub.merge(old[["location_name", "year", column]], on=["location_name", "year"], how="inner", validate="one_to_one")
    merged["previous_all_age_count"] = merged[column]
    merged["new_45plus_count"] = merged.val
    merged["fraction_of_previous_all_age_count"] = merged.val / merged[column]
    comparisons.append(merged.drop(columns=[column, "val"]))
comparison = pd.concat(comparisons, ignore_index=True)
comparison.to_csv(ROOT / "new_45plus_vs_previous_all_age_counts.csv", index=False)
identity = df.pivot(index=["location_id", "sex_id", "age_id", "year", "metric_name"], columns="outcome", values="val").dropna(subset=["dalys", "ylds", "ylls"])
identity_error = (identity.dalys - identity.ylds - identity.ylls).abs()
ui = df[df.metric_name.eq("Rate")].copy()
ui["relative_ui_width"] = (ui.upper - ui.lower) / ui.val
ui[ui.year.eq(2023)].groupby(["location_name", "sex_name", "outcome"]).relative_ui_width.agg(["min", "median", "max"]).reset_index().to_csv(ROOT / "relative_ui_width_2023.csv", index=False)
saudi_counts = summaries[(summaries.location_name == "Saudi Arabia") & (summaries.year == 2023) & summaries.outcome.isin(["prevalence", "incidence"])]
saudi_rates = rates[(rates.location_name == "Saudi Arabia") & (rates.year == 2023) & rates.outcome.isin(["prevalence", "incidence", "deaths"])]
saudi_counts.to_csv(ROOT / "saudi_age_counts_2023.csv", index=False)
saudi_rates.to_csv(ROOT / "saudi_age_rates_2023.csv", index=False)
manifest = {
    "audit_date": "2026-09-27", "source_relative_path": str(SOURCE.relative_to(REPO)),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "bytes": SOURCE.stat().st_size,
    "rows": len(df), "original_columns": [c for c in df.columns if c not in ["outcome", "age_start", "age_group"]],
    "locations": sorted(df.location_name.unique().tolist()), "age_groups": df[["age_start", "age_group"]].drop_duplicates().sort_values("age_start").age_group.tolist(),
    "sexes": sorted(df.sex_name.unique().tolist()), "missing_values": int(df.isna().sum().sum()),
    "duplicate_keys": int(df.duplicated(keys).sum()), "nonfinite_values": int((~np.isfinite(df[["val", "lower", "upper"]])).sum().sum()),
    "nonpositive_values": int((df.val <= 0).sum()), "ui_order_failures": int(((df.lower > df.val) | (df.val > df.upper)).sum()),
    "coverage": coverage.to_dict(orient="records"), "maximum_relative_implied_population_range_across_outcomes": float(pop_check.relative_range.max()),
    "maximum_absolute_daly_identity_difference": float(identity_error.max()),
    "un_matched_prevalence_cells": len(matched), "un_unmatched_locations": sorted(joined.loc[joined.population_persons.isna(), "location_name"].unique().tolist()),
    "source_citation_verbatim": SOURCE.with_name("citation.txt").read_text().strip(),
    "notes": ["Descriptive feasibility audit, not forecasting or independent validation.", "Implied denominators are calculated from point Number/Rate pairs, not independent population observations.", "Age-stratum bounds must not be summed and labeled an aggregate 95% UI."]
}
(ROOT / "audit_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in manifest.items() if k not in ["coverage", "original_columns", "source_citation_verbatim"]}, indent=2))
print("SAUDI_2023_COUNTS_BY_AGE", saudi_counts.to_string(index=False))
print("SAUDI_2023_RATES", saudi_rates.to_string(index=False))
print("POPULATION_SOURCE_COMPARISON_2023", pop_comparison[pop_comparison.year.eq(2023)].to_string(index=False))
print("SAUDI_NEW_VS_OLD_2023", comparison[(comparison.location_name == "Saudi Arabia") & (comparison.year == 2023)].to_string(index=False))
