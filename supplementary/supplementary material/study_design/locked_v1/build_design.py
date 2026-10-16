"""Build and validate the frozen design inputs, without fitting forecast models."""

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data/processed/design_v1"
CONFIG = json.loads((HERE / "design.json").read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def history_at(panel, origin):
    """The minimum time guard; future model tests must also exercise fitted code."""
    return panel.loc[panel.year.le(origin)].copy()


def write_csv(table, name):
    table.to_csv(OUT / name, index=False, float_format="%.15g")


def build_panel():
    source = ROOT / CONFIG["regional_source"]
    data = pd.read_csv(source)
    assert len(data) == 68992
    assert set(data.population_group_name) == {"All Population"}
    assert data.cause_id.nunique() == 1
    assert set(data.cause_id) == {544}
    assert set(data.measure_name) == set(CONFIG["outcomes"])
    assert set(data.metric_name) == {"Rate", "Number"}
    assert set(data.sex_name) == set(CONFIG["sexes"])
    countries = pd.DataFrame(CONFIG["countries"])
    actual = data[["location_id", "location_name"]].drop_duplicates().sort_values("location_id")
    expected = countries[["gbd_id", "name"]].rename(columns={"gbd_id": "location_id", "name": "location_name"}).sort_values("location_id")
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    data["age"] = data.age_name.str.replace(" years", "", regex=False)
    assert set(data.age) == set(CONFIG["ages"])
    data["age_start"] = data.age.str.extract(r"^(\d+)").astype(int)
    data["outcome"] = data.measure_name.map(CONFIG["outcomes"])
    data["sex"] = data.sex_name
    keys = ["location_id", "location_name", "sex", "age", "age_start", "outcome", "year"]
    assert not data.duplicated(keys + ["metric_name"]).any()
    assert np.isfinite(data[["val", "lower", "upper"]].to_numpy()).all()
    assert data.val.gt(0).all() and data.lower.ge(0).all()
    assert data.lower.le(data.val).all() and data.val.le(data.upper).all()
    pieces = []
    for metric, prefix in [("Rate", "rate"), ("Number", "count")]:
        part = data.loc[data.metric_name.eq(metric), keys + ["val", "lower", "upper"]].copy()
        pieces.append(part.rename(columns={"val": prefix, "lower": prefix + "_lower", "upper": prefix + "_upper"}))
    panel = pieces[0].merge(pieces[1], on=keys, validate="one_to_one")
    assert len(panel) * 2 == len(data)
    for outcome, group in panel.groupby("outcome"):
        first = 1980 if outcome in ["deaths", "ylls"] else 1990
        assert set(group.year) == set(range(first, 2024))
        assert len(group) == 7 * 2 * 11 * (2024 - first)
    panel["implied_population"] = panel["count"] / panel.rate * 100000
    panel["gbd_release"] = CONFIG["gbd_release"]
    panel["rate_unit"] = "per_100000_population"
    panel["source_id"] = "D1_age_specific_export"
    panel["analysis_period"] = np.where(panel.year.ge(1990), "common", "longer_mortality_history")
    panel = panel.sort_values(keys).reset_index(drop=True)
    pop_keys = ["location_id", "sex", "age", "year"]
    populations = panel.groupby(pop_keys).implied_population.agg(["min", "max", "mean"])
    spread = (populations["max"] - populations["min"]) / populations["mean"]
    assert spread.max() < 1e-9
    common = panel[panel.year.ge(1990)]
    counts = common.pivot(index=pop_keys, columns="outcome", values="count")
    assert np.allclose(counts.dalys, counts.ylds + counts.ylls, rtol=1e-12, atol=1e-9)
    write_csv(panel, "regional_outcomes.csv")
    write_csv(countries, "country_crosswalk.csv")
    return panel, float(spread.max())


def build_population(panel):
    columns = ["LocID", "Time", "AgeGrpStart", "Variant", "PopMale", "PopFemale"]
    crosswalk = pd.DataFrame(CONFIG["countries"])
    pieces = []
    for chunk in pd.read_csv(ROOT / CONFIG["population_source"], usecols=columns, chunksize=250000):
        selected = chunk.loc[
            chunk.LocID.isin(crosswalk.un_id)
            & chunk.Time.between(1990, 2028)
            & chunk.AgeGrpStart.ge(45)
            & chunk.Variant.eq("Medium")
        ].copy()
        if not selected.empty:
            pieces.append(selected)
    raw = pd.concat(pieces, ignore_index=True)
    assert not raw.duplicated(["LocID", "Time", "AgeGrpStart"]).any()
    raw["age_start"] = raw.AgeGrpStart.clip(upper=95).astype(int)
    grouped = raw.groupby(["LocID", "Time", "age_start"], as_index=False)[["PopMale", "PopFemale"]].sum()
    population = grouped.melt(id_vars=["LocID", "Time", "age_start"], var_name="sex", value_name="population_thousands")
    population["sex"] = population.sex.map({"PopMale": "Male", "PopFemale": "Female"})
    population["population_persons"] = population.population_thousands * 1000
    population["age"] = population.age_start.map(lambda a: "95+" if a == 95 else f"{a}-{a+4}")
    population = population.merge(crosswalk, left_on="LocID", right_on="un_id", validate="many_to_one")
    population = population.rename(columns={"Time": "year", "gbd_id": "location_id", "name": "location_name"})
    population["period_type"] = np.where(population.year.le(2023), "estimate", "projection")
    population["source_id"] = "D2_UN_WPP2024_medium"
    population = population[["location_id", "location_name", "un_id", "iso3", "year", "sex", "age", "age_start", "population_thousands", "population_persons", "period_type", "source_id"]]
    keys = ["location_id", "sex", "age", "year"]
    assert len(population) == 7 * 2 * 11 * 39
    assert not population.duplicated(keys).any()
    assert population.population_persons.gt(0).all()
    core = panel[panel.outcome.eq("prevalence")]
    joined = core.merge(population[keys + ["population_persons"]], on=keys, how="left", validate="one_to_one")
    assert joined.population_persons.notna().all()
    write_csv(population.sort_values(keys), "un_population_1990_2028.csv")
    return population


def build_calendar():
    cal = CONFIG["calendar"]
    stages = []
    for origin in cal["selection_origins"] + cal["reliability_origins"] + [cal["projection_origin"]]:
        stage = "historical_selection" if origin in cal["selection_origins"] else "nested_reliability"
        if origin == cal["primary_origin"]:
            stage = "final_and_nested_reliability"
        if origin == cal["projection_origin"]:
            stage = "projection"
        stages.append({
            "stage": stage, "origin": origin, "history_start": 1990, "history_end": origin,
            "forecast_start": origin + 1, "forecast_end": origin + 5,
            "first_training_window_origin": 1997, "last_training_window_origin": origin - 5,
            "training_windows_per_age_sex": origin - 2001,
            "inner_origins": "|".join(map(str, range(2003, origin - 4))),
            "family_selection_origins": "|".join(str(o) for o in cal["selection_origins"] if o + 5 <= origin),
            "observed_scoring_available": origin != 2023,
        })
    splits = pd.DataFrame(stages)
    assert splits.origin.is_unique
    write_csv(splits, "evaluation_calendar.csv")
    windows = []
    fit_origins = list(range(2003, 2019)) + [2023]
    for origin in fit_origins:
        for u in range(1997, origin - 4):
            windows.append({"fit_origin": origin, "window_origin": u, "input_start": u - 7,
                            "input_end": u, "label_start": u + 1, "label_end": u + 5})
    windows = pd.DataFrame(windows)
    assert windows.label_end.le(windows.fit_origin).all()
    assert windows.input_end.lt(windows.label_start).all()
    assert (windows.input_end - windows.input_start + 1).eq(8).all()
    assert (windows.label_end - windows.label_start + 1).eq(5).all()
    assert len(windows[windows.fit_origin.eq(2018)]) == 17
    assert len(windows[windows.fit_origin.eq(2023)]) == 22
    write_csv(windows, "training_window_index.csv")
    learning = pd.DataFrame([
        {"origin": 2018, "history_years": n, "history_start": 2019 - n,
         "windows_per_age_sex": max(0, n - 12), "included": n >= 13,
         "settings": "fixed_defaults_no_full_history_target_tuning"}
        for n in [10, 15, 20, 29]
    ])
    write_csv(learning, "learning_curve_feasibility.csv")
    return len(windows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / CONFIG[k] for k in ["regional_source", "global_source", "population_source"]]
    sources += sorted((ROOT / "supporting_data/2026-09-26/raw").glob("*"))
    sources += [ROOT / "More data/IHME-GBD_2023_DATA-e22a15d9-1/citation.txt"]
    sources = sorted({p for p in sources if p.is_file()})
    design_files = [HERE / name for name in ["protocol.md", "design.json", "build_design.py"]]
    source_hashes = {relative(p): digest(p) for p in sources}
    design_hashes = {relative(p): digest(p) for p in design_files}
    manifest_path = HERE / "lock_manifest.json"
    if manifest_path.exists():
        locked = json.loads(manifest_path.read_text())
        assert locked["source_sha256"] == source_hashes, "Sources differ from lock; use a new version."
        assert locked["design_sha256"] == design_hashes, "Design differs from lock; use a new version."
    panel, denominator_spread = build_panel()
    population = build_population(panel)
    window_rows = build_calendar()
    past = history_at(panel, 2018)
    perturbed = panel.copy()
    perturbed.loc[perturbed.year.gt(2018), "rate"] *= 100
    pd.testing.assert_frame_equal(past, history_at(perturbed, 2018))
    global_data = pd.read_csv(ROOT / CONFIG["global_source"])
    assert len(global_data) == 6902 and global_data.location_id.nunique() == 203
    assert not global_data.duplicated(["location_id", "year"]).any()
    assert source_hashes == {relative(p): digest(p) for p in sources}, "Raw input changed during preparation."
    validation = {
        "study_id": CONFIG["study_id"], "design_version": CONFIG["version"], "status": "passed",
        "models_fitted": False, "regional_panel_rows": len(panel),
        "common_period_rows": int(panel.year.ge(1990).sum()),
        "prevalence_rows": int(panel.outcome.eq("prevalence").sum()),
        "incidence_rows": int(panel.outcome.eq("incidence").sum()),
        "population_rows": len(population), "countries_with_matched_populations": 7,
        "window_index_rows": window_rows, "windows_per_stratum_at_2018": 17,
        "primary_target_training_windows_at_2018": 17 * 2 * 11,
        "primary_donor_training_windows_at_2018": 17 * 2 * 11 * 6,
        "maximum_relative_denominator_disagreement": denominator_spread,
        "checks": ["source_hashes_preserved", "expected_country_sex_age_outcome_coverage",
                   "unique_keys", "source_bounds_ordered", "positive_finite_points",
                   "daly_component_identity", "population_units_and_age_alignment",
                   "jordan_population_match", "completed_training_labels_only",
                   "history_accessor_future_perturbation_invariance", "global_panel_scope"],
        "not_yet_validated": CONFIG["required_future_model_checks"],
        "note": "The time-accessor check is not a substitute for future-perturbation tests of fitted models.",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    (OUT / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
    outputs = sorted(p for p in OUT.glob("*") if p.is_file())
    manifest = {
        "study_id": CONFIG["study_id"], "version": CONFIG["version"], "lock_date": CONFIG["lock_date"],
        "source_sha256": source_hashes, "design_sha256": design_hashes,
        "output_sha256": {relative(p): digest(p) for p in outputs},
        "status": "locked_locally_not_registered", "models_fitted": False,
    }
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Regenerated outputs differ from lock."
    else:
        with manifest_path.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
