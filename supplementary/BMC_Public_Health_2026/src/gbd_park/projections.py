"""Cutoff-2023 regional projections and fixed-population joint transformations."""

import numpy as np
import pandas as pd

from gbd_park.secondary import context_config
from gbd_park.demography import rate_ratio_views
from gbd_park.population_sensitivity import aggregate, conditional_intervals, CONTEXT, NODE

OUTCOMES = ("prevalence", "incidence")
ROLES = ("tcn_adapted", "local_champion", "nonneural_champion")
SCENARIOS = ("un_medium_unaligned", "gbd_2023_aligned_un_growth")
CUTOFF = 2023


def projection_tasks(config):
    return [{"id": f"{country['iso3']}_{outcome}", "target": country["name"], "outcome": outcome}
            for country in config["countries"] if country["gcc"] for outcome in OUTCOMES]


def underlying_families(config):
    return config["models"]["local_order"] + config["models"]["nonneural_order"] + [
        "tcn_adapted", "tcn_unadapted", "tcn_intercept"]


def working_context(panel, config, target, outcome, origin):
    """Origin-restricted actual observations; public future values cannot enter fits."""
    if outcome not in OUTCOMES or not 1990 <= origin <= CUTOFF:
        raise ValueError("Projection fitting requires prevalence/incidence and a 1990–2023 origin")
    copied = context_config(config, target)
    if copied["calendar"]["history_start"] != 1990 or copied["calendar"]["history_end"] != CUTOFF:
        raise ValueError("Projection history calendar must remain locked")
    selected = panel.loc[panel.outcome.eq(outcome) & panel.year.between(1990, origin)
                         & panel.location_name.isin([c["name"] for c in copied["countries"]])
                         & panel.sex.isin(copied["sexes"]) & panel.age.isin(copied["ages"])].copy()
    if selected.empty:
        raise ValueError("Empty projection fit history")
    selected["source_outcome"] = outcome
    selected["outcome"] = "prevalence"
    return selected, copied


def validate_rate_grid(frame, config, target, outcome, origins, families):
    keys = ["origin", "family", "sex", "age", "horizon"]
    expected = pd.MultiIndex.from_product([origins, families, config["sexes"], config["ages"],
                                          config["calendar"]["horizons"]], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if actual.has_duplicates or len(actual) != len(expected) or not expected.isin(actual).all():
        raise ValueError("Incomplete or duplicate projection/historical rate grid")
    if not frame.target.eq(target).all() or not frame.outcome.eq(outcome).all():
        raise ValueError("Unexpected projection target/outcome")
    if not frame.forecast_year.eq(frame.origin+frame.horizon).all():
        raise ValueError("Incorrect projection horizon/year relationship")
    values = frame[["prediction", "log_prediction"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or frame.prediction.le(0).any():
        raise ValueError("Invalid projection rates")
    np.testing.assert_allclose(np.log(frame.prediction), frame.log_prediction, rtol=1e-12, atol=1e-10)


def historical_ledger(early, issued, config, target, outcome):
    """Use original issued settings, never the new projection candidate refits."""
    families = underlying_families(config)
    fields = ["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year", "family",
              "setting_id", "prediction", "log_prediction", "parameter_count", "status"]
    first = early.loc[early.origin.between(2003, 2013) & early.family.isin(families), fields].copy()
    later = issued.loc[issued.origin.between(2014, 2018) & issued.family.isin(families), fields].copy()
    first["history_source"] = "original_prequential_2003_2013"
    later["history_source"] = "original_issued_2014_2018"
    combined = pd.concat([first, later], ignore_index=True)
    validate_rate_grid(combined, config, target, outcome, list(range(2003, 2019)), families)
    if combined.forecast_year.gt(CUTOFF).any():
        raise ValueError("Historical residual source exceeds the projection cutoff")
    return combined


def champion_views(points, mapping, config, target, outcome):
    """Copy the frozen selected family at 2023 without inventing future truth."""
    validate_rate_grid(points, config, target, outcome, [CUTOFF], underlying_families(config))
    if "observed_rate" in points:
        raise ValueError("Verification observations cannot enter projection ledgers")
    selected = mapping.loc[mapping.fit_origin.eq(CUTOFF)].copy()
    expected = {(role, sex) for role in ROLES[1:] for sex in config["sexes"]}
    if (len(selected) != 4 or selected.duplicated(["role", "sex"]).any()
            or set(zip(selected.role, selected.sex)) != expected):
        raise ValueError("Projection champions require both sexes and both roles")
    if not selected.last_selection_target_year.eq(2018).all():
        raise ValueError("Projection family selection must retain the 2009–2013 development origins")
    if not selected.selection_origins.eq("2009|2010|2011|2012|2013").all():
        raise ValueError("Unexpected projection family-selection origin list")
    frames = []
    for row in selected.itertuples():
        allowed = config["models"]["local_order" if row.role == "local_champion" else "nonneural_order"]
        if row.source_family not in allowed:
            raise ValueError("Champion family is outside its comparator group")
        part = points.loc[points.family.eq(row.source_family) & points.sex.eq(row.sex)].copy()
        if part.setting_id.nunique() != 1:
            raise ValueError("Ambiguous current champion settings")
        part["family"], part["source_family"] = row.role, row.source_family
        part["last_family_selection_target_year"] = row.last_selection_target_year
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def population_scenarios(handoff, config, target, outcome):
    """Verify the prepared handoff formulas and exact ages; do not derive ASR counts."""
    source = handoff.loc[handoff.location_name.eq(target) & handoff.outcome.eq(outcome)].copy()
    source = source.rename(columns={"location_name": "target", "year": "forecast_year"})
    source["horizon"] = source.forecast_year-CUTOFF
    keys = ["scenario", "sex", "age", "horizon"]
    expected = pd.MultiIndex.from_product([SCENARIOS, config["sexes"], config["ages"],
                                          config["calendar"]["horizons"]], names=keys)
    actual = pd.MultiIndex.from_frame(source[keys])
    if actual.has_duplicates or len(actual) != len(expected) or not expected.isin(actual).all():
        raise ValueError("Incomplete, duplicate or unexpected projection population cells")
    fields = ["population", "gbd_baseline", "un_baseline", "un_future", "rate", "count"]
    if not np.isfinite(source[fields]).all().all() or source[fields].le(0).any().any():
        raise ValueError("Population scenarios require positive finite persons and baseline values")
    if not source.origin.eq(CUTOFF).all() or not source.unit.eq("persons").all():
        raise ValueError("Incorrect projection population cutoff or units")
    np.testing.assert_allclose(source.gbd_baseline, source["count"]/source.rate*100000, rtol=1e-12, atol=1e-7)
    expected_population = np.where(source.scenario.eq(SCENARIOS[0]), source.un_future,
                                    source.gbd_baseline*source.un_future/source.un_baseline)
    np.testing.assert_allclose(source.population, expected_population, rtol=1e-12, atol=1e-7)
    for _, group in source.groupby(["sex", "age"]):
        for column in ["gbd_baseline", "un_baseline", "rate", "count"]:
            if group[column].nunique() != 1:
                raise ValueError("Scenario baseline must agree across years and scenarios")
    return source


def _ratio_statistics(frame, config, draw=False):
    ids = ["target", "outcome", "origin", "horizon", "forecast_year", "family", "age"]
    if draw:
        ids += ["residual_origin"]
    result = rate_ratio_views(frame, config, value="rate_draw" if draw else "prediction", identifiers=ids)
    result = result.rename(columns={"male_female_rate_ratio": "value", "age": "age_group"})
    result["scenario"] = "not_applicable_rate_ratio"
    result["measure"], result["unit"], result["sex"] = "sex_rate_ratio", "rate_ratio", "Male/Female"
    result["node"] = "Male_Female__"+result.age_group
    result["hierarchy_level"] = "age_sex_ratio"
    return result[CONTEXT+NODE+["value"]+(["residual_origin"] if draw else [])]


def conditional_statistics(points, draws, populations, config):
    """Whole-block burden/ratio transformations, with fixed scenario populations."""
    points = points.loc[points.family.isin(ROLES)].copy()
    draws = draws.loc[draws.family.isin(ROLES)].copy()
    if "observed_rate" in points or "observed_rate" in draws:
        raise ValueError("Projected distributions must not contain future verification values")
    if points.target.nunique() != 1 or points.outcome.nunique() != 1:
        raise ValueError("Conditional projections require one target and one actual outcome")
    target, outcome = points.target.iloc[0], points.outcome.iloc[0]
    if outcome not in OUTCOMES:
        raise ValueError("Conditional projections require age-specific prevalence or incidence")
    validate_rate_grid(points, config, target, outcome, [CUTOFF], list(ROLES))
    if (len(draws) != 5280 or set(draws.family) != set(ROLES)
            or not draws.target.eq(target).all() or not draws.outcome.eq(outcome).all()
            or not draws.origin.eq(CUTOFF).all()
            or not draws.forecast_year.eq(draws.origin+draws.horizon).all()):
        raise ValueError("Point and draw families, cutoff and actual estimand must match exactly")
    population_keys = ["scenario", "sex", "age", "horizon"]
    expected_population = pd.MultiIndex.from_product(
        [SCENARIOS, config["sexes"], config["ages"], config["calendar"]["horizons"]], names=population_keys)
    actual_population = pd.MultiIndex.from_frame(populations[population_keys])
    if (actual_population.has_duplicates or len(actual_population) != len(expected_population)
            or not expected_population.isin(actual_population).all()
            or not populations.target.eq(target).all() or not populations.outcome.eq(outcome).all()
            or not populations.origin.eq(CUTOFF).all()
            or not populations.forecast_year.eq(CUTOFF+populations.horizon).all()):
        raise ValueError("Conditional populations must match the complete projection scenario grid")
    expected_blocks = set(range(2003, 2019))
    for (_, family), group in draws.groupby(["origin", "family"]):
        if len(group) != 1760 or set(group.residual_origin) != expected_blocks:
            raise ValueError("Projection transformations require sixteen intact rate blocks")
        for _, block in group.groupby("residual_origin"):
            expected = {(sex, age, h) for sex in config["sexes"] for age in config["ages"]
                        for h in config["calendar"]["horizons"]}
            if len(block) != 110 or set(zip(block.sex, block.age, block.horizon)) != expected:
                raise ValueError("Incomplete projection rate block")
    keys = ["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year"]
    point_statistics, draw_statistics = [_ratio_statistics(points, config)], [_ratio_statistics(draws, config, True)]
    for scenario in SCENARIOS:
        population = populations.loc[populations.scenario.eq(scenario), keys+["population"]]
        if population.duplicated(keys).any():
            raise ValueError("Duplicate projection population input")
        for frame, value, draw in [(points, "prediction", False), (draws, "rate_draw", True)]:
            joined = frame.merge(population, on=keys, how="left", validate="many_to_one")
            if not np.isfinite(joined.population).all() or joined.population.le(0).any():
                raise ValueError("Missing or invalid projection population matches")
            joined["scenario"] = scenario
            joined["count"] = joined[value]*joined.population/100000
            statistics = aggregate(joined, config, CONTEXT+(["residual_origin"] if draw else []))
            (draw_statistics if draw else point_statistics).append(statistics)
    predicted = pd.concat(point_statistics, ignore_index=True)
    transformed = pd.concat(draw_statistics, ignore_index=True)
    intervals = conditional_intervals(transformed)
    intervals.loc[intervals.measure.eq("sex_rate_ratio"), "uncertainty_scope"] = "joint_rate_error_blocks"
    predicted["information_role"] = "projection_from_2023_data_cutoff"
    return predicted, intervals, transformed


def baseline_statistics(panel, populations, config, target, outcome):
    """Native 2023 burden and scenario-consistent baseline burden stay distinct."""
    base = panel.loc[panel.location_name.eq(target) & panel.outcome.eq(outcome) & panel.year.eq(CUTOFF),
                     ["sex", "age", "rate", "count"]].copy()
    expected = {(s, a) for s in config["sexes"] for a in config["ages"]}
    if len(base) != 22 or set(zip(base.sex, base.age)) != expected:
        raise ValueError("Incomplete 2023 source baseline")
    base["target"], base["outcome"] = target, outcome
    base["origin"], base["horizon"], base["forecast_year"] = CUTOFF, 0, CUTOFF
    base["family"], base["scenario"] = "GBD_2023_baseline", "native_gbd_2023"
    outputs = [aggregate(base, config)]
    for scenario, column in zip(SCENARIOS, ["un_baseline", "gbd_baseline"]):
        pop = populations.loc[populations.scenario.eq(scenario) & populations.horizon.eq(1),
                              ["sex", "age", column, "rate", "count"]]
        matched = base.merge(pop, on=["sex", "age"], how="left", validate="one_to_one", suffixes=("", "_handoff"))
        np.testing.assert_allclose(matched.rate, matched.rate_handoff, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(matched["count"], matched.count_handoff, rtol=1e-12, atol=1e-10)
        matched["scenario"], matched["count"] = scenario, matched.rate*matched[column]/100000
        outputs.append(aggregate(matched, config))
    result = pd.concat(outputs, ignore_index=True)
    result["information_role"] = "GBD_2023_rates_with_native_or_scenario_baseline_population"
    return result
