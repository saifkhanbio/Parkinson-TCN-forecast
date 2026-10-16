"""Supporting mortality/disability adapters for the frozen regional experiment.

The immutable models use a prevalence channel internally. Selection of the
actual outcome and common historical period always precedes that alias.
"""

from gbd_park.secondary import (
    actual_evaluation as secondary_evaluation,
    context_config,
    internal_outcome,
    restore_outcome,
)


OUTCOMES = ("deaths", "ylds", "ylls", "dalys")
ROLE = "prespecified_supporting_mortality_disability"


def supporting_tasks(config):
    """Four separately fitted outcomes for each of the six GCC targets."""
    return [
        {"id": f"{country['iso3']}_{outcome}", "target": country["name"],
         "outcome": outcome}
        for country in config["countries"] if country["gcc"]
        for outcome in OUTCOMES
    ]


def working_context(panel, config, target, outcome, donor_countries=None, origin=None):
    """Copy the actual outcome, countries and 1990+ origin-restricted history."""
    if outcome not in OUTCOMES:
        raise ValueError("Supporting fits require deaths, ylds, ylls, or dalys")
    copied = context_config(config, target, donor_countries)
    countries = [country["name"] for country in copied["countries"]]
    start = int(copied["calendar"]["history_start"])
    if start != 1990:
        raise ValueError("Supporting comparison requires the locked 1990 history start")
    cutoff = int(copied["calendar"]["history_end"]) if origin is None else int(origin)
    if cutoff < start or cutoff > int(copied["calendar"]["history_end"]):
        raise ValueError("Origin lies outside the locked common history")
    selected = (
        panel.outcome.eq(outcome)
        & panel.location_name.isin(countries)
        & panel.year.between(start, cutoff)
        & panel.sex.isin(copied["sexes"])
        & panel.age.isin(copied["ages"])
    )
    working = panel.loc[selected].copy()
    if working.empty:
        raise ValueError("Actual supporting outcome panel is empty")
    working["source_outcome"] = outcome
    working["outcome"] = "prevalence"
    return working, copied


def actual_evaluation(scored, weights, config, outcome):
    """Reuse numerical scoring while explicitly labeling supporting endpoints."""
    if outcome not in OUTCOMES:
        raise ValueError("Unknown supporting outcome")
    tables, verdict = secondary_evaluation(scored, weights, config, outcome)
    for old, new in [("primary_by_family", "endpoint_by_family"),
                     ("primary_age_scores", "endpoint_age_scores")]:
        tables[new] = tables.pop(old)
    verdict["endpoint_role"] = ROLE
    verdict["all_four_endpoint_comparisons_favor_tcn"] = verdict.pop("joint_success")
    verdict["both_comparators_favor_tcn_by_sex"] = verdict.pop("success_by_sex")
    verdict["multiplicity_adjusted_confirmatory_claim"] = False
    return tables, verdict
