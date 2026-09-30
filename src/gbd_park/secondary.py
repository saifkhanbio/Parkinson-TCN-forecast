"""Transparent target/outcome adapters for the locked regional experiment.

Existing immutable model implementations use an internal ``prevalence``
label. This module selects the actual outcome before applying that label and
restores the actual outcome on every public prediction and score table.
"""

import copy
import hashlib
import json

import numpy as np
import pandas as pd

from gbd_park.evaluation import (champion_forecasts, matched_tcn_harm,
                                 origin_population_weights, primary_contrasts, summarize_scores)
from gbd_park.local import settings_grid as local_grid
from gbd_park.pooled import settings_grid as nonneural_grid
from gbd_park.scoring import score_forecasts, select_family, select_settings


def secondary_tasks(config):
    """The eleven prespecified new GCC/outcome experiments, Saudi prevalence excluded."""
    tasks = []
    for country in config["countries"]:
        if not country["gcc"]:
            continue
        for outcome in ["prevalence", "incidence"]:
            if country["name"] == config["primary_target"] and outcome == "prevalence":
                continue
            tasks.append({"id": f"{country['iso3']}_{outcome}", "target": country["name"], "outcome": outcome})
    return tasks


def context_config(config, target, donor_countries=None):
    copied = copy.deepcopy(config)
    names = {country["name"] for country in config["countries"]}
    if target not in names:
        raise ValueError("Unknown regional target")
    copied["primary_target"] = target
    if donor_countries is not None:
        donors = list(donor_countries)
        if not donors or len(set(donors)) != len(donors) or target in donors or not set(donors).issubset(names):
            raise ValueError("Donors must be distinct known countries excluding the target")
        retained = set(donors) | {target}
        copied["countries"] = [country for country in copied["countries"] if country["name"] in retained]
    return copied


def working_context(panel, config, target, outcome, donor_countries=None, origin=None):
    """Select actual outcome/countries/cutoff, then relabel only a copied working panel."""
    if outcome not in {"prevalence", "incidence"}:
        raise ValueError("The core secondary experiment supports prevalence and incidence")
    copied = context_config(config, target, donor_countries)
    countries = [country["name"] for country in copied["countries"]]
    select = panel.outcome.eq(outcome) & panel.location_name.isin(countries)
    if origin is not None:
        select &= panel.year.le(origin)
    working = panel.loc[select].copy()
    if working.empty:
        raise ValueError("Actual outcome panel is empty")
    working["source_outcome"] = outcome
    working["outcome"] = "prevalence"
    return working, copied


def restore_outcome(frame, outcome):
    """Restore the actual estimand on a copied public table."""
    copied = frame.copy()
    if "outcome" in copied:
        present = set(copied.outcome.dropna().unique())
        if not present.issubset({"prevalence", outcome}):
            raise ValueError("Unexpected mixed outcomes during restoration")
        copied["outcome"] = outcome
    return copied


def internal_outcome(frame, outcome):
    copied = frame.copy()
    if "outcome" in copied:
        if copied.outcome.isna().any() or not copied.outcome.eq(outcome).all():
            raise ValueError("Public table contains a different actual outcome")
        copied["outcome"] = "prevalence"
    return copied


def actual_truth(panel, target, outcome, maximum_year):
    selected = panel.loc[panel.location_name.eq(target) & panel.outcome.eq(outcome)
                         & panel.year.le(maximum_year)].copy()
    return selected.rename(columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})


def score_actual(predictions, panel, config, target, outcome, maximum_year):
    if not predictions.target.eq(target).all() or not predictions.outcome.eq(outcome).all():
        raise ValueError("Prediction estimand does not match its verification source")
    return score_forecasts(predictions, actual_truth(panel, target, outcome, maximum_year),
                           config["ages"], config["calendar"]["horizons"], maximum_year)


def baseline_choices(scores, config, origins, group):
    if group not in {"local", "nonneural"}:
        raise ValueError("Unknown baseline family group")
    grid = local_grid(config) if group == "local" else nonneural_grid(config)
    rows = []
    for origin in origins:
        inner = list(range(config["calendar"]["inner_first_origin"], origin-4))
        for sex in config["sexes"]:
            for family in config["models"][group + "_order"]:
                eligible = scores.loc[scores.sex.eq(sex) & scores.family.eq(family)
                                      & scores.origin.isin(inner) & scores.horizon.eq(5)]
                settings = {item["setting_id"] for item in grid if item["family"] == family}
                expected = {(past, age) for past in inner for age in config["ages"]}
                if set(eligible.setting_id) != settings:
                    raise ValueError("Incomplete historical candidate settings")
                if (not np.isfinite(eligible[["absolute_log_error", "parameter_count", "grid_order"]]).all().all()
                        or eligible.absolute_log_error.lt(0).any()):
                    raise ValueError("Invalid eligible historical selection evidence")
                for _, part in eligible.groupby("setting_id"):
                    if len(part) != len(expected) or set(zip(part.origin, part.age)) != expected:
                        raise ValueError("Incomplete historical candidate age/origin grid")
                ident, loss, used = select_settings(scores, config, origin, sex, family)
                if used != inner:
                    raise ValueError("Unexpected inner selection origins")
                rows.append({"origin": origin, "sex": sex, "family": family, "setting_id": ident,
                             "inner_loss": loss, "inner_origins": "|".join(map(str, inner)),
                             "last_inner_label_year": max(inner)+5,
                             "selection_status": "frozen_completed_blocks"})
    return pd.DataFrame(rows)


def family_mappings(development_scores, config, origins):
    mappings = []
    for origin in origins:
        for group in ["local", "nonneural"]:
            comparison_config = copy.deepcopy(config)
            comparison_config["models"]["local_order"] = config["models"][group + "_order"]
            selected = development_scores.loc[development_scores.family.isin(config["models"][group + "_order"])]
            for sex in config["sexes"]:
                allowed = [o for o in config["calendar"]["selection_origins"] if o + 5 <= origin]
                eligible = selected.loc[selected.sex.eq(sex) & selected.origin.isin(allowed)
                                        & selected.horizon.eq(5)]
                if (not np.isfinite(eligible[["absolute_log_error", "parameter_count"]]).all().all()
                        or eligible.absolute_log_error.lt(0).any()
                        or eligible.duplicated(["family", "origin", "age"]).any()):
                    raise ValueError("Invalid eligible family-selection evidence")
                expected = {(past, age) for past in allowed for age in config["ages"]}
                for family in config["models"][group + "_order"]:
                    part = eligible.loc[eligible.family.eq(family)]
                    if len(part) != len(expected) or set(zip(part.origin, part.age)) != expected:
                        raise ValueError("Incomplete family-selection age/origin grid")
                choice = select_family(selected, comparison_config, origin, sex)
                mappings.append({"fit_origin": origin, "role": group + "_champion", "sex": sex,
                                 "source_family": choice["selected_family"],
                                 "last_selection_target_year": choice["last_selection_target_year"],
                                 "selection_loss": choice["selection_loss"],
                                 "selection_origins": choice["selection_origins"]})
    return pd.DataFrame(mappings)


def actual_champions(predictions, mappings, config, outcome):
    return restore_outcome(champion_forecasts(internal_outcome(predictions, outcome), mappings, config), outcome)


def actual_population_weights(panel, config, origins, outcome):
    # Both outcomes use the same origin-year prevalence-implied denominator.
    return restore_outcome(origin_population_weights(panel, config, origins), outcome)


def actual_evaluation(scored, weights, config, outcome):
    internal = internal_outcome(scored, outcome)
    tables = {name: restore_outcome(frame, outcome) for name, frame in
              summarize_scores(internal, internal_outcome(weights, outcome), config).items()}
    contrasts, ages, verdict = primary_contrasts(internal, config)
    verdict.update(target=config["primary_target"], outcome=outcome,
                   endpoint_role="prespecified_secondary_replication",
                   saudi_prevalence_primary_result_replaced=False)
    tables["endpoint_contrasts"] = restore_outcome(contrasts, outcome)
    tables["endpoint_age_contrasts"] = restore_outcome(ages, outcome)
    harm, summary = matched_tcn_harm(internal, config)
    tables["matched_adaptation_cells"] = restore_outcome(harm, outcome)
    tables["matched_adaptation_summary"] = restore_outcome(summary, outcome)
    return tables, verdict


def job_fingerprint(job):
    """Deterministic identity including target, actual outcome, donors, and device."""
    return hashlib.sha256(json.dumps(job, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
