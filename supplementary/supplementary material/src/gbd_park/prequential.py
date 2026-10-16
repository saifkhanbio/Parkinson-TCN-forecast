"""Select historical forecasts without using the outcomes they predict."""

import numpy as np
import pandas as pd

from gbd_park.local import settings_grid as local_grid
from gbd_park.pooled import settings_grid as nonneural_grid
from gbd_park.scoring import select_settings


def cold_setting(config, family, group):
    """Find the configured cold-start setting within the fixed candidate grid."""
    defaults = config["cold_start_defaults"]
    if group == "local":
        specs = [s for s in local_grid(config) if s["family"] == family]
        if family == "log_trend":
            wanted = {"window": defaults["trend_years"]}
        elif family == "age_smooth_trend":
            wanted = {"window": defaults["age_smooth_years"], "penalty": defaults["age_smooth_penalty"]}
        else:
            wanted = {}
        chosen = [s for s in specs if s["setting"] == wanted]
    elif group == "nonneural":
        specs = [s for s in nonneural_grid(config) if s["family"] == family]
        if "ridge" in family:
            wanted = {"algorithm": "ridge", "alpha": defaults["ridge_penalty"]}
        else:
            wanted = {"algorithm": "boosting", "depth": defaults["boosting_depth"],
                      "trees": defaults["boosting_trees"], "min_leaf": defaults["boosting_min_leaf"]}
        penalty = defaults["adaptation_penalty"] if family.endswith("_adapted") else None
        chosen = [s for s in specs if s["base"] == wanted and s["penalty"] == penalty]
    else:
        raise ValueError("Unknown historical model group")
    if len(chosen) != 1:
        raise ValueError("Cold-start setting is missing or ambiguous")
    return chosen[0]["setting_id"]


def select_prequential(predictions, scores, config, origin, group):
    """One setting per sex/family at an origin, using only completed inner blocks."""
    if group not in {"local", "nonneural"}:
        raise ValueError("Unknown historical model group")
    if "observed_rate" in predictions:
        raise ValueError("Prediction ledger must be separate from verification values")
    if origin < config["calendar"]["inner_first_origin"]:
        raise ValueError("Origin precedes the residual-history start")
    selected, decisions = [], []
    families = config["models"][group + "_order"]
    specs = local_grid(config) if group == "local" else nonneural_grid(config)
    inner = list(range(config["calendar"]["inner_first_origin"], origin - 4))
    expected = {(age, horizon) for age in config["ages"] for horizon in config["calendar"]["horizons"]}
    for sex in config["sexes"]:
        for family in families:
            if inner:
                eligible = scores.loc[scores.sex.eq(sex) & scores.family.eq(family)
                                      & scores.origin.isin(inner) & scores.horizon.eq(5)]
                if set(eligible.setting_id) != {s["setting_id"] for s in specs if s["family"] == family}:
                    raise ValueError("Incomplete historical candidate setting grid")
                ident, loss, used = select_settings(scores, config, origin, sex, family)
                status = "tuned_completed_blocks"
                assert used == inner
            else:
                ident, loss = cold_setting(config, family, group), None
                status = "cold_start_defaults"
            part = predictions.loc[predictions.origin.eq(origin) & predictions.sex.eq(sex)
                                   & predictions.family.eq(family) & predictions.setting_id.eq(ident)].copy()
            if len(part) != len(expected) or set(zip(part.age, part.horizon)) != expected:
                raise ValueError("Incomplete selected historical forecast grid")
            if not part.forecast_year.eq(part.origin + part.horizon).all():
                raise ValueError("Historical forecast year mismatch")
            if not np.isfinite(part.prediction).all() or not part.prediction.gt(0).all():
                raise ValueError("Invalid selected forecast values")
            part["selection_status"] = status
            part["last_inner_label_year"] = max(inner)+5 if inner else np.nan
            selected.append(part)
            decisions.append({"origin": origin, "sex": sex, "family": family, "setting_id": ident,
                              "selection_status": status, "inner_origins": "|".join(map(str, inner)),
                              "last_inner_label_year": max(inner)+5 if inner else None, "inner_loss": loss})
    return pd.concat(selected, ignore_index=True), decisions


def champion_history(scored, config, family_by_sex, name):
    """Join the currently selected sex-specific families by common past origin.

    Within each family, the historical settings remain those selected at that
    historical origin. This function never substitutes the current setting.
    """
    if set(family_by_sex) != set(config["sexes"]):
        raise ValueError("Champion mapping must specify every configured sex")
    if name in set(scored.family):
        raise ValueError("Champion name collides with a source family")
    parts = []
    for sex in config["sexes"]:
        part = scored.loc[scored.sex.eq(sex) & scored.family.eq(family_by_sex[sex])].copy()
        if part.empty:
            raise ValueError("Selected champion family is missing")
        part["source_family"] = part.family
        part["family"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
