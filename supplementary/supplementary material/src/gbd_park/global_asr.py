"""Separate full-age ASR forecast adapters; no regional scientific files change."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from gbd_park.local import forecast_setting, settings_grid as local_grid
from gbd_park.pooled import (balanced_weights, build_examples, forecast_origin,
                             target_inputs)
from gbd_park.tcn import (fit_tcn, predict_changes, save_checkpoint,
                          state_fingerprint)
from gbd_park.tcn_forecasting import make_forecasts

OUTCOMES = ("prevalence", "incidence")
SCOPES = ("global", "regional")
AGE = "Full-age ASR"
ROLE = "prespecified_supporting_separate_global_asr_benchmark"
LOCAL_FAMILIES = ("persistence", "log_trend", "damped_ets", "arima")
LEARNED_FAMILIES = ("pooled_ridge", "pooled_boosting", "donor_ridge_unadapted",
                    "donor_ridge_adapted", "donor_boosting_unadapted",
                    "donor_boosting_adapted", "tcn_unadapted", "tcn_adapted",
                    "tcn_intercept")
ALIASES = {
    "Taiwan": "China, Taiwan Province of China",
    "Democratic People's Republic of Korea": "Dem. People's Republic of Korea",
    "Micronesia (Federated States of)": "Micronesia (Fed. States of)",
    "Palestine": "State of Palestine",
    "Hong Kong Special Administrative Region of China": "China, Hong Kong SAR",
    "Macao Special Administrative Region of China": "China, Macao SAR",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_un_registry(path):
    """Read country/area types from preserved WPP input, not numeric-ID guesses."""
    columns = ["LocID", "ISO3_code", "Location", "LocTypeID", "LocTypeName", "ParentID"]
    parts = [chunk.drop_duplicates() for chunk in pd.read_csv(
        path, usecols=columns, chunksize=100000,
        dtype={"ISO3_code": str, "Location": str, "LocTypeName": str,
               "LocTypeID": str, "ParentID": str, "LocID": str})]
    return pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)


def location_registry(wide, un):
    locs = wide[["location_id", "location_name"]].drop_duplicates().copy()
    if locs.location_id.duplicated().any() or locs.location_name.duplicated().any():
        raise ValueError("GBD IDs and names must have a one-to-one mapping")
    source = un[["LocID", "ISO3_code", "Location", "LocTypeID", "LocTypeName", "ParentID"]].dropna(subset=["Location"]).drop_duplicates()
    locs["un_location_name"] = locs.location_name.map(lambda name: ALIASES.get(name, name))
    source = source.loc[source.Location.isin(locs.un_location_name)]
    if source.Location.duplicated().any():
        raise ValueError("Requested UN location names are ambiguous")
    result = locs.merge(source, left_on="un_location_name", right_on="Location", how="left", validate="one_to_one")
    if (result.ISO3_code.isna().any() or result.ISO3_code.duplicated().any()
            or not result.LocTypeName.eq("Country/Area").all()
            or not result.ISO3_code.str.fullmatch("[A-Z]{3}").all()):
        raise ValueError("Unresolved, aggregate or duplicate-ISO geographic locations")
    for column in ["LocID", "LocTypeID", "ParentID"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    result["match_method"] = np.where(result.location_name.eq(result.un_location_name), "exact_name", "explicit_alias")
    result["geography_role"] = "supplied_GBD_country_or_territory"
    result["boundary_compatibility"] = "UN_identity_match_not_independent_GBD_boundary_verification"
    return result.rename(columns={"LocID": "un_location_id", "ISO3_code": "iso3",
                                  "LocTypeID": "un_location_type_id", "LocTypeName": "un_location_type",
                                  "ParentID": "un_parent_id"}).drop(columns="Location").sort_values("location_id").reset_index(drop=True)


def asr_panel(wide):
    """Validate the full grid and select only explicitly standardized-rate fields."""
    required = ["location_id", "location_name", "year"]
    if wide[required].isna().any().any() or wide.duplicated(["location_id", "year"]).any():
        raise ValueError("Missing or duplicate global source keys")
    for _, group in wide.groupby("location_id"):
        if len(group) != 34 or set(group.year) != set(range(1990, 2024)) or group.location_name.nunique() != 1:
            raise ValueError("Each global location requires every year 1990–2023")
    rows = []
    for outcome in OUTCOMES:
        for sex in ["Male", "Female"]:
            column = f"{outcome}_rate_age_std_{sex.lower()}"
            part = wide[required + [column]].rename(columns={column: "rate"}).copy()
            if not np.isfinite(part.rate).all() or part.rate.le(0).any():
                raise ValueError("ASRs must be finite and positive")
            part["sex"], part["outcome"], part["age"] = sex, outcome, AGE
            part["unit"], part["gbd_release"] = "ASR_per_100000", "GBD 2023 as labeled"
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _xlsx_rows(archive, filename):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    strings = []
    if "xl/sharedStrings.xml" in archive.namelist():
        tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(item.itertext()) for item in tree]
    with archive.open(filename) as stream:
        for event, row in ET.iterparse(stream, events=["end"]):
            if row.tag != ns + "row":
                continue
            result = {}
            for cell in row:
                letter = "".join(c for c in cell.attrib["r"] if c.isalpha())
                index = 0
                for char in letter:
                    index = index * 26 + ord(char) - 64
                kind = cell.attrib.get("t")
                raw = cell.find(ns + "v")
                if kind == "inlineStr":
                    value = "".join(cell.find(ns + "is").itertext())
                elif raw is not None:
                    value = strings[int(raw.text)] if kind == "s" else raw.text
                else:
                    value = ""
                result[index - 1] = value
            yield [result.get(index, "") for index in range(max(result, default=-1) + 1)]
            row.clear()


def replay_workbook(panel, workbook):
    """Compare every chosen CSV ASR with retained workbook raw-sheet ASR values."""
    records = []
    with zipfile.ZipFile(workbook) as archive:
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rels = {item.attrib["Id"]: item.attrib["Target"].lstrip("/")
                for item in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        sheets = ET.fromstring(archive.read("xl/workbook.xml")).find("s:sheets", ns)
        filenames = {}
        for sheet in sheets:
            rel = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            path = rels[rel]
            filenames[sheet.attrib["name"]] = path if path.startswith("xl/") else "xl/" + path
        metadata = dict(row[:2] for row in _xlsx_rows(archive, filenames["README"]) if len(row) >= 2)
        dictionary = dict(row[:2] for row in _xlsx_rows(archive, filenames["Data_Dictionary"]) if len(row) >= 2)
        if ("GBD 2023" not in metadata.get("Source", "") or "544" not in metadata.get("Cause", "")
                or "100,000" not in dictionary.get("rate_age_std_[sex]", "")
                or "GBD world standard" not in dictionary.get("rate_age_std_[sex]", "")):
            raise ValueError("Workbook metadata do not identify required ASR estimand")
        rows = iter(_xlsx_rows(archive, filenames["Raw_Data"]))
        header = next(rows)
        for values in rows:
            row = dict(zip(header, values))
            if row.get("measure") not in OUTCOMES:
                continue
            if row.get("cause_id") != "544" or row.get("age_group") != "all_ages":
                raise ValueError("Unexpected raw workbook disease or age metadata")
            for sex in ["Male", "Female"]:
                records.append({"location_id": int(row["location_id"]), "year": int(row["year"]),
                                "outcome": row["measure"], "sex": sex,
                                "workbook_rate": float(row[f"rate_age_std_{sex.lower()}"])})
    raw = pd.DataFrame(records)
    keys = ["location_id", "year", "outcome", "sex"]
    matched = panel.merge(raw, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not matched._merge.eq("both").all() or not np.allclose(matched.rate, matched.workbook_rate, rtol=0, atol=1e-12):
        raise ValueError("CSV ASRs do not exactly replay from workbook raw ASR fields")
    return {"passed": True, "matched_asr_cells": len(matched), "maximum_absolute_difference": float(
        abs(matched.rate - matched.workbook_rate).max()), "metadata": metadata, "dictionary": dictionary,
        "independent_native_export_verified": False}


def tasks(config):
    return [{"id": f"{country['iso3']}_{outcome}", "target": country["name"], "outcome": outcome}
            for country in config["countries"] if country["gcc"] for outcome in OUTCOMES]


def context(panel, config, registry, target, outcome, scope, origin=2018):
    if outcome not in OUTCOMES or scope not in SCOPES or origin != 2018:
        raise ValueError("Global benchmark requires its fixed outcome, scope and origin")
    cfg = deepcopy(config)
    if scope == "global":
        cfg["countries"] = [{"name": row.location_name, "gbd_id": int(row.location_id),
                             "iso3": row.iso3} for row in registry.itertuples()]
    names = [country["name"] for country in cfg["countries"]]
    if target not in names or len(names) != len(set(names)):
        raise ValueError("Target or source country registry invalid")
    cfg["primary_target"], cfg["ages"] = target, [AGE]
    cfg["donors"]["primary"] = f"{scope}_all_other_supplied_countries_territories"
    cfg["models"]["local_order"] = list(LOCAL_FAMILIES)
    cfg["models"]["log_trend_windows"] = [config["cold_start_defaults"]["trend_years"]]
    cfg["models"]["ridge_penalties"] = [config["cold_start_defaults"]["ridge_penalty"]]
    for key, default in [("depth", "boosting_depth"), ("trees", "boosting_trees"), ("min_leaf", "boosting_min_leaf")]:
        cfg["models"]["boosting"][key] = [config["cold_start_defaults"][default]]
    cfg["adaptation"]["penalties"] = [config["cold_start_defaults"]["adaptation_penalty"]]
    work = panel.loc[panel.location_name.isin(names) & panel.outcome.eq(outcome)
                     & panel.year.between(1990, origin)].copy()
    keys = ["location_name", "sex", "age", "year"]
    expected = pd.MultiIndex.from_product([names, cfg["sexes"], [AGE], range(1990, origin + 1)], names=keys)
    actual = pd.MultiIndex.from_frame(work[keys])
    if actual.has_duplicates or len(actual) != len(expected) or not expected.isin(actual).all():
        raise ValueError("Incomplete pre-origin ASR country/sex/year grid")
    if not np.isfinite(work.rate).all() or work.rate.le(0).any():
        raise ValueError("Pre-origin ASRs must be positive finite values")
    work["source_outcome"], work["outcome"] = outcome, "prevalence"
    return work, cfg


def restore(rows, outcome, scope):
    result = pd.DataFrame(rows).copy()
    result["outcome"], result["donor_scope"] = outcome, scope
    result["analysis_role"], result["unit"] = ROLE, "ASR_per_100000"
    if "donor_strategy" in result:
        result["donor_strategy"] = f"{scope}_all_other_supplied_countries_territories"
    return result


def local_forecasts(work, cfg, outcome):
    rows, fits, arima = [], [], []
    for sex in cfg["sexes"]:
        for spec in local_grid(cfg):
            forecast, fitted, audited = forecast_setting(work, cfg, 2018, sex, spec, cfg["primary_target"])
            rows.extend(forecast); fits.extend(fitted); arima.extend(audited)
    return {"predictions": restore(rows, outcome, "local"), "fits": fits, "arima": arima}


def nonneural_forecasts(work, cfg, outcome, scope, checkpoint_dir):
    forecast, fits, adaptation, metadata = forecast_origin(work, cfg, 2018, checkpoint_dir)
    return {"predictions": restore(forecast, outcome, scope), "fits": fits,
            "adaptation": adaptation, "training_meta": pd.DataFrame(metadata)}


def fixed_base(config):
    defaults = config["cold_start_defaults"]
    return {"channels": defaults["tcn_channels"], "weight_decay": defaults["tcn_weight_decay"],
            "epochs": defaults["tcn_epochs"]}


def fit_seed(work, cfg, seed, device, checkpoint_path=None, base=None):
    """Actual one-ASR TCN, never using the age-specific wrapper's fixed count."""
    started = time.perf_counter()
    base = fixed_base(cfg) if base is None else base
    target = cfg["primary_target"]
    countries = [country["name"] for country in cfg["countries"] if country["name"] != target]
    x, y, meta = build_examples(work, cfg, 2018, countries)
    current_x, levels, current_meta = target_inputs(work, cfg, 2018, target)
    target_x, target_y, target_meta = build_examples(work, cfg, 2018, [target])
    if meta.country.eq(target).any() or meta.label_end.gt(2018).any() or target_meta.label_end.gt(2018).any():
        raise ValueError("Target-exclusion or future-label violation")
    # The one-category ASR representation has 50 fewer head parameters than
    # the original eleven-category regional representation.
    channels = int(base["channels"])
    count = 6 * channels * channels + 11 * channels + 20
    audit = {"origin": 2018, "seed": seed, "base": base, "device": str(device), "status": "ok", "reason": "",
             "countries": countries, "target_in_base_fit": False, "training_windows": len(meta),
             "target_windows": len(target_meta), "maximum_label_year": int(meta.label_end.max()),
             "last_target_label_year": int(target_meta.label_end.max()), "parameter_count": count,
             "minimum_input_year": int(meta.input_start.min())}
    try:
        fitted = fit_tcn(x, y, meta, base, cfg, seed, device)
        before = state_fingerprint(fitted)
        current = predict_changes(fitted, current_x)
        historical = predict_changes(fitted, target_x)
        after = state_fingerprint(fitted)
        if before != after or fitted["parameter_count"] != count:
            raise AssertionError("ASR TCN mutated or violated parameter accounting")
        if checkpoint_path is not None:
            save_checkpoint(fitted, checkpoint_path)
        audit.update(fingerprint_before=before, fingerprint_after=after,
                     feature_mean=fitted["scaler"].mean_.tolist(), feature_scale=fitted["scaler"].scale_.tolist(),
                     training_losses=fitted["training_losses"])
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        current, historical = np.zeros_like(target_y[:len(levels)]), np.zeros_like(target_y)
        audit.update(status="fallback", reason=f"{type(exc).__name__}: {exc}",
                     fingerprint_before="unavailable", fingerprint_after="unavailable")
    audit["elapsed_seconds"] = time.perf_counter() - started
    return {"origin": 2018, "base": base, "seed": seed, "current_changes": current,
            "target_changes": historical, "target_y": target_y, "levels": levels,
            "current_meta": current_meta, "target_meta": target_meta, "audit": audit,
            "training_meta": meta.assign(sample_weight=balanced_weights(meta), fit_origin=2018)}


def ensemble_forecasts(payloads, cfg, outcome, scope):
    if sorted(item["seed"] for item in payloads) != sorted(cfg["models"]["tcn"]["ensemble_seeds"]):
        raise ValueError("Every fixed ensemble seed, including failures, must be retained")
    if len({item["audit"]["device"] for item in payloads}) != 1:
        raise ValueError("TCN ensemble cannot mix devices")
    rows, adaptation = make_forecasts(payloads, cfg, cfg["adaptation"]["penalties"], include_intercept=True)
    result = restore(rows, outcome, scope)
    count = payloads[0]["audit"]["parameter_count"]
    result["parameter_count"] = count + result.family.map({"tcn_unadapted": 0, "tcn_adapted": 2, "tcn_intercept": 1})
    result["fitted_seed_parameters"] = count * len(payloads)
    return result, adaptation


def check_forecasts(frame, config, single_case=False):
    keys = ["target", "outcome", "donor_scope", "family", "sex", "horizon"]
    if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Empty, missing or duplicate ASR forecast coordinates")
    if (not frame.origin.eq(2018).all() or not frame.forecast_year.eq(2018 + frame.horizon).all()
            or not frame.age.eq(AGE).all() or not frame.unit.eq("ASR_per_100000").all()
            or not np.isfinite(frame.prediction).all() or frame.prediction.le(0).any()
            or not np.allclose(frame.log_prediction, np.log(frame.prediction), atol=1e-12, rtol=1e-12)):
        raise ValueError("Invalid ASR forecast values or estimand")
    views = [("local", family) for family in LOCAL_FAMILIES]
    views += [(scope, family) for scope in SCOPES for family in LEARNED_FAMILIES]
    expected = {(scope, family, sex, horizon) for scope, family in views for sex in config["sexes"] for horizon in range(1, 6)}
    for _, group in frame.groupby(["target", "outcome"]):
        if set(map(tuple, group[["donor_scope", "family", "sex", "horizon"]].to_numpy())) != expected:
            raise ValueError("Incomplete ASR case forecast grid")
    actual_cases = set(map(tuple, frame[["target", "outcome"]].drop_duplicates().to_numpy()))
    allowed = {(task["target"], task["outcome"]) for task in tasks(config)}
    if single_case:
        if len(actual_cases) != 1 or not actual_cases.issubset(allowed):
            raise ValueError("Unexpected single ASR case")
    elif actual_cases != allowed:
        raise ValueError("Every GCC/outcome ASR case is required before scoring")


def score_forecasts(points, panel, config):
    check_forecasts(points, config)
    actual = panel[["location_name", "outcome", "sex", "year", "rate"]].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})
    result = points.merge(actual, on=["target", "outcome", "sex", "forecast_year"], how="left", validate="many_to_one")
    if not np.isfinite(result.observed_rate).all() or result.observed_rate.le(0).any():
        raise ValueError("Missing ASR verification outcome")
    result["absolute_log_error"] = np.abs(result.log_prediction - np.log(result.observed_rate))
    result["absolute_rate_error"] = np.abs(result.prediction - result.observed_rate)
    return result
