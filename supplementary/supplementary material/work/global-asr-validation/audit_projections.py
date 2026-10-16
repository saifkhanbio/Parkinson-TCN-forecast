"""Independent projection arithmetic/lineage audit, importing no projection code."""
import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from gbd_park.adaptation import fit_adaptation, correction, regularized_location
from gbd_park.local import settings_grid as local_grid
from gbd_park.pooled import settings_grid as nonneural_grid, build_examples, balanced_weights, target_inputs, predict_changes as pooled_predict
from gbd_park.tcn import load_checkpoint, predict_changes, state_fingerprint
from gbd_park.tcn_forecasting import make_forecasts

CONTEXT = ["target", "outcome", "origin", "horizon", "forecast_year", "family", "scenario"]
NODE = ["measure", "node", "sex", "age_group", "unit", "hierarchy_level"]
COORD = ["sex", "age", "horizon"]
RATE_KEY = ["origin", "family"] + COORD
ROLES = ["tcn_adapted", "local_champion", "nonneural_champion"]
SCENARIOS = ["un_medium_unaligned", "gbd_2023_aligned_un_growth"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def checked_hashes(root, hashes):
    for name, expected in hashes.items():
        if sha(Path(root) / name) != expected:
            raise ValueError(f"Changed artifact {name}")


def compare(actual, expected, keys, numeric, text=()):
    left, right = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if left.index.has_duplicates or right.index.has_duplicates or not left.index.equals(right.index):
        raise ValueError(f"Audit keys mismatch: {keys}")
    for column in numeric:
        np.testing.assert_allclose(left[column], right[column], rtol=3e-10, atol=2e-9, equal_nan=False)
    for column in text:
        if not left[column].fillna("").astype(str).equals(right[column].fillna("").astype(str)):
            raise ValueError(f"Audit text mismatch: {column}")


def tasks(config):
    return [{"id": f"{country['iso3']}_{outcome}", "target": country["name"], "outcome": outcome}
            for country in config["countries"] if country["gcc"] for outcome in ["prevalence", "incidence"]]


def paths(task):
    if task["id"] == "SAU_prevalence":
        return {"local": ROOT / "results/local_baselines_v1/candidate_predictions.csv",
                "nonneural": ROOT / "results/nonneural_v1/candidate_predictions.csv",
                "tcn": ROOT / "results/tcn_v1/candidate_predictions.csv",
                "early": ROOT / "results/intervals_v1/prequential_predictions.csv",
                "issued": ROOT / "results/primary_v1/predictions.csv"}
    base = ROOT / "results/secondary_v1/trials" / task["id"]
    return {**{kind: base / f"candidate_{kind}_predictions.csv" for kind in ["local", "nonneural", "tcn"]},
            "early": base / "prequential_predictions.csv", "issued": base / "predictions.csv"}


def attach_truth(points, panel, task):
    source = panel.loc[panel.location_name.eq(task["target"]) & panel.outcome.eq(task["outcome"])
                       & panel.year.le(2023), ["sex", "age", "year", "rate"]].rename(
        columns={"year": "forecast_year", "rate": "observed_rate"})
    if points.forecast_year.gt(2023).any():
        raise ValueError("Future projection outcomes cannot be joined")
    joined = points.merge(source, on=["sex", "age", "forecast_year"], how="left", validate="many_to_one")
    if not np.isfinite(joined.observed_rate).all() or joined.observed_rate.le(0).any():
        raise ValueError("Historical source join incomplete")
    joined["absolute_log_error"] = np.abs(np.log(joined.prediction) - np.log(joined.observed_rate))
    joined["absolute_rate_error"] = np.abs(joined.prediction - joined.observed_rate)
    return joined


def selection_audit(directory, panel, task, config, historical):
    references = paths(task)
    decisions = read(directory / "settings_decisions.csv")
    if (not decisions.origin.eq(2023).all() or not decisions.last_inner_label_year.eq(2023).all()
            or not decisions.inner_origins.eq("|".join(map(str, range(2003, 2019)))).all()):
        raise ValueError("Incorrect settings selection calendar")
    counts = {}
    for kind in ["local", "nonneural", "tcn"]:
        candidate = read(directory / f"historical_candidate_{kind}_predictions.csv")
        marker = json.loads((directory / f"historical_candidate_{kind}_ledger.json").read_text())
        if marker["sha256"] != sha(directory / f"historical_candidate_{kind}_predictions.csv"):
            raise ValueError("Uncommitted candidate ledger")
        old = read(references[kind])
        compare(candidate.loc[candidate.origin.le(2013)], old, RATE_KEY+["setting_id"],
                ["prediction", "log_prediction"], ["target", "outcome"])
        if set(candidate.origin) != set(range(2003, 2019)):
            raise ValueError("Candidate origin extension incomplete")
        fresh = attach_truth(candidate, panel, task)
        compare(read(directory / f"historical_candidate_{kind}_scores.csv"), fresh,
                RATE_KEY+["setting_id"], ["prediction", "observed_rate", "absolute_log_error", "absolute_rate_error"])
        counts[kind] = len(fresh)
        if kind != "tcn":
            grid = local_grid(config) if kind == "local" else nonneural_grid(config)
            for sex in config["sexes"]:
                for family in config["models"][kind+"_order"]:
                    chosen = fresh.loc[fresh.sex.eq(sex) & fresh.family.eq(family) & fresh.horizon.eq(5)]
                    expected_ids = {spec["setting_id"] for spec in grid if spec["family"] == family}
                    if set(chosen.setting_id) != expected_ids:
                        raise ValueError("Candidate setting family incomplete")
                    if not chosen.groupby("setting_id").size().eq(16*11).all():
                        raise ValueError("Candidate age/origin cells incomplete")
                    winners = chosen.groupby("setting_id", as_index=False).agg(
                        loss=("absolute_log_error", "mean"), parameters=("parameter_count", "mean"), order=("grid_order", "first"))
                    winner = winners.sort_values(["loss", "parameters", "order"]).iloc[0]
                    saved = decisions.loc[decisions.sex.eq(sex) & decisions.family.eq(family)].iloc[0]
                    if saved.setting_id != winner.setting_id:
                        raise ValueError("Manual baseline setting selection mismatch")
                    np.testing.assert_allclose(saved.inner_loss, winner.loss, rtol=1e-10, atol=1e-12)
        else:
            saved = json.loads((directory / "tcn_choice.json").read_text())
            spec = config["models"]["tcn"]
            bases = [dict(channels=channels, weight_decay=decay, epochs=epochs)
                     for channels, decay, epochs in itertools.product(spec["channels"], spec["weight_decay"], spec["epochs"])]
            candidates = []
            for index, base in enumerate(bases):
                base_id = "__".join(f"{key}={base[key]}" for key in ["channels", "weight_decay", "epochs"])
                losses, penalties = [], {}
                for sex in config["sexes"]:
                    comparisons = []
                    for pi, penalty in enumerate(config["adaptation"]["penalties"]):
                        part = fresh.loc[fresh.base_id.eq(base_id) & fresh.family.eq("tcn_adapted")
                                         & fresh.sex.eq(sex) & fresh.horizon.eq(5) & fresh.adaptation_penalty.eq(penalty)]
                        if len(part) != 16*11:
                            raise ValueError("TCN settings grid incomplete")
                        comparisons.append((part.absolute_log_error.mean(), pi, penalty))
                    loss, _, penalty = min(comparisons)
                    losses.append(loss); penalties[sex] = penalty
                channels = base["channels"]
                candidates.append((float(np.mean(losses)), 6*channels**2+11*channels+70, index, base, penalties))
            winner = min(candidates, key=lambda item: item[:3])
            if (saved["base"] != winner[3] or saved["penalties"] != winner[4]
                    or saved["inner_origins"] != list(range(2003, 2019)) or saved["last_inner_label_year"] != 2023):
                raise ValueError("Manual neural settings selection mismatch")
            np.testing.assert_allclose(saved["loss"], winner[0], atol=1e-12, rtol=1e-10)
    mapping = read(directory / "champion_family_mappings.csv")
    if (not mapping.fit_origin.eq(2023).all() or not mapping.last_selection_target_year.eq(2018).all()
            or not mapping.selection_origins.eq("2009|2010|2011|2012|2013").all()):
        raise ValueError("Family selection has changed its locked development calendar")
    for sex in config["sexes"]:
        for kind in ["local", "nonneural"]:
            allowed = config["models"][kind+"_order"]
            selected = historical.loc[historical.sex.eq(sex) & historical.origin.between(2009, 2013)
                                      & historical.horizon.eq(5) & historical.family.isin(allowed)]
            aggregate = selected.groupby("family", as_index=False).agg(
                loss=("absolute_log_error", "mean"), parameters=("parameter_count", "mean"))
            aggregate["order"] = aggregate.family.map({family: i for i, family in enumerate(allowed)})
            winner = aggregate.sort_values(["loss", "parameters", "order"]).iloc[0]
            saved = mapping.loc[mapping.sex.eq(sex) & mapping.role.eq(kind+"_champion")].iloc[0]
            if saved.source_family != winner.family:
                raise ValueError("Manual comparator family choice mismatch")
            np.testing.assert_allclose(saved.selection_loss, winner.loss, rtol=1e-10, atol=1e-12)
    return mapping, counts


def manual_burden(cells, config, identifiers, value="count"):
    """Explicit node sums and ratios, without production aggregation helpers."""
    bottom = [(sex, age) for sex in config["sexes"] for age in config["ages"]]
    table = cells.pivot(index=identifiers, columns=["sex", "age"], values=value).reindex(columns=pd.MultiIndex.from_tuples(bottom))
    values = table.to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid manual burden cell grid")
    definitions = [(f"{sex}__{age}", "age_sex", sex, age, [i]) for i, (sex, age) in enumerate(bottom)]
    for sex in config["sexes"]:
        for label, starts in config["age_groups"].items():
            take = [i for i, (s, a) in enumerate(bottom) if s == sex and int(a.split("-")[0].rstrip("+")) in starts]
            definitions.append((sex+"__"+label, "broad_age_sex", sex, label, take))
        definitions.append((sex+"__45+", "sex_total", sex, "45+", [i for i, (s, _) in enumerate(bottom) if s == sex]))
    definitions.append(("Both__45+", "grand_total", "Both", "45+", list(range(22))))
    frames = []
    for node, level, sex, age, indices in definitions:
        frame = table.index.to_frame(index=False)
        frame["node"], frame["hierarchy_level"], frame["sex"], frame["age_group"] = node, level, sex, age
        frame["measure"], frame["unit"], frame["value"] = "count", "modeled_number", values[:, indices].sum(axis=1)
        frames.append(frame)
    for sex in config["sexes"]:
        denominator = values[:, [i for i, (s, _) in enumerate(bottom) if s == sex]].sum(axis=1)
        for threshold in [65, 80]:
            take = [i for i, (s, a) in enumerate(bottom) if s == sex and int(a.split("-")[0].rstrip("+")) >= threshold]
            frame = table.index.to_frame(index=False)
            label = f"{threshold}+_within_45+"
            frame["node"], frame["hierarchy_level"], frame["sex"], frame["age_group"] = sex+"__"+label, "sex_total", sex, label
            frame["measure"], frame["unit"], frame["value"] = "age_share", "percent", 100*values[:, take].sum(axis=1)/denominator
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def manual_ratios(frame, draw=False):
    ids = [key for key in CONTEXT if key != "scenario"] + ["age"] + (["residual_origin"] if draw else [])
    pivot = frame.pivot(index=ids, columns="sex", values="rate_draw" if draw else "prediction")
    result = pivot.index.to_frame(index=False).rename(columns={"age": "age_group"})
    result["value"] = (pivot.Male/pivot.Female).to_numpy()
    result["scenario"], result["measure"], result["unit"] = "not_applicable_rate_ratio", "sex_rate_ratio", "rate_ratio"
    result["sex"], result["node"], result["hierarchy_level"] = "Male/Female", "Male_Female__"+result.age_group, "age_sex_ratio"
    return result


def quantile_table(draws, keys, value="value"):
    records = []
    grouped = draws.groupby(keys, sort=False)[value]
    quantiles = grouped.quantile([.025, .1, .25, .5, .75, .9, .975], interpolation="linear").unstack()
    for level, lo, hi in [(.5, .25, .75), (.8, .1, .9), (.95, .025, .975)]:
        frame = quantiles[[lo, .5, hi]].rename(columns={lo: "lower", .5: "median", hi: "upper"}).reset_index()
        frame["level"] = level
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def checkpoint_audit(out, directory, task, config, panel):
    cfg = deepcopy(config); cfg["primary_target"] = task["target"]
    all_jobs = []
    for marker in sorted((out / "jobs" / task["id"]).glob("*/complete.json")):
        record = json.loads(marker.read_text())
        job = record["job"]
        if (record["job_sha256"] != fingerprint(job) or marker.parent.name != fingerprint(job)
                or record["config_sha256"] != fingerprint(config) or job["target"] != task["target"]
                or job["outcome"] != task["outcome"] or job["origin"] not in [2014, 2015, 2016, 2017, 2018, 2023]):
            raise ValueError("Cached job configuration/identity/calendar mismatch")
        checked_hashes(marker.parent, record["artifact_sha256"])
        result = joblib.load(marker.parent / "result.joblib")
        if result["job"] != job or result["actual_outcome"] != task["outcome"]:
            raise ValueError("Cached result actual estimand mismatch")
        all_jobs.append((job, marker.parent, result))
        if job["kind"] == "tcn":
            audit = result["audit"]
            if (job["device"] != "cpu" or audit["device"] != "cpu" or audit["maximum_label_year"] != job["origin"]
                    or task["target"] in audit["countries"] or len(audit["countries"]) != 6
                    or result["training_meta"].label_end.gt(job["origin"]).any()
                    or result["target_meta"].label_end.gt(job["origin"]).any()):
                raise ValueError("Cached neural source exclusion or calendar mismatch")
    # Link every newly fitted historical candidate to its cached source result;
    # this replay uses the old frozen ensemble primitive, never projection code.
    for kind in ["local", "nonneural", "tcn"]:
        recent_rows = []
        for job, _, result in all_jobs:
            if job["origin"] == 2023 or job["kind"] != kind:
                continue
            rows = make_forecasts([result], cfg, cfg["adaptation"]["penalties"])[0] if kind == "tcn" else result["forecasts"]
            recent_rows.extend(rows)
        recent = pd.DataFrame(recent_rows)
        recent["outcome"] = task["outcome"]
        saved = read(directory / f"historical_candidate_{kind}_predictions.csv")
        compare(saved.loc[saved.origin.ge(2014)], recent, RATE_KEY+["setting_id"],
                ["prediction", "log_prediction"], ["target", "outcome", "status"])
    current = [(job, folder, result) for job, folder, result in all_jobs if job["origin"] == 2023]
    seeds = [(job, folder, result) for job, folder, result in current if job["kind"] == "tcn"]
    if sorted(job["seed"] for job, _, _ in seeds) != sorted(config["models"]["tcn"]["ensemble_seeds"]):
        raise ValueError("Current projection five-seed ensemble incomplete")
    work = panel.loc[panel.outcome.eq(task["outcome"]) & panel.year.between(1990, 2023)].copy()
    work["outcome"] = "prevalence"
    cx, levels, meta = target_inputs(work, cfg, 2023, task["target"])
    donors = [country["name"] for country in config["countries"] if country["name"] != task["target"]]
    x, _, training = build_examples(work, cfg, 2023, donors)
    tx, ty, target_meta = build_examples(work, cfg, 2023, [task["target"]])
    weights = balanced_weights(training)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x-mean)**2, axis=0, weights=weights)
    replays = 0
    for job, folder, payload in seeds:
        if payload["audit"]["status"] == "ok":
            fitted = load_checkpoint(folder / "checkpoint.joblib")
            if state_fingerprint(fitted) != payload["audit"]["fingerprint_before"]:
                raise ValueError("Current projection checkpoint fingerprint changed")
            np.testing.assert_allclose(fitted["scaler"].mean_, mean, atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(fitted["scaler"].var_, variance, atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(predict_changes(fitted, cx), payload["current_changes"], atol=2e-7, rtol=2e-6)
            np.testing.assert_allclose(predict_changes(fitted, tx), payload["target_changes"], atol=2e-7, rtol=2e-6)
            np.testing.assert_allclose(payload["target_y"], ty, atol=1e-13, rtol=1e-13)
            replays += 1
    saved_points = read(directory / "predictions.csv")
    choice = json.loads((directory / "tcn_choice.json").read_text())
    mean_current = np.mean([item[2]["current_changes"] for item in seeds], axis=0)
    mean_target = np.mean([item[2]["target_changes"] for item in seeds], axis=0)
    any_failed = any(item[2]["audit"]["status"] != "ok" for item in seeds)
    neural_rows = []
    for sex in config["sexes"]:
        selected, target_selected = meta.sex.eq(sex).to_numpy(), target_meta.sex.eq(sex).to_numpy()
        penalty = choice["penalties"][sex]
        for family in ["tcn_unadapted", "tcn_adapted", "tcn_intercept"]:
            changes = mean_current[selected].copy()
            if any_failed:
                changes[:] = 0
            elif family == "tcn_adapted":
                changes += correction(fit_adaptation(ty[target_selected], mean_target[target_selected], penalty))
            elif family == "tcn_intercept":
                changes += regularized_location((ty[target_selected]-mean_target[target_selected]).ravel(), penalty)
            for index, row in enumerate(meta.loc[selected].itertuples()):
                for h in range(1, 6):
                    neural_rows.append(dict(origin=2023, family=family, sex=sex, age=row.age, horizon=h,
                                            log_prediction=levels[selected][index]+changes[index, h-1]))
    compare(saved_points.loc[saved_points.family.str.startswith("tcn_")], pd.DataFrame(neural_rows), RATE_KEY, ["log_prediction"])
    for job, folder, result in current:
        if job["kind"] == "tcn":
            continue
        forecast = pd.DataFrame(result["forecasts"])
        for _, point in saved_points.loc[saved_points.family.isin(config["models"]["local_order"] if job["kind"] == "local" else config["models"]["nonneural_order"])].iterrows():
            if job["kind"] == "local" and point.sex != job["sex"]:
                continue
            matching = forecast.loc[forecast.family.eq(point.family) & forecast.sex.eq(point.sex)
                                    & forecast.age.eq(point.age) & forecast.horizon.eq(point.horizon) & forecast.setting_id.eq(point.setting_id)]
            if len(matching) != 1:
                raise ValueError("Current selected forecast cannot be traced to its cached fitted setting")
            np.testing.assert_allclose(matching.prediction.iloc[0], point.prediction, rtol=1e-12, atol=1e-12)
        if job["kind"] == "nonneural":
            for fit in result["fits"]:
                if fit["status"] != "ok":
                    continue
                fitted = joblib.load(folder / "checkpoints" / (fit["model_id"]+".joblib"))
                nx, _, nm = build_examples(work, cfg, 2023, fit["countries"])
                nw = balanced_weights(nm)
                nm_mean = np.average(nx, axis=0, weights=nw)
                np.testing.assert_allclose(fitted["scaler"].mean_, nm_mean, atol=1e-11, rtol=1e-11)
                np.testing.assert_allclose(fitted["scaler"].var_, np.average((nx-nm_mean)**2, axis=0, weights=nw), atol=1e-11, rtol=1e-11)
                predicted = pooled_predict(fitted, cx)
                family = fit["pool"]+"_"+fit["base"]["algorithm"]+("_unadapted" if fit["pool"] == "donor" else "")
                expected = forecast.loc[forecast.model_id.eq(fit["model_id"]) & forecast.family.eq(family)]
                for i, row in meta.iterrows():
                    part = expected.loc[expected.sex.eq(row.sex) & expected.age.eq(row.age)].sort_values("horizon")
                    np.testing.assert_allclose(part.log_prediction, levels[i]+predicted[i], atol=1e-10, rtol=1e-10)
                if fit["pool"] == "donor":
                    historical_prediction = pooled_predict(fitted, tx)
                    for calibration in result["calibrations"]:
                        if calibration["model_id"] != fit["model_id"] or calibration["status"] != "ok":
                            continue
                        sex = calibration["sex"]
                        target_selected = target_meta.sex.eq(sex).to_numpy()
                        fitted_correction = fit_adaptation(ty[target_selected], historical_prediction[target_selected], calibration["penalty"])
                        np.testing.assert_allclose([calibration["b0"], calibration["b1"]],
                                                   [fitted_correction["b0"], fitted_correction["b1"]], atol=1e-9, rtol=1e-8)
                        selected = forecast.loc[forecast.model_id.eq(fit["model_id"])
                                                & forecast.setting_id.eq(calibration["setting_id"]) & forecast.sex.eq(sex)]
                        for i, row in meta.loc[meta.sex.eq(sex)].iterrows():
                            part = selected.loc[selected.age.eq(row.age)].sort_values("horizon")
                            np.testing.assert_allclose(part.log_prediction, levels[i]+predicted[i]+correction(fitted_correction), atol=1e-9, rtol=1e-9)
                replays += 1
    return {"job_identities": len(all_jobs), "checkpoint_replays": replays}


def audit_case(task, config, out):
    directory = out / "trials" / task["id"]
    marker = json.loads((directory / "issued_commit.json").read_text())
    checked_hashes(directory, marker["artifact_sha256"])
    checked_hashes(ROOT, json.loads((directory / "consumed_sources.json").read_text()))
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    families = config["models"]["local_order"]+config["models"]["nonneural_order"]+["tcn_adapted", "tcn_unadapted", "tcn_intercept"]
    reference = paths(task)
    early, later = read(reference["early"]), read(reference["issued"])
    fields = ["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year", "family", "setting_id",
              "prediction", "log_prediction", "parameter_count", "status"]
    historical = pd.concat([early.loc[early.origin.between(2003, 2013) & early.family.isin(families), fields],
                            later.loc[later.origin.between(2014, 2018) & later.family.isin(families), fields]], ignore_index=True)
    if len(historical) != 24640 or historical.duplicated(RATE_KEY).any():
        raise ValueError("Original historical forecast blocks incomplete")
    compare(read(directory / "original_historical_predictions.csv"), historical, RATE_KEY,
            ["prediction", "log_prediction", "parameter_count"], ["setting_id", "status", "target", "outcome"])
    historical = attach_truth(historical, panel, task)
    compare(read(directory / "original_historical_scores.csv"), historical, RATE_KEY,
            ["prediction", "observed_rate", "absolute_log_error", "absolute_rate_error"])
    mapping, candidate_counts = selection_audit(directory, panel, task, config, historical)
    points = read(directory / "predictions.csv")
    all_families = families + ROLES[1:]
    expected = pd.MultiIndex.from_product([[2023], all_families, config["sexes"], config["ages"], range(1, 6)], names=RATE_KEY)
    actual = pd.MultiIndex.from_frame(points[RATE_KEY])
    if (len(actual) != 1760 or actual.has_duplicates or not expected.isin(actual).all() or "observed_rate" in points
            or not points.forecast_year.eq(points.origin+points.horizon).all()
            or not points.target.eq(task["target"]).all() or not points.outcome.eq(task["outcome"]).all()):
        raise ValueError("Projection point ledger incomplete or contains future truth")
    np.testing.assert_allclose(np.log(points.prediction), points.log_prediction, rtol=1e-12, atol=1e-12)
    for row in mapping.itertuples():
        current = points.loc[points.family.eq(row.source_family) & points.sex.eq(row.sex)].copy()
        current["family"] = row.role
        compare(points.loc[points.family.eq(row.role) & points.sex.eq(row.sex)], current, RATE_KEY,
                ["prediction", "log_prediction"], ["setting_id"])
    coord = pd.MultiIndex.from_product([config["sexes"], config["ages"], range(1, 6)], names=COORD)
    history_index = pd.MultiIndex.from_product([range(2003, 2019), config["sexes"], config["ages"], range(1, 6)], names=["origin"]+COORD)
    draw_parts, interval_parts = [], []
    for family in all_families:
        selected = historical.loc[historical.family.eq(family)].copy()
        if family in ROLES[1:]:
            selected = pd.concat([historical.loc[historical.family.eq(row.source_family) & historical.sex.eq(row.sex)]
                                  for row in mapping.loc[mapping.role.eq(family)].itertuples()], ignore_index=True)
        selected = selected.set_index(["origin"]+COORD).reindex(history_index)
        residuals = (np.log(selected.observed_rate)-selected.log_prediction).to_numpy().reshape(16,110)
        centered = residuals-residuals.mean(axis=0)
        bank = joblib.load(directory / "banks" / (family+".joblib"))
        if bank["n_blocks"] != 16 or bank["origins"] != list(range(2003, 2019)) or bank["fit_origin"] != 2023:
            raise ValueError("Projection residual bank history differs")
        np.testing.assert_allclose(bank["raw_residuals"], residuals, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(bank["center"], residuals.mean(axis=0), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(bank["centered_residuals"], centered, rtol=1e-11, atol=1e-12)
        current = points.loc[points.family.eq(family)].set_index(COORD).reindex(coord)
        logs = current.log_prediction.to_numpy()[None,:]+centered
        rate_values = np.exp(logs)
        for i, origin in enumerate(range(2003, 2019)):
            frame = current.reset_index()[["target", "outcome", "origin", "family", "forecast_year"]+COORD].copy()
            frame["residual_origin"], frame["log_draw"], frame["rate_draw"] = origin, logs[i], rate_values[i]
            draw_parts.append(frame)
        for scale, values in [("log_rate", logs), ("rate", rate_values)]:
            for level in [.5,.8,.95]:
                bounds = np.quantile(values, [(1-level)/2,.5,(1+level)/2], axis=0)
                frame = current.reset_index()[RATE_KEY].copy()
                frame["scale"], frame["level"] = scale, level
                frame["lower"], frame["median"], frame["upper"] = bounds
                interval_parts.append(frame)
    draws = pd.concat(draw_parts, ignore_index=True)
    intervals = pd.concat(interval_parts, ignore_index=True)
    compare(read(directory / "joint_draws.csv.gz"), draws, RATE_KEY+["residual_origin"], ["log_draw", "rate_draw"])
    compare(read(directory / "intervals.csv"), intervals, RATE_KEY+["scale", "level"], ["lower", "median", "upper"])
    baseline = panel.loc[panel.location_name.eq(task["target"]) & panel.outcome.eq(task["outcome"])
                         & panel.year.eq(2023), ["sex", "age", "rate", "count"]].copy()
    un = pd.read_csv(ROOT / "data/processed/design_v1/un_population_1990_2028.csv")
    un = un.loc[un.location_name.eq(task["target"]) & un.year.between(2023,2028), ["sex", "age", "year", "population_persons"]]
    raw_pop = baseline.merge(un.loc[un.year.eq(2023),["sex","age","population_persons"]].rename(columns={"population_persons":"un_baseline"}), on=["sex","age"])
    raw_pop = raw_pop.merge(un.loc[un.year.gt(2023)].rename(columns={"year":"forecast_year","population_persons":"un_future"}), on=["sex","age"])
    raw_pop["gbd_baseline"] = raw_pop["count"]/raw_pop.rate*100000
    populations = []
    for scenario in SCENARIOS:
        frame = raw_pop.copy();frame["scenario"] = scenario
        frame["population"] = frame.un_future if scenario == SCENARIOS[0] else frame.gbd_baseline*frame.un_future/frame.un_baseline
        frame["horizon"] = frame.forecast_year-2023
        populations.append(frame)
    populations = pd.concat(populations,ignore_index=True)
    compare(read(directory/"population_scenarios.csv"),populations,["scenario","sex","age","horizon"],
            ["population","gbd_baseline","un_baseline","un_future","rate","count"])
    role_points, role_draws = points.loc[points.family.isin(ROLES)], draws.loc[draws.family.isin(ROLES)]
    burden_points, burden_draws = [manual_ratios(role_points)], [manual_ratios(role_draws,True)]
    for scenario in SCENARIOS:
        pop = populations.loc[populations.scenario.eq(scenario),["sex","age","horizon","population"]]
        for source,value,is_draw in [(role_points,"prediction",False),(role_draws,"rate_draw",True)]:
            frame=source.merge(pop,on=["sex","age","horizon"],validate="many_to_one")
            frame["scenario"],frame["count"] = scenario,frame[value]*frame.population/100000
            aggregate=manual_burden(frame,config,CONTEXT+(["residual_origin"] if is_draw else []))
            (burden_draws if is_draw else burden_points).append(aggregate)
    burden_points,burden_draws=pd.concat(burden_points,ignore_index=True),pd.concat(burden_draws,ignore_index=True)
    compare(read(directory/"burden_predictions.csv"),burden_points,CONTEXT+NODE,["value"])
    compare(read(directory/"burden_joint_statistics.csv.gz"),burden_draws,CONTEXT+NODE+["residual_origin"],["value"])
    compare(read(directory/"burden_intervals.csv"),quantile_table(burden_draws,CONTEXT+NODE),CONTEXT+NODE+["level"],["lower","median","upper"])
    baseline_parts=[]
    for scenario,column in [("native_gbd_2023",None),(SCENARIOS[0],"un_baseline"),(SCENARIOS[1],"gbd_baseline")]:
        frame=baseline.copy()
        if column:
            one=populations.loc[populations.scenario.eq(scenario)&populations.horizon.eq(1),["sex","age",column]]
            frame=frame.merge(one,on=["sex","age"],validate="one_to_one")
            frame["count"]=frame.rate*frame[column]/100000
        frame["target"],frame["outcome"],frame["origin"],frame["horizon"],frame["forecast_year"]=task["target"],task["outcome"],2023,0,2023
        frame["family"],frame["scenario"]="GBD_2023_baseline",scenario
        baseline_parts.append(manual_burden(frame,config,CONTEXT))
    compare(read(directory/"baseline_2023_burden.csv"),pd.concat(baseline_parts,ignore_index=True),CONTEXT+NODE,["value"])
    checkpoint_checks=checkpoint_audit(out,directory,task,config,panel)
    return {"case":task["id"],"passed":True,"historical_rows":len(historical),"candidate_rows":candidate_counts,
            "rate_points":len(points),"rate_draws":len(draws),"rate_intervals":len(intervals),
            "burden_points":len(burden_points),"burden_draw_statistics":len(burden_draws),**checkpoint_checks}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="results/projections_v1")
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args();out=ROOT/args.run
    manifest=json.loads((out/"run_manifest.json").read_text())
    if manifest["status"]!="complete": raise ValueError("Audit requires completed projection run")
    checked_hashes(out,manifest["output_sha256"])
    checked_hashes(ROOT,manifest["identity"]["source_sha256"])
    checked_hashes(ROOT,manifest["code_sha256"])
    config=json.loads((ROOT/"study_design/locked_v1/design.json").read_text())
    cases=tasks(config)
    marker=json.loads((out/"global_issued_commit.json").read_text())
    if set(marker["artifact_sha256"])!={f"trials/{task['id']}/issued_commit.json" for task in cases}:
        raise ValueError("Global projection issuance does not contain exactly twelve cases")
    checked_hashes(out,marker["artifact_sha256"])
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as executor:
        futures={executor.submit(audit_case,task,config,out):task for task in cases}
        for future in as_completed(futures):
            results.append(future.result());print("Independently verified "+futures[future]["id"],flush=True)
    result={"passed":True,"created_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.perf_counter()-started,
            "run_manifest_sha256":sha(out/"run_manifest.json"),"audit_code_sha256":sha(__file__),
            "cases":sorted(results,key=lambda row:row["case"]),"future_verification_used":False,"model_fits":0,
            "imports_new_projection_module_or_runner":False}
    result["frozen_helper_sha256"]={str(path.relative_to(ROOT)):sha(path) for path in [
        ROOT/"src/gbd_park/adaptation.py",ROOT/"src/gbd_park/local.py",ROOT/"src/gbd_park/pooled.py",
        ROOT/"src/gbd_park/tcn.py",ROOT/"src/gbd_park/tcn_forecasting.py"]}
    destination=ROOT/"work/global-asr-validation/projections_independent_audit.json"
    destination.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":True,"cases":len(results),"elapsed_seconds":result["elapsed_seconds"]}),flush=True)


if __name__=="__main__": main()
