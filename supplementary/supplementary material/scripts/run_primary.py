"""Stage 5: frozen Saudi primary and nested reliability rate evaluation."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import torch
from gbd_park.local import forecast_setting, settings_grid as local_grid
from gbd_park.pooled import forecast_origin, settings_grid as nonneural_grid, country_pool
from gbd_park.scoring import score_forecasts, select_settings, select_family
from gbd_park.intervals import apply_bank, score_intervals
from gbd_park.tcn_forecasting import make_forecasts, select_tcn_settings, base_id
from run_tcn import fit_job as neural_fit_job
from run_local_baselines import check_lock, sha, now, json_lines

PRIOR = ["local_baselines_v1", "nonneural_v1", "tcn_v1", "intervals_v1"]


def evaluation_origins(config):
    origins = config["calendar"]["reliability_origins"]
    if len(origins) != len(set(origins)):
        raise ValueError("Repeated reliability origin")
    primary = config["calendar"]["primary_origin"]
    if primary not in origins:
        raise ValueError("Primary origin must occur once in the reliability calendar")
    return sorted(origins)


def select_baseline_settings(scores, config, origins, group):
    if group not in {"local", "nonneural"}:
        raise ValueError("Unknown baseline group")
    specs = local_grid(config) if group == "local" else nonneural_grid(config)
    decisions = []
    for origin in origins:
        inner = list(range(config["calendar"]["inner_first_origin"], origin - 4))
        for sex in config["sexes"]:
            for family in config["models"][group + "_order"]:
                eligible = scores.loc[scores.sex.eq(sex) & scores.family.eq(family)
                                      & scores.origin.isin(inner) & scores.horizon.eq(5)]
                wanted = {s["setting_id"] for s in specs if s["family"] == family}
                if set(eligible.setting_id) != wanted:
                    raise ValueError("Incomplete candidate setting grid")
                if (not np.isfinite(eligible[["absolute_log_error", "parameter_count", "grid_order"]]).all().all()
                        or eligible.absolute_log_error.lt(0).any()):
                    raise ValueError("Invalid eligible historical selection evidence")
                expected = {(o, a) for o in inner for a in config["ages"]}
                for _, cells in eligible.groupby("setting_id"):
                    if len(cells) != len(expected) or set(zip(cells.origin, cells.age)) != expected:
                        raise ValueError("Incomplete candidate age/origin grid")
                ident, loss, used = select_settings(scores, config, origin, sex, family)
                assert used == inner and max(inner) + 5 <= origin
                decisions.append({"origin": origin, "sex": sex, "family": family,
                                  "setting_id": ident, "inner_loss": loss,
                                  "inner_origins": "|".join(map(str, inner)),
                                  "last_inner_label_year": max(inner) + 5,
                                  "selection_status": "frozen_completed_blocks"})
    return pd.DataFrame(decisions)


def validate_forecast_ledger(frame, config, families):
    if "observed_rate" in frame:
        raise ValueError("Verification values in prediction ledger")
    keys = ["origin", "sex", "family", "age", "horizon"]
    expected = {(o, s, f, a, h) for o in evaluation_origins(config)
                for s in config["sexes"] for f in families for a in config["ages"]
                for h in config["calendar"]["horizons"]}
    if len(frame) != len(expected) or set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError("Incomplete or duplicated evaluation forecast grid")
    if not frame.forecast_year.eq(frame.origin + frame.horizon).all():
        raise ValueError("Incorrect forecast year")
    if not frame.target.eq(config["primary_target"]).all() or not frame.outcome.eq("prevalence").all():
        raise ValueError("Unexpected target or outcome")
    if not np.isfinite(frame[["prediction", "log_prediction"]]).all().all() or not frame.prediction.gt(0).all():
        raise ValueError("Invalid forecast values")
    np.testing.assert_allclose(np.log(frame.prediction), frame.log_prediction, atol=1e-13)
    if not frame.last_inner_label_year.le(frame.origin).all():
        raise ValueError("Future labels used in settings selection")


def verify_committed_outputs(out, hashes):
    required = {"predictions.csv", "intervals.csv", "joint_draws.csv"}
    if not required.issubset(hashes):
        raise ValueError("Point and interval ledgers must be committed before scoring")
    for name, expected in hashes.items():
        path = Path(out) / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Committed artifact missing or changed: {name}")


def verify_scoring_order(events):
    names = [row["event"] for row in events]
    required = ["settings_frozen", "forecast_fitting_started", "predictions_committed",
                "intervals_committed", "evaluation_scoring_started"]
    if any(names.count(name) != 1 for name in required):
        raise ValueError("Required evaluation event missing or duplicated")
    indices = [names.index(name) for name in required]
    if indices != sorted(indices):
        raise ValueError("Forecasts and intervals must precede evaluation scoring")


def verify_prior():
    hashes, code, records = {}, {}, {}
    for name in PRIOR:
        directory = ROOT / "results" / name
        record = json.loads((directory / "run_manifest.json").read_text())
        assert record["status"] == "complete" and not record["final_period_scored"]
        for filename, expected in record["output_sha256"].items():
            assert sha(directory / filename) == expected, filename
        for filename, expected in record["code_sha256"].items():
            assert sha(ROOT / filename) == expected, filename
            code[filename] = expected
        hashes[name] = sha(directory / "run_manifest.json")
        records[name] = record
    return hashes, code, records


def local_job(config, origin, sex, specs):
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(origin) & panel.outcome.eq("prevalence")].copy()
    rows, fits, audit = [], [], []
    for spec in specs:
        predicted, fitted, arima = forecast_setting(panel, config, origin, sex, spec,
                                                     target=config["primary_target"])
        rows.extend(predicted)
        fits.extend(fitted)
        audit.extend(arima)
    return {"origin": origin, "sex": sex, "forecasts": rows, "fits": fits, "arima": audit}


def nonneural_job(config, origin, specs, out):
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(origin) & panel.outcome.eq("prevalence")].copy()
    bases = []
    for spec in specs:
        if spec["base"] not in bases:
            bases.append(spec["base"])
    directory = out / "nonneural_checkpoints" / str(origin)
    directory.mkdir(parents=True)
    rows, fits, calibrations, training = forecast_origin(panel, config, origin, directory, bases)
    return {"origin": origin, "forecasts": rows, "fits": fits,
            "calibrations": calibrations, "training": training}


def main():
    from gbd_park.evaluation import (champion_forecasts, origin_population_weights, summarize_scores,
                                     primary_contrasts, matched_tcn_harm)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/primary_v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one through four CPU workers")
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing run")
    check_lock()
    previous, old_code, records = verify_prior()
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    origins = evaluation_origins(config)
    families = config["models"]["local_order"] + config["models"]["nonneural_order"] + ["tcn_adapted", "tcn_unadapted", "tcn_intercept"]
    all_families = families + ["local_champion", "nonneural_champion"]
    test_path = ROOT / "work/primary-validation/tests.json"
    tests = json.loads(test_path.read_text())
    assert tests["passed"]
    required = ["src/gbd_park/evaluation.py", "scripts/run_primary.py", "tests/test_primary.py",
                "study_design/primary_evaluation_implementation.md"]
    assert set(required).issubset(tests["tested_code_sha256"])
    for filename, expected in tests["tested_code_sha256"].items():
        assert sha(ROOT / filename) == expected, f"Changed after tests: {filename}"
    code = {**old_code, **{name: sha(ROOT / name) for name in required}}
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__,
                "torch": torch.__version__, "joblib": joblib.__version__}
    for record in records.values():
        for key, value in record["versions"].items():
            if key in versions:
                assert versions[key] == value, f"Changed numerical environment: {key}"
    assert records["tcn_v1"]["device"] == "cpu"
    out.mkdir(parents=True)
    for name in ["checkpoints", "payloads", "nonneural_checkpoints"]:
        (out / name).mkdir()
    (out / "tests_at_run.json").write_bytes(test_path.read_bytes())
    manifest = {"run_id": out.name, "status": "running", "created_utc": now(),
                "protocol_version": config["version"], "config_sha256": sha(config_path),
                "source_lock_hash": sha(ROOT / "study_design/locked_v1/lock_manifest.json"),
                "prior_manifests_sha256": previous, "code_sha256": code,
                "test_report_sha256": sha(test_path), "device": "cpu", "workers": args.workers,
                "versions": versions, "origins": origins, "primary_origin": config["calendar"]["primary_origin"],
                "families": families, "reporting_roles": all_families[-2:],
                "maximum_scored_year": 2023, "final_period_scored": False,
                "primary_period_previously_inspected_descriptively": True,
                "ensemble_seeds": config["models"]["tcn"]["ensemble_seeds"],
                "role": "Saudi prevalence primary and nested reliability evaluation; rates only"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started, events = time.perf_counter(), []

    def event(name, **details):
        events.append({"event": name, "time_utc": now(), **copy.deepcopy(details)})
        (out / "events.json").write_text(json.dumps(events, indent=2) + "\n")

    decisions = []
    for group, run in [("local", "local_baselines_v1"), ("nonneural", "nonneural_v1")]:
        scores = pd.read_csv(ROOT / "results" / run / "candidate_scores.csv")
        assert scores.forecast_year.max() <= 2018
        decisions.append(select_baseline_settings(scores, config, origins, group))
    decisions = pd.concat(decisions, ignore_index=True)
    assert len(decisions) == 110
    decisions.to_csv(out / "settings_decisions.csv", index=False)
    neural_scores = pd.read_csv(ROOT / "results/tcn_v1/candidate_scores.csv")
    assert neural_scores.seed_or_ensemble.astype(str).eq("11").all()
    assert neural_scores.forecast_year.max() <= 2018
    choices = [select_tcn_settings(neural_scores, config, origin) for origin in origins]
    previous_choices = json.loads((ROOT / "results/tcn_v1/choices_for_later_evaluation.json").read_text())
    assert len(previous_choices) == len(choices) == len(origins)
    for chosen, previous_choice in zip(choices, previous_choices):
        for key in ["fit_origin", "base", "penalties", "inner_origins", "last_inner_label_year", "status"]:
            assert chosen[key] == previous_choice[key]
        np.testing.assert_allclose(chosen["loss"], previous_choice["loss"], atol=1e-14)
    (out / "tcn_choices.json").write_text(json.dumps(choices, indent=2) + "\n")
    mappings_path = ROOT / "results/intervals_v1/champion_family_mappings.csv"
    mappings = pd.read_csv(mappings_path)
    for role, run in [("local", "local_baselines_v1"), ("nonneural", "nonneural_v1")]:
        champion = pd.read_csv(ROOT / "results" / run / f"{role}_champions_for_later_evaluation.csv")
        historical = pd.read_csv(ROOT / "results" / run / "development_scores.csv")
        selection_config = copy.deepcopy(config)
        selection_config["models"]["local_order"] = config["models"][role + "_order"]
        assert len(champion) == len(origins) * len(config["sexes"])
        for row in champion.itertuples():
            reconstructed = select_family(historical, selection_config, row.fit_origin, row.sex)
            assert reconstructed["selected_family"] == row.selected_family
            assert reconstructed["selection_origins"] == str(row.selection_origins)
            assert reconstructed["last_selection_target_year"] == row.last_selection_target_year
            decision = decisions.loc[decisions.origin.eq(row.fit_origin) & decisions.sex.eq(row.sex)
                                      & decisions.family.eq(row.selected_family)]
            assert len(decision) == 1 and decision.iloc[0].setting_id == row.setting_id
            mapped = mappings.loc[mappings.fit_origin.eq(row.fit_origin) & mappings.sex.eq(row.sex)
                                   & mappings.role.eq(role + "_champion")]
            assert len(mapped) == 1 and mapped.iloc[0].source_family == row.selected_family
    (out / "champion_family_mappings.csv").write_bytes(mappings_path.read_bytes())
    # Only issuance-year denominators enter secondary population-weighted rate metrics.
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(max(origins))].copy()
    weights = origin_population_weights(panel, config, origins)
    weights.to_csv(out / "origin_population_weights.csv", index=False)
    del panel
    frozen_names = ["settings_decisions.csv", "tcn_choices.json", "champion_family_mappings.csv", "origin_population_weights.csv"]
    frozen_hashes = {name: sha(out / name) for name in frozen_names}
    event("settings_frozen", hashes=frozen_hashes)
    event("forecast_fitting_started", origins=origins, workers=args.workers)
    local_specs = {s["setting_id"]: s for s in local_grid(config)}
    nonneural_specs = {s["setting_id"]: s for s in nonneural_grid(config)}
    local_results, nonneural_results, payloads = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {}
        for origin in origins:
            for sex in config["sexes"]:
                part = decisions.loc[decisions.origin.eq(origin) & decisions.sex.eq(sex)
                                     & decisions.family.isin(config["models"]["local_order"])]
                futures[pool.submit(local_job, config, origin, sex, [local_specs[s] for s in part.setting_id])] = "local"
            part = decisions.loc[decisions.origin.eq(origin) & decisions.family.isin(config["models"]["nonneural_order"])]
            futures[pool.submit(nonneural_job, config, origin, [nonneural_specs[s] for s in part.setting_id], out)] = "nonneural"
        for choice in choices:
            for seed in manifest["ensemble_seeds"]:
                futures[pool.submit(neural_fit_job, config, choice["fit_origin"], choice["base"], seed, "cpu", out)] = "tcn"
        for index, future in enumerate(as_completed(futures), 1):
            kind = futures[future]
            {"local": local_results, "nonneural": nonneural_results, "tcn": payloads}[kind].append(future.result())
            print(f"Evaluation fitting: {index}/{len(futures)} jobs ({kind}); {time.perf_counter()-started:.1f}s", flush=True)
    local_results.sort(key=lambda r: (r["origin"], r["sex"]))
    nonneural_results.sort(key=lambda r: r["origin"])
    payloads.sort(key=lambda r: (r["origin"], r["seed"]))
    neural_rows, calibrations, seed_rows = [], [], []
    for choice in choices:
        group = [p for p in payloads if p["origin"] == choice["fit_origin"]]
        assert sorted(p["seed"] for p in group) == manifest["ensemble_seeds"]
        assert all(p["base"] == choice["base"] for p in group)
        rows, corrections = make_forecasts(group, config, choice["penalties"], include_intercept=True)
        for row in rows:
            row.update(last_inner_label_year=choice["last_inner_label_year"], selection_status="frozen_completed_blocks")
        neural_rows.extend(rows)
        calibrations.extend(corrections)
        for payload in group:
            rows, _ = make_forecasts([payload], config, [])
            seed_rows.extend(rows)
    json_lines(out / "local_fit_audit.jsonl", [r for result in local_results for r in result["fits"]])
    pd.DataFrame([r for result in local_results for r in result["arima"]]).to_csv(out / "arima_candidate_audit.csv", index=False)
    json_lines(out / "nonneural_fit_audit.jsonl", [r for result in nonneural_results for r in result["fits"]])
    json_lines(out / "nonneural_adaptation_audit.jsonl", [r for result in nonneural_results for r in result["calibrations"]])
    json_lines(out / "tcn_fit_audit.jsonl", [p["audit"] for p in payloads])
    json_lines(out / "tcn_adaptation_audit.jsonl", calibrations)
    neural_training = pd.concat([next(p["training_meta"] for p in payloads if p["origin"] == o) for o in origins], ignore_index=True)
    assert neural_training.label_end.le(neural_training.fit_origin).all()
    assert not neural_training.country.eq(config["primary_target"]).any()
    neural_training.to_csv(out / "tcn_training_audit.csv", index=False)
    pooled_training = pd.DataFrame([r for result in nonneural_results for r in result["training"]])
    assert pooled_training.label_end.le(pooled_training.fit_origin).all()
    assert not pooled_training.loc[pooled_training.pool.eq("donor"), "country"].eq(config["primary_target"]).any()
    for _, training_group in pooled_training.groupby(["fit_origin", "pool"]):
        np.testing.assert_allclose(training_group.groupby("country").sample_weight.sum(),
                                   len(training_group) / training_group.country.nunique(), rtol=1e-12)
    pooled_training.to_csv(out / "nonneural_training_audit.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "seed_predictions.csv", index=False)
    raw_local = pd.DataFrame([r for result in local_results for r in result["forecasts"]])
    raw_nonneural = pd.DataFrame([r for result in nonneural_results for r in result["forecasts"]])
    # Auxiliary bases/penalties are unscored; selection was frozen before fitting.
    raw_nonneural.to_csv(out / "nonneural_auxiliary_predictions.csv", index=False)
    selected = pd.concat([raw_local, raw_nonneural], ignore_index=True).merge(
        decisions, on=["origin", "sex", "family", "setting_id"], how="inner", validate="many_to_one")
    selected["seed_or_ensemble"] = np.where(selected.family.isin(config["models"]["local_order"]),
                                             "deterministic_statistical", "11_for_boosting;ridge_deterministic")
    selected.loc[selected.family.isin(config["models"]["local_order"]), "adaptation"] = "none"
    selected.loc[selected.family.isin(config["models"]["local_order"]), "donor_strategy"] = "target_only"
    selected.loc[selected.family.str.startswith("pooled_"), "donor_strategy"] = "pooled_target_plus_six_regional"
    points = pd.concat([selected, pd.DataFrame(neural_rows)], ignore_index=True)
    points["source_family"] = points.family
    points["run_id"], points["protocol_version"] = out.name, config["version"]
    points["config_sha256"], points["source_lock_hash"] = manifest["config_sha256"], manifest["source_lock_hash"]
    points["population_scenario"] = "not_applicable_rate_forecast"
    points["residual_block_count"] = points.origin - 2007
    assignments = []
    for family in families:
        countries = ([config["primary_target"]] if family in config["models"]["local_order"]
                     else country_pool(config, config["primary_target"], "pooled" if family.startswith("pooled_") else "donor"))
        digest = hashlib.sha256(json.dumps(countries).encode()).hexdigest()
        points.loc[points.family.eq(family), "donor_list_sha256"] = digest
        assignments.append({"family": family, "countries": "|".join(countries), "country_list_sha256": digest})
    pd.DataFrame(assignments).to_csv(out / "country_assignments.csv", index=False)
    validate_forecast_ledger(points, config, families)
    points = pd.concat([points, champion_forecasts(points, mappings, config)], ignore_index=True)
    points = points.sort_values(["origin", "family", "sex", "age", "horizon"]).reset_index(drop=True)
    validate_forecast_ledger(points, config, all_families)
    assert len(points) == 8800
    points.to_csv(out / "predictions.csv", index=False)
    event("predictions_committed", rows=len(points), sha256=sha(out / "predictions.csv"))
    interval_tables, draw_tables, bank_refs = [], [], []
    for (origin, family), part in points.groupby(["origin", "family"]):
        bank_path = ROOT / "results/intervals_v1/banks" / f"origin{origin}__{family}.joblib"
        bank = joblib.load(bank_path)
        assert bank["n_blocks"] == origin - 2007 and max(bank["origins"]) + 5 <= origin
        for sex in config["sexes"]:
            assert part.loc[part.sex.eq(sex), "source_family"].eq(bank["source_family_by_sex"][sex]).all()
        intervals, draws = apply_bank(part, bank, config)
        for frame in [intervals, draws]:
            frame["run_id"] = out.name
            frame["source_family"] = frame.sex.map(bank["source_family_by_sex"])
        interval_tables.append(intervals)
        draw_tables.append(draws)
        bank_refs.append({"origin": origin, "family": family, "path": str(bank_path.relative_to(ROOT)),
                          "sha256": sha(bank_path), "n_blocks": bank["n_blocks"]})
    intervals, draws = pd.concat(interval_tables, ignore_index=True), pd.concat(draw_tables, ignore_index=True)
    assert len(intervals) == 52800 and len(draws) == 79200
    intervals.to_csv(out / "intervals.csv", index=False)
    draws.to_csv(out / "joint_draws.csv", index=False)
    pd.DataFrame(bank_refs).to_csv(out / "bank_references.csv", index=False)
    frozen_hashes.update({name: sha(out / name) for name in ["predictions.csv", "intervals.csv", "joint_draws.csv", "bank_references.csv", "country_assignments.csv"]})
    (out / "pre_score_commit.json").write_text(json.dumps({"committed_utc": now(), "artifact_sha256": frozen_hashes}, indent=2) + "\n")
    event("intervals_committed", rows=len(intervals), draws=len(draws), hashes=frozen_hashes)
    verify_committed_outputs(out, frozen_hashes)
    event("evaluation_scoring_started", maximum_verification_year=2023)
    verify_scoring_order(events)
    # First evaluation join: all point forecasts and intervals have already been saved.
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    truth = panel.loc[panel.location_name.eq(config["primary_target"]) & panel.outcome.eq("prevalence")].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})
    scored = score_forecasts(pd.read_csv(out / "predictions.csv"), truth, config["ages"], config["calendar"]["horizons"], 2023)
    scored.to_csv(out / "point_scores.csv", index=False)
    cells, wis = score_intervals(pd.read_csv(out / "intervals.csv"), truth, config, 2023)
    assert len(cells) == 52800 and len(wis) == 17600
    cells.to_csv(out / "interval_scores.csv", index=False)
    wis.to_csv(out / "wis_scores.csv", index=False)
    for name, frame in summarize_scores(scored, weights, config).items():
        frame.to_csv(out / f"{name}.csv", index=False)
    contrasts, age_contrasts, verdict = primary_contrasts(scored, config)
    contrasts.to_csv(out / "primary_contrasts.csv", index=False)
    age_contrasts.to_csv(out / "primary_age_contrasts.csv", index=False)
    (out / "primary_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    paired, harm = matched_tcn_harm(scored, config)
    paired.to_csv(out / "matched_adaptation_cells.csv", index=False)
    harm.to_csv(out / "matched_adaptation_summary.csv", index=False)
    cells.groupby(["origin", "sex", "family", "horizon", "scale", "level"], as_index=False).agg(
        coverage=("covered", "mean"), mean_width=("width", "mean"),
        mean_interval_score=("interval_score", "mean"), age_cells=("covered", "size"),
        n_blocks=("n_blocks", "first")).to_csv(out / "interval_summary_by_origin.csv", index=False)
    wis.groupby(["origin", "sex", "family", "horizon", "scale"], as_index=False).agg(
        mean_wis_50_80=("wis_50_80", "mean"), mean_wis_50_80_95=("wis_50_80_95", "mean"),
        n_blocks=("n_blocks", "first")).to_csv(out / "wis_summary_by_origin.csv", index=False)
    verify_committed_outputs(out, frozen_hashes)
    check_lock()
    assert verify_prior()[0] == previous
    assert all(sha(ROOT / name) == digest for name, digest in code.items())
    neural_fits = [p["audit"] for p in payloads]
    pooled_fits = [r for result in nonneural_results for r in result["fits"]]
    pooled_corrections = [r for result in nonneural_results for r in result["calibrations"]]
    local_fits = [r for result in local_results for r in result["fits"]]
    assert len(neural_fits) == 25 and len(local_fits) == 550
    assert all(f["fingerprint_before"] == f["fingerprint_after"] for f in neural_fits)
    assert all(f["base_fingerprint_before"] == f["base_fingerprint_after"] for f in pooled_fits)
    assert all(f["maximum_label_year"] <= f["origin"] for f in neural_fits + pooled_fits)
    assert all(c["last_target_label_year"] <= c["origin"] for c in calibrations + pooled_corrections)
    assert all(config["primary_target"] not in f["countries"] for f in neural_fits)
    assert all((config["primary_target"] in f["countries"]) == (f["pool"] == "pooled") for f in pooled_fits)
    validation = {"passed": True, "unit_tests": tests["tests_run"], "origins": origins,
                  "unique_model_forecasts": 7700, "champion_view_forecasts": 1100,
                  "prediction_rows": len(points), "interval_rows": len(cells), "joint_draw_rows": len(draws),
                  "wis_rows": len(wis), "tcn_fits": len(neural_fits), "nonneural_fits": len(pooled_fits),
                  "local_age_fits": len(local_fits), "tcn_fit_failures": sum(f["status"] != "ok" for f in neural_fits),
                  "nonneural_fit_failures": sum(f["status"] != "ok" for f in pooled_fits),
                  "local_age_fit_fallbacks": sum(f["status"] != "ok" for f in local_fits),
                  "tcn_adaptation_failures": sum(c["status"] != "ok" for c in calibrations),
                  "nonneural_adaptation_failures": sum(c["status"] != "ok" for c in pooled_corrections),
                  "selected_fallback_cells": int(points.loc[points.family.isin(families), "status"].eq("fallback").sum()),
                  "primary_origin_fitted_once": True, "all_five_seeds_retained": True,
                  "frozen_settings_and_banks": True, "point_and_intervals_committed_before_scoring": True,
                  "maximum_scored_year": 2023, "final_period_scored": True,
                  "prior_stages_and_lock_unchanged": True, "device": "cpu", "workers": args.workers,
                  "elapsed_seconds": time.perf_counter()-started}
    (out / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
    event("evaluation_complete", elapsed_seconds=validation["elapsed_seconds"])
    manifest.update(status="complete", final_period_scored=True, completed_utc=now())
    manifest["output_sha256"] = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*"))
                                 if path.is_file() and path.name != "run_manifest.json"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
