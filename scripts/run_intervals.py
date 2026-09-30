"""Stage 4: complete prequential histories and freeze joint residual banks."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"

import argparse
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
import torch
from gbd_park.intervals import build_residual_bank, apply_bank, bank_quantiles, score_intervals
from gbd_park.prequential import champion_history, select_prequential
from gbd_park.scoring import score_forecasts
from gbd_park.tcn_forecasting import base_id, make_forecasts, select_tcn_settings
from run_tcn import fit_job
from run_local_baselines import check_lock, sha, now, json_lines

PRIOR = ["local_baselines_v1", "nonneural_v1", "tcn_v1"]


def verify_prior():
    hashes, code = {}, {}
    for run in PRIOR:
        directory = ROOT / "results" / run
        record = json.loads((directory / "run_manifest.json").read_text())
        assert record["status"] == "complete" and not record["final_period_scored"]
        for path, expected in record["output_sha256"].items():
            assert sha(directory / path) == expected, path
        for path, expected in record["code_sha256"].items():
            assert sha(ROOT / path) == expected, path
            code[path] = expected
        hashes[run] = sha(directory / "run_manifest.json")
    return hashes, code


def bank_long(bank):
    blocks = []
    for index, residual_origin in enumerate(bank["origins"]):
        part = bank["coords"].copy()
        part["fit_origin"], part["family"] = bank["fit_origin"], bank["family"]
        part["residual_origin"] = residual_origin
        part["residual_verification_year"] = residual_origin + part.horizon
        part["raw_log_residual"] = bank["raw_residuals"][index]
        part["coordinate_mean"] = bank["center"]
        part["centered_log_residual"] = bank["centered_residuals"][index]
        part["n_blocks"] = bank["n_blocks"]
        part["source_family"] = part.sex.map(bank["source_family_by_sex"])
        blocks.append(part)
    return pd.concat(blocks, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/intervals_v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one through four CPU workers")
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing run")
    check_lock()
    previous, old_code = verify_prior()
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    test_path = ROOT / "work/interval-validation/tests.json"
    tests = json.loads(test_path.read_text())
    assert tests["passed"]
    for filename, expected in tests["tested_code_sha256"].items():
        assert sha(ROOT / filename) == expected, f"Changed after testing: {filename}"
    required = ["src/gbd_park/intervals.py", "src/gbd_park/prequential.py", "scripts/run_intervals.py",
                "tests/test_intervals.py", "study_design/intervals_implementation.md"]
    assert set(required).issubset(tests["tested_code_sha256"]), "Test current runner and specification first"
    code = {**old_code, **{name: sha(ROOT / name) for name in required}}
    neural = ROOT / "results/tcn_v1"
    old_neural = json.loads((neural / "run_manifest.json").read_text())
    assert old_neural["device"] == "cpu", "Early ensembles must use the stage-3 numerical device"
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "torch": torch.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__}
    assert all(value == old_neural["versions"][key] for key, value in versions.items()), "Neural execution environment changed"
    families = config["models"]["local_order"] + config["models"]["nonneural_order"] + ["tcn_adapted", "tcn_unadapted", "tcn_intercept"]
    out.mkdir(parents=True)
    for name in ["checkpoints", "payloads", "banks"]:
        (out / name).mkdir()
    (out / "tests_at_run.json").write_bytes(test_path.read_bytes())
    manifest = {"run_id": out.name, "status": "running", "created_utc": now(),
                "protocol_version": config["version"], "config_sha256": sha(config_path),
                "source_lock_hash": sha(ROOT / "study_design/locked_v1/lock_manifest.json"),
                "prior_manifests_sha256": previous, "code_sha256": code,
                "test_report_sha256": sha(test_path), "device": "cpu", "workers": args.workers, "versions": versions,
                "historical_origins": list(range(2003, 2014)), "bank_fit_origins": list(range(2014, 2019)),
                "development_preview_origins": [2012, 2013], "families": families,
                "maximum_scored_year": 2018, "final_period_scored": False,
                "role": "prequential calibration preparation and development-only interval diagnostics"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started, events = time.perf_counter(), []

    def event(name, **details):
        events.append({"event": name, "time_utc": now(), **details})
        (out / "events.json").write_text(json.dumps(events, indent=2) + "\n")

    event("historical_selection_started")
    frames, choices = [], []
    for group, name in [("local", "local_baselines_v1"), ("nonneural", "nonneural_v1")]:
        directory = ROOT / "results" / name
        candidates = pd.read_csv(directory / "candidate_predictions.csv")
        candidate_scores = pd.read_csv(directory / "candidate_scores.csv")
        previous_development = pd.read_csv(directory / "development_predictions.csv")
        for origin in manifest["historical_origins"]:
            picked, decisions = select_prequential(candidates, candidate_scores, config, origin, group)
            choices.extend(decisions)
            artifact = "candidate_predictions.csv"
            if origin in config["calendar"]["selection_origins"]:
                reference = previous_development[previous_development.origin.eq(origin)].copy()
                keys = ["sex", "family", "age", "horizon"]
                left, right = picked.set_index(keys).sort_index(), reference.set_index(keys).sort_index()
                assert left.index.equals(right.index) and left.setting_id.eq(right.setting_id).all()
                np.testing.assert_allclose(left.prediction, right.prediction, rtol=1e-12)
                picked = reference.merge(pd.DataFrame(decisions)[["origin", "sex", "family", "setting_id", "selection_status", "last_inner_label_year"]],
                                         on=["origin", "sex", "family", "setting_id"], validate="many_to_one")
                artifact = "development_predictions.csv"
            picked["source_artifact"] = f"results/{name}/{artifact}"
            frames.append(picked)
    pd.DataFrame(choices).to_csv(out / "historical_setting_decisions.csv", index=False)
    neural_scores = pd.read_csv(neural / "candidate_scores.csv")
    assert neural_scores.seed_or_ensemble.astype(str).eq("11").all()
    neural_choices = [select_tcn_settings(neural_scores, config, origin) for origin in manifest["historical_origins"]]
    (out / "historical_tcn_choices.json").write_text(json.dumps(neural_choices, indent=2) + "\n")
    old_choices = json.loads((neural / "tuning_decisions.json").read_text())
    for old in old_choices:
        current = next(choice for choice in neural_choices if choice["fit_origin"] == old["fit_origin"])
        for key in ["base", "penalties", "inner_origins", "last_inner_label_year", "status"]:
            assert current[key] == old[key]
        np.testing.assert_allclose(current["loss"], old["loss"], rtol=1e-12, atol=1e-14)
    early = [c for c in neural_choices if c["fit_origin"] <= 2008]
    jobs = [(c["fit_origin"], c["base"], seed) for c in early for seed in config["models"]["tcn"]["ensemble_seeds"] if seed != 11]
    event("early_ensemble_fitting_started", new_fits=len(jobs))
    payloads = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(fit_job, config, origin, base, seed, "cpu", out) for origin, base, seed in jobs]
        for future in as_completed(futures):
            payloads.append(future.result())
            if len(payloads) % 4 == 0:
                print(f"Early ensemble seeds: {len(payloads)}/{len(jobs)} fits", flush=True)
    assert len(payloads) == 24
    fits = [p["audit"] for p in payloads]
    early_records, early_calibrations, seed_references = [], [], []
    for choice in early:
        origin, base = choice["fit_origin"], choice["base"]
        saved_name = f"origin{origin}__{base_id(base)}__seed=11.joblib"
        reused = joblib.load(neural / "payloads" / saved_name)
        group = [reused] + [p for p in payloads if p["origin"] == origin]
        assert all(p["base"] == base for p in group)
        assert sorted(p["seed"] for p in group) == sorted(config["models"]["tcn"]["ensemble_seeds"])
        rows, calibrations = make_forecasts(group, config, choice["penalties"], include_intercept=True)
        for row in rows:
            row.update(selection_status=choice["status"], last_inner_label_year=choice["last_inner_label_year"],
                       source_artifact=f"{args.output}/early_tcn_predictions.csv")
        early_records.extend(rows)
        early_calibrations.extend(calibrations)
        for payload in group:
            run = "tcn_v1" if payload["seed"] == 11 else out.name
            filename = f"origin{origin}__{base_id(base)}__seed={payload['seed']}.joblib"
            payload_path = (neural if payload["seed"] == 11 else out) / "payloads" / filename
            seed_references.append({"origin": origin, "seed": payload["seed"], "base": base,
                                    "source_run": run, "payload_path": str(payload_path.relative_to(ROOT)),
                                    "state_fingerprint": payload["audit"]["fingerprint_before"],
                                    "status": payload["audit"]["status"]})
    json_lines(out / "new_tcn_fit_audit.jsonl", fits)
    json_lines(out / "early_tcn_adaptation_audit.jsonl", early_calibrations)
    json_lines(out / "early_seed_references.jsonl", seed_references)
    early_frame = pd.DataFrame(early_records)
    early_frame.to_csv(out / "early_tcn_predictions.csv", index=False)
    early_frame["run_id"] = out.name
    frames.append(early_frame)
    old_tcn = pd.read_csv(neural / "development_predictions.csv")
    old_tcn["source_artifact"] = "results/tcn_v1/development_predictions.csv"
    for choice in neural_choices:
        mask = old_tcn.origin.eq(choice["fit_origin"])
        old_tcn.loc[mask, "selection_status"] = choice["status"]
        old_tcn.loc[mask, "last_inner_label_year"] = choice["last_inner_label_year"]
    frames.append(old_tcn)
    prequential = pd.concat(frames, ignore_index=True)
    prequential["source_run_id"] = prequential.run_id
    prequential["run_id"] = out.name
    prequential["protocol_version"] = config["version"]
    prequential["config_sha256"] = manifest["config_sha256"]
    prequential["source_lock_hash"] = manifest["source_lock_hash"]
    columns = ["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year", "family", "setting_id",
               "log_prediction", "prediction", "status", "fallback_reason", "parameter_count", "seed_or_ensemble",
               "donor_strategy", "selection_status", "last_inner_label_year", "ensemble_fingerprint", "source_run_id",
               "source_artifact", "run_id", "protocol_version", "config_sha256", "source_lock_hash"]
    prequential = prequential[columns].sort_values(["origin", "family", "sex", "age", "horizon"]).reset_index(drop=True)
    assert len(prequential) == 16940 and prequential.forecast_year.max() == 2018
    assert prequential.last_inner_label_year.fillna(prequential.origin).le(prequential.origin).all()
    assert not prequential.duplicated(["origin", "family", "sex", "age", "horizon"]).any()
    assert set(prequential.family) == set(families)
    prequential.to_csv(out / "prequential_predictions.csv", index=False)
    forecast_hash = sha(out / "prequential_predictions.csv")
    event("prequential_predictions_committed", rows=len(prequential), sha256=forecast_hash)
    donors = []
    country_names = [country["name"] for country in config["countries"]]
    for origin in manifest["historical_origins"]:
        for family in families:
            countries = ([config["primary_target"]] if family in config["models"]["local_order"] else
                         country_names if family.startswith("pooled_") else [name for name in country_names if name != config["primary_target"]])
            donors.append({"origin": origin, "family": family, "countries": "|".join(countries),
                           "country_list_sha256": hashlib.sha256(json.dumps(countries).encode()).hexdigest()})
    pd.DataFrame(donors).to_csv(out / "historical_country_assignments.csv", index=False)
    event("historical_residual_scoring_started", maximum_verification_year=2018)
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    truth = panel[panel.location_name.eq(config["primary_target"]) & panel.outcome.eq("prevalence") & panel.year.le(2018)].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})
    scores = score_forecasts(pd.read_csv(out / "prequential_predictions.csv"), truth,
                            config["ages"], config["calendar"]["horizons"], 2018)
    scores["log_residual"] = np.log(scores.observed_rate) - scores.log_prediction
    scores.to_csv(out / "prequential_scores.csv", index=False)
    local_choices = pd.read_csv(ROOT / "results/local_baselines_v1/local_champions_for_later_evaluation.csv")
    nonneural_choices = pd.read_csv(ROOT / "results/nonneural_v1/nonneural_champions_for_later_evaluation.csv")
    summaries, block_tables, quantiles, mappings = [], [], [], []
    for fit_origin in manifest["bank_fit_origins"]:
        sources = {family: (scores, {sex: family for sex in config["sexes"]}) for family in families}
        for name, choices_table in [("local_champion", local_choices), ("nonneural_champion", nonneural_choices)]:
            eligible = choices_table[choices_table.fit_origin.eq(fit_origin)]
            assert len(eligible) == 2 and eligible.last_selection_target_year.le(fit_origin).all()
            mapping = eligible.set_index("sex").selected_family.to_dict()
            sources[name] = (champion_history(scores, config, mapping, name), mapping)
            mappings.extend({"fit_origin": fit_origin, "role": name, "sex": sex, "source_family": family,
                             "last_selection_target_year": int(eligible.last_selection_target_year.max())}
                            for sex, family in mapping.items())
        for family, (source, mapping) in sources.items():
            bank = build_residual_bank(source, config, fit_origin, family)
            bank["source_family_by_sex"] = mapping
            assert bank["status"] == "ok" and bank["n_blocks"] == fit_origin-2007
            np.testing.assert_allclose(bank["centered_residuals"].mean(axis=0), 0, atol=1e-14)
            joblib.dump(bank, out / "banks" / f"origin{fit_origin}__{family}.joblib", compress=3)
            block_tables.append(bank_long(bank))
            quantiles.append(bank_quantiles(bank, config))
            summaries.append({"fit_origin": fit_origin, "family": family, "n_blocks": bank["n_blocks"],
                              "first_residual_origin": min(bank["origins"]), "last_residual_origin": max(bank["origins"]),
                              "last_residual_label_year": max(bank["origins"])+5, "coordinates": len(bank["coords"]),
                              "status": bank["status"], "source_family_by_sex": json.dumps(mapping, sort_keys=True)})
    block_frame = pd.concat(block_tables, ignore_index=True)
    assert len(summaries) == 80 and len(block_frame) == 79200
    block_frame.to_csv(out / "residual_blocks.csv", index=False)
    pd.concat(quantiles, ignore_index=True).to_csv(out / "calibration_quantiles.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "bank_summary.csv", index=False)
    pd.DataFrame(mappings).to_csv(out / "champion_family_mappings.csv", index=False)
    event("residual_banks_committed", banks=len(summaries), final_origin_blocks=11)
    previews, draws = [], []
    for origin in manifest["development_preview_origins"]:
        for family in families:
            bank = build_residual_bank(scores, config, origin, family)
            points = prequential[prequential.origin.eq(origin) & prequential.family.eq(family)]
            intervals, joint = apply_bank(points, bank, config)
            previews.append(intervals)
            draws.append(joint)
    preview = pd.concat(previews, ignore_index=True)
    joint_draws = pd.concat(draws, ignore_index=True)
    assert len(preview) == 18480 and len(joint_draws) == 16940
    preview["evaluation_role"] = "development_diagnostic_only"
    preview.to_csv(out / "development_intervals.csv", index=False)
    joint_draws.to_csv(out / "development_joint_draws.csv", index=False)
    event("development_intervals_committed", rows=len(preview), sha256=sha(out / "development_intervals.csv"))
    event("development_interval_scoring_started", origins=[2012, 2013], maximum_verification_year=2018)
    interval_scores, wis_scores = score_intervals(pd.read_csv(out / "development_intervals.csv"), truth, config, 2018)
    assert len(interval_scores) == 18480 and len(wis_scores) == 6160
    interval_scores.to_csv(out / "development_interval_scores.csv", index=False)
    wis_scores.to_csv(out / "development_wis_scores.csv", index=False)
    interval_scores.groupby(["origin", "sex", "family", "horizon", "scale", "level"], as_index=False).agg(
        coverage=("covered", "mean"), mean_width=("width", "mean"), mean_interval_score=("interval_score", "mean"),
        age_cells=("covered", "count"), n_blocks=("n_blocks", "first")).to_csv(out / "development_interval_summary.csv", index=False)
    wis_scores.groupby(["origin", "sex", "family", "horizon", "scale"], as_index=False).agg(
        mean_wis_50_80=("wis_50_80", "mean"), mean_wis_50_80_95=("wis_50_80_95", "mean"),
        mean_median_absolute_error=("median_absolute_error", "mean"), n_blocks=("n_blocks", "first")).to_csv(
            out / "development_wis_summary.csv", index=False)
    check_lock()
    assert verify_prior()[0] == previous
    assert all(sha(ROOT / name) == value for name, value in code.items())
    assert sha(out / "prequential_predictions.csv") == forecast_hash
    validation = {"passed": True, "unit_tests": tests["tests_run"], "prequential_forecasts": len(prequential),
                  "method_families": len(families), "historical_origins": 11, "new_tcn_fits": len(fits),
                  "new_tcn_failures": sum(row["status"] != "ok" for row in fits),
                  "early_adaptation_failures": sum(row["status"] != "ok" for row in early_calibrations),
                  "prequential_fallback_cells": int(prequential.status.eq("fallback").sum()),
                  "banks": len(summaries), "bank_residual_rows": len(block_frame),
                  "bank_block_counts": [7, 8, 9, 10, 11], "development_interval_rows": len(preview),
                  "development_interval_origins": [2012, 2013], "preview_block_counts": [5, 6],
                  "maximum_scored_year": 2018, "final_period_scored": False,
                  "final_origin_point_forecasts_produced": False, "reliability_coverage_evaluated": False,
                  "complete_joint_blocks": True, "coordinate_mean_centering": True,
                  "distinct_scale_quantiles_and_medians": True, "point_forecasts_unchanged": True,
                  "prior_stages_and_lock_unchanged": True, "elapsed_seconds": time.perf_counter()-started}
    (out / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
    event("calibration_preparation_complete", elapsed_seconds=validation["elapsed_seconds"])
    manifest.update(status="complete", completed_utc=now())
    manifest["output_sha256"] = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*"))
                                 if path.is_file() and path.name != "run_manifest.json"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
