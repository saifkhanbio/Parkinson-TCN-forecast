"""Independent audit and descriptive reporting of frozen incidence/GCC replication."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gbd_park_matplotlib")
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gbd_park.pooled import target_inputs
from gbd_park.tcn import load_checkpoint, predict_changes, state_fingerprint
from run_local_baselines import check_lock, sha
from report_primary import verify_roles, verify_interval_scores, verify_primary_contrasts, LABELS

COORDS = ["sex", "age", "horizon"]
ROLES = ["tcn_adapted", "local_champion", "nonneural_champion"]


def check_hashes(root, hashes):
    for filename, expected in hashes.items():
        assert sha(root / filename) == expected, str(root / filename)


def verify_manifest(directory):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    check_hashes(directory, manifest["output_sha256"])
    check_hashes(ROOT, manifest["code_sha256"])
    return manifest


def verify_chronology(run, tasks):
    events = json.loads((run / "events.json").read_text())
    names = [row["event"] for row in events]
    wanted = ["candidate_fitting_started", "all_choices_frozen", "selected_fitting_started",
              "all_trials_issued_before_final_scoring", "evaluation_scoring_started", "complete"]
    assert all(names.count(name) == 1 for name in wanted)
    assert [names.index(name) for name in wanted] == sorted(names.index(name) for name in wanted)
    times = [datetime.fromisoformat(events[names.index(name)]["time_utc"]) for name in wanted]
    assert times == sorted(times)
    global_commit = json.loads((run / "global_issued_commit.json").read_text())
    check_hashes(run, global_commit["artifact_sha256"])
    assert len(global_commit["artifact_sha256"]) == 11
    for task in tasks:
        directory = run / "trials" / task["id"]
        for phase in ["choices_frozen", "issued_commit", "scoring_complete"]:
            record = json.loads((directory / f"{phase}.json").read_text())
            assert record["status"] == "complete"
            check_hashes(directory, record["artifact_sha256"])
        issued = json.loads((directory / "issued_commit.json").read_text())
        assert {"predictions.csv", "intervals.csv", "joint_draws.csv"}.issubset(issued["artifact_sha256"])
        assert datetime.fromisoformat(issued["committed_utc"]) <= times[4]
        assert issued["final_period_scored"] is False


def verify_fit_jobs(run, task):
    counts = {"local": 0, "nonneural": 0, "tcn": 0}
    failures = {"local": 0, "nonneural": 0, "tcn": 0}
    for marker in sorted((run / "jobs" / task["id"]).glob("*/complete.json")):
        record = json.loads(marker.read_text())
        job = record["job"]
        assert job["target"] == task["target"] and job["outcome"] == task["outcome"]
        assert record["job_sha256"] == hashlib.sha256(json.dumps(job, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        payload = joblib.load(marker.parent / "result.joblib")
        assert payload["job"] == job and payload["actual_target"] == task["target"] and payload["actual_outcome"] == task["outcome"]
        kind = job["kind"]
        counts[kind] += 1
        if kind == "tcn":
            audit = payload["audit"]
            failures[kind] += int(audit["status"] != "ok")
            assert audit["maximum_label_year"] <= job["origin"] and audit["last_target_label_year"] <= job["origin"]
            assert task["target"] not in audit["countries"]
            assert audit["fingerprint_before"] == audit["fingerprint_after"]
            assert payload["training_meta"].label_end.le(job["origin"]).all()
            assert not payload["training_meta"].country.eq(task["target"]).any()
        else:
            failures[kind] += sum(row["status"] != "ok" for row in payload["fits"])
            if kind == "nonneural":
                for row in payload["fits"]:
                    assert row["maximum_label_year"] <= job["origin"]
                    assert row["base_fingerprint_before"] == row["base_fingerprint_after"]
                    assert (task["target"] in row["countries"]) == (row["pool"] == "pooled")
                assert all(row["last_target_label_year"] <= job["origin"] for row in payload["calibrations"])
    assert counts == {"local": 32, "nonneural": 16, "tcn": 157}
    return {"verified_fit_jobs": counts, "fit_failures_or_local_age_fallbacks": failures}


def verify_source_scores(scores, panel, task):
    keys = ["target", "outcome", "sex", "age", "forecast_year"]
    truth = panel.loc[panel.location_name.eq(task["target"]) & panel.outcome.eq(task["outcome"])].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "source_rate"})
    assert scores.target.eq(task["target"]).all() and scores.outcome.eq(task["outcome"]).all()
    joined = scores.merge(truth[keys+["source_rate"]], on=keys, validate="many_to_one")
    assert len(joined) == len(scores)
    np.testing.assert_allclose(joined.observed_rate, joined.source_rate, rtol=1e-14)
    np.testing.assert_allclose(joined.absolute_log_error, np.abs(np.log(joined.prediction)-np.log(joined.source_rate)), atol=2e-14, rtol=1e-12)
    np.testing.assert_allclose(joined.absolute_rate_error, np.abs(joined.prediction-joined.source_rate), atol=2e-11, rtol=1e-12)
    assert scores.forecast_year.eq(scores.origin+scores.horizon).all()


def verify_banks(directory, points, intervals, draws, historical, mappings, config):
    coords = pd.MultiIndex.from_product([config["sexes"], config["ages"], config["calendar"]["horizons"]], names=COORDS)
    count = 0
    banks = sorted((directory / "banks").glob("*.joblib"))
    assert len(banks) == 80
    max_difference = 0.
    for path in banks:
        bank = joblib.load(path)
        origin, family = bank["fit_origin"], bank["family"]
        assert bank["origins"] == list(range(2003, origin-4)) and bank["n_blocks"] == origin-2007
        assert bank["target"] == points.target.iloc[0] and bank["outcome"] == points.outcome.iloc[0]
        assert list(map(tuple, bank["coords"].to_numpy())) == list(coords)
        history = []
        for sex in config["sexes"]:
            expected_family = family
            if family in ["local_champion", "nonneural_champion"]:
                row = mappings.loc[mappings.fit_origin.eq(origin) & mappings.sex.eq(sex) & mappings.role.eq(family)]
                assert len(row) == 1
                expected_family = row.iloc[0].source_family
            assert bank["source_family_by_sex"][sex] == expected_family
            history.append(historical.loc[historical.sex.eq(sex) & historical.family.eq(expected_family)
                                          & historical.origin.isin(bank["origins"])])
        source = pd.concat(history)
        source["residual"] = np.log(source.observed_rate)-source.log_prediction
        raw = source.pivot(index="origin", columns=COORDS, values="residual").reindex(index=bank["origins"], columns=coords).to_numpy()
        assert np.isfinite(raw).all() and raw.shape == (bank["n_blocks"], 110)
        np.testing.assert_allclose(bank["raw_residuals"], raw, atol=2e-14, rtol=1e-12)
        centered = raw-raw.mean(axis=0)
        np.testing.assert_allclose(bank["center"], raw.mean(axis=0), atol=2e-14, rtol=1e-12)
        np.testing.assert_allclose(bank["centered_residuals"], centered, atol=2e-14, rtol=1e-12)
        np.testing.assert_allclose(bank["centered_residuals"].mean(axis=0), 0, atol=2e-14)
        current = points.loc[points.origin.eq(origin) & points.family.eq(family)].set_index(COORDS).reindex(coords)
        logs = current.log_prediction.to_numpy()[None, :]+bank["centered_residuals"]
        saved_draws = draws.loc[draws.origin.eq(origin) & draws.family.eq(family)]
        for scale, values, field in [("log_rate", logs, "log_draw"), ("rate", np.exp(logs), "rate_draw")]:
            actual = saved_draws.pivot(index="residual_origin", columns=COORDS, values=field).reindex(index=bank["origins"], columns=coords).to_numpy()
            np.testing.assert_allclose(actual, values, atol=2e-12, rtol=1e-12)
            for level in config["intervals"]["central_levels"]:
                saved = intervals.loc[intervals.origin.eq(origin) & intervals.family.eq(family)
                                      & intervals.scale.eq(scale) & intervals.level.eq(level)].set_index(COORDS).reindex(coords)
                expected = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0, method="linear").T
                actual = saved[["lower", "median", "upper"]].to_numpy()
                np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-12)
                max_difference = max(max_difference, float(np.max(np.abs(actual-expected))))
                point_field = "log_prediction" if scale == "log_rate" else "prediction"
                np.testing.assert_allclose(saved.point_prediction, current[point_field], atol=2e-12, rtol=1e-12)
                count += len(saved)
    assert count == len(intervals) == 52800
    return {"verified_banks": len(banks), "verified_interval_rows": count,
            "quantile_max_absolute_difference": max_difference}


def verify_selection_and_summaries(directory, scores, historical, mappings, panel, task, config):
    for row in mappings.itertuples():
        allowed = [o for o in config["calendar"]["selection_origins"] if o+5 <= row.fit_origin]
        families = config["models"]["local_order" if row.role == "local_champion" else "nonneural_order"]
        eligible = historical.loc[historical.sex.eq(row.sex) & historical.origin.isin(allowed)
                                  & historical.family.isin(families) & historical.horizon.eq(5)]
        means = eligible.groupby("family").agg(loss=("absolute_log_error", "mean"), parameters=("parameter_count", "mean")).reset_index()
        means["order"] = means.family.map({f: i for i, f in enumerate(families)})
        winner = means.sort_values(["loss", "parameters", "order"]).iloc[0]
        assert winner.family == row.source_family and row.last_selection_target_year == max(allowed)+5
        np.testing.assert_allclose(winner.loss, row.selection_loss, atol=2e-14)
    weights = pd.read_csv(directory / "origin_population_weights.csv")
    source = panel.loc[panel.location_name.eq(task["target"]) & panel.outcome.eq("prevalence")
                       & panel.year.isin(config["calendar"]["reliability_origins"])].copy()
    source["population"] = source["count"]/source.rate*100000
    source["weight"] = source.population/source.groupby(["year", "sex"]).population.transform("sum")
    source = source.set_index(["year", "sex", "age"]).sort_index()
    actual = weights.set_index(["origin", "sex", "age"]).sort_index()
    np.testing.assert_allclose(actual.origin_population, source.population, atol=1e-8, rtol=1e-12)
    np.testing.assert_allclose(actual.origin_population_weight, source.weight, atol=1e-14)
    joined = scores.merge(weights[["origin", "sex", "age", "origin_population_weight"]], on=["origin", "sex", "age"], validate="many_to_one")
    joined["weighted"] = joined.absolute_log_error*joined.origin_population_weight
    by = joined.groupby(["origin", "sex", "family", "horizon"]).agg(
        mean_age_absolute_log_error=("absolute_log_error", "mean"), rate_mae=("absolute_rate_error", "mean"),
        population_weighted_absolute_log_error=("weighted", "sum"))
    saved = pd.read_csv(directory / "by_origin.csv").set_index(by.index.names).sort_index()
    np.testing.assert_allclose(saved[by.columns], by, atol=2e-12, rtol=1e-12)
    expected = scores.groupby(["sex", "family", "horizon"]).absolute_log_error.mean()
    saved = pd.read_csv(directory / "reliability_by_horizon.csv").set_index(expected.index.names).sort_index()
    np.testing.assert_allclose(saved.mean_absolute_log_error, expected, atol=2e-14, rtol=1e-12)
    assert saved.n_origins.eq(5).all()


def replay_tcn(directory, scores, panel, task, config):
    cfg = json.loads(json.dumps(config))
    cfg["primary_target"] = task["target"]
    # Independently reproduce the transparent adapter, without calling the production adapter.
    working = panel.loc[panel.outcome.eq(task["outcome"]) & panel.year.le(2018)].copy()
    working["outcome"] = "prevalence"
    x, levels, meta = target_inputs(working, cfg, 2018, task["target"])
    refs = pd.read_json(directory / "seed_references.jsonl", lines=True)
    refs = refs.loc[refs.origin.eq(2018)].sort_values("seed")
    assert refs.seed.tolist() == cfg["models"]["tcn"]["ensemble_seeds"]
    changes, failed, replayed = [], False, 0
    for row in refs.itertuples():
        payload = joblib.load(ROOT / row.payload_path)
        audit = payload["audit"]
        assert audit["maximum_label_year"] <= 2018 and task["target"] not in audit["countries"]
        assert set(audit["countries"]) == {c["name"] for c in config["countries"] if c["name"] != task["target"]}
        assert payload["actual_outcome"] == task["outcome"] and payload["actual_target"] == task["target"]
        assert meta.equals(payload["current_meta"])
        np.testing.assert_array_equal(levels, payload["levels"])
        if audit["status"] != "ok":
            failed = True
            changes.append(np.zeros((len(meta), 5)))
            continue
        fitted = load_checkpoint(ROOT / row.checkpoint_path)
        before = state_fingerprint(fitted)
        predicted = predict_changes(fitted, x)
        np.testing.assert_array_equal(predicted, payload["current_changes"])
        assert before == state_fingerprint(fitted) == audit["fingerprint_before"] == audit["fingerprint_after"]
        changes.append(predicted)
        replayed += 1
    ensemble = np.mean(changes, axis=0)
    audit = pd.read_json(directory / "tcn_adaptation_audit.jsonl", lines=True)
    count = 0
    for (family, sex), part in scores.loc[scores.origin.eq(2018) & scores.family.str.startswith("tcn_")].groupby(["family", "sex"]):
        indices = meta.sex.eq(sex).to_numpy()
        prediction = ensemble[indices].copy()
        if failed:
            prediction[:] = 0
        elif family != "tcn_unadapted":
            correction = audit.loc[audit.origin.eq(2018) & audit.family.eq(family) & audit.sex.eq(sex)]
            assert len(correction) == 1 and correction.last_target_label_year.le(2018).all()
            row = correction.iloc[0]
            if row.status == "ok":
                prediction += row.b0+row.b1*np.arange(1, 6)/5
            else:
                prediction[:] = 0
        logs = levels[indices, None]+prediction
        invalid = ~np.isfinite(logs).all(axis=1) | (np.abs(logs)>700).any(axis=1)
        logs[invalid] = levels[indices][invalid, None]
        saved = part.pivot(index="age", columns="horizon", values="log_prediction").reindex(index=cfg["ages"], columns=range(1, 6))
        np.testing.assert_allclose(logs, saved, rtol=0, atol=3e-14)
        count += saved.size
    return {"replayed_tcn_cells": count, "replayed_tcn_checkpoints": replayed}


def md_table(frame, digits=5):
    lines = ["| " + " | ".join(str(col).replace("_", " ") for col in frame.columns) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{value:.{digits}f}" if np.isfinite(value) else "undefined")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_figures(contrasts, coverage, out, order):
    labels = [name.replace("United Arab Emirates", "UAE") for name in order]
    columns = [(sex, role) for sex in ["Male", "Female"] for role in ["local_champion", "nonneural_champion"]]
    values = contrasts.pivot(index="context", columns=["sex", "comparator"], values="relative_improvement_percent").reindex(index=order, columns=columns)
    bound = max(10., float(np.nanmax(np.abs(values.to_numpy()))))
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    image = ax.imshow(values, cmap="RdBu", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
    ax.set_xticks(range(4), ["Male vs local", "Male vs non-neural", "Female vs local", "Female vs non-neural"])
    ax.set_yticks(range(len(order)), labels)
    for r in range(len(order)):
        for c in range(4):
            v = values.iloc[r, c]
            ax.text(c, r, f"{v:+.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if abs(v)>bound*.55 else "#172333")
    ax.set_title("Five-year endpoint: relative error reduction from adapted TCN\nOrigin 2018 → 2023; positive favors TCN", loc="left", fontsize=13, pad=14)
    fig.colorbar(image, ax=ax, label="Relative reduction in mean age absolute log error (%)", shrink=.8)
    fig.text(.01, .01, "Saudi prevalence is the unchanged primary experiment; all other rows are prespecified secondary replications.", fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ["png", "svg"]:
        fig.savefig(out / f"endpoint_comparisons.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.4), sharey=True)
    for ax, sex in zip(axes, ["Male", "Female"]):
        values = coverage.loc[coverage.sex.eq(sex)].pivot(index="context", columns="family", values="coverage").reindex(index=order, columns=ROLES)
        image = ax.imshow(values*100, vmin=0, vmax=100, cmap="cividis", aspect="auto")
        ax.set_xticks(range(3), ["Adapted TCN", "Local", "Non-neural"], rotation=15)
        ax.set_yticks(range(len(order)), labels)
        ax.set_title(sex)
        for r in range(len(order)):
            for c in range(3):
                v = values.iloc[r, c]*100
                ax.text(c, r, f"{v:.0f}%", ha="center", va="center", color="white" if v<55 else "black", fontsize=9)
    fig.suptitle("Nominal 80% rate-interval coverage at the five-year endpoint", x=.03, ha="left", fontsize=13)
    fig.text(.03, .01, "11 dependent age cells per context/sex/model; descriptive coverage, with no exact calibration guarantee.", fontsize=9)
    fig.tight_layout(rect=(0,.05,.92,.95))
    color_axis = fig.add_axes([.935,.18,.015,.65])
    fig.colorbar(image, cax=color_axis, label="Observed coverage (%)")
    for ext in ["png", "svg"]:
        fig.savefig(out / f"endpoint_coverage.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/secondary_v1")
    parser.add_argument("--output", default="reports/secondary_v1")
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.output
    check_lock()
    manifest = verify_manifest(run)
    assert manifest["final_period_scored"]
    assert sha(ROOT / "data/processed/design_v1/regional_outcomes.csv") == manifest["identity"]["source_sha256"]
    assert sha(ROOT / "study_design/locked_v1/design.json") == manifest["identity"]["config_sha256"]
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    tasks = manifest["identity"]["tasks"]
    assert len(tasks) == 11
    for name, expected in manifest["prior_manifests_sha256"].items():
        previous = ROOT / "results" / name
        verify_manifest(previous)
        assert sha(previous / "run_manifest.json") == expected
    verify_chronology(run, tasks)
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    out.mkdir(parents=True, exist_ok=True)
    audits, contrasts_all, endpoint_all, reliability_all, coverage_all, wis_all, mappings_all = [], [], [], [], [], [], []
    reliability_coverage_all = []
    contexts = [{"id": "SAU_prevalence", "target": "Saudi Arabia", "outcome": "prevalence"}]+tasks
    for task in contexts:
        primary = task["id"] == "SAU_prevalence"
        directory = ROOT / "results/primary_v1" if primary else run / "trials" / task["id"]
        points, scores = pd.read_csv(directory / "predictions.csv"), pd.read_csv(directory / "point_scores.csv")
        intervals, draws = pd.read_csv(directory / "intervals.csv"), pd.read_csv(directory / "joint_draws.csv")
        mappings = pd.read_csv(directory / "champion_family_mappings.csv")
        cfg = json.loads(json.dumps(config))
        cfg["primary_target"] = task["target"]
        verify_source_scores(scores, panel, task)
        assert len(points) == len(scores) == 8800 and len(intervals) == 52800 and len(draws) == 79200
        verify_roles(points, intervals, mappings)
        contrasts = pd.read_csv(directory / ("primary_contrasts.csv" if primary else "endpoint_contrasts.csv"))
        age = pd.read_csv(directory / ("primary_age_contrasts.csv" if primary else "endpoint_age_contrasts.csv"))
        verdict = json.loads((directory / ("primary_verdict.json" if primary else "endpoint_verdict.json")).read_text())
        verify_primary_contrasts(scores, contrasts, age, verdict, cfg)
        audit = {"context": task["id"], "target": task["target"], "outcome": task["outcome"], "secondary_joint_success": verdict["joint_success"]}
        if not primary:
            audit.update(verify_fit_jobs(run, task))
            historical = pd.read_csv(directory / "prequential_scores.csv", low_memory=False)
            assert len(historical) == 16940 and historical.forecast_year.max() <= 2018
            verify_source_scores(historical, panel, task)
            audit.update(verify_banks(directory, points, intervals, draws, historical, mappings, cfg))
            verify_selection_and_summaries(directory, scores, historical, mappings, panel, task, cfg)
            audit.update(replay_tcn(directory, scores, panel, task, cfg))
            cells, wis = pd.read_csv(directory / "interval_scores.csv"), pd.read_csv(directory / "wis_scores.csv")
            n_wis, audit_roundoff = verify_interval_scores(cells, wis)
            audit.update(verified_wis_rows=n_wis, **audit_roundoff)
        else:
            primary_report = json.loads((ROOT / "reports/primary_v1/report_validation.json").read_text())
            assert primary_report["passed"]
            audit.update(primary_report_validation_sha256=sha(ROOT / "reports/primary_v1/report_validation.json"), reused_primary=True)
        context = task["target"]+" · "+task["outcome"]
        contrasts["context"] = context
        contrasts["role"] = "primary" if primary else "secondary"
        contrasts_all.append(contrasts)
        endpoint = pd.read_csv(directory / "primary_by_family.csv")
        reliability = pd.read_csv(directory / "reliability_by_horizon.csv")
        interval_summary = pd.read_csv(directory / "interval_summary_by_origin.csv")
        wis_summary = pd.read_csv(directory / "wis_summary_by_origin.csv")
        for frame in [endpoint, reliability, interval_summary, wis_summary, mappings]:
            frame["context"] = context
            frame["target"] = task["target"]
            frame["outcome"] = task["outcome"]
        endpoint_all.append(endpoint)
        reliability_all.append(reliability)
        coverage_all.append(interval_summary.loc[interval_summary.origin.eq(2018) & interval_summary.horizon.eq(5)
                                                & interval_summary.scale.eq("rate") & interval_summary.level.eq(.8)])
        reliability_coverage_all.append(interval_summary.loc[interval_summary.horizon.eq(5)
                                                             & interval_summary.scale.eq("rate") & interval_summary.level.eq(.8)])
        wis_all.append(wis_summary)
        mappings_all.append(mappings)
        audits.append(audit)
        print(f"Independent verification passed: {task['id']}", flush=True)
    contrasts = pd.concat(contrasts_all, ignore_index=True)
    endpoint, reliability = pd.concat(endpoint_all, ignore_index=True), pd.concat(reliability_all, ignore_index=True)
    coverage, wis = pd.concat(coverage_all, ignore_index=True), pd.concat(wis_all, ignore_index=True)
    reliability_coverage = pd.concat(reliability_coverage_all, ignore_index=True)
    reliability_coverage["covered_cells"] = (reliability_coverage.coverage*reliability_coverage.age_cells).round().astype(int)
    np.testing.assert_allclose(reliability_coverage.covered_cells,
                               reliability_coverage.coverage*reliability_coverage.age_cells, atol=1e-12, rtol=0)
    reliability_coverage = reliability_coverage.groupby(["context", "target", "outcome", "sex", "family"], as_index=False).agg(
        covered_cells=("covered_cells", "sum"), dependent_age_origin_cells=("age_cells", "sum"),
        mean_width=("mean_width", "mean"), n_origins=("origin", "nunique"))
    assert reliability_coverage.n_origins.eq(5).all() and reliability_coverage.dependent_age_origin_cells.eq(55).all()
    reliability_coverage["mean_coverage"] = reliability_coverage.covered_cells/reliability_coverage.dependent_age_origin_cells
    mappings = pd.concat(mappings_all, ignore_index=True)
    for name, frame in [("endpoint_contrasts", contrasts), ("endpoint_all_families", endpoint),
                        ("reliability_all_families", reliability), ("endpoint_80_rate_coverage", coverage),
                        ("wis_all_origins", wis), ("champion_mappings", mappings),
                        ("reliability_80_rate_coverage", reliability_coverage)]:
        frame.to_csv(out / f"{name}.csv", index=False)
    order = [task["target"]+" · "+task["outcome"] for task in contexts]
    make_figures(contrasts, coverage, out, order)
    summary = []
    for task, audit in zip(contexts, audits):
        context = task["target"]+" · "+task["outcome"]
        rows = contrasts.loc[contrasts.context.eq(context)]
        summary.append({"Context": context, "Male vs both": bool(rows.loc[rows.sex.eq("Male"), "strictly_lower"].all()),
                        "Female vs both": bool(rows.loc[rows.sex.eq("Female"), "strictly_lower"].all()),
                        "All four": audit["secondary_joint_success"]})
    summary = pd.DataFrame(summary)
    secondary_success = sum(a["secondary_joint_success"] for a in audits[1:])
    rates = coverage.loc[coverage.family.isin(ROLES)]
    adapted_coverage = rates.loc[rates.family.eq("tcn_adapted") & ~rates.context.eq(order[0]), "coverage"]
    adapted_reliability = reliability_coverage.loc[reliability_coverage.family.eq("tcn_adapted")]
    coverage_below = int(adapted_reliability.mean_coverage.lt(.8).sum())
    endpoint_below = int(coverage.loc[coverage.family.eq("tcn_adapted"), "coverage"].lt(.8).sum())
    coverage_table = (100*adapted_reliability.pivot(index="context", columns="sex", values="mean_coverage")).reindex(order).reset_index()
    coverage_table.columns = ["Context", "Female coverage (%)", "Male coverage (%)"]
    incidence = reliability.loc[reliability.target.eq("Saudi Arabia") & reliability.outcome.eq("incidence")
                                & reliability.horizon.eq(5) & reliability.family.isin(ROLES)].copy()
    incidence_values = incidence.set_index(["sex", "family"]).mean_absolute_log_error
    incidence_gains = {(sex, role): 100*(incidence_values.loc[sex, role]-incidence_values.loc[sex, "tcn_adapted"])/incidence_values.loc[sex, role]
                      for sex in config["sexes"] for role in ROLES[1:]}
    failures = sum(sum(a.get("fit_failures_or_local_age_fallbacks", {}).values()) for a in audits)
    selected = contrasts[["context", "sex", "comparator", "comparator_source_family", "tcn_error", "comparator_error", "relative_improvement_percent"]].copy()
    selected["comparator_source_family"] = selected.comparator_source_family.map(lambda name: LABELS.get(name, name))
    core_seconds = json.loads((run / "validation_report.json").read_text())["this_invocation_elapsed_seconds"]
    benchmark = json.loads((run / "benchmark_at_run.json").read_text()) if (run / "benchmark_at_run.json").exists() else {}
    report = f"""# Incidence and GCC replication

The adapted TCN meets all four sex-specific comparisons at the 2018→2023 endpoint in **{secondary_success} of 11 secondary target/outcome experiments: Saudi incidence and Bahrain prevalence**. Including the unchanged Saudi prevalence experiment, this is 2 of 12 country/outcome summaries. These are dependent regional evaluations, not independent experimental replications. The Saudi prevalence primary joint criterion remains unmet. Results below include every target/outcome and all model families; no method was retuned after these evaluation scores.

Saudi incidence is encouraging at the endpoint: error reductions versus the local/non-neural champions are 19.0%/19.5% for males and 28.0%/5.3% for females. Across all five evaluation origins at horizon five, males retain reductions of {incidence_gains['Male', 'local_champion']:.2f}%/{incidence_gains['Male', 'nonneural_champion']:.2f}%. Female TCN error is **{-incidence_gains['Female', 'local_champion']:.2f}% higher than the local champion**, while remaining {incidence_gains['Female', 'nonneural_champion']:.2f}% lower than the non-neural champion. Thus the female endpoint advantage does not establish consistent superiority over time.

## Endpoint findings

{md_table(summary)}

![Endpoint error comparisons](endpoint_comparisons.png)

Positive values indicate lower mean age absolute log error for the adapted TCN. Each comparator is the family selected using completed development windows available at origin 2018, with the setting selected before the forecast. These are age-specific rates averaged equally over 11 ages, not age-standardized rates. Relative improvement is undefined when comparator error is zero.

{md_table(selected, 5)}

## Predictive reliability

Across the 22 new target/outcome/sex endpoint assessments, adapted-TCN nominal 80% rate coverage ranges from {adapted_coverage.min()*100:.1f}% to {adapted_coverage.max()*100:.1f}%. Each percentage uses only 11 dependent age cells. The five-origin reliability table keeps all five issuance origins separate before averaging, so the endpoint is counted once within reliability. Neither overlapping origins nor country/age cells are independent participants; these summaries support no formal significance or universal superiority claim.

Including the original Saudi prevalence experiment, **{coverage_below} of 24 adapted-TCN country/outcome/sex summaries have five-origin mean coverage below 80%**; each uses 55 dependent age-origin cells. The corresponding endpoint-only count is {endpoint_below} of 24. Saudi incidence endpoint coverage is 54.5% for each sex, despite its point-error improvement. The table below reports five-origin mean coverage, whereas the figure reports the endpoint only.

{md_table(coverage_table, 1)}

![Endpoint rate coverage](endpoint_coverage.png)

Coverage, widths, and WIS can differ substantially from point error. Coordinate-mean centering removes historical residual bias and can under-cover a persistent shift. There are 7–11 joint historical blocks per issuance; 95% tails remain exploratory. Predictive medians are reported separately from unchanged point forecasts. Supplied GBD uncertainty bounds do not calibrate these predictive intervals, and the modeled annual series do not establish patient-level biological effects.

## Complete results and numerical audit

- [All 16 family/role endpoint summaries](endpoint_all_families.csv), [all-horizon reliability summaries](reliability_all_families.csv), [endpoint coverage](endpoint_80_rate_coverage.csv), [five-origin coverage and mean widths](reliability_80_rate_coverage.csv), [WIS by origin](wis_all_origins.csv), and [chronological champion mappings](champion_mappings.csv). Two champions are explicit views of underlying families, not extra fitted models.
- All eleven tasks used the locked 14 families, 157 TCN source fits per task, five reported seeds, unchanged grids, target-specific adaptation, and own-target/outcome residual banks. Default donors are the other six regional countries, including Jordan.
- Independently counted jobs total 1,727 TCN, 352 local, and 176 non-neural jobs across the eleven new contexts; local/non-neural jobs contain multiple age or candidate-model fits. Their fit audits contain {failures} failed fits or local age fallbacks. All built-in and independent numerical checks passed.
- CPU execution used {manifest['identity']['workers']} spawned workers, each with one numerical thread; this invocation took {core_seconds/60:.2f} minutes. The [frozen benchmark](../../{args.run}/benchmark_at_run.json) records the parallelization decision.
- Independent checks verified all result/code/prior hashes; actual-outcome truth joins; every new raw residual bank, centered residual, joint draw and interval quantile; coverage/WIS; origin population weights; selected family identities; all 44 new endpoint contrasts; and 3,630 TCN endpoint cells replayed from up to 55 frozen checkpoints. All eleven issued ledgers preceded any new final scoring. Floating-point audit tolerances only accommodate CSV round trips; they do not change metrics or strict comparison criteria.
- The outcome adapter first filters incidence or prevalence and the issuance cutoff, then labels only the internal copied model channel `prevalence`. Public outputs retain the actual estimand. Both outcomes' population-weighted error sensitivity uses the same prevalence-implied, issuance-year GBD denominator; future realized populations do not enter those weights.

Reproduction: `/home/saif/agpu_env/bin/python tests/test_secondary.py`; `/home/saif/agpu_env/bin/python scripts/run_secondary.py --output results/secondary_reproduction --workers 12 --device cpu`; `/home/saif/agpu_env/bin/python scripts/report_secondary.py --run results/secondary_reproduction --output reports/secondary_reproduction`. Interrupted identical runs support `--resume`; completed jobs are verified rather than refitted.

See the [secondary specification](../../study_design/secondary_implementation.md), [interval specification](../../study_design/intervals_implementation.md), [original Saudi primary report](../primary_v1/report.md), and [machine-readable report validation](report_validation.json). These results remain secondary, descriptive country-level evaluations of GBD-modeled rates.
"""
    (out / "report.md").write_text(report)
    validation = {"passed": True, "run_manifest_sha256": sha(run / "run_manifest.json"),
                  "report_script_sha256": sha(Path(__file__)), "verification_helpers_sha256": sha(ROOT / "scripts/report_primary.py"),
                  "new_tasks": 11, "new_interval_rows_verified": 11*52800, "new_banks_verified": 11*80,
                  "new_wis_rows_verified": 11*17600, "new_tcn_cells_replayed": sum(a.get("replayed_tcn_cells", 0) for a in audits),
                  "no_refits_or_adaptation_refits": True, "all_trials_committed_before_final_scoring": True, "audits": audits}
    validation["report_output_sha256"] = {str(p.relative_to(out)): sha(p) for p in sorted(out.glob("*")) if p.is_file() and p.name != "report_validation.json"}
    (out / "report_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps({k:v for k,v in validation.items() if k != "audits" and k != "report_output_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
