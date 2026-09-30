"""Independently audit and report stage-3 development artifacts without refitting."""

import json
import os
from pathlib import Path
import sys

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gbd_park_matplotlib")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gbd_park.pooled import target_inputs
from gbd_park.tcn import load_checkpoint, predict_changes, state_fingerprint
from gbd_park.tcn_forecasting import base_id
from run_local_baselines import check_lock, sha

RUN = ROOT / "results/tcn_v1"
LOCAL = ROOT / "results/local_baselines_v1"
NONNEURAL = ROOT / "results/nonneural_v1"
OUT = ROOT / "reports/tcn_v1"
LABELS = {"tcn_adapted": "TCN, two-coefficient adaptation",
          "tcn_unadapted": "TCN, no adaptation", "tcn_intercept": "TCN, intercept adaptation"}
BASELINE_LABELS = {
    "persistence": "Persistence", "log_trend": "Log-linear trend", "damped_ets": "Damped ETS",
    "arima": "ARIMA", "age_smooth_trend": "Age-smoothed trend", "pooled_ridge": "Pooled ridge",
    "pooled_boosting": "Pooled boosting", "donor_ridge_unadapted": "Donor ridge, unadapted",
    "donor_ridge_adapted": "Donor ridge, adapted", "donor_boosting_unadapted": "Donor boosting, unadapted",
    "donor_boosting_adapted": "Donor boosting, adapted",
}


def verify_run(directory):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["status"] == "complete", "Reporting requires a complete immutable run"
    for name, expected in manifest["output_sha256"].items():
        assert sha(directory / name) == expected, name
    for name, expected in manifest["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    return manifest


def verify_previous_manifests(manifest):
    expected = manifest["prior_manifests_sha256"]
    for name, value in expected.items():
        path = ROOT / "results" / name
        if path.is_dir():
            path = path / "run_manifest.json"
        assert sha(path) == value, name
    return {directory.name: verify_run(directory) for directory in [LOCAL, NONNEURAL]}


def replay_checkpoints(scores, config, decisions):
    """Rebuild origin-2013 log forecasts using five saved models and saved coefficients."""
    origin = 2013
    choice = next(row for row in decisions if row["fit_origin"] == origin)
    ident = base_id(choice["base"])
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(origin) & panel.outcome.eq("prevalence")]
    x, levels, meta = target_inputs(panel, config, origin, config["primary_target"])
    changes, payloads, frozen_checks = [], [], 0
    for seed in config["models"]["tcn"]["ensemble_seeds"]:
        filename = f"origin{origin}__{ident}__seed={seed}.joblib"
        payload = joblib.load(RUN / "payloads" / filename)
        assert payload["origin"] == origin and payload["seed"] == seed and payload["base"] == choice["base"]
        np.testing.assert_array_equal(levels, payload["levels"])
        assert meta.equals(payload["current_meta"])
        payloads.append(payload)
        if payload["audit"]["status"] != "ok":
            changes.append(np.zeros((len(meta), 5)))
            continue
        fitted = load_checkpoint(RUN / "checkpoints" / filename)
        before = state_fingerprint(fitted)
        prediction = predict_changes(fitted, x)
        np.testing.assert_array_equal(prediction, payload["current_changes"])
        assert state_fingerprint(fitted) == before == payload["audit"]["fingerprint_before"]
        assert before == payload["audit"]["fingerprint_after"]
        changes.append(prediction)
        frozen_checks += 1
    ensemble = np.mean(changes, axis=0)
    failed_seed = any(payload["audit"]["status"] != "ok" for payload in payloads)
    calibration = pd.read_json(RUN / "development_adaptation_audit.jsonl", lines=True)
    replayed = 0
    for (family, sex), part in scores.loc[scores.origin.eq(origin)].groupby(["family", "sex"]):
        indices = meta.sex.eq(sex).to_numpy()
        changes_for_sex = ensemble[indices].copy()
        assert part.base_id.eq(ident).all()
        if failed_seed:
            changes_for_sex[:] = 0
        elif family != "tcn_unadapted":
            cal = calibration.loc[calibration.origin.eq(origin) & calibration.family.eq(family)
                                  & calibration.sex.eq(sex) & calibration.base_id.eq(ident)]
            assert len(cal) == 1
            row = cal.iloc[0]
            if row.status == "ok":
                changes_for_sex += row.b0 + row.b1 * np.arange(1, 6) / 5
            else:
                changes_for_sex[:] = 0
        logs = levels[indices, None] + changes_for_sex
        invalid = ~np.isfinite(logs).all(axis=1) | (np.abs(logs) > 700).any(axis=1)
        logs[invalid] = levels[indices][invalid, None]
        expected = part.pivot(index="age", columns="horizon", values="log_prediction").reindex(
            index=meta.loc[indices, "age"], columns=range(1, 6)).to_numpy()
        np.testing.assert_allclose(logs, expected, rtol=0, atol=2e-14)
        replayed += expected.size
    return {"replayed_forecast_cells": replayed, "frozen_checkpoints_replayed": frozen_checks,
            "failed_seeds_retained": len(payloads) - frozen_checks}


def matched_effects(scores):
    keys = ["origin", "sex", "age", "horizon", "base_id", "ensemble_fingerprint"]
    errors = scores.pivot(index=keys, columns="family", values="absolute_log_error")
    assert errors.notna().all().all()
    rows = []
    for family in ["tcn_adapted", "tcn_intercept"]:
        part = errors.reset_index()[keys].copy()
        part["family"] = family
        part["adapted_error"] = errors[family].to_numpy()
        part["matched_unadapted_error"] = errors.tcn_unadapted.to_numpy()
        part["adaptation_loss_change"] = part.adapted_error - part.matched_unadapted_error
        rows.append(part)
    comparisons = pd.concat(rows, ignore_index=True)
    summary = comparisons.loc[comparisons.horizon.eq(5)].groupby(["sex", "family"], as_index=False).agg(
        mean_loss_change=("adaptation_loss_change", "mean"),
        worse_cells=("adaptation_loss_change", lambda value: int((value > 0).sum())),
        cells=("adaptation_loss_change", "size"))
    return comparisons, summary


def seed_spread(seed_predictions, decisions, config):
    seed_predictions = seed_predictions.copy()
    seed_predictions["seed"] = seed_predictions.seed_or_ensemble.astype(int)
    selected = []
    for decision in decisions:
        origin = decision["fit_origin"]
        if origin not in config["calendar"]["selection_origins"]:
            continue
        part = seed_predictions.loc[seed_predictions.origin.eq(origin)
                                    & seed_predictions.base_id.eq(base_id(decision["base"]))].copy()
        if "family" in part:
            part = part.loc[part.family.eq("tcn_unadapted")]
        selected.append(part)
    values = pd.concat(selected, ignore_index=True)
    assert set(values.seed.unique()) == set(config["models"]["tcn"]["ensemble_seeds"])
    group = values.groupby(["origin", "sex", "age", "horizon"])
    assert group.seed.nunique().eq(5).all() and group.size().eq(5).all()
    spread = group.log_prediction.agg(seed_sd="std", minimum="min", maximum="max").reset_index()
    spread["seed_range"] = spread.maximum - spread.minimum
    summary = spread.loc[spread.horizon.eq(5)].groupby("sex", as_index=False).agg(
        mean_seed_log_sd=("seed_sd", "mean"), maximum_seed_log_sd=("seed_sd", "max"),
        mean_seed_log_range=("seed_range", "mean"), cells=("seed_sd", "size"))
    return spread, summary


def main():
    check_lock()
    manifest = verify_run(RUN)
    verify_previous_manifests(manifest)
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    validation = json.loads((RUN / "validation_report.json").read_text())
    assert validation["passed"]
    tests_source = RUN / "tests_at_run.json"
    assert sha(tests_source) == manifest["test_report_sha256"]
    tests_bytes = tests_source.read_bytes()
    tests = json.loads(tests_bytes)
    predictions = pd.read_csv(RUN / "candidate_predictions.csv")
    scores = pd.read_csv(RUN / "development_scores.csv")
    summary = pd.read_csv(RUN / "development_summary.csv")
    origins = pd.read_csv(RUN / "scores_by_origin.csv")
    decisions = json.loads((RUN / "tuning_decisions.json").read_text())
    later = json.loads((RUN / "choices_for_later_evaluation.json").read_text())
    seed_predictions = pd.read_csv(RUN / "seed_predictions.csv")
    assert "observed_rate" not in predictions and "observed_rate" not in seed_predictions
    assert predictions.forecast_year.max() <= 2018 and scores.forecast_year.max() == 2018
    assert set(scores.origin.unique()) == set(config["calendar"]["selection_origins"])
    for choice in decisions + later:
        assert choice["last_inner_label_year"] is None or choice["last_inner_label_year"] <= choice["fit_origin"]
        assert all(origin + 5 <= choice["fit_origin"] for origin in choice["inner_origins"])
    np.testing.assert_allclose(scores.absolute_log_error,
                               np.abs(np.log(scores.prediction / scores.observed_rate)), atol=1e-14)
    calculated = scores.loc[scores.horizon.eq(5)].groupby(["sex", "family"]).absolute_log_error.mean()
    expected = summary.loc[summary.horizon.eq(5)].set_index(["sex", "family"]).mean_absolute_log_error
    np.testing.assert_allclose(calculated.sort_index(), expected.sort_index(), rtol=1e-12)
    events = [event["event"] for event in json.loads((RUN / "events.json").read_text())]
    assert events.index("candidate_predictions_committed") < events.index("candidate_scoring_started")
    assert events.index("development_predictions_committed") < events.index("development_scoring_started")
    replay = replay_checkpoints(scores, config, decisions)
    pairs, matched = matched_effects(scores)
    recorded_pairs = pd.read_csv(RUN / "matched_adaptation_comparisons.csv")
    pair_keys = ["origin", "sex", "age", "horizon", "ensemble_fingerprint", "family"]
    comparison_fields = ["adapted_error", "matched_unadapted_error", "adaptation_loss_change"]
    recalculated_pairs = pairs.set_index(pair_keys).sort_index()
    recorded_pairs = recorded_pairs.set_index(pair_keys).sort_index()
    assert recalculated_pairs.index.equals(recorded_pairs.index)
    np.testing.assert_allclose(recalculated_pairs[comparison_fields], recorded_pairs[comparison_fields], atol=1e-14)
    spread, spread_summary = seed_spread(seed_predictions, decisions, config)

    local_choice = pd.read_csv(LOCAL / "local_champions_for_later_evaluation.csv")
    local_choice = local_choice.loc[local_choice.fit_origin.eq(2018)].set_index("sex")
    nonneural_choice = pd.read_csv(NONNEURAL / "nonneural_champions_for_later_evaluation.csv")
    nonneural_choice = nonneural_choice.loc[nonneural_choice.fit_origin.eq(2018)].set_index("sex")
    baseline_summary = {"local": pd.read_csv(LOCAL / "development_summary.csv"),
                        "nonneural": pd.read_csv(NONNEURAL / "development_summary.csv")}
    baseline_origins = {"local": pd.read_csv(LOCAL / "scores_by_origin.csv"),
                        "nonneural": pd.read_csv(NONNEURAL / "scores_by_origin.csv")}
    choices = {"local": local_choice, "nonneural": nonneural_choice}
    contrasts = []
    for sex in config["sexes"]:
        tcn_loss = expected.loc[sex, "tcn_adapted"]
        for group in ["local", "nonneural"]:
            family = choices[group].loc[sex, "selected_family"]
            table = baseline_summary[group]
            loss = table.loc[table.sex.eq(sex) & table.family.eq(family) & table.horizon.eq(5), "mean_absolute_log_error"].item()
            contrasts.append({"sex": sex, "comparator_group": group, "comparator_family": family,
                              "comparator_error": loss, "adapted_tcn_error": tcn_loss,
                              "loss_difference_tcn_minus_comparator": tcn_loss - loss})
    contrast = pd.DataFrame(contrasts)
    final_choice = next(choice for choice in later if choice["fit_origin"] == 2018)

    benchmark_path = RUN / "benchmark_at_run.json"
    assert sha(benchmark_path) == manifest["benchmark_sha256"]
    benchmark = json.loads(benchmark_path.read_text())
    assert benchmark["status"] == "complete"
    hardware = benchmark["hardware_before"]
    gpu_names = ", ".join(gpu["name"] for gpu in hardware["gpu"]["gpus"])
    mode_lines = ["| Execution mode | Four-fit batch time, seconds | Fits/second | Sampled peak GPU memory |",
                  "|---|---:|---:|---:|"]
    for mode in benchmark["modes"]:
        peak = mode.get("sampled_peak_total_gpu_memory_fraction")
        memory = "Not applicable" if mode["device"] == "cpu" else (f"{peak:.1%}" if peak is not None else "Unavailable")
        mode_lines.append(f"| {mode['device']}, {mode['workers']} workers | "
                          f"{mode['whole_batch_seconds_including_spawn_shutdown_and_monitoring']:.1f} | "
                          f"{mode['fits_per_second']:.4f} | {memory} |")
    recommendation = benchmark["recommendation"]
    result_lines = ["| TCN configuration | Male | Female |", "|---|---:|---:|"]
    for family, label in LABELS.items():
        result_lines.append(f"| {label} | {expected.loc['Male', family]:.6f} | {expected.loc['Female', family]:.6f} |")
    contrast_lines = ["| Sex | Comparator selected for origin 2018 | Comparator error | Adapted TCN error | TCN − comparator |",
                      "|---|---|---:|---:|---:|"]
    for row in contrast.itertuples():
        contrast_lines.append(f"| {row.sex} | {BASELINE_LABELS[row.comparator_family]} | {row.comparator_error:.6f} | "
                              f"{row.adapted_tcn_error:.6f} | {row.loss_difference_tcn_minus_comparator:+.6f} |")
    matched_lines = ["| Sex | Correction versus the same ensemble without correction | Mean error change | Cells with higher error |",
                     "|---|---|---:|---:|"]
    for row in matched.itertuples():
        matched_lines.append(f"| {row.sex} | {LABELS[row.family]} | {row.mean_loss_change:+.6f} | {row.worse_cells}/{row.cells} |")
    spread_lines = ["| Sex | Mean seed SD of log prediction | Largest seed SD | Mean seed range |", "|---|---:|---:|---:|"]
    for row in spread_summary.itertuples():
        spread_lines.append(f"| {row.sex} | {row.mean_seed_log_sd:.6f} | {row.maximum_seed_log_sd:.6f} | {row.mean_seed_log_range:.6f} |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tests_at_run.json").write_bytes(tests_bytes)
    contrast.to_csv(OUT / "development_comparator_contrasts.csv", index=False)
    pairs.to_csv(OUT / "matched_adaptation_recalculated.csv", index=False)
    matched.to_csv(OUT / "matched_adaptation_summary.csv", index=False)
    spread.to_csv(OUT / "seed_prediction_spread.csv", index=False)
    spread_summary.to_csv(OUT / "seed_spread_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    maximum = 0
    colors = {"tcn_adapted": "#0369a1", "tcn_unadapted": "#a855f7", "tcn_intercept": "#08916a"}
    for axis, sex in zip(axes, config["sexes"]):
        for family, label in LABELS.items():
            part = origins.loc[origins.sex.eq(sex) & origins.family.eq(family) & origins.horizon.eq(5)].sort_values("origin")
            axis.plot(part.origin, part.mean_age_absolute_log_error, color=colors[family],
                      marker="o", markersize=3, linestyle="--" if family == "tcn_unadapted" else "-", label=label)
            maximum = max(maximum, part.mean_age_absolute_log_error.max())
        for group, color, marker in [("local", "#475569", "s"), ("nonneural", "#c65f00", "D")]:
            family = choices[group].loc[sex, "selected_family"]
            table = baseline_origins[group]
            part = table.loc[table.sex.eq(sex) & table.family.eq(family) & table.horizon.eq(5)].sort_values("origin")
            axis.plot(part.origin, part.mean_age_absolute_log_error, color=color, marker=marker, markersize=3,
                      label=f"Selected {group} comparator")
            maximum = max(maximum, part.mean_age_absolute_log_error.max())
        axis.set_title(sex)
        axis.set_xticks(config["calendar"]["selection_origins"])
        axis.set_xlabel("Forecast origin (five-year endpoint)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0, maximum * 1.08)
    axes[0].set_ylabel("Mean absolute log error across 11 ages")
    fig.suptitle("Saudi prevalence: historical TCN development errors", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.15, 1, 0.94])
    fig.savefig(OUT / "development_errors.png", dpi=180)
    fig.savefig(OUT / "development_errors.svg")
    plt.close(fig)

    report = f"""# Stage 3 — Compact TCN and limited Saudi adaptation

Completed under [locked design v1.0](../../study_design/locked_v1/protocol.md), using `/home/saif/agpu_env/bin/python`. This stage evaluates historical development forecasts of Saudi age-specific Parkinson’s prevalence. The final 2019–2023 forecast evaluation, interval assessment, and future projections remain pending.

## Historical development results

The metric is mean absolute log error at horizon five, averaged equally across eleven age bands and development origins 2009–2013 (endpoints 2014–2018). Lower is better. Values are log-scale losses, not percentage errors. The three TCN rows use the same chronologically selected base settings and five-seed ensemble; the latter two are matched ablations of the primary two-coefficient procedure.

{chr(10).join(result_lines)}

![TCN development errors and selected comparators](development_errors.png)

Comparison with the sex-specific comparator families selected for the later 2018-origin evaluation:

{chr(10).join(contrast_lines)}

These are descriptive development comparisons, not final test results or evidence of statistical or clinical significance. The comparator curves show the families selected for origin 2018 across their historical development forecasts; they do not represent a new family selection at each plotted origin. No final-period outcome selected the TCN settings or comparator families.

Using only eligible completed inner blocks, the TCN setting prepared for the later 2018-origin fit is **{final_choice['base']['channels']} channels, weight decay {final_choice['base']['weight_decay']}, {final_choice['base']['epochs']} epochs**. Its adaptation penalties are **{final_choice['penalties']['Male']} for males** and **{final_choice['penalties']['Female']} for females**. This is a saved choice, not an already fitted final-origin model or evaluated forecast.

## Matched adaptation and seed variation

Corrections are fitted after averaging the five seed predictions in log space. Encoder, head, and source scaler remain frozen. Negative changes below mean improvement over the exact same ensemble without correction. The intercept-only correction uses the primary procedure’s selected penalty without separate outcome-driven retuning.

{chr(10).join(matched_lines)}

Each row comprises 55 dependent age–origin cells. Counts describe where correction increased error; they are not independent participants and do not support a binomial significance test. Alternative donor pools and GCC target replications remain later analyses.

Unadapted predictions from the five prescribed seeds provide an optimization-stability diagnostic:

{chr(10).join(spread_lines)}

Seed SD uses the sample standard deviation across all five seeds for each age–origin cell at horizon five. The table summarizes those 55 cells per sex. Seed spread is not a prediction interval, GBD uncertainty estimate, or measure of biological variability. No seed was selected according to forecast accuracy.

## Parallel execution and measured hardware

The benchmark trained the same four donor-only real-data jobs at origin 2013, each using 32 channels and 50 epochs. It measured whole-batch time including worker startup, shutdown, and monitoring. Hardware: **{hardware['cpu_model']}** ({hardware['logical_cpus']} logical CPUs), **{gpu_names}**, PyTorch **{hardware['torch']}**, CUDA build **{hardware['torch_cuda_build']}**.

{chr(10).join(mode_lines)}

The throughput recommendation was **{recommendation['device']} with {recommendation['workers']} workers**, subject to sampled total GPU memory below 60% for GPU candidates. The development run used **{manifest['device']} with {manifest['workers']} workers** and took **{validation['elapsed_seconds']:.1f} seconds**. CPU and GPU are allowed to produce different fitted weights; each run records its actual device, consistent with [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.8/notes/randomness.html). Repeated same-seed CUDA smoke fits produced identical state fingerprints and predictions. Memory sampling can miss brief peaks; allocator peaks are also retained in the benchmark.

## Validation and preserved evidence

- All {tests['tests_run']} implementation tests passed. They cover causal inputs, parameter counts, deterministic fitting, source-only scaling, country/sex exclusions, temporal eligibility, frozen adaptation, missing-cell handling, and checkpoint preservation, with the exact assertions recorded in the test source.
- Saved {validation['candidate_predictions']:,} candidate forecast cells and {validation['development_predictions']:,} selected development forecast cells. Recorded {validation['base_fits']:,} source fits; fit failures: {validation['failures']}; selected persistence-fallback cells: {validation['selected_fallback_cells']}.
- The primary pool excludes all Saudi ages and both sexes from pretraining and scaling. Two coefficients per target sex are then shared across ages. Models contain 1,782 or 6,566 parameters, below the locked cap of 10,000.
- Every reported ensemble retains the prescribed seeds 11, 23, 37, 53, and 71. Training and correction labels end no later than each fit origin; setting choices use completed five-year inner blocks only.
- Forecast ledgers contain no verification values and were committed before scoring. Independent metric calculations reproduced the summaries. Saved checkpoints and coefficients reproduced {replay['replayed_forecast_cells']} origin-2013 forecast cells without refitting; {replay['frozen_checkpoints_replayed']} frozen model fingerprints were checked.
- Source/configuration hashes, locked artifacts, and earlier local/non-neural run manifests passed verification. Maximum scored year remains 2018. No forecast interval or final-origin score is produced here.

## Limits and next stage

These models forecast revised GBD-estimated rates rather than individual diagnoses, survival, or biological mechanisms. Age, sex, countries, and overlapping forecast windows are dependent. Descriptive 2023 outcomes were inspected during study design; later evaluation is retrospective and is not described as blinded. Development gains or losses do not establish the primary final-period result.

Stage 4 builds eligible historical ensemble forecasts and regional interval banks. In particular, the early origins 2003–2008 currently supply seed-11 tuning candidates; they still need chronologically selected five-seed ensembles for the interval procedure. Stage 5 then evaluates nested reliability and the Saudi final origin under the frozen rules. Incidence, donor comparisons, GCC replication, demographic analyses, and projections follow their assigned stages.

## Artifacts and reproduction

- [Forecast candidates](../../results/tcn_v1/candidate_predictions.csv), [development forecasts](../../results/tcn_v1/development_predictions.csv), [age-specific scores](../../results/tcn_v1/development_scores.csv), and [summary](../../results/tcn_v1/development_summary.csv).
- [Development choices](../../results/tcn_v1/tuning_decisions.json), [later-evaluation settings](../../results/tcn_v1/choices_for_later_evaluation.json), [seed forecasts](../../results/tcn_v1/seed_predictions.csv), and [matched adaptation contrasts](matched_adaptation_summary.csv).
- [Saved model checkpoints](../../results/tcn_v1/checkpoints/), [payloads](../../results/tcn_v1/payloads/), [validation](../../results/tcn_v1/validation_report.json), and [immutable run manifest](../../results/tcn_v1/run_manifest.json).
- [Frozen hardware benchmark](../../results/tcn_v1/benchmark_at_run.json), [test evidence](tests_at_run.json), [report verification](report_validation.json), and [implementation specification](../../study_design/tcn_implementation.md).
- [Independent implementation review](../../work/tcn-validation/review.md).

```bash
/home/saif/agpu_env/bin/python tests/test_tcn.py
/home/saif/agpu_env/bin/python scripts/report_tcn.py
```

The report reads the completed `tcn_v1` artifacts without training or altering them.
"""
    (OUT / "report.md").write_text(report)
    report_validation = {"passed": True, "run_and_code_hashes_verified": True,
                         "locked_design_and_prior_runs_unchanged": True,
                         "independent_metric_recalculation_passed": True,
                         "prediction_before_score_order_passed": True,
                         "maximum_scored_year": 2018, **replay,
                         "run_manifest_sha256": sha(RUN / "run_manifest.json"),
                         "benchmark_report_sha256": sha(benchmark_path),
                         "report_script_sha256": sha(Path(__file__)),
                         "output_sha256": {path.name: sha(path) for path in sorted(OUT.iterdir())
                                           if path.is_file() and path.name != "report_validation.json"}}
    (OUT / "report_validation.json").write_text(json.dumps(report_validation, indent=2) + "\n")
    print(json.dumps(report_validation, indent=2))


if __name__ == "__main__":
    main()
