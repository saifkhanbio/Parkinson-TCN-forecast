"""Audit and report immutable stage-2 development results without refitting."""

import json
import os
from pathlib import Path
import sys

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
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
from gbd_park.adaptation import correction
from gbd_park.pooled import predict_changes, target_inputs
from run_local_baselines import check_lock, sha

RUN = ROOT / "results/nonneural_v1"
LOCAL = ROOT / "results/local_baselines_v1"
OUT = ROOT / "reports/nonneural_v1"
LABELS = {
    "pooled_ridge": "Pooled ridge",
    "pooled_boosting": "Pooled boosting",
    "donor_ridge_unadapted": "Donor ridge, unadapted",
    "donor_ridge_adapted": "Donor ridge, adapted",
    "donor_boosting_unadapted": "Donor boosting, unadapted",
    "donor_boosting_adapted": "Donor boosting, adapted",
}
LOCAL_LABELS = {"damped_ets": "Damped ETS", "arima": "ARIMA"}


def verify_run(directory):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    for name, expected in manifest["output_sha256"].items():
        assert sha(directory / name) == expected, name
    for name, expected in manifest["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    return manifest


def replay_checkpoints(scores, config):
    """Replay every selected 2013 family/sex using saved base and correction."""
    origin = 2013
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel[panel.year.le(origin) & panel.outcome.eq("prevalence")]
    x, levels, meta = target_inputs(panel, config, origin, config["primary_target"])
    calibrations = pd.read_json(RUN / "adaptation_audit.jsonl", lines=True)
    cells = 0
    for (_, sex, ident, model_id), part in scores[scores.origin.eq(origin)].groupby(
            ["family", "sex", "setting_id", "model_id"]):
        assert part.status.eq("ok").all(), "Replay expects successful selected fits"
        fitted = joblib.load(RUN / "checkpoints" / str(origin) / f"{model_id}.joblib")
        before = joblib.hash(fitted, hash_name="sha1")
        indices = meta.sex.eq(sex).to_numpy()
        log_rates = levels[indices, None] + predict_changes(fitted, x[indices])
        if part.adaptation.eq("two_parameter").all():
            selected = calibrations[calibrations.origin.eq(origin) & calibrations.sex.eq(sex)
                                    & calibrations.setting_id.eq(ident)]
            assert len(selected) == 1
            log_rates += correction(selected.iloc[0].to_dict())
        expected = part.pivot(index="age", columns="horizon", values="prediction").reindex(
            index=meta.loc[indices, "age"], columns=range(1, 6)).to_numpy()
        np.testing.assert_allclose(np.exp(log_rates), expected, rtol=1e-12)
        assert joblib.hash(fitted, hash_name="sha1") == before
        cells += expected.size
    return cells


def main():
    check_lock()
    manifest = verify_run(RUN)
    verify_run(LOCAL)
    assert sha(LOCAL / "run_manifest.json") == manifest["stage1_manifest_sha256"]
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    validation = json.loads((RUN / "validation_report.json").read_text())
    # Preserve the original test evidence when later reproductions rerun tests.
    test_source = OUT / "tests_at_run.json"
    if not test_source.exists():
        test_source = ROOT / "work/nonneural-validation/tests.json"
    assert sha(test_source) == manifest["test_report_sha256"]
    test_bytes = test_source.read_bytes()
    tests = json.loads(test_bytes)
    predictions = pd.read_csv(RUN / "candidate_predictions.csv")
    scores = pd.read_csv(RUN / "development_scores.csv")
    summary = pd.read_csv(RUN / "development_summary.csv")
    origins = pd.read_csv(RUN / "scores_by_origin.csv")
    pairs = pd.read_csv(RUN / "matched_adaptation_comparisons.csv")
    champions = pd.read_csv(RUN / "nonneural_champions_for_later_evaluation.csv")
    local_champions = pd.read_csv(LOCAL / "local_champions_for_later_evaluation.csv")
    local_summary = pd.read_csv(LOCAL / "development_summary.csv")
    fits = pd.read_json(RUN / "base_fit_audit.jsonl", lines=True)
    calibrations = pd.read_json(RUN / "adaptation_audit.jsonl", lines=True)
    assert "observed_rate" not in predictions
    assert predictions.forecast_year.max() == scores.forecast_year.max() == 2018
    assert champions.last_inner_label_year.le(champions.fit_origin).all()
    assert champions.last_selection_target_year.le(champions.fit_origin).all()
    np.testing.assert_allclose(scores.absolute_log_error,
                               np.abs(np.log(scores.prediction / scores.observed_rate)), atol=1e-14)
    independently = scores[scores.horizon.eq(5)].groupby(["sex", "family"]).absolute_log_error.mean()
    expected = summary[summary.horizon.eq(5)].set_index(["sex", "family"]).mean_absolute_log_error
    np.testing.assert_allclose(independently.sort_index(), expected.sort_index(), rtol=1e-12)
    np.testing.assert_allclose(pairs.adaptation_loss_change,
                               pairs.adapted_error - pairs.matched_unadapted_error, atol=1e-14)
    order = [e["event"] for e in json.loads((RUN / "events.json").read_text())]
    assert order.index("candidate_predictions_committed") < order.index("development_scoring_started")
    replayed = replay_checkpoints(scores, config)
    final_choice = champions[champions.fit_origin.eq(2018)].set_index("sex")
    local_choice = local_champions[local_champions.fit_origin.eq(2018)].set_index("sex")
    table = expected.unstack("sex")
    lines = ["| Non-neural family | Male | Female |", "|---|---:|---:|"]
    for family, label in LABELS.items():
        lines.append(f"| {label} | {table.loc[family, 'Male']:.6f} | {table.loc[family, 'Female']:.6f} |")
    contrasts = []
    for sex in config["sexes"]:
        nonneural = final_choice.loc[sex, "selected_family"]
        local = local_choice.loc[sex, "selected_family"]
        nonneural_loss = expected.loc[sex, nonneural]
        local_loss = local_summary.loc[local_summary.sex.eq(sex) & local_summary.family.eq(local)
                                       & local_summary.horizon.eq(5), "mean_absolute_log_error"].item()
        contrasts.append({"sex": sex, "nonneural_family": nonneural, "nonneural_loss": nonneural_loss,
                          "local_family": local, "local_loss": local_loss,
                          "loss_difference": nonneural_loss - local_loss})
    contrast_lines = ["| Sex | Selected non-neural comparator | Error | Selected local comparator | Error | Difference (non-neural − local) |",
                      "|---|---|---:|---|---:|---:|"]
    for row in contrasts:
        contrast_lines.append(f"| {row['sex']} | {LABELS[row['nonneural_family']]} | {row['nonneural_loss']:.6f} | "
                              f"{LOCAL_LABELS[row['local_family']]} | {row['local_loss']:.6f} | {row['loss_difference']:+.6f} |")
    matched = pairs[pairs.horizon.eq(5)].groupby(["sex", "family"], as_index=False).agg(
        mean_loss_change=("adaptation_loss_change", "mean"),
        worse_cells=("adaptation_loss_change", lambda x: int((x > 0).sum())),
        cells=("adaptation_loss_change", "size"))
    match_lines = ["| Sex | Adapted family | Mean error change | Age–origin cells with higher error |",
                   "|---|---|---:|---:|"]
    for row in matched.itertuples():
        match_lines.append(f"| {row.sex} | {LABELS[row.family]} | {row.mean_loss_change:+.6f} | "
                           f"{row.worse_cells}/{row.cells} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tests_at_run.json").write_bytes(test_bytes)
    pd.DataFrame(contrasts).to_csv(OUT / "development_comparator_contrasts.csv", index=False)
    matched.to_csv(OUT / "matched_adaptation_summary.csv", index=False)
    colors = ["#1d4ed8", "#d97706", "#0f766e", "#0f766e", "#9333ea", "#9333ea"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    local_origins = pd.read_csv(LOCAL / "scores_by_origin.csv")
    maximum = origins.loc[origins.horizon.eq(5), "mean_age_absolute_log_error"].max()
    for ax, sex in zip(axes, config["sexes"]):
        for (family, label), color in zip(LABELS.items(), colors):
            part = origins[origins.sex.eq(sex) & origins.family.eq(family) & origins.horizon.eq(5)].sort_values("origin")
            style = "--" if family.endswith("_unadapted") else "-"
            ax.plot(part.origin, part.mean_age_absolute_log_error, color=color, linestyle=style,
                    marker="o", markersize=3, label=label)
        local = local_choice.loc[sex, "selected_family"]
        part = local_origins[local_origins.sex.eq(sex) & local_origins.family.eq(local)
                             & local_origins.horizon.eq(5)].sort_values("origin")
        ax.plot(part.origin, part.mean_age_absolute_log_error, color="#333333", linewidth=2,
                marker="s", markersize=3, label="Selected local comparator")
        maximum = max(maximum, part.mean_age_absolute_log_error.max())
        ax.set_title(f"{sex} · local comparator: {LOCAL_LABELS[local]}")
        ax.set_xticks(range(2009, 2014))
        ax.set_xlabel("Forecast origin (five-year endpoint)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0, maximum * 1.08)
    axes[0].set_ylabel("Mean absolute log error across 11 ages")
    fig.suptitle("Saudi prevalence: historical non-neural development errors", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.14, 1, 0.94])
    fig.savefig(OUT / "development_errors.png", dpi=180)
    fig.savefig(OUT / "development_errors.svg")
    plt.close(fig)
    warnings_count = int(fits.warnings.map(bool).sum())
    adaptation_failures = int(calibrations.status.ne("ok").sum())
    report = f"""# Stage 2 — Pooled and donor-only non-neural controls

Completed under [locked design v1.0](../../study_design/locked_v1/protocol.md), using `/home/saif/agpu_env/bin/python`. This stage covers Saudi age-specific prevalence and historical development only. The primary 2019–2023 evaluation, intervals and future projections remain pending.

## Development results

The non-neural comparator selected for the later 2018-origin evaluation is **{LABELS[final_choice.loc['Male', 'selected_family']]} for males** and **{LABELS[final_choice.loc['Female', 'selected_family']]} for females**. Settings and families were selected from completed historical windows; no final-period outcome chose these comparators.

Mean absolute log error at horizon five, averaged equally across eleven age bands and five development origins (2009–2013; endpoints 2014–2018), is shown below. Lower is better. These values are not percentage errors.

{chr(10).join(lines)}

![Non-neural historical development errors](development_errors.png)

Comparison with the local families selected in stage 1:

{chr(10).join(contrast_lines)}

These are descriptive development comparisons used for model selection, not final test estimates or evidence of statistical or clinical significance. A selected comparator can vary at earlier reliability origins because fewer historical errors are then available. The local-comparator line shows the family selected for origin 2018 across its development origins, not a separately nested family choice at each plotted origin.

## Does limited Saudi adaptation help?

For each chronologically selected adapted setting, compare its forecasts with the exact same donor checkpoint without correction. Negative changes mean adaptation reduced error. This isolates the correction; comparing independently tuned adapted and unadapted families would also change base settings.

{chr(10).join(match_lines)}

Each row contains 55 dependent age–origin cells at horizon five. Cell counts describe where adaptation harmed forecasts; they are not independent participants or a basis for a binomial significance test. Results are conditional on the selected adapted settings and this historical period. Alternative donor pools and GCC target replications remain later analyses.

## Implementation and validation

- All {tests['tests_run']} implementation tests passed: chronological label eligibility, future-data invariance, donor fitting independent of Saudi histories, sex-specific adaptation, training-only scaling, balanced weights, exact correction checks against an independent optimizer, missing-cell rejection, and retained failure records.
- Saved {validation['candidate_predictions']:,} candidate forecast cells across 78 settings and eleven origins. Chronological tuning retained {validation['selected_development_predictions']:,} development cells across six families, two sexes, eleven ages and five horizons.
- Fitted {validation['base_fits']:,} base models and {validation['adaptations']:,} two-coefficient corrections. Base-fit failures: {validation['base_fit_failures']}; adaptation failures: {adaptation_failures}; selected persistence-fallback cells: {validation['selected_forecast_fallback_cells']}; base fits with recorded warnings: {warnings_count}.
- The pooled models used Saudi Arabia plus six regional countries. Donor-only models excluded all Saudi ages and both sexes from preprocessing and fitting. Each sex then received its own two-parameter correction shared over ages.
- Every fitted donor object retained its fingerprint after adaptation. All training/adaptation labels ended by the fit origin. Saved country/window weights, source-only scaling moments and correction coefficients allow inspection.
- Forecast ledgers were committed before scoring and contain no verification values. Independent recalculation reproduced the development error summaries; saved checkpoints and coefficients reproduced {replayed} selected forecast cells at origin 2013 without refitting.
- Original inputs, locked design and stage-1 artifacts passed their hash checks. Four CPU workers completed the modeling/scoring run in {validation['elapsed_seconds']:.1f} seconds within `agpu`. This does not measure neural GPU runtime.

## Limits and next stage

The primary model forecasts GBD-estimated rates, not individual diagnoses or biological mechanisms. GBD histories are retrospectively revised, and descriptive 2023 values were inspected during design; the final evaluation is not claimed to be blinded. This stage evaluates the fixed six-country donor pool only. Historical correction gains do not establish a neural transfer advantage, interval calibration, or final-period performance.

Stage 2 is complete. Stage 3 implements the compact TCN, GPU smoke test, frozen-weight Saudi adaptation and prescribed seed ensemble. The locked architecture/grid and final evaluation cutoffs remain unchanged.

## Artifacts and reproduction

- [Candidate predictions](../../results/nonneural_v1/candidate_predictions.csv), [selected development predictions](../../results/nonneural_v1/development_predictions.csv), [age-specific scores](../../results/nonneural_v1/development_scores.csv), and [summary](../../results/nonneural_v1/development_summary.csv).
- [Matched adaptation comparisons](../../results/nonneural_v1/matched_adaptation_comparisons.csv), [tuning choices](../../results/nonneural_v1/tuning_decisions.csv), and [comparators for later evaluation](../../results/nonneural_v1/nonneural_champions_for_later_evaluation.csv).
- [Base-fit audit](../../results/nonneural_v1/base_fit_audit.jsonl), [correction audit](../../results/nonneural_v1/adaptation_audit.jsonl), [training windows](../../results/nonneural_v1/training_window_audit.csv), and [saved checkpoints](../../results/nonneural_v1/checkpoints/).
- [Run validation](../../results/nonneural_v1/validation_report.json), [immutable run manifest](../../results/nonneural_v1/run_manifest.json), [original test evidence](tests_at_run.json), [report verification](report_validation.json), and [implementation specification](../../study_design/nonneural_implementation.md).

```bash
/home/saif/agpu_env/bin/python tests/test_nonneural.py
/home/saif/agpu_env/bin/python scripts/run_nonneural.py --output results/nonneural_reproduction
/home/saif/agpu_env/bin/python scripts/report_nonneural.py
```

The runner refuses an existing result directory. The report command reads the original `nonneural_v1` run without training or changing it.
"""
    (OUT / "report.md").write_text(report)
    report_validation = {
        "passed": True, "run_and_code_hashes_verified": True,
        "locked_design_and_stage1_unchanged": True,
        "independent_metric_recalculation_passed": True,
        "saved_checkpoint_replayed_cells": replayed,
        "prediction_before_score_order_passed": True,
        "maximum_scored_year": 2018,
        "run_manifest_sha256": sha(RUN / "run_manifest.json"),
        "report_script_sha256": sha(Path(__file__)),
        "output_sha256": {p.name: sha(p) for p in sorted(OUT.iterdir())
                          if p.is_file() and p.name != "report_validation.json"},
    }
    (OUT / "report_validation.json").write_text(json.dumps(report_validation, indent=2) + "\n")
    print(json.dumps(report_validation, indent=2))


if __name__ == "__main__":
    main()
