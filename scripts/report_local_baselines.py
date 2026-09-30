"""Create a report from immutable stage-1 outputs, without refitting models."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gbd_park_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/local_baselines_v1"
OUT = ROOT / "reports/local_baselines_v1"


def main():
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    for filename, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((RUN / filename).read_bytes()).hexdigest() == digest, filename
    for filename, digest in manifest["code_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
    validation = json.loads((RUN / "validation_report.json").read_text())
    summary = pd.read_csv(RUN / "development_summary.csv")
    origins = pd.read_csv(RUN / "scores_by_origin.csv")
    scores = pd.read_csv(RUN / "development_scores.csv")
    predictions = pd.read_csv(RUN / "candidate_predictions.csv")
    assert "observed_rate" not in predictions
    assert predictions.forecast_year.max() == 2018
    np.testing.assert_allclose(scores.absolute_log_error, np.abs(np.log(scores.prediction / scores.observed_rate)), atol=1e-14)
    independently = scores[scores.horizon.eq(5)].groupby(["sex", "family"]).absolute_log_error.mean()
    expected = summary[summary.horizon.eq(5)].set_index(["sex", "family"]).mean_absolute_log_error
    np.testing.assert_allclose(independently.sort_index(), expected.sort_index(), rtol=1e-12)
    labels = {"persistence": "Persistence", "log_trend": "Log-linear trend", "damped_ets": "Damped ETS",
              "arima": "ARIMA", "age_smooth_trend": "Age-smoothed trend"}
    colors = {"persistence": "#777777", "log_trend": "#d97706", "damped_ets": "#1d4ed8",
              "arima": "#0f766e", "age_smooth_trend": "#9333ea"}
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for ax, sex in zip(axes, ["Male", "Female"]):
        for family, label in labels.items():
            part = origins[origins.sex.eq(sex) & origins.family.eq(family) & origins.horizon.eq(5)].sort_values("origin")
            ax.plot(part.origin, part.mean_age_absolute_log_error, color=colors[family], marker="o", markersize=4, label=label)
        ax.set_title(sex)
        ax.set_xticks(range(2009, 2014))
        ax.set_xlabel("Forecast origin (five-year endpoint)")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, origins.loc[origins.horizon.eq(5), "mean_age_absolute_log_error"].max() * 1.08)
    axes[0].set_ylabel("Mean absolute log error across 11 ages")
    fig.suptitle("Saudi prevalence: historical development errors", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    fig.savefig(OUT / "development_errors.png", dpi=180)
    fig.savefig(OUT / "development_errors.svg")
    plt.close(fig)
    table = expected.unstack("sex")
    lines = ["| Model | Male | Female |", "|---|---:|---:|"]
    for family, label in labels.items():
        lines.append(f"| {label} | {table.loc[family, 'Male']:.6f} | {table.loc[family, 'Female']:.6f} |")
    champions = pd.read_csv(RUN / "local_champions_for_later_evaluation.csv")
    assert champions.last_inner_label_year.le(champions.fit_origin).all()
    assert champions.last_selection_target_year.le(champions.fit_origin).all()
    final_choice = champions[champions.fit_origin.eq(2018)].set_index("sex")
    events = json.loads((RUN / "events.json").read_text())
    order = [e["event"] for e in events]
    assert order.index("candidate_predictions_committed") < order.index("development_scoring_started")
    report = f"""# Stage 1 — Saudi-only prevalence baselines

Completed 29 September 2026 under [locked design v1.0](../../study_design/locked_v1/protocol.md), using the `agpu` environment. This report covers historical development only; no 2019–2023 final evaluation scores or future projections were produced.

## Result and interpretation

The local comparator selected for the later 2018-origin evaluation is **{labels[final_choice.loc['Male', 'selected_family']]} for males** and **{labels[final_choice.loc['Female', 'selected_family']]} for females**. Selection used the prespecified historical origins 2009–2013, whose five-year endpoints are 2014–2018. The female ARIMA and ETS scores are close; this selection is not a significance claim. Comparator choices at earlier reliability origins vary and are saved separately using their own information cutoffs.

The table reports mean absolute log error, averaged equally across eleven age bands and the five development origins, at horizon five. Lower is better. These numbers are **not percentage errors**, and results over these dependent observations do not establish general superiority.

{chr(10).join(lines)}

![Historical development errors](development_errors.png)

The full age-specific errors and original-rate errors are available in the score tables. The interval procedure and final-period reliability evaluation belong to subsequent stages. These local results provide strong comparators for pooled and transfer models; they do not establish that borrowing information helps.

## Completed checks

- All {validation['unit_tests']} scientific/implementation tests passed, including fitted-model invariance to future-row and other-country changes, exact log-trend extrapolation, metric examples, temporal tuning rules and retained numerical-failure records.
- Saved {validation['candidate_prediction_rows']:,} candidate forecasts from origins 2003–2013; selected {validation['selected_development_rows']:,} historical development forecasts across five families, two sexes, eleven ages and five horizons.
- Every required forecast cell is present. Selected development forecasts used **zero persistence fallbacks**.
- Raw files, locked design and prepared-input hashes are unchanged. Model/source/configuration versions and fitted parameters are recorded.
- Prediction ledgers were saved before scoring, remain free of verification values, and retained their hashes after scoring. Independent recalculation agrees with the reported log-error summaries.
- Four CPU workers completed fitting and scoring in approximately {validation['elapsed_seconds']:.1f} seconds. Statistical baselines run on CPU within `agpu`; GPU benchmarking of the actual neural implementation remains pending.

## Numerical limitations

Of 3,872 ARIMA candidate fits, 724 were rejected: 723 did not converge and one had a linear-algebra failure. Every age–sex–origin series still had an eligible candidate, giving 242 selected ARIMA fits. The order audit retains all candidates and their warnings; the selection result is conditional on the converged candidate set. No candidate was removed because of its forecast error. ETS fits used the optimizer and initialization documented before inspecting development results.

GBD histories are modeled, retrospectively revised estimates. The earlier design stage inspected descriptive 2023 values; no claim of a blinded final test is made. The current development scores use verification years no later than 2018 and should not be presented as final held-out results.

## Artifacts and next stage

- [Prediction ledger](../../results/local_baselines_v1/candidate_predictions.csv) and [selected development predictions](../../results/local_baselines_v1/development_predictions.csv).
- [Age-specific scores](../../results/local_baselines_v1/development_scores.csv), [summary](../../results/local_baselines_v1/development_summary.csv), and [origin/horizon scores](../../results/local_baselines_v1/scores_by_origin.csv).
- [Tuning decisions](../../results/local_baselines_v1/tuning_decisions.csv) and [local comparator choices](../../results/local_baselines_v1/local_champions_for_later_evaluation.csv).
- [ARIMA audit](../../results/local_baselines_v1/arima_candidate_audit.csv), [validation](../../results/local_baselines_v1/validation_report.json), and [run manifest](../../results/local_baselines_v1/run_manifest.json).
- [Implementation conventions and commands](../../study_design/local_baselines_implementation.md).

**Stage 1 is complete. Stage 2 is next:** implement pooled ridge/boosting and donor-only controls with matched inputs, chronological tuning, country/sex/age weights, and the same limited target calibration. Final-period scoring remains at stage 5 after the complete comparison and uncertainty procedures are implemented.
"""
    (OUT / "report.md").write_text(report)
    validation_out = {"run_hashes_verified": True, "prediction_ledger_has_no_truth": True,
                      "independent_metric_recalculation_passed": True, "maximum_scored_year": 2018,
                      "prediction_before_score_order_passed": True}
    (OUT / "report_validation.json").write_text(json.dumps(validation_out, indent=2) + "\n")
    print(json.dumps(validation_out))


if __name__ == "__main__":
    main()
