"""Audit and report a completed prespecified donor-pool sensitivity run."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gbd_park_matplotlib")
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
from gbd_park.pooled import target_inputs
from gbd_park.tcn import configure_torch, load_checkpoint, predict_changes
ARM_ORDER = ["all_six", "gcc_five", "similar_three", "random_three_101", "random_three_211",
             "random_three_307", "random_three_401", "random_three_503"]
ARM_LABELS = {"all_six": "All six (GPU reference)", "gcc_five": "GCC five", "similar_three": "Similar three",
              **{f"random_three_{seed}": f"Random three: {seed}" for seed in [101, 211, 307, 401, 503]}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_table(frame):
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(row) + " |" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/donor_comparisons_gpu_v1")
    parser.add_argument("--output", default="reports/donor_comparisons_gpu_v1")
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.output
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Report only a complete donor experiment")
    for name, expected in manifest["output_sha256"].items():
        if sha(run / name) != expected:
            raise ValueError(f"Output artifact changed: {name}")
    for name, expected in manifest["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Source artifact changed: {name}")
    if sha(ROOT / "results/primary_v1/run_manifest.json") != manifest["primary_reference_sha256"]:
        raise ValueError("Primary reference changed")
    scored = pd.read_csv(run / "point_scores.csv")
    predictions = pd.read_csv(run / "predictions.csv")
    cells = pd.read_csv(run / "interval_scores.csv")
    wis = pd.read_csv(run / "wis_scores.csv")
    harm = pd.read_csv(run / "donor_harm_cells.csv")
    adaptation = pd.read_csv(run / "matched_adaptation_cells.csv")
    contrasts = pd.read_csv(run / "endpoint_contrasts.csv")
    validation = json.loads((run / "validation_report.json").read_text())
    plans = json.loads((run / "donor_decisions.json").read_text())
    fits = json.loads((run / "fit_audit.json").read_text())
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    independently_computed = np.abs(np.log(scored.prediction) - np.log(scored.observed_rate))
    np.testing.assert_allclose(scored.absolute_log_error, independently_computed, atol=2e-12, rtol=1e-12)
    if predictions.duplicated(["arm", "origin", "family", "sex", "age", "horizon"]).any():
        raise ValueError("Duplicate donor prediction coordinates")
    if set(predictions.arm) != set(ARM_ORDER) or set(predictions.origin) != set(range(2014, 2019)):
        raise ValueError("Unexpected arms or evaluation origins")
    if not predictions.seed_or_ensemble.eq("11|23|37|53|71").all():
        raise ValueError("Reported forecasts omit fixed ensemble seeds")
    for plan in plans:
        if "Saudi Arabia" in plan["countries"] or plan["contributing_sexes"] != ["Male", "Female"]:
            raise ValueError("Invalid target exclusion or donor-sex scope")
        if plan["maximum_feature_year"] is not None and plan["maximum_feature_year"] > plan["origin"]:
            raise ValueError("Donor selection used future features")
    for fit in fits:
        if ("Saudi Arabia" in fit["countries"] or fit["maximum_label_year"] > fit["origin"]
                or fit["fingerprint_before"] != fit["fingerprint_after"]):
            raise ValueError("Source fit violated exclusion, chronology, or frozen-state rules")
    if not cells.n_blocks.eq(cells.origin - 2007).all():
        raise ValueError("Interval residual-block counts disagree with chronology")
    # Replay selected source models and complete adapted forecasts without
    # refitting, using the same original origin-restricted feature construction.
    origin = config["calendar"]["primary_origin"]
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(origin) & panel.outcome.eq("prevalence")]
    current_x, levels, current_meta = target_inputs(panel, config, origin, config["primary_target"])
    calibration = pd.read_csv(run / "adaptation_audit.csv")
    reproduced, replayed = {}, 0
    for arm in ["all_six", "similar_three"]:
        for sex in config["sexes"]:
            selected = predictions.loc[predictions.arm.eq(arm) & predictions.origin.eq(origin)
                                       & predictions.sex.eq(sex) & predictions.family.eq("tcn_adapted")]
            keys = selected.source_cache_keys.unique()
            if len(keys) != 1:
                raise ValueError("A selected sex forecast references multiple source ensembles")
            source_predictions, failed = [], False
            for key in keys[0].split("|"):
                payload = joblib.load(run / "cache" / f"{key}.joblib")
                if payload["audit"]["status"] != "ok":
                    failed = True
                    continue
                if key not in reproduced:
                    configure_torch(payload["seed"], "cpu")
                    fitted = load_checkpoint(run / "checkpoints" / f"{key}.joblib")
                    reproduced[key] = predict_changes(fitted, current_x)
                    np.testing.assert_array_equal(reproduced[key], payload["current_changes"])
                source_predictions.append(reproduced[key])
            if failed:
                continue
            correction = calibration.loc[calibration.arm.eq(arm) & calibration.origin.eq(origin)
                                         & calibration.sex.eq(sex) & calibration.family.eq("tcn_adapted")
                                         & calibration.seed_or_ensemble.eq("11|23|37|53|71")]
            if len(correction) != 1:
                raise ValueError("Missing or duplicate saved selected correction")
            correction = correction.iloc[0]
            mask = current_meta.sex.eq(sex).to_numpy()
            expected = levels[mask, None] + np.mean(source_predictions, axis=0)[mask]
            expected += correction.b0 + correction.b1 * np.arange(1, 6) / 5
            order = pd.MultiIndex.from_product([config["ages"], config["calendar"]["horizons"]], names=["age", "horizon"])
            actual = selected.set_index(["age", "horizon"]).reindex(order).log_prediction.to_numpy().reshape(11, 5)
            np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=1e-13)
            replayed += expected.size
    historical = pd.read_csv(run / "historical_scores.csv")
    interval_frame = pd.read_csv(run / "intervals.csv")
    joint_draw_frame = pd.read_csv(run / "joint_draws.csv")
    quantile_max_difference = 0.
    joint_draw_max_difference = 0.
    checked_interval_rows, checked_joint_draws = 0, 0
    coordinate_index = pd.MultiIndex.from_product([config["sexes"], config["ages"], config["calendar"]["horizons"]],
                                                 names=["sex", "age", "horizon"])
    for (arm, origin, family), issued in predictions.groupby(["arm", "origin", "family"], sort=True):
        past = historical.loc[historical.arm.eq(arm) & historical.family.eq(family)]
        error_blocks = []
        for residual_origin in range(2003, origin - 4):
            block = past.loc[past.origin.eq(residual_origin)].set_index(["sex", "age", "horizon"]).reindex(coordinate_index)
            error_blocks.append(np.log(block.observed_rate.to_numpy()) - block.log_prediction.to_numpy())
        error_blocks = np.stack(error_blocks)
        error_blocks -= error_blocks.mean(axis=0)
        current = issued.set_index(["sex", "age", "horizon"]).reindex(coordinate_index)
        log_draws = current.log_prediction.to_numpy()[None, :] + error_blocks
        joint_index = pd.MultiIndex.from_product([range(2003, origin - 4), config["sexes"], config["ages"],
                                                  config["calendar"]["horizons"]],
                                                 names=["residual_origin", "sex", "age", "horizon"])
        actual_draws = joint_draw_frame.loc[joint_draw_frame.arm.eq(arm) & joint_draw_frame.origin.eq(origin)
                                            & joint_draw_frame.family.eq(family)].set_index(joint_index.names).reindex(joint_index)
        expected_draws = np.column_stack([log_draws.ravel(), np.exp(log_draws).ravel()])
        np.testing.assert_allclose(actual_draws[["log_draw", "rate_draw"]], expected_draws, atol=1e-10, rtol=1e-12)
        joint_draw_max_difference = max(joint_draw_max_difference, float(np.max(
            np.abs(actual_draws[["log_draw", "rate_draw"]].to_numpy() - expected_draws))))
        checked_joint_draws += len(actual_draws)
        for scale, values in [("log_rate", log_draws), ("rate", np.exp(log_draws))]:
            for level in [.5, .8, .95]:
                expected = np.quantile(values, [(1 - level) / 2, .5, (1 + level) / 2], axis=0, method="linear").T
                actual = interval_frame.loc[interval_frame.arm.eq(arm) & interval_frame.origin.eq(origin)
                                            & interval_frame.family.eq(family) & interval_frame.scale.eq(scale)
                                            & interval_frame.level.eq(level)].set_index(["sex", "age", "horizon"]).reindex(coordinate_index)
                actual = actual[["lower", "median", "upper"]].to_numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)
                quantile_max_difference = max(quantile_max_difference, float(np.max(np.abs(actual - expected))))
                checked_interval_rows += len(actual)
    if checked_interval_rows != len(interval_frame) or checked_joint_draws != len(joint_draw_frame):
        raise ValueError("Independent reconstruction did not cover the complete interval/draw ledger")
    # Direct scoring formulas, without importing the production scoring helpers.
    alpha = 1 - cells.level.to_numpy()
    observed = cells.observed_value.to_numpy()
    lower, upper = cells.lower.to_numpy(), cells.upper.to_numpy()
    independent_interval_score = (upper - lower + 2 / alpha * np.maximum(lower - observed, 0)
                                  + 2 / alpha * np.maximum(observed - upper, 0))
    np.testing.assert_allclose(cells.interval_score, independent_interval_score, atol=1e-10, rtol=1e-11)
    np.testing.assert_array_equal(cells.covered.astype(bool), (observed >= lower) & (observed <= upper))
    score_keys = ["arm", "origin", "family", "sex", "age", "horizon", "scale"]
    checks = cells[score_keys + ["level"]].copy()
    checks["independent_interval_score"] = independent_interval_score
    pivot = checks.pivot(index=score_keys, columns="level", values="independent_interval_score")
    ordered_wis = wis.set_index(score_keys).reindex(pivot.index)
    median_error = np.abs(ordered_wis.observed_value - ordered_wis["median"])
    wis_differences = []
    for levels, name in [([.5, .8], "wis_50_80"), ([.5, .8, .95], "wis_50_80_95")]:
        independent_wis = (.5 * median_error + sum((1 - level) / 2 * pivot[level] for level in levels)) / (len(levels) + .5)
        np.testing.assert_allclose(ordered_wis[name], independent_wis, atol=1e-10, rtol=1e-11)
        wis_differences.append(float(np.max(np.abs(ordered_wis[name] - independent_wis))))
    endpoint = scored.loc[scored.family.eq("tcn_adapted") & scored.origin.eq(2018) & scored.horizon.eq(5)]
    endpoint_table = endpoint.groupby(["arm", "sex"]).absolute_log_error.mean().unstack("sex").reindex(ARM_ORDER)
    reliability = scored.loc[scored.family.eq("tcn_adapted") & scored.horizon.eq(5)].groupby(
        ["arm", "sex"]).absolute_log_error.mean().unstack("sex").reindex(ARM_ORDER)
    control = endpoint_table.loc["all_six"]
    differences = endpoint_table.subtract(control, axis=1)
    endpoint_percent_change = 100 * endpoint_table.divide(control, axis=1).subtract(1)
    rolling_percent_change = 100 * reliability.divide(reliability.loc["all_six"], axis=1).subtract(1)
    endpoint_report = pd.DataFrame({"Donor pool": [ARM_LABELS[arm] for arm in ARM_ORDER],
                                   "Male ALE": [f"{value:.6f}" for value in endpoint_table.Male],
                                   "Female ALE": [f"{value:.6f}" for value in endpoint_table.Female],
                                   "Male change vs all six": [f"{value:+.6f}" for value in differences.Male],
                                   "Female change vs all six": [f"{value:+.6f}" for value in differences.Female]})
    reliability_report = pd.DataFrame({"Donor pool": [ARM_LABELS[arm] for arm in ARM_ORDER],
                                      "Male mean ALE": [f"{value:.6f}" for value in reliability.Male],
                                      "Female mean ALE": [f"{value:.6f}" for value in reliability.Female]})
    adaptation_endpoint = adaptation.loc[adaptation.origin.eq(2018) & adaptation.horizon.eq(5)].groupby(
        ["arm", "sex"]).agg(mean_change=("adaptation_loss_change", "mean"), harmed_fraction=("harmed", "mean"))
    donor_harm_endpoint = harm.loc[harm.family.eq("tcn_adapted") & harm.origin.eq(2018) & harm.horizon.eq(5)].groupby(
        ["arm", "sex"]).harmed.sum()
    cal = cells.loc[cells.family.eq("tcn_adapted") & cells.horizon.eq(5) & cells.scale.eq("rate") & cells.level.eq(.8)]
    coverage = cal.groupby(["arm", "sex"]).covered.mean()
    principal_wis = wis.loc[wis.family.eq("tcn_adapted") & wis.horizon.eq(5) & wis.scale.eq("log_rate")].groupby(
        ["arm", "sex"]).wis_50_80.mean()
    calibration_report = pd.DataFrame([
        {"Donor pool": ARM_LABELS[arm], "Sex": sex, "80% rate coverage": f"{100 * coverage.loc[(arm, sex)]:.1f}%",
         "Mean log-scale WIS50/80": f"{principal_wis.loc[(arm, sex)]:.6f}",
         "Adaptation ALE change": f"{adaptation_endpoint.loc[(arm, sex), 'mean_change']:+.6f}"}
        for arm in ARM_ORDER for sex in ["Male", "Female"]])
    selected = pd.DataFrame([{"Sex": plan["sex"], "Donors": ", ".join(plan["countries"])}
                             for plan in plans if plan["origin"] == 2018 and plan["arm"] == "similar_three"])
    device = pd.read_csv(run / "gpu_reference_vs_cpu_primary.csv")
    device_endpoint = device.loc[device.family.eq("tcn_adapted") & device.origin.eq(2018) & device.horizon.eq(5)]
    device_differences = device_endpoint.groupby("sex").device_procedure_loss_change.mean()
    male_nonneural_error = contrasts.loc[contrasts.arm.eq("all_six") & contrasts.sex.eq("Male")
                                         & contrasts.comparator.eq("nonneural_champion"), "comparator_error"].iloc[0]
    out.mkdir(parents=True, exist_ok=True)
    endpoint_table.to_csv(out / "endpoint_age_mean_errors.csv")
    differences.to_csv(out / "endpoint_changes_vs_gpu_reference.csv")
    endpoint_percent_change.to_csv(out / "endpoint_percent_changes_vs_gpu_reference.csv")
    rolling_percent_change.to_csv(out / "rolling_percent_changes_vs_gpu_reference.csv")
    rolling_by_origin = scored.loc[scored.family.eq("tcn_adapted") & scored.horizon.eq(5)].groupby(
        ["arm", "sex", "origin"], as_index=False).absolute_log_error.mean()
    rolling_by_origin.to_csv(out / "rolling_origin_age_mean_errors.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True)
    for axis, sex in zip(axes, ["Male", "Female"]):
        colors = ["#244E70"] + ["#63A3A1"] * 2 + ["#A7BFC7"] * 5
        axis.barh(np.arange(len(ARM_ORDER)), endpoint_table[sex], color=colors)
        axis.set_yticks(np.arange(len(ARM_ORDER)), [ARM_LABELS[arm] for arm in ARM_ORDER])
        axis.invert_yaxis()
        axis.set_title(sex)
        axis.set_xlabel("45+ mean absolute log error (2018 origin, horizon 5)")
        axis.grid(axis="x", alpha=.25)
        axis.set_axisbelow(True)
        reference_rows = contrasts.loc[contrasts.arm.eq("all_six") & contrasts.sex.eq(sex)]
        for row, color in zip(reference_rows.itertuples(), ["#A64B3C", "#AA8424"]):
            axis.axvline(row.comparator_error, color=color, linestyle="--", linewidth=1.3,
                         label=row.comparator.replace("_", " "))
        axis.legend(fontsize=8, loc="lower right")
    maximum = max(float(endpoint_table.to_numpy().max()), float(contrasts.comparator_error.max()))
    for axis in axes:
        axis.set_xlim(0, maximum * 1.12)
    fig.suptitle("Prespecified donor sensitivity: common GPU reference; frozen CPU comparators")
    fig.tight_layout()
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(out / f"donor_endpoint_comparison.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for axis, sex in zip(axes, ["Male", "Female"]):
        for index, arm in enumerate(ARM_ORDER):
            part = rolling_by_origin.loc[rolling_by_origin.arm.eq(arm) & rolling_by_origin.sex.eq(sex)].sort_values("origin")
            major = arm in ["all_six", "gcc_five", "similar_three"]
            color = {"all_six": "#244E70", "gcc_five": "#009E73", "similar_three": "#D55E00"}.get(arm, "#AAB4BC")
            label = ARM_LABELS[arm] if major else "Five fixed random arms" if arm == "random_three_101" else None
            axis.plot(part.origin, part.absolute_log_error, marker="o" if major else None,
                      linewidth=2 if major else 1, color=color, label=label, alpha=1 if major else .65,
                      zorder=3 if major else 1)
        axis.set_title(sex)
        axis.set_xticks(range(2014, 2019))
        axis.set_xlabel("Forecast origin (five-year horizon)")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("45+ mean absolute log error")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Donor sensitivity across dependent rolling forecast origins")
    fig.tight_layout()
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(out / f"donor_rolling_origins.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    report = f"""# Prespecified donor-pool comparisons

This is a secondary sensitivity analysis. Every donor arm, including the all-six reference, was fitted on `{manifest['device']}`. The completed CPU Saudi prevalence primary result remains unchanged; no donor pool or random seed was selected using final accuracy.

## Interpretation

- **Male forecasts benefit from the GCC-only pool:** endpoint ALE is {abs(endpoint_percent_change.loc['gcc_five', 'Male']):.2f}% lower than the same-GPU all-six reference, and the five-origin mean is {abs(rolling_percent_change.loc['gcc_five', 'Male']):.2f}% lower. Similarity selection improves the rolling mean by {abs(rolling_percent_change.loc['similar_three', 'Male']):.2f}% but makes the endpoint {endpoint_percent_change.loc['similar_three', 'Male']:.2f}% worse. This distinction prevents a favorable average from replacing the prespecified endpoint.
- **Female forecasts benefit from retaining the broader pool at the endpoint:** GCC-only increases error {endpoint_percent_change.loc['gcc_five', 'Female']:.2f}%; similar-three increases it {endpoint_percent_change.loc['similar_three', 'Female']:.2f}%, with worse errors in {int(donor_harm_endpoint.loc[('similar_three', 'Female')])}/11 age bands. Similar-three also increases the rolling mean {rolling_percent_change.loc['similar_three', 'Female']:.2f}%. Historical resemblance did not guarantee useful transfer.
- **Donor restriction and adaptation harm are distinct:** for the female similarity pool, applying the two-parameter correction adds {adaptation_endpoint.loc[('similar_three', 'Female'), 'mean_change']:.6f} ALE relative to its own unchanged, unadapted source ensemble; it worsens {round(11*adaptation_endpoint.loc[('similar_three', 'Female'), 'harmed_fraction'])}/11 age bands. Male adaptation lowers endpoint mean error in all eight pools, although individual ages can worsen.
- **Random subsets are unstable across evaluation periods:** all five have worse endpoint errors than all-six for both sexes. Female random arms 101 and 503 have better rolling means but worse endpoints; they remain prespecified sensitivity arms, with no winning seed selected afterward.
- **Interval calibration remains inadequate:** every adapted arm's rolling 80% interval coverage is below 80% (range {100*coverage.min():.1f}%–{100*coverage.max():.1f}%). Better point accuracy does not establish reliable uncertainty estimates.

## Five-year endpoint at origin 2018

ALE is the equal-age mean absolute log error across eleven included bands, ages 45–49 through 95+. Negative changes indicate improvement over the newly fitted all-six GPU reference. All eight arms are retained.

{markdown_table(endpoint_report)}

![Endpoint errors and frozen comparator references](donor_endpoint_comparison.png)

## Nested reliability

The following averages use the five separate origins 2014–2018 at horizon five. The 2018 forecast is included once and also supplies the endpoint above; these overlapping windows are dependent.

{markdown_table(reliability_report)}

![Five-year errors across rolling origins](donor_rolling_origins.png)

## Calibration and adaptation

Coverage uses 55 dependent age/origin cells per sex at horizon five. Each prediction has seven through eleven joint historical residual blocks. Coverage is descriptive, with no independent-cell confidence intervals or exact distribution-free guarantee. Adaptation changes compare the same source ensemble before and after two-parameter correction at the 2018 endpoint; positive values indicate adaptation harm.

{markdown_table(calibration_report)}

## Similarity-selected countries at origin 2018

{markdown_table(selected)}

Selection used only the corresponding sex's eight-year history. Every selected donor contributed both sexes to source fitting. Full historical distances, feature scaling, masks, ranks, and countries are retained in `donor_decisions.json`.

## Numerical reference and audit

The all-six GPU procedure's endpoint ALE change relative to the frozen CPU primary TCN was {device_differences.loc['Male']:+.6f} for males and {device_differences.loc['Female']:+.6f} for females. This comparison includes numerical-device effects on fitting and historical tuning; it does not replace the original primary procedure.

The all-six GPU male ALE ({endpoint_table.loc['all_six', 'Male']:.6f}) is slightly higher than the frozen non-neural champion's {male_nonneural_error:.6f}, whereas the original CPU primary TCN was lower. Thus that small ranking margin is sensitive to the numerical procedure. Donor-arm comparisons use a common GPU reference, but comparisons against the original CPU champions are supplementary. These data do not establish that one particular country causes transfer benefit or harm.

The run used {validation['unique_source_fits']:,} unique source fits, with {validation['source_failures']} source failures. It saved {validation['evaluation_forecasts']:,} evaluation forecast values, {validation['interval_rows']:,} interval rows, and {validation['joint_draw_rows']:,} joint draws. Source pools exclude Saudi Arabia, source training includes both donor sexes, completed labels obey each fit cutoff, and all reported forecasts retain the five fixed model seeds. Failed models and dependent-cell harm summaries remain in the audit.

The report independently checked stored artifact/source hashes, primary-reference identity, point errors, country exclusions, temporal cutoffs, ensemble identities, frozen source fingerprints, and residual-block counts. It replayed {len(reproduced)} selected checkpoints and {replayed} adapted forecast values, reconstructed all {checked_joint_draws:,} joint draws and all {checked_interval_rows:,} interval rows independently on both scales, and verified interval scores, coverage and both WIS formulas across the complete scoring ledger. Underlying rates are modeled GBD estimates; donor similarity and transfer differences do not establish biological mechanisms or causation.

Detailed outputs include `endpoint_contrasts.csv` against both frozen champions, `donor_harm_cells.csv` against the GPU all-six procedure, `borrowing_harm_cells.csv`, `matched_adaptation_cells.csv`, and complete point/interval/WIS ledgers.
"""
    (out / "report.md").write_text(report)
    audit = {"passed": True, "run_manifest_sha256": sha(run / "run_manifest.json"),
             "independent_absolute_log_error_max_difference": float(np.max(np.abs(independently_computed - scored.absolute_log_error))),
             "primary_reference_unchanged": True, "all_eight_arms_reported": True,
             "point_and_interval_rows": [len(predictions), len(cells)], "fit_audits": len(fits),
             "checkpoints_replayed": len(reproduced), "adapted_forecasts_replayed": replayed,
             "independent_interval_rows_checked": checked_interval_rows,
             "independent_joint_draws_checked": checked_joint_draws,
             "independent_wis_rows_checked": len(wis),
             "independent_quantile_max_absolute_difference": quantile_max_difference,
             "independent_joint_draw_max_absolute_difference": joint_draw_max_difference,
             "independent_wis_max_absolute_difference": max(wis_differences),
             "report_source_sha256": sha(__file__),
             "report_output_sha256": {path.name: sha(path) for path in sorted(out.iterdir())
                                       if path.is_file() and path.name != "validation.json"}}
    (out / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
