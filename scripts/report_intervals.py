"""Independently audit stage-4 residual banks and historical interval diagnostics."""

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
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gbd_park.pooled import target_inputs
from gbd_park.tcn import load_checkpoint, predict_changes, state_fingerprint
from run_local_baselines import check_lock, sha

RUN = ROOT / "results/intervals_v1"
OUT = ROOT / "reports/intervals_v1"
COORDS = ["sex", "age", "horizon"]
LABELS = {
    "persistence": "Persistence", "log_trend": "Log-linear trend", "damped_ets": "Damped ETS",
    "arima": "ARIMA", "age_smooth_trend": "Age-smoothed trend", "pooled_ridge": "Pooled ridge",
    "pooled_boosting": "Pooled boosting", "donor_ridge_unadapted": "Donor ridge, unadapted",
    "donor_ridge_adapted": "Donor ridge, adapted", "donor_boosting_unadapted": "Donor boosting, unadapted",
    "donor_boosting_adapted": "Donor boosting, adapted", "tcn_adapted": "TCN, two-coefficient adaptation",
    "tcn_unadapted": "TCN, no adaptation", "tcn_intercept": "TCN, intercept adaptation",
    "local_champion": "Selected local family", "nonneural_champion": "Selected non-neural family",
}


def verify_run(directory):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["status"] == "complete" and not manifest["final_period_scored"]
    for name, expected in manifest["output_sha256"].items():
        assert sha(directory / name) == expected, name
    for name, expected in manifest["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    return manifest


def coordinate_index(config):
    return pd.MultiIndex.from_product([config["sexes"], config["ages"], config["calendar"]["horizons"]], names=COORDS)


def verify_residual_banks(scores, blocks, summaries, mappings, config, manifest):
    expected_families = set(manifest["families"] + ["local_champion", "nonneural_champion"])
    expected_pairs = {(origin, family) for origin in range(2014, 2019) for family in expected_families}
    assert set(zip(summaries.fit_origin, summaries.family)) == expected_pairs and len(summaries) == 80
    assert summaries.n_blocks.eq(summaries.fit_origin - 2007).all()
    assert summaries.coordinates.eq(110).all() and summaries.status.eq("ok").all()
    assert summaries.first_residual_origin.eq(2003).all()
    assert summaries.last_residual_origin.eq(summaries.fit_origin - 5).all()
    assert summaries.last_residual_label_year.le(summaries.fit_origin).all()
    assert mappings.last_selection_target_year.le(mappings.fit_origin).all()
    bank_keys = ["fit_origin", "family", "residual_origin"] + COORDS
    assert not blocks.duplicated(bank_keys).any() and len(blocks) == 79200
    assert blocks.groupby(["fit_origin", "family", "residual_origin"]).size().eq(110).all()
    assert set(map(tuple, blocks[COORDS].drop_duplicates().to_numpy())) == set(coordinate_index(config))
    assert blocks.residual_verification_year.eq(blocks.residual_origin + blocks.horizon).all()
    assert blocks.residual_verification_year.le(blocks.fit_origin).all()
    source = scores.rename(columns={"origin": "residual_origin", "family": "source_family"})
    source_keys = ["residual_origin", "source_family"] + COORDS
    joined = blocks.merge(source[source_keys + ["prediction", "log_prediction", "observed_rate"]],
                          on=source_keys, how="left", validate="many_to_one")
    assert len(joined) == len(blocks) and joined.observed_rate.notna().all()
    residual = np.log(joined.observed_rate.to_numpy()) - joined.log_prediction.to_numpy()
    np.testing.assert_allclose(joined.raw_log_residual, residual, rtol=1e-12, atol=1e-14)
    group_keys = ["fit_origin", "family"] + COORDS
    mean = joined.groupby(group_keys).raw_log_residual.transform("mean")
    np.testing.assert_allclose(joined.coordinate_mean, mean, atol=1e-14)
    np.testing.assert_allclose(joined.centered_log_residual, joined.raw_log_residual - mean, atol=1e-14)
    np.testing.assert_allclose(joined.groupby(group_keys).centered_log_residual.mean(), 0, atol=1e-14)
    mapped = {(int(row.fit_origin), row.role, row.sex): row.source_family for row in mappings.itertuples()}
    for row in summaries.itertuples():
        group = blocks.loc[blocks.fit_origin.eq(row.fit_origin) & blocks.family.eq(row.family)]
        origins = list(range(2003, row.fit_origin - 4))
        assert sorted(group.residual_origin.unique()) == origins
        declared_mapping = json.loads(row.source_family_by_sex)
        for sex in config["sexes"]:
            expected_family = mapped[(row.fit_origin, row.family, sex)] if row.family.endswith("_champion") else row.family
            assert declared_mapping[sex] == expected_family
            assert group.loc[group.sex.eq(sex), "source_family"].eq(expected_family).all()
        bank = joblib.load(RUN / "banks" / f"origin{row.fit_origin}__{row.family}.joblib")
        assert bank["origins"] == origins and bank["source_family_by_sex"] == declared_mapping
        assert list(map(tuple, bank["coords"].to_numpy())) == list(coordinate_index(config))
        for field, array_name in [("raw_log_residual", "raw_residuals"), ("centered_log_residual", "centered_residuals")]:
            array = group.pivot(index="residual_origin", columns=COORDS, values=field).reindex(
                index=origins, columns=coordinate_index(config)).to_numpy()
            np.testing.assert_allclose(array, bank[array_name], atol=1e-14)
    for name, source_run in [("local_champion", "local_baselines_v1"), ("nonneural_champion", "nonneural_v1")]:
        filename = "local_champions_for_later_evaluation.csv" if name == "local_champion" else "nonneural_champions_for_later_evaluation.csv"
        original = pd.read_csv(ROOT / "results" / source_run / filename)
        for row in original.itertuples():
            assert mapped[(row.fit_origin, name, row.sex)] == row.selected_family
    return {"verified_banks": len(summaries), "verified_residual_rows": len(blocks)}


def verify_preview_quantiles(prequential, scores, intervals, draws, config, families):
    cells = 0
    for origin in [2012, 2013]:
        residual_origins = list(range(2003, origin - 4))
        for family in families:
            part = draws.loc[draws.origin.eq(origin) & draws.family.eq(family)]
            assert len(part) == len(residual_origins) * 110
            assert not part.duplicated(["residual_origin"] + COORDS).any()
            points = prequential.loc[prequential.origin.eq(origin) & prequential.family.eq(family)].set_index(COORDS).reindex(coordinate_index(config))
            history = scores.loc[scores.family.eq(family) & scores.origin.isin(residual_origins)].copy()
            history["independent_residual"] = np.log(history.observed_rate) - history.log_prediction
            residual = history.pivot(index="origin", columns=COORDS, values="independent_residual").reindex(
                index=residual_origins, columns=coordinate_index(config)).to_numpy()
            centered = residual - residual.mean(axis=0)
            expected_logs = centered + points.log_prediction.to_numpy()[None, :]
            matrices = {}
            for scale, field in [("log_rate", "log_draw"), ("rate", "rate_draw")]:
                matrices[scale] = part.pivot(index="residual_origin", columns=COORDS, values=field).reindex(
                    index=residual_origins, columns=coordinate_index(config)).to_numpy()
            np.testing.assert_allclose(matrices["log_rate"], expected_logs, atol=1e-14)
            np.testing.assert_allclose(matrices["rate"], np.exp(expected_logs), rtol=1e-12)
            for scale, matrix in matrices.items():
                for level in config["intervals"]["central_levels"]:
                    selected = intervals.loc[intervals.origin.eq(origin) & intervals.family.eq(family)
                                             & intervals.scale.eq(scale) & intervals.level.eq(level)].set_index(COORDS).reindex(coordinate_index(config))
                    expected = np.quantile(matrix, [(1-level)/2, 0.5, (1+level)/2], axis=0, method="linear").T
                    np.testing.assert_allclose(selected[["lower", "median", "upper"]], expected, rtol=1e-12, atol=1e-13)
                    original_points = points.log_prediction if scale == "log_rate" else points.prediction
                    np.testing.assert_allclose(selected.point_prediction, original_points, rtol=1e-12, atol=1e-14)
                    assert selected.n_blocks.eq(len(residual_origins)).all() and selected.status.eq("ok").all()
                    cells += len(selected)
    assert cells == len(intervals) == 18480
    return cells


def verify_interval_scores(cells, wis):
    observed = cells.observed_rate.to_numpy()
    expected_observed = np.where(cells.scale.eq("log_rate"), np.log(observed), observed)
    np.testing.assert_allclose(cells.observed_value, expected_observed, atol=1e-13)
    alpha = 1 - cells.level.to_numpy()
    width = cells.upper.to_numpy() - cells.lower.to_numpy()
    penalties = np.where(expected_observed < cells.lower, (cells.lower - expected_observed) * 2 / alpha, 0)
    penalties += np.where(expected_observed > cells.upper, (expected_observed - cells.upper) * 2 / alpha, 0)
    expected_score = width + penalties
    expected_coverage = ((expected_observed >= cells.lower) & (expected_observed <= cells.upper)).astype(float)
    np.testing.assert_allclose(cells.width, width, atol=1e-12)
    np.testing.assert_allclose(cells.interval_score, expected_score, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(cells.covered, expected_coverage)
    keys = ["target", "outcome", "origin", "family"] + COORDS + ["forecast_year", "scale"]
    indexed = wis.set_index(keys)
    cell_copy = cells.copy()
    cell_copy["independent_weighted_interval_score"] = alpha * expected_score / 2
    terms = cell_copy.pivot(index=keys, columns="level", values="independent_weighted_interval_score").reindex(index=indexed.index)
    medians = cell_copy.pivot(index=keys, columns="level", values="median").reindex(index=indexed.index)
    repeated_medians = np.broadcast_to(indexed["median"].to_numpy()[:, None], medians.shape)
    np.testing.assert_allclose(medians.to_numpy(), repeated_medians, atol=1e-13)
    median_error = np.abs(indexed.observed_value - indexed["median"])
    np.testing.assert_allclose(indexed.median_absolute_error, median_error, atol=1e-12)
    for field, levels in [("wis_50_80", [0.5, 0.8]), ("wis_50_80_95", [0.5, 0.8, 0.95])]:
        expected = (0.5 * median_error + terms[levels].sum(axis=1)) / (len(levels) + 0.5)
        np.testing.assert_allclose(indexed[field], expected, rtol=1e-12, atol=1e-12)
    return len(wis)


def replay_early_tcn(prequential, config):
    origin = 2008
    refs = pd.read_json(RUN / "early_seed_references.jsonl", lines=True)
    refs = refs.loc[refs.origin.eq(origin)].sort_values("seed")
    assert refs.seed.tolist() == config["models"]["tcn"]["ensemble_seeds"]
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(origin) & panel.outcome.eq("prevalence")]
    x, levels, meta = target_inputs(panel, config, origin, config["primary_target"])
    predictions, failed, checkpoints = [], False, 0
    for ref in refs.itertuples():
        payload_path = ROOT / ref.payload_path
        payload = joblib.load(payload_path)
        assert payload["origin"] == origin and payload["seed"] == ref.seed
        assert meta.equals(payload["current_meta"])
        np.testing.assert_array_equal(levels, payload["levels"])
        if ref.status != "ok":
            failed = True
            predictions.append(np.zeros((110 // 5, 5)))
            continue
        fitted = load_checkpoint(payload_path.parent.parent / "checkpoints" / payload_path.name)
        before = state_fingerprint(fitted)
        prediction = predict_changes(fitted, x)
        np.testing.assert_array_equal(prediction, payload["current_changes"])
        assert state_fingerprint(fitted) == before == ref.state_fingerprint
        assert before == payload["audit"]["fingerprint_before"] == payload["audit"]["fingerprint_after"]
        predictions.append(prediction)
        checkpoints += 1
    ensemble = np.mean(predictions, axis=0)
    calibrations = pd.read_json(RUN / "early_tcn_adaptation_audit.jsonl", lines=True)
    selected = prequential.loc[prequential.origin.eq(origin) & prequential.family.str.startswith("tcn_")]
    count = 0
    for (family, sex), group in selected.groupby(["family", "sex"]):
        indices = meta.sex.eq(sex).to_numpy()
        changes = ensemble[indices].copy()
        if failed:
            changes[:] = 0
        elif family != "tcn_unadapted":
            cal = calibrations.loc[calibrations.origin.eq(origin) & calibrations.family.eq(family) & calibrations.sex.eq(sex)]
            assert len(cal) == 1
            row = cal.iloc[0]
            if row.status == "ok":
                changes += row.b0 + row.b1 * np.arange(1, 6) / 5
            else:
                changes[:] = 0
        logs = levels[indices, None] + changes
        invalid = ~np.isfinite(logs).all(axis=1) | (np.abs(logs) > 700).any(axis=1)
        logs[invalid] = levels[indices][invalid, None]
        expected = group.pivot(index="age", columns="horizon", values="log_prediction").reindex(
            index=config["ages"], columns=config["calendar"]["horizons"]).to_numpy()
        np.testing.assert_allclose(logs, expected, rtol=0, atol=2e-14)
        count += expected.size
    return {"early_tcn_replayed_cells": count, "early_frozen_checkpoints_replayed": checkpoints}


def main():
    check_lock()
    manifest = verify_run(RUN)
    for name, expected in manifest["prior_manifests_sha256"].items():
        directory = ROOT / "results" / name
        verify_run(directory)
        assert sha(directory / "run_manifest.json") == expected
    tests_path = RUN / "tests_at_run.json"
    assert sha(tests_path) == manifest["test_report_sha256"]
    tests = json.loads(tests_path.read_text())
    validation = json.loads((RUN / "validation_report.json").read_text())
    assert validation["passed"] and not validation["final_origin_point_forecasts_produced"]
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    prequential = pd.read_csv(RUN / "prequential_predictions.csv")
    scores = pd.read_csv(RUN / "prequential_scores.csv")
    blocks = pd.read_csv(RUN / "residual_blocks.csv")
    summaries = pd.read_csv(RUN / "bank_summary.csv")
    mappings = pd.read_csv(RUN / "champion_family_mappings.csv")
    intervals = pd.read_csv(RUN / "development_intervals.csv")
    draws = pd.read_csv(RUN / "development_joint_draws.csv")
    cells = pd.read_csv(RUN / "development_interval_scores.csv")
    wis = pd.read_csv(RUN / "development_wis_scores.csv")
    assert "observed_rate" not in prequential and "observed_rate" not in intervals
    assert len(prequential) == 16940 and len(wis) == 6160
    assert prequential.forecast_year.max() == scores.forecast_year.max() == intervals.forecast_year.max() == 2018
    assert set(intervals.origin) == {2012, 2013}
    np.testing.assert_allclose(scores.log_residual, np.log(scores.observed_rate) - scores.log_prediction, atol=1e-14)
    np.testing.assert_allclose(scores.log_prediction, np.log(scores.prediction), atol=1e-14)
    source = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    source = source.loc[source.year.le(2018) & source.location_name.eq(config["primary_target"])
                        & source.outcome.eq("prevalence")].rename(columns={"location_name": "target", "year": "forecast_year", "rate": "source_rate"})
    truth_keys = ["target", "outcome", "sex", "age", "forecast_year"]
    verified_truth = scores.merge(source[truth_keys + ["source_rate"]], on=truth_keys, how="left", validate="many_to_one")
    np.testing.assert_allclose(verified_truth.observed_rate, verified_truth.source_rate, rtol=1e-12)
    bank_validation = verify_residual_banks(scores, blocks, summaries, mappings, config, manifest)
    interval_count = verify_preview_quantiles(prequential, scores, intervals, draws, config, manifest["families"])
    wis_count = verify_interval_scores(cells, wis)
    replay = replay_early_tcn(prequential, config)
    events = [row["event"] for row in json.loads((RUN / "events.json").read_text())]
    assert events.index("prequential_predictions_committed") < events.index("historical_residual_scoring_started")
    assert events.index("development_intervals_committed") < events.index("development_interval_scoring_started")
    focus = cells.loc[cells.horizon.eq(5) & cells.scale.eq("rate") & cells.level.eq(0.8)]
    diagnostic = focus.groupby(["sex", "family"], as_index=False).agg(
        coverage_80=("covered", "mean"), covered_cells=("covered", "sum"), cells=("covered", "size"),
        mean_width_80=("width", "mean"), mean_interval_score_80=("interval_score", "mean"))
    score_summary = wis.loc[wis.horizon.eq(5) & wis.scale.eq("rate")].groupby(["sex", "family"], as_index=False).agg(
        mean_wis_50_80=("wis_50_80", "mean"), mean_wis_50_80_95=("wis_50_80_95", "mean"),
        mean_median_absolute_error=("median_absolute_error", "mean"))
    diagnostic = diagnostic.merge(score_summary, on=["sex", "family"], validate="one_to_one")
    assert len(diagnostic) == 28 and diagnostic.cells.eq(22).all()
    diagnostic["family_order"] = diagnostic.family.map({family: index for index, family in enumerate(manifest["families"])})
    diagnostic = diagnostic.sort_values(["sex", "family_order"]).drop(columns="family_order")
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostic.to_csv(OUT / "horizon5_rate_interval_diagnostics.csv", index=False)
    final_mapping = mappings.loc[mappings.fit_origin.eq(2018)].copy()
    final_mapping.to_csv(OUT / "origin2018_champion_bank_mappings.csv", index=False)
    (OUT / "tests_at_run.json").write_bytes(tests_path.read_bytes())
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    labels = [LABELS[family] for family in manifest["families"]]
    y = np.arange(len(labels))
    for axis, sex, color in zip(axes, config["sexes"], ["#2563a6", "#138a72"]):
        part = diagnostic.loc[diagnostic.sex.eq(sex)].set_index("family").reindex(manifest["families"])
        axis.barh(y, part.coverage_80, height=0.64, color=color, alpha=0.85)
        axis.axvline(0.8, color="#8b2331", linewidth=1.8, linestyle="--", label="Nominal 80%")
        for index, value in enumerate(part.coverage_80):
            axis.text(value + 0.012, index, f"{value:.1%}", va="center", fontsize=8)
        axis.set_title(sex, fontsize=12)
        axis.set_xlim(0, 1.08)
        axis.set_xticks(np.arange(0, 1.01, 0.2))
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_xlabel("Observed coverage of central 80% rate interval")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Historical interval diagnostic: 22 dependent age–origin cells per sex", fontsize=14)
    fig.text(0.5, 0.928, "Origins 2012 and 2013 · horizon five · five and six calibration blocks", ha="center", fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", frameon=False)
    fig.tight_layout(rect=[0, 0.055, 1, 0.915])
    fig.savefig(OUT / "development_coverage.png", dpi=180)
    fig.savefig(OUT / "development_coverage.svg")
    plt.close(fig)
    table_lines = []
    for sex in config["sexes"]:
        table_lines.extend([f"### {sex}", "", "| Family | 80% coverage | Covered cells | Mean interval width | Mean 50/80 WIS |",
                            "|---|---:|---:|---:|---:|"])
        part = diagnostic.loc[diagnostic.sex.eq(sex)].set_index("family")
        for family in manifest["families"]:
            row = part.loc[family]
            table_lines.append(f"| {LABELS[family]} | {row.coverage_80:.1%} | {int(row.covered_cells)}/{int(row.cells)} | "
                               f"{row.mean_width_80:.3f} | {row.mean_wis_50_80:.3f} |")
        table_lines.append("")
    mapping_lines = ["| Bank role | Sex | Historical source family |", "|---|---|---|"]
    for row in final_mapping.itertuples():
        mapping_lines.append(f"| {LABELS[row.role]} | {row.sex} | {LABELS[row.source_family]} |")
    tcn_diagnostic = diagnostic.loc[diagnostic.family.eq("tcn_adapted")].set_index("sex")
    below = int(diagnostic.coverage_80.lt(0.8).sum())
    report = f"""# Stage 4 — Joint residual banks and historical interval diagnostics

Completed under [locked design v1.0](../../study_design/locked_v1/protocol.md). This stage completes historical forecast ensembles and prepares calibration banks for later evaluation. It does not fit the final-origin point forecasts or evaluate 2019–2023 outcomes.

## Main finding and scope

The calibration infrastructure is ready, while preliminary interval reliability remains limited. In the two eligible development previews, **{below} of 28 sex–family summaries had coverage below the nominal 80% level**. Adapted TCN coverage was **{tcn_diagnostic.loc['Male', 'coverage_80']:.1%} for males** and **{tcn_diagnostic.loc['Female', 'coverage_80']:.1%} for females**. These findings are reported with the prespecified method unchanged; they do not establish coverage in the later reliability period.

The preview uses origins **2012 and 2013**, with **five and six** fully completed historical residual blocks. Each horizon-five summary contains only **22 dependent age–origin cells per sex**. It is a development diagnostic, not an independent validation sample or a significance test. All fourteen method families are shown, including weaker interval results.

## Prepared historical forecasts and banks

- Saved **{validation['prequential_forecasts']:,} prequential point forecasts** across eleven origins (2003–2013), fourteen families, both sexes, eleven ages, and five horizons. Existing development forecasts remain unchanged.
- Completed **{validation['new_tcn_fits']} additional neural fits** for seeds 23, 37, 53, and 71 at early origins 2003–2008. Each early ensemble reuses its frozen seed-11 model from stage 3 and averages all five seeds before its Saudi correction.
- Saved **80 banks**: fourteen method families plus the sex-specific local and non-neural champion roles, at fit origins 2014–2018. They contain **7, 8, 9, 10, and 11 complete joint blocks**, respectively, each with 110 age–sex–horizon coordinates.
- The **{validation['bank_residual_rows']:,} residual rows** preserve origin pairing. Each residual is `log(observed GBD rate) − log(point forecast)`; its coordinate-wise historical arithmetic mean is subtracted before adding the joint deviations to a current forecast.
- Banks for origin 2018 contain eleven eligible origins through 2013, whose labels end by 2018. Their calibration offsets and multipliers are saved without generating a final-origin point forecast.

The champion-bank mappings prepared for origin 2018 are:

{chr(10).join(mapping_lines)}

Each mapped bank uses that currently selected family’s **chronologically generated historical forecasts**, including the settings that were eligible at each historical origin. It does not apply current hyperparameters retrospectively. Male and female blocks are paired by the same historical residual origin, even when their selected families differ. Comparator-family selection reuses development information; this is disclosed and does not imply an exact conformal guarantee.

## Preliminary interval coverage, width, and WIS

![Historical coverage across all method families](development_coverage.png)

These tables pool the two preview origins equally. Width and WIS are on the rate scale, per 100,000 population. Width describes sharpness; a narrow interval can have poor coverage. Lower WIS is better because it combines width and penalties for misses, with the predictive median included. The saved tables also retain log-scale and 50/80/95 results.

{chr(10).join(table_lines)}

Low coverage must remain visible alongside point-forecast accuracy. Centering removes historical coordinate-level mean errors; later bias or changes in the error distribution can therefore fall outside the resulting bands. The small, overlapping block set also limits tail estimation. These diagnostics do not justify claiming calibrated intervals, increasing the effective sample size by resampling blocks, or changing the locked interval method after viewing results.

## Scientific interpretation of the intervals

Log-rate quantiles and rate quantiles are each computed by deterministic linear interpolation on their own scale. Interpolated rate quantiles are not obtained by exponentiating interpolated log quantiles. The original point forecast remains separate from the predictive median, which can differ after centering and transformation.

The complete joint draw identifier is retained for later sex ratios, age shares, and population-conditional count sums. Such derived quantities must transform whole blocks before summarizing; summing marginal medians is not the median of a joint sum. These bands describe historical forecast errors of revised GBD point estimates, not all uncertainty in true disease burden. Source lower/upper bounds and neural seed variability are not forecast intervals.

The principal interval score uses central 50% and 80% intervals. Including 95% is a supplementary sparse-tail analysis. WIS uses weight 0.5 for the absolute median error and `(1 − level)/2` for each interval score, divided by `number of intervals + 0.5`, following the [WIS methodology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618) and its [correction](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010592). No distribution-free coverage guarantee is claimed.

## Validation and execution

- All **{tests['tests_run']} tests passed**, including chronology, complete joint grids, residual signs and centering, source-bound exclusion, separate scale quantiles, unavailable intervals, matched historical selection, and hand-calculated coverage/interval/WIS examples.
- New TCN fit failures: **{validation['new_tcn_failures']}**; early correction failures: **{validation['early_adaptation_failures']}**; historical persistence-fallback cells: **{validation['prequential_fallback_cells']}**. Fallback provenance and unavailable reasons remain in the ledgers.
- Independent report checks reproduced every saved bank’s residuals and centering, every champion-family mapping, **{interval_count:,} preview interval rows**, and **{wis_count:,} coordinate/scale WIS rows**. Saved early checkpoints reproduced **{replay['early_tcn_replayed_cells']} origin-2008 neural forecast cells** without refitting.
- Point forecasts were committed before verification values were joined. Preview intervals and their draws were committed before interval scoring. Original inputs, the locked design, and all three earlier run manifests passed their hash checks.
- Four CPU workers in `agpu` completed the preparation run in **{validation['elapsed_seconds']:.1f} seconds**, retaining the numerical device/software used in stage 3. The maximum scored year remains **2018**.

## Next stage

Stage 5 applies the frozen procedures and banks to the Saudi nested reliability origins and the primary 2018-origin forecast. It will report both sexes, local/non-neural comparator contrasts, age-specific errors, interval coverage and WIS, and the primary five-year endpoint. The eleven-block bank at origin 2018 is available now; its reliability is still to be evaluated. Subsequent incidence, GCC, donor-selection, demographic, and projection analyses remain separate.

## Artifacts and reproduction

- [Historical point ledger](../../results/intervals_v1/prequential_predictions.csv), [historical scores](../../results/intervals_v1/prequential_scores.csv), [setting choices](../../results/intervals_v1/historical_setting_decisions.csv), and [TCN choices](../../results/intervals_v1/historical_tcn_choices.json).
- [Bank summary](../../results/intervals_v1/bank_summary.csv), [joint residual rows](../../results/intervals_v1/residual_blocks.csv), [calibration quantiles](../../results/intervals_v1/calibration_quantiles.csv), [champion mappings](../../results/intervals_v1/champion_family_mappings.csv), and [serialized banks](../../results/intervals_v1/banks/).
- [Preview intervals](../../results/intervals_v1/development_intervals.csv), [joint draws](../../results/intervals_v1/development_joint_draws.csv), [interval scores](../../results/intervals_v1/development_interval_scores.csv), [WIS scores](../../results/intervals_v1/development_wis_scores.csv), and [horizon-five diagnostic table](horizon5_rate_interval_diagnostics.csv).
- [Early seed references](../../results/intervals_v1/early_seed_references.jsonl), [run validation](../../results/intervals_v1/validation_report.json), [immutable manifest](../../results/intervals_v1/run_manifest.json), [report verification](report_validation.json), and [original test evidence](tests_at_run.json).
- [Implementation specification](../../study_design/intervals_implementation.md) and [independent review](../../work/interval-validation/review.md).

```bash
/home/saif/agpu_env/bin/python tests/test_intervals.py
/home/saif/agpu_env/bin/python scripts/run_intervals.py --output results/intervals_reproduction
/home/saif/agpu_env/bin/python scripts/report_intervals.py
```

The runner refuses an existing output directory. The report command reads the original `intervals_v1` artifacts without refitting or changing them.
"""
    (OUT / "report.md").write_text(report)
    report_validation = {"passed": True, "run_and_prior_artifact_hashes_verified": True,
                         "source_truth_verified": True, "joint_block_mapping_and_centering_verified": True,
                         "independent_preview_quantile_rows": interval_count,
                         "independent_wis_rows": wis_count, "prediction_and_interval_before_scoring_order_verified": True,
                         "maximum_scored_year": 2018, **bank_validation, **replay,
                         "run_manifest_sha256": sha(RUN / "run_manifest.json"),
                         "report_script_sha256": sha(Path(__file__)),
                         "output_sha256": {path.name: sha(path) for path in sorted(OUT.iterdir())
                                           if path.is_file() and path.name != "report_validation.json"}}
    (OUT / "report_validation.json").write_text(json.dumps(report_validation, indent=2) + "\n")
    print(json.dumps(report_validation, indent=2))


if __name__ == "__main__":
    main()
