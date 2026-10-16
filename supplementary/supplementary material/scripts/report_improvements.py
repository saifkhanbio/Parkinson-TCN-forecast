"""Independently audit and report the bounded post-result exploratory analysis."""

import json
import os
from datetime import datetime
from pathlib import Path
import sys

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gbd_park_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_local_baselines import sha

RUN = ROOT / "results/improvements_v1"
OUT = ROOT / "reports/improvements_v1"
COORDS = ["sex", "age", "horizon"]
POINTS = ["tcn_v1_unchanged", "tcn_age_group_correction"]
LABELS = {POINTS[0]: "Original TCN", POINTS[1]: "Age-offset procedure"}
VARIANTS = ["0", "0.5", "1", "selected"]


def read(name, directory=RUN):
    return pd.read_csv(directory / name, float_precision="round_trip")


def lines(name):
    return [json.loads(line) for line in (RUN / name).read_text().splitlines() if line.strip()]


def verify_integrity(amendment):
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["final_period_scored"]
    for name, expected in manifest["output_sha256"].items():
        assert sha(RUN / name) == expected, name
    for name, expected in manifest["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for old_run in amendment["original_runs"]:
        directory = ROOT / "results" / old_run
        assert sha(directory / "run_manifest.json") == manifest["prior_manifest_sha256"][old_run]
        old = json.loads((directory / "run_manifest.json").read_text())
        for name, expected in old["output_sha256"].items():
            assert sha(directory / name) == expected, f"{old_run}/{name}"
        for name, expected in old["code_sha256"].items():
            assert sha(ROOT / name) == expected, name
    lock = ROOT / "study_design/exploratory_v1_1.lock.json"
    assert sha(lock) == manifest["amendment_lock_sha256"]
    for name, expected in json.loads(lock.read_text())["document_sha256"].items():
        assert sha(ROOT / name) == expected
    for row in lines("frozen_source_references.jsonl"):
        assert sha(ROOT / row["path"]) == row["sha256"]
    commit = json.loads((RUN / "pre_score_commit.json").read_text())
    for name, expected in commit["sha256"].items():
        assert sha(RUN / name) == expected
    events = json.loads((RUN / "events.json").read_text())
    names = [row["event"] for row in events]
    order = ["age_candidates_committed", "historical_candidate_scoring_started",
             "point_predictions_committed", "development_intervals_committed",
             "evaluation_predictions_and_choices_committed", "exploratory_evaluation_scoring_started"]
    assert all(names.count(name) == 1 for name in order)
    indices = [names.index(name) for name in order]
    assert indices == sorted(indices)
    times = [datetime.fromisoformat(events[index]["time_utc"]) for index in indices]
    assert times == sorted(times)
    assert datetime.fromisoformat(commit["committed_utc"]) <= times[-1]
    assert events[names.index("evaluation_predictions_and_choices_committed")]["hashes"] == commit["sha256"]
    return manifest


def audit_point_scores(scores, point_predictions):
    panel = read("regional_outcomes.csv", ROOT / "data/processed/design_v1")
    truth = panel[panel.location_name.eq("Saudi Arabia") & panel.outcome.eq("prevalence")].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "true_rate"})
    keys = ["target", "outcome", "sex", "age", "forecast_year"]
    joined = scores.merge(truth[keys + ["true_rate"]], on=keys, validate="many_to_one")
    np.testing.assert_array_equal(joined.observed_rate, joined.true_rate)
    np.testing.assert_allclose(scores.absolute_log_error,
                               np.abs(scores.log_prediction - np.log(scores.observed_rate)), atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(scores.signed_log_error,
                               scores.log_prediction - np.log(scores.observed_rate), atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(scores.absolute_rate_error,
                               np.abs(scores.prediction - scores.observed_rate), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(scores.signed_rate_error,
                               scores.prediction - scores.observed_rate, atol=1e-12, rtol=1e-12)
    keys = ["origin", "family", "sex", "age", "horizon"]
    expected = point_predictions[point_predictions.origin.between(2014, 2018)].set_index(keys).sort_index()
    actual = scores.set_index(keys).sort_index()
    assert len(actual) == 1100 and actual.index.equals(expected.index)
    np.testing.assert_array_equal(actual[["log_prediction", "prediction"]], expected[["log_prediction", "prediction"]])
    groups = ["origin", "family", "sex", "horizon"]
    calculated = scores.groupby(groups).agg(mean_absolute_log_error=("absolute_log_error", "mean"),
                                            rate_mae=("absolute_rate_error", "mean"),
                                            mean_signed_log_error=("signed_log_error", "mean"),
                                            mean_signed_rate_error=("signed_rate_error", "mean"))
    saved = read("point_summary.csv").set_index(groups).sort_index()
    np.testing.assert_allclose(saved[calculated.columns], calculated, atol=1e-12, rtol=1e-12)
    return calculated.reset_index()


def audit_selections(config, amendment):
    age_scores = read("historical_age_candidate_scores.csv")
    candidates = read("age_candidate_predictions.csv")
    selected = read("point_predictions.csv")
    for choice in lines("age_setting_choices.jsonl"):
        origin, sex = choice["origin"], choice["sex"]
        origins = list(range(2003, origin - 4))
        assert choice["inner_origins"] == origins
        if origins:
            losses = []
            for order, penalty in enumerate(amendment["age_candidates"]):
                setting = "age_penalty=" + ("none" if penalty is None else str(penalty))
                part = age_scores[age_scores.origin.isin(origins) & age_scores.sex.eq(sex)
                                  & age_scores.horizon.eq(5) & age_scores.setting_id.eq(setting)]
                assert len(part) == len(origins) * 11 and (part.forecast_year <= origin).all()
                losses.append((float(part.absolute_log_error.mean()), order, setting))
            loss, _, setting = min(losses)
            assert choice["setting_id"] == setting and choice["last_label_year"] <= origin
            np.testing.assert_allclose(choice["loss"], loss, atol=1e-14)
        else:
            assert choice["penalty"] == 100 and choice["last_label_year"] is None
        source = candidates[candidates.origin.eq(origin) & candidates.sex.eq(sex)
                            & candidates.setting_id.eq(choice["setting_id"])].set_index(["age", "horizon"]).sort_index()
        actual = selected[selected.origin.eq(origin) & selected.sex.eq(sex)
                          & selected.family.eq(POINTS[1])].set_index(["age", "horizon"]).sort_index()
        np.testing.assert_array_equal(source.log_prediction, actual.log_prediction)
    development_wis = read("development_wis_scores.csv")
    for choice in lines("retention_choices.jsonl"):
        origins = [o for o in amendment["interval_development_origins"] if o + 5 <= choice["origin"]]
        assert choice["inner_origins"] == origins
        if origins:
            losses = []
            for retention in amendment["bias_retention_candidates"]:
                family = f"{choice['point_family']}__retention={retention:g}"
                part = development_wis[development_wis.origin.isin(origins) & development_wis.sex.eq(choice["sex"])
                                       & development_wis.horizon.eq(5) & development_wis.scale.eq("log_rate")
                                       & development_wis.family.eq(family)]
                assert len(part) == len(origins) * 11 and (part.forecast_year <= choice["origin"]).all()
                losses.append((float(part.wis_50_80.mean()), retention))
            loss, retention = min(losses)
            assert choice["retention"] == retention and choice["last_label_year"] <= choice["origin"]
            np.testing.assert_allclose(choice["loss"], loss, atol=1e-14)
        else:
            assert choice["retention"] == 0 and choice["last_label_year"] is None
    for audit in lines("age_correction_audit.jsonl"):
        assert audit["last_target_label_year"] <= audit["origin"] and not audit["original_correction_refitted"]
        for value in audit["offsets"].values():
            if audit["status"] == "ok":
                assert abs(value["offset"]) <= value["absolute_offset_bound"] + 1e-12
            else:
                assert value["offset"] == 0
    return {"age_choices_independently_replayed": len(lines("age_setting_choices.jsonl")),
            "retention_choices_independently_replayed": len(lines("retention_choices.jsonl"))}


def audit_joint_quantiles(points, intervals, draws, config):
    history = read("historical_selected_scores.csv")
    coords = pd.MultiIndex.from_product([config["sexes"], config["ages"], range(1, 6)], names=COORDS)
    count = 0
    for (origin, family), part in intervals.groupby(["origin", "family"]):
        source_family = part.source_point_family.unique()
        assert len(source_family) == 1
        history_part = history[history.family.eq(source_family[0]) & history.origin.between(2003, origin - 5)]
        residual_origins = list(range(2003, origin - 4))
        raw = []
        for residual_origin in residual_origins:
            block = history_part[history_part.origin.eq(residual_origin)].set_index(COORDS).reindex(coords)
            assert len(block) == 110 and (block.forecast_year <= origin).all()
            raw.append(np.log(block.observed_rate.to_numpy()) - block.log_prediction.to_numpy())
        raw = np.asarray(raw)
        center = raw.mean(axis=0)
        fractions = part.groupby("sex").bias_retention.agg(lambda x: x.unique()[0])
        assert part.groupby("sex").bias_retention.nunique().eq(1).all()
        retention = np.asarray([fractions.loc[sex] for sex, _, _ in coords])
        expected_residuals = raw - center + retention[None, :] * center
        current = points[points.origin.eq(origin) & points.family.eq(source_family[0])].set_index(COORDS).reindex(coords)
        expected_logs = current.log_prediction.to_numpy()[None, :] + expected_residuals
        joint = draws[draws.origin.eq(origin) & draws.family.eq(family)]
        for scale, values, field in [("log_rate", expected_logs, "log_draw"), ("rate", np.exp(expected_logs), "rate_draw")]:
            actual = joint.pivot(index="residual_origin", columns=COORDS, values=field).reindex(
                index=residual_origins, columns=coords).to_numpy()
            np.testing.assert_allclose(actual, values, atol=1e-12, rtol=1e-12)
            for level in [0.5, 0.8, 0.95]:
                cell = part[part.scale.eq(scale) & part.level.eq(level)].set_index(COORDS).reindex(coords)
                expected = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0, method="linear").T
                np.testing.assert_allclose(cell[["lower", "median", "upper"]], expected, atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(cell.point_prediction, current.log_prediction if scale == "log_rate" else current.prediction)
                assert cell.n_blocks.eq(len(residual_origins)).all()
                count += len(cell)
    assert count == len(intervals) == 26400 and len(draws) == 39600
    return count


def audit_interval_scores(cells, wis):
    observed = np.where(cells.scale.eq("log_rate"), np.log(cells.observed_rate), cells.observed_rate)
    alpha = 1 - cells.level.to_numpy()
    width = cells.upper.to_numpy() - cells.lower.to_numpy()
    value = width + 2 / alpha * (np.maximum(cells.lower.to_numpy() - observed, 0)
                               + np.maximum(observed - cells.upper.to_numpy(), 0))
    np.testing.assert_allclose(cells.observed_value, observed, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(cells.width, width, atol=1e-11, rtol=1e-12)
    np.testing.assert_allclose(cells.interval_score, value, atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(cells.covered, ((observed >= cells.lower) & (observed <= cells.upper)).astype(float))
    keys = ["target", "outcome", "origin", "family", "sex", "age", "horizon", "forecast_year", "scale"]
    index = wis.set_index(keys)
    temporary = cells[keys + ["level"]].copy()
    temporary["weighted_score"] = alpha * value / 2
    scores = temporary.pivot(index=keys, columns="level", values="weighted_score").reindex(index.index)
    for field, levels in [("wis_50_80", [.5, .8]), ("wis_50_80_95", [.5, .8, .95])]:
        expected = (.5 * np.abs(index.observed_value - index["median"]) + scores[levels].sum(axis=1)) / (len(levels) + .5)
        np.testing.assert_allclose(index[field], expected, atol=1e-10, rtol=1e-12)
    return len(wis)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def interval_table(cells, wis, scope):
    selection = cells[cells.horizon.eq(5) & cells.scale.eq("rate") & cells.level.eq(.8)]
    w = wis[wis.horizon.eq(5) & wis.scale.eq("rate")]
    if scope == "endpoint":
        selection, w = selection[selection.origin.eq(2018)], w[w.origin.eq(2018)]
    groups = ["family", "sex"]
    summary = selection.groupby(groups).agg(coverage=("covered", "mean"), width=("width", "mean"))
    summary["wis"] = w.groupby(groups).wis_50_80.mean()
    rows = []
    for sex in ["Male", "Female"]:
        for family in POINTS:
            for variant in VARIANTS:
                row = summary.loc[(family + "__retention=" + variant, sex)]
                rows.append([sex, LABELS[family], variant, f"{100*row.coverage:.1f}%", f"{row.width:.2f}", f"{row.wis:.2f}"])
    return table(["Sex", "Point procedure", "Retention r", "80% coverage", "Mean width", "WIS (50/80)"], rows), summary


def make_figures(point_summary, scores, endpoint, reliability, ages):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    limit = float(scores[scores.horizon.eq(5)].signed_log_error.abs().max())
    for col, sex in enumerate(["Male", "Female"]):
        for family, color, marker in [(POINTS[0], "#1676a8", "o"), (POINTS[1], "#cb7937", "x")]:
            part = point_summary[point_summary.sex.eq(sex) & point_summary.horizon.eq(5) & point_summary.family.eq(family)]
            axs[0, col].plot(part.origin, part.mean_absolute_log_error, label=LABELS[family], color=color,
                             marker=marker, markersize=8, linestyle="-" if family == POINTS[0] else "--")
        axs[0, col].set(title=f"{sex}: five-year point error", xlabel="Fit origin", ylabel="Mean absolute log error", xticks=range(2014, 2019))
        axs[0, col].legend(fontsize=9)
        values = scores[scores.sex.eq(sex) & scores.horizon.eq(5) & scores.family.eq(POINTS[0])].pivot(
            index="age", columns="origin", values="signed_log_error").reindex(index=ages, columns=range(2014, 2019))
        artist = axs[1, col].imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axs[1, col].set(title=f"{sex}: age-specific signed log error (both procedures)", xlabel="Fit origin",
                        xticks=range(5), xticklabels=range(2014, 2019), yticks=range(11), yticklabels=ages)
    fig.colorbar(artist, ax=axs[1, :], label="Signed log error (positive = overprediction)", shrink=.9)
    fig.suptitle("Exploratory age offsets: chronologically selected point forecasts unchanged", fontsize=14)
    fig.savefig(OUT / "point_errors_and_age_bias.png", dpi=180)
    fig.savefig(OUT / "point_errors_and_age_bias.svg")
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True, sharey=True)
    labels = [f"Original\nr={r}" for r in VARIANTS] + [f"Age offsets\nr={r}" for r in VARIANTS]
    for row, (name, summary) in enumerate([("2018 → 2023; 11 ages", endpoint), ("Five origins; 55 age–origin cells", reliability)]):
        for col, sex in enumerate(["Male", "Female"]):
            values = [summary.loc[(family + "__retention=" + variant, sex), "coverage"] for family in POINTS for variant in VARIANTS]
            ax = axs[row, col]
            bars = ax.bar(range(8), values, color=["#1676a8"] * 4 + ["#cb7937"] * 4)
            ax.axhline(.8, color="#a02e37", linestyle="--", label="Nominal 80%")
            ax.bar_label(bars, labels=[f"{100*v:.0f}%" for v in values], fontsize=9, padding=3)
            ax.set(title=f"{sex}: {name}", ylim=(0, 1), xticks=range(8), xticklabels=labels, ylabel="Empirical 80% coverage")
            ax.tick_params(axis="x", labelsize=8)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Bias retention shifts intervals; sparse dependent blocks provide no coverage guarantee", fontsize=14)
    fig.savefig(OUT / "interval_coverage_all_variants.png", dpi=180)
    fig.savefig(OUT / "interval_coverage_all_variants.svg")
    plt.close(fig)


def main():
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    amendment = json.loads((ROOT / "study_design/exploratory_v1_1.json").read_text())
    manifest = verify_integrity(amendment)
    points, scores = read("point_predictions.csv"), read("evaluation_point_scores.csv")
    intervals, draws = read("interval_predictions.csv"), read("joint_draws.csv")
    cells, wis = read("evaluation_interval_scores.csv"), read("evaluation_wis_scores.csv")
    point_summary = audit_point_scores(scores, points)
    selection_validation = audit_selections(config, amendment)
    interval_count = audit_joint_quantiles(points, intervals, draws, config)
    wis_count = audit_interval_scores(cells, wis)
    development_wis_count = audit_interval_scores(read("development_interval_scores.csv"), read("development_wis_scores.csv"))
    verdict = json.loads((ROOT / "results/primary_v1/primary_verdict.json").read_text())
    assert verdict["joint_success"] is False
    keys = ["origin", "sex", "age", "horizon"]
    original = scores[scores.family.eq(POINTS[0])].set_index(keys).sort_index()
    improved = scores[scores.family.eq(POINTS[1])].set_index(keys).sort_index()
    np.testing.assert_array_equal(original.log_prediction, improved.log_prediction)
    # Compare against the committed original prediction ledger. The original
    # score ledger used a different CSV parser and may differ by one ulp.
    previous = read("predictions.csv", ROOT / "results/primary_v1")
    previous = previous[previous.family.eq("tcn_adapted")].set_index(keys).sort_index()
    np.testing.assert_array_equal(original.log_prediction, previous.log_prediction)
    OUT.mkdir(parents=True, exist_ok=True)
    endpoint_table, endpoint = interval_table(cells, wis, "endpoint")
    reliability_table, reliability = interval_table(cells, wis, "reliability")
    make_figures(point_summary, scores, endpoint, reliability, config["ages"])
    rows = []
    for sex in config["sexes"]:
        for family in POINTS:
            selected = point_summary[point_summary.sex.eq(sex) & point_summary.family.eq(family) & point_summary.horizon.eq(5)]
            final = selected[selected.origin.eq(2018)].iloc[0]
            rows.append([sex, LABELS[family], f"{final.mean_absolute_log_error:.6f}", f"{selected.mean_absolute_log_error.mean():.6f}",
                         f"{final.mean_signed_log_error:+.6f}"])
    point_table = table(["Sex", "Point procedure", "2018→2023 ALE", "Five-origin mean ALE", "2018→2023 signed log error"], rows)
    age = pd.DataFrame(lines("age_setting_choices.jsonl"))
    retention = pd.DataFrame(lines("retention_choices.jsonl"))
    choice_rows = []
    for origin in range(2014, 2019):
        for sex in config["sexes"]:
            selected = age[age.origin.eq(origin) & age.sex.eq(sex)].iloc[0]
            r = retention[retention.origin.eq(origin) & retention.sex.eq(sex)].set_index("point_family")
            choice_rows.append([origin, sex, "None" if pd.isna(selected.penalty) else f"{selected.penalty:g}",
                                f"{r.loc[POINTS[0], 'retention']:g}", f"{r.loc[POINTS[1], 'retention']:g}",
                                ", ".join(map(str, r.loc[POINTS[0], "inner_origins"])) or "Cold default"])
    choice_table = table(["Origin", "Sex", "Age penalty", "Original r", "Age procedure r", "Interval tuning origins"], choice_rows)
    source_validation = json.loads((RUN / "validation_report.json").read_text())
    text = f"""# Exploratory age correction and interval bias retention

**The prespecified primary conclusion remains unchanged: the joint success criterion was not met.** This amendment was written after the original 2019–2023 evaluation results were viewed. All comparisons here are exploratory, including those with chronological tuning. Both sexes select zero additional age offsets at every evaluation origin (2014–2018); their point forecasts are identical to the original adapted TCN. At final origin 2018, both procedures select fully centered intervals (r=0) for both sexes. The two proposed changes therefore provide no selected point-forecast improvement and do not resolve interval undercoverage.

## Frozen procedures and interpretation

The [registered amendment](../../study_design/exploratory_v1_1.md) preserves all original source models, five-seed ensembles, selected architectures, and two-coefficient sex corrections. Three broad age-group offsets (45–64, 65–79, 80+) use penalties 1, 10, or 100, alongside a zero-offset option. Only complete earlier five-year forecast blocks select settings. The cold start uses penalty 100 in 2003–2007; zero offsets are selected for both sexes from 2008 onward. An age-offset fit failure retains the unchanged v1 point forecast and records the failure; **{source_validation['correction_fallbacks']} failures occurred**. No source model was retrained and no original correction was refitted.

The interval variants retain r=0, 0.5, or 1 of each coordinate's historical mean residual. They preserve the same paired residual origin across both sexes, all 11 ages, and all five horizons. Retention changes location, not log-scale dispersion; rate-scale widths may change after exponentiation. Each point procedure has its own honest historical residual bank. Thus its intervals can differ even when current points coincide: the age procedure used small cold-start offsets in 2003–2007. The unchanged procedure with r=0 reproduces the original v1 intervals exactly.

Selection of r uses five-year log-scale WIS (50/80 levels) from interval origins 2012 and 2013 only after their outcomes are complete. Consequently origins 2014–2016 use r=0 by default, 2017 has one tuning origin, and 2018 has two. Ties favor less retention. All fixed alternatives remain reported; none is promoted based on its 2019–2023 result. Quantiles are interpolated independently on log and rate scales after transforming each draw. Predictive medians may differ from the unchanged point forecast.

## Point accuracy and signed bias

ALE is equal-age absolute log error. The five-origin mean includes origin 2018 once. Positive signed log error indicates overprediction; negative indicates underprediction.

{point_table}

![Point errors and age-specific signed bias](point_errors_and_age_bias.png)

The heatmaps describe forecast error against modeled GBD point estimates. Opposite signs across ages can cancel in a sex-level mean; the full age-by-sex-by-horizon-by-origin cells remain in the [score ledger](../../results/improvements_v1/evaluation_point_scores.csv). These patterns do not establish biological mechanisms or individual risk.

## Interval performance at the primary endpoint

Each row below represents the 11 dependent age cells at horizon 5, origin 2018 (year 2023). Width and WIS are on the rate scale, per 100,000. WIS uses the median plus the 50% and 80% intervals; smaller is better.

{endpoint_table}

For the unchanged male forecasts, full bias retention lowers rate WIS from {endpoint.loc[(POINTS[0]+'__retention=0','Male'),'wis']:.2f} to {endpoint.loc[(POINTS[0]+'__retention=1','Male'),'wis']:.2f}, while 80% coverage falls from {endpoint.loc[(POINTS[0]+'__retention=0','Male'),'coverage']:.1%} to {endpoint.loc[(POINTS[0]+'__retention=1','Male'),'coverage']:.1%}. This illustrates why a lower aggregate proper score does not imply adequate nominal coverage in every subgroup. These fixed alternatives are sensitivity results, not newly selected winners.

## Reliability across five evaluation origins

These equal-age, equal-origin summaries contain 55 dependent age–origin cells per sex at horizon 5. The five origins use 7–11 historical joint residual blocks. Overlapping periods and age dependence preclude interpreting the cells as independent replicates; no exact or distribution-free coverage guarantee is asserted.

{reliability_table}

![Coverage for every fixed and selected interval variant](interval_coverage_all_variants.png)

All horizons, both scales, and 50%, 80%, and 95% intervals are retained in the [interval summary](../../results/improvements_v1/interval_summary.csv) and [WIS summary](../../results/improvements_v1/wis_summary.csv). The 95% results remain supplementary because 7–11 empirical residual blocks provide sparse tail information.

## Chronological evaluation choices

“None” denotes exactly zero additional age offsets; it does not denote an unpenalized fit.

{choice_table}

The only nonzero selected retention during evaluation is r=0.5 for the unchanged female procedure in 2017, selected using origin 2012 alone. This difference is retained; it is not corrected after observing its evaluation result.

## Validation and scope

The run completed in {source_validation['elapsed_seconds']:.2f} seconds using eight CPU workers for independent correction batches. Eleven synthetic tests cover future perturbations, missing seeds, convex shrinkage bounds, zero-offset equivalence, pairing of joint blocks, and chronological selection. The independent report audit replays all 32 age choices and 20 retention choices; reconstructs all 39,600 paired draws and 26,400 interval rows from historical residuals; recalculates point errors, interval scores and WIS; and checks saved predictions/settings precede evaluation scoring. All original run manifests, source-model references, and outputs remain hash verified. See [report validation](validation.json) and [run validation](../../results/improvements_v1/validation_report.json).

GBD source-estimate bounds are exported separately in [source_bounds_separate.csv](../../results/improvements_v1/source_bounds_separate.csv). They were not used as forecast intervals, calibration targets, or additional observations. Rates only are evaluated here; demographic/count analyses, additional outcomes, and donor sensitivity are separate work. Neither the exploratory tuning nor its diagnostic figures change the [original primary verdict](../primary_v1/report.md).
"""
    (OUT / "report.md").write_text(text)
    validation = {"passed": True, "role": "independent_post_result_exploratory_audit", "original_joint_success": False,
                  "original_primary_verdict_preserved": True, "all_original_and_new_run_hashes_verified": True,
                  "report_code_sha256": sha(Path(__file__)), "run_manifest_sha256": sha(RUN / "run_manifest.json"),
                  "point_cells_recalculated": len(scores), "interval_cells_reconstructed": interval_count,
                  "evaluation_wis_cells_recalculated": wis_count, "development_wis_cells_recalculated": development_wis_count,
                  "joint_draw_cells_reconstructed": len(draws), "all_evaluation_age_offsets_zero": True,
                  **selection_validation}
    validation["report_artifact_sha256"] = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "validation.json"}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
