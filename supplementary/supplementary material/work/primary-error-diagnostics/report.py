"""Read-only, post-evaluation age/sex error diagnostic; no forecasting or tuning."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gbd_park_matplotlib")
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/primary_error_diagnostics"
RUN = ROOT / "results/primary_v1"
AGES = ["80-84", "85-89", "90-94", "95+"]
SEXES = ["Male", "Female"]
FAMILIES = ["tcn_adapted", "local_champion", "nonneural_champion"]
LABELS = dict(tcn_adapted="Adapted TCN", local_champion="Local champion",
              nonneural_champion="Non-neural champion")
COLORS = dict(tcn_adapted="#0072B2", local_champion="#D55E00", nonneural_champion="#009E73")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def main():
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["device"] == "cpu"
    inputs = [RUN / "run_manifest.json", RUN / "predictions.csv", RUN / "point_scores.csv",
              ROOT / "study_design/locked_v1/design.json", ROOT / "study_design/locked_v1/lock_manifest.json",
              ROOT / "data/processed/design_v1/regional_outcomes.csv"]
    before = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    for name in ["predictions.csv", "point_scores.csv"]:
        assert sha(RUN / name) == manifest["output_sha256"][name]
    lock = json.loads(inputs[4].read_text())
    assert sha(inputs[4]) == manifest["source_lock_hash"]
    for section in ["design_sha256", "output_sha256"]:
        for name, digest in lock[section].items():
            assert sha(ROOT / name) == digest
    config = json.loads(inputs[3].read_text())
    assert sha(inputs[3]) == manifest["config_sha256"]
    scores = pd.read_csv(RUN / "point_scores.csv", float_precision="round_trip")
    issued = pd.read_csv(RUN / "predictions.csv", float_precision="round_trip")
    keep = scores.origin.eq(2018) & scores.family.isin(FAMILIES)
    scores = scores.loc[keep].copy()
    assert len(scores) == 3 * 2 * 11 * 5
    keys = ["origin", "family", "sex", "age", "horizon"]
    assert not scores.duplicated(keys).any()
    check = scores.merge(issued[keys + ["prediction", "log_prediction"]], on=keys,
                         validate="one_to_one", suffixes=("", "_issued"))
    csv_tolerance = 8 * np.finfo(float).eps
    np.testing.assert_allclose(check.prediction, check.prediction_issued, rtol=csv_tolerance, atol=0)
    np.testing.assert_allclose(check.log_prediction, check.log_prediction_issued, rtol=csv_tolerance, atol=0)
    forecast_roundtrip_max_difference = float(abs(check.prediction - check.prediction_issued).max())
    panel = pd.read_csv(inputs[-1], float_precision="round_trip")
    history = panel.loc[panel.location_name.eq("Saudi Arabia") & panel.outcome.eq("prevalence")
                        & panel.year.between(2006, 2023)].copy()
    assert len(history) == 2 * 11 * 18 and not history.duplicated(["sex", "age", "year"]).any()
    check = scores.merge(history[["sex", "age", "year", "rate"]],
                         left_on=["sex", "age", "forecast_year"], right_on=["sex", "age", "year"], validate="many_to_one")
    np.testing.assert_allclose(check.observed_rate, check.rate, rtol=1e-12, atol=0)
    scores["signed_log_error"] = scores.log_prediction - np.log(scores.observed_rate)
    scores["signed_rate_error"] = scores.prediction - scores.observed_rate
    scores["signed_percent_error"] = 100 * np.expm1(scores.signed_log_error)
    scores["age_group"] = np.where(scores.age.isin(AGES), "80+", "45-79")
    np.testing.assert_allclose(abs(scores.signed_log_error), scores.absolute_log_error, atol=2e-15, rtol=1e-12)
    endpoint = scores.loc[scores.horizon.eq(5)].copy()
    summaries = []
    for (sex, family, horizon), part in scores.groupby(["sex", "family", "horizon"], sort=False):
        older = part.loc[part.age_group.eq("80+")]
        younger = part.loc[part.age_group.eq("45-79")]
        total = part.absolute_log_error.sum()
        summaries.append(dict(sex=sex, family=family, source_family=part.source_family.iloc[0], horizon=horizon,
            forecast_year=2018+horizon, age45plus_mean_absolute_log_error=total/11,
            older_mean_absolute_log_error=older.absolute_log_error.mean(),
            younger_mean_absolute_log_error=younger.absolute_log_error.mean(),
            older_contribution_to_age45plus_mean=older.absolute_log_error.sum()/11,
            younger_contribution_to_age45plus_mean=younger.absolute_log_error.sum()/11,
            older_share_of_total_error_percent=100*older.absolute_log_error.sum()/total if total > 0 else np.nan,
            older_mean_signed_log_error=older.signed_log_error.mean(),
            older_overpredicted_cells=int(older.signed_log_error.gt(0).sum()),
            older_underpredicted_cells=int(older.signed_log_error.lt(0).sum()),
            older_min_signed_percent_error=older.signed_percent_error.min(),
            older_max_signed_percent_error=older.signed_percent_error.max()))
    summary = pd.DataFrame(summaries)
    np.testing.assert_allclose(summary.older_contribution_to_age45plus_mean + summary.younger_contribution_to_age45plus_mean,
                               summary.age45plus_mean_absolute_log_error, rtol=1e-14)
    slopes = []
    for (sex, age), part in history.loc[history.age.isin(AGES)].groupby(["sex", "age"]):
        record = dict(sex=sex, age=age)
        for label, first, last in [("pre", 2006, 2018), ("recent_pre", 2011, 2018), ("post", 2019, 2023)]:
            segment = part.loc[part.year.between(first, last)].sort_values("year")
            years = segment.year.to_numpy(dtype=float)
            log_rate = np.log(segment.rate.to_numpy())
            slope = np.dot(years-years.mean(), log_rate-log_rate.mean()) / np.sum((years-years.mean())**2)
            record.update({label+"_first_year": first, label+"_last_year": last,
                           label+"_n_years": len(segment), label+"_log_slope_per_year": slope,
                           label+"_annual_percent_change": 100*np.expm1(slope)})
        record["post_minus_pre_log_slope"] = record["post_log_slope_per_year"] - record["pre_log_slope_per_year"]
        slopes.append(record)
    slopes = pd.DataFrame(slopes)
    annual = history.loc[history.age.isin(AGES), ["sex", "age", "year", "rate"]].sort_values(["sex", "age", "year"])
    annual["annual_log_change"] = np.log(annual.rate).groupby([annual.sex, annual.age]).diff()
    annual["annual_percent_change"] = 100 * np.expm1(annual.annual_log_change)
    contrasts = []
    for sex in SEXES:
        neural = endpoint.loc[endpoint.sex.eq(sex) & endpoint.family.eq("tcn_adapted")].set_index("age")
        for family in FAMILIES[1:]:
            comparator = endpoint.loc[endpoint.sex.eq(sex) & endpoint.family.eq(family)].set_index("age")
            difference = neural.absolute_log_error - comparator.absolute_log_error
            for age_group, mask in [("80+", difference.index.isin(AGES)), ("45-79", ~difference.index.isin(AGES))]:
                contrasts.append(dict(sex=sex, comparator=family, age_group=age_group,
                    contribution_to_tcn_minus_comparator_mean_ALE=difference.loc[mask].sum()/11))
    contrasts = pd.DataFrame(contrasts)
    OUT.mkdir(parents=True, exist_ok=True)
    history.loc[history.age.isin(AGES), ["sex", "age", "year", "rate"]].to_csv(OUT / "observed_trajectories_2006_2023.csv", index=False)
    scores.to_csv(OUT / "origin2018_signed_errors_all_horizons.csv", index=False)
    endpoint.loc[endpoint.age.isin(AGES)].to_csv(OUT / "endpoint_80plus_signed_errors.csv", index=False)
    summary.to_csv(OUT / "age_group_error_contributions.csv", index=False)
    slopes.to_csv(OUT / "observed_log_slopes.csv", index=False)
    annual.to_csv(OUT / "observed_annual_changes.csv", index=False)
    contrasts.to_csv(OUT / "endpoint_contrast_contributions.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.3), sharex=True)
    for row, sex in enumerate(SEXES):
        for col, age in enumerate(AGES):
            axis = axes[row, col]
            observed = history.loc[history.sex.eq(sex) & history.age.eq(age)].sort_values("year")
            axis.axvspan(2018, 2023, color="#CBD5E1", alpha=.25, linewidth=0)
            axis.axvline(2018, color="#7A7A7A", linewidth=.8, linestyle=":")
            axis.plot(observed.year, observed.rate, color="#222222", linewidth=2, label="GBD point estimate")
            anchor = observed.loc[observed.year.eq(2018), "rate"].iloc[0]
            for family, style in zip(FAMILIES, ["--", "-.", ":"]):
                part = scores.loc[scores.sex.eq(sex) & scores.age.eq(age) & scores.family.eq(family)].sort_values("horizon")
                axis.plot([2018, *part.forecast_year], [anchor, *part.prediction], color=COLORS[family],
                          linestyle=style, linewidth=1.9, marker="o", markersize=2.8, label=LABELS[family])
            axis.set_title(f"{sex}, {age}")
            axis.set_xticks([2006, 2010, 2014, 2018, 2023])
            axis.grid(alpha=.2)
            if col == 0:
                axis.set_ylabel("Prevalence per 100,000")
            if row == 1:
                axis.set_xlabel("Year")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .96), ncol=4, frameon=False)
    fig.suptitle("Saudi prevalence ages 80+: observed trajectories and forecasts issued in 2018", y=.995)
    fig.text(.5, .01, "Shaded years are the forecast period. Panels use their own rate scales; lines begin at the observed 2018 value.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .025, 1, .92))
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"age80plus_trajectories.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    endpoint_summary = summary.loc[summary.horizon.eq(5)].set_index(["sex", "family"]).reindex(
        pd.MultiIndex.from_product([SEXES, FAMILIES], names=["sex", "family"])).reset_index()
    slopes = slopes.set_index(["sex", "age"]).reindex(pd.MultiIndex.from_product([SEXES, AGES], names=["sex", "age"])).reset_index()
    display = pd.DataFrame([{"Sex": row.sex, "Procedure": LABELS[row.family], "45+ ALE (11 bands)": f"{row.age45plus_mean_absolute_log_error:.6f}",
        "80+ share of error": f"{row.older_share_of_total_error_percent:.2f}%", "80+ mean signed log error": f"{row.older_mean_signed_log_error:+.5f}",
        "80+ over/under": f"{row.older_overpredicted_cells}/{row.older_underpredicted_cells}"} for row in endpoint_summary.itertuples()])
    slope_display = pd.DataFrame([{"Sex": row.sex, "Age": row.age, "2006–2018": f"{100*row.pre_log_slope_per_year:+.3f}",
        "2011–2018": f"{100*row.recent_pre_log_slope_per_year:+.3f}", "2019–2023": f"{100*row.post_log_slope_per_year:+.3f}"}
        for row in slopes.itertuples()])
    percentages = endpoint.loc[endpoint.age.isin(AGES)].pivot(index=["sex", "age"], columns="family", values="signed_percent_error")
    percentages = percentages.reindex(pd.MultiIndex.from_product([SEXES, AGES], names=["sex", "age"]))
    percent_display = percentages[FAMILIES].map(lambda value: f"{value:+.2f}%").rename(columns=LABELS).reset_index().rename(columns={"sex": "Sex", "age": "Age"})
    female_share = endpoint_summary.loc[endpoint_summary.sex.eq("Female") & endpoint_summary.family.eq("tcn_adapted"), "older_share_of_total_error_percent"].iloc[0]
    female_margin = contrasts.loc[contrasts.sex.eq("Female") & contrasts.comparator.eq("nonneural_champion")].set_index("age_group").contribution_to_tcn_minus_comparator_mean_ALE
    male_last = annual.loc[annual.sex.eq("Male") & annual.year.eq(2023), "annual_percent_change"]
    report = f"""# Original Saudi prevalence error diagnostic

The four age bands 80–84, 85–89, 90–94 and 95+ account for **{female_share:.4f}%** of the adapted TCN's female 2023 endpoint absolute-log-error sum. All three procedures overpredict every female 80+ band. Male TCN and local forecasts underpredict all four bands; the non-neural male forecast underpredicts three and slightly overpredicts 95+.

This is a post-evaluation description of the original **CPU primary run**, using its saved origin-2018 forecasts. No model is fitted, selected, corrected or replaced. “Observed” below means the GBD 2023 modeled point estimate used for verification.

## Contribution to the primary error

ALE is the equal-weight mean across all 11 included age bands, from 45–49 through 95+. The 80+ percentage is `sum(ALE in four older bands) / sum(ALE in all 11 included age bands) × 100`; it measures error concentration, not population share or disease burden. The four bands represent 36.36% of equally weighted age cells. Signed error is `log(prediction/observed)`; positive values indicate overprediction. Over/under counts are out of four.

{table(display)}

For females, the 80+ bands contribute **{female_margin.loc['80+']:+.6f}** to the TCN-minus-non-neural difference in mean ALE across the 11 age bands at ages 45+; ages 45–79 contribute **{female_margin.loc['45-79']:+.6f}**. Their sum is **{female_margin.sum():+.6f}**, so the poorer older-age performance outweighs the younger-age improvement.

## Trajectories and direction

![Observed trajectories and saved forecasts](age80plus_trajectories.png)

The following are 2023 percentage errors, `100 × (prediction/observed − 1)`. Male local/non-neural champions are damped ETS/pooled boosting; female champions are ARIMA/adapted donor ridge, as selected in the original chronological procedure.

{table(percent_display)}

## Descriptive changes in slope

Slopes are ordinary least-squares slopes of **observed log rate against year**, multiplied by 100 (log points per year). The long pre-period has 13 values, the recent pre-period has eight, and the post-period has five. The recent pre-period aligns with the neural input-window length. The post-period excludes the 2018–2019 transition. These summaries are calculated after evaluation and never used for model selection; they are not formal change-point tests.

{table(slope_display)}

Female older-age modeled rates flatten markedly after 2018 relative to both historical windows, while the saved forecasts continue upward. This pattern is consistent with extrapolation overshoot across several model classes. Male slope changes vary by age: the plotted trajectories include a **{male_last.min():.2f}%–{male_last.max():.2f}% rise during 2022–2023** across the four bands, leaving most endpoint forecasts below the modeled rates. A period-average slope masks this late movement. A correction shared over all ages can leave age-dependent error, but this diagnostic does not identify its cause.

GBD estimates can reflect changes in underlying information, estimation procedures, or the modeled disease trajectory. These data alone cannot separate those explanations, identify a biological mechanism, infer individual progression or survival, or establish a causal sex difference. Small, dependent age/year cells do not support independent-cell significance claims. The fixed primary result and prespecified secondary comparisons remain unchanged.

## Reproduction and tables

Run `/home/saif/agpu_env/bin/python work/primary-error-diagnostics/report.py`. The report checks the original prediction/score hashes, locked prepared data, forecast agreement within CSV machine precision, verification values, and error-contribution sums. It rechecks source hashes after writing new report files. CSVs retain all five forecast horizons, full-precision signed errors, slope definitions and the contributions to comparator differences. See [validation](validation.json), [error contributions](age_group_error_contributions.csv), [older-age errors](endpoint_80plus_signed_errors.csv) and [observed slopes](observed_log_slopes.csv).
"""
    (OUT / "report.md").write_text(report)
    assert all(sha(ROOT / name) == digest for name, digest in before.items())
    validation = dict(passed=True, role="post_evaluation_read_only_diagnostic", new_model_fits=0,
        original_inputs_unchanged=True, input_sha256=before, diagnostic_source_sha256=sha(__file__),
        forecast_roundtrip_max_absolute_difference=forecast_roundtrip_max_difference,
        forecast_csv_relative_tolerance=csv_tolerance,
        female_tcn_80plus_share_of_endpoint_error_percent=float(female_share),
        forecast_cells_checked=len(scores), endpoint_older_cells=24, observed_older_rate_values=144,
        checks=["immutable primary forecast and score hashes", "locked design and prepared data hashes",
                "saved forecasts match scored forecasts within CSV machine precision", "verification values match prepared data",
                "signed and absolute log errors agree", "age-group contributions sum to primary mean error",
                "source hashes unchanged after diagnostic"],
        output_sha256={path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "validation.json"})
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"passed": True, "female_80plus_error_share_percent": float(female_share),
                      "report": str((OUT / "report.md").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
