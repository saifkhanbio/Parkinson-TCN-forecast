"""Read-only reconstruction of frozen v1.3 states; no model fitting or tuning."""
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gbd_park.dynamic_joint import _decode

OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/reliability_v1_3"
FAMILY = "robust_dynamic__joint"
inputs = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    inputs[str(path.relative_to(ROOT))] = sha(path)
    return path


def read(path):
    return pd.read_csv(source(path), float_precision="round_trip", low_memory=False)


def pct(estimate, observed):
    return 100 * (estimate / observed - 1)


panel = read(ROOT / "data/processed/design_v1/regional_outcomes.csv")
source(ROOT / "src/gbd_park/dynamic_joint.py")
source(ROOT / "study_design/reliability_v1_3.json")
weights, tracking, cells, slopes, issued, decomposition, interval_checks = [], [], [], [], [], [], []
for directory in sorted(RESULTS.glob("*_*")):
    path = directory / "dynamic_diagnostics.json"
    if not path.is_file():
        continue
    for audit in json.loads(source(path).read_text()):
        w = np.asarray(audit["filter_weights"])
        radial = np.asarray(audit["radial_weights"])
        weights.append(dict(target=audit["target"], outcome=audit["outcome"], origin=audit["origin"],
                            updates=len(w), first_weight=w[0], last_weight=w[-1],
                            minimum_weight=w.min(), median_weight=np.median(w), maximum_weight=w.max(),
                            updates_below_point1=int((w < .1).sum()),
                            last_observation_variance_multiplier=1/w[-1],
                            radial_minimum=radial.min(), radial_median=np.median(radial),
                            radial_clipped_years=int((radial < 1).sum())))

for outcome in ["prevalence", "incidence"]:
    directory = RESULTS / ("SAU_" + outcome)
    rates = read(directory / "rate_point_scores.csv")
    rates = rates.loc[rates.family.eq(FAMILY)]
    burden = read(directory / "burden_point_scores.csv")
    burden = burden.loc[burden.family.eq(FAMILY) & burden.measure.eq("count")]
    bounds = read(directory / "burden_interval_scores.csv.gz")
    selected = bounds.loc[bounds.family.eq(FAMILY) & bounds.node.eq("Both__45+")
                          & bounds.horizon.eq(5) & bounds.level.eq(.8)].copy()
    assert len(selected) == 5 and selected.origin.nunique() == 5
    interval_checks.extend(selected[["target", "outcome", "origin", "lower", "median", "upper", "observed",
                                     "covered", "lower_miss", "upper_miss"]].to_dict("records"))
    for origin in range(2014, 2019):
        saved = joblib.load(source(directory / "draws" / f"origin{origin}__dynamic_state.joblib"))
        f = saved["fitted_state"]
        dimension = len(f["sigma"])
        z, means = f["transformed_history"], f["filtered_means"]
        actual = _decode(z, f)
        fitted = _decode(means[:, :dimension], f)
        expected = pd.MultiIndex.from_product([f["years"], f["sexes"], f["ages"]],
                                               names=["year", "sex", "age"])
        observed = panel.loc[panel.location_name.eq(f["target"]) & panel.outcome.eq(outcome)
                             & panel.year.le(origin)].set_index(["year", "sex", "age"]).reindex(expected)
        for values, column in zip(actual, ["rate", "count", "implied_population"]):
            np.testing.assert_allclose(values.ravel(), observed[column], rtol=1e-10, atol=1e-8)
        last_prior = f["transition"] @ means[-2]
        for yi, year in enumerate(f["years"]):
            for si, sex in enumerate(f["sexes"]):
                sl = slice(si * 11, (si + 1) * 11)
                record = dict(target=f["target"], outcome=outcome, origin=origin, year=int(year), sex=sex,
                              filter_weight=np.nan if yi == 0 else f["filter_weights"][yi-1],
                              observed_count=actual[1][yi, sl].sum(), filtered_count=fitted[1][yi, sl].sum(),
                              observed_population=actual[2][yi, sl].sum(), filtered_population=fitted[2][yi, sl].sum(),
                              rate_mean_signed_log_error=np.log(fitted[0][yi, sl]/actual[0][yi, sl]).mean(),
                              rate_mean_absolute_log_error=np.abs(np.log(fitted[0][yi, sl]/actual[0][yi, sl])).mean())
                for name in ["count", "population"]:
                    record[name+"_percent_error"] = pct(record["filtered_"+name], record["observed_"+name])
                tracking.append(record)
        for si, sex in enumerate(f["sexes"]):
            j = si * 22
            first_delta, last_delta = z[1, j]-z[0, j], z[-1, j]-z[-2, j]
            innovation, update = z[-1, j]-last_prior[j], means[-1, j]-last_prior[j]
            slopes.append(dict(outcome=outcome, origin=origin, sex=sex,
                               first_observed_log_count_change=first_delta,
                               first_filtered_count_slope=means[1, dimension+j],
                               first_slope_fraction_of_observed_change=means[1, dimension+j]/first_delta,
                               last_observed_log_count_change=last_delta,
                               final_filtered_count_slope=means[-1, dimension+j],
                               final_slope_fraction_of_observed_change=means[-1, dimension+j]/last_delta,
                               final_count_level_innovation=innovation,
                               final_count_level_correction=update,
                               final_correction_fraction_of_innovation=update/innovation))
            for ai, age in enumerate(f["ages"]):
                j = si*11+ai
                row = dict(outcome=outcome, origin=origin, sex=sex, age=age)
                for index, name in enumerate(["rate", "count", "population"]):
                    row["observed_"+name], row["filtered_"+name] = actual[index][-1,j], fitted[index][-1,j]
                    row[name+"_percent_error"] = pct(fitted[index][-1,j], actual[index][-1,j])
                cells.append(row)
        for horizon in [1, 5]:
            r = rates.loc[rates.origin.eq(origin) & rates.horizon.eq(horizon)]
            b = burden.loc[burden.origin.eq(origin) & burden.horizon.eq(horizon) & burden.node.eq("Both__45+")].iloc[0]
            np.testing.assert_allclose(saved["point_counts"][horizon-1].sum(), b.value, rtol=1e-12)
            ordered = r.set_index(["sex", "age"]).reindex(pd.MultiIndex.from_product([f["sexes"], f["ages"]]))
            np.testing.assert_allclose(saved["point_rates"][horizon-1], ordered.prediction, rtol=1e-12)
            for sex, group in r.groupby("sex"):
                issued.append(dict(outcome=outcome, origin=origin, horizon=horizon, sex=sex,
                                   rate_mean_signed_log_error=group.signed_log_error.mean(),
                                   rate_mean_absolute_log_error=group.absolute_log_error.mean(),
                                   count_both45plus_prediction=b.value, count_both45plus_observed=b.observed,
                                   count_both45plus_percent_error=pct(b.value, b.observed)))
            if horizon == 5:
                initial_gap = np.log(fitted[1][-1].sum()/actual[1][-1].sum())
                modeled_growth = np.log(b.value/fitted[1][-1].sum())
                observed_growth = np.log(b.observed/actual[1][-1].sum())
                error = np.log(b.value/b.observed)
                np.testing.assert_allclose(error, initial_gap+modeled_growth-observed_growth, atol=1e-14)
                decomposition.append(dict(outcome=outcome, origin=origin,
                    initial_count_gap_log=initial_gap, forecast_five_year_growth_log=modeled_growth,
                    observed_five_year_growth_log=observed_growth, endpoint_count_log_error=error,
                    initial_gap_fraction_of_endpoint_log_error=initial_gap/error))

tables = dict(weights=pd.DataFrame(weights), tracking=pd.DataFrame(tracking),
              last_year_cells=pd.DataFrame(cells), slopes=pd.DataFrame(slopes),
              issued=pd.DataFrame(issued), decomposition=pd.DataFrame(decomposition),
              intervals=pd.DataFrame(interval_checks))
outputs = []
for name, frame in tables.items():
    path = OUT / ("dynamic_fit_diagnostic_"+name+".csv")
    frame.to_csv(path, index=False)
    outputs.append(path)
report = OUT / "dynamic_fit_diagnostic.md"
if report.is_file():
    outputs.append(report)
assert all(sha(ROOT/path) == digest for path, digest in inputs.items()), "Source mutated during diagnosis"
validation = dict(status="pass", no_model_fits=True, no_settings_changed=True,
                  source_sha256=inputs,
                  output_sha256={str(path.relative_to(ROOT)):sha(path) for path in outputs},
                  script_sha256=sha(Path(__file__)), rows={k:len(v) for k,v in tables.items()},
                  checks=["Transformed observed histories reconstruct the immutable source rate/count/population panel",
                          "Saved zero-innovation point arrays agree with issued rate and count ledgers",
                          "Endpoint log-count error equals starting gap plus forecast-minus-observed growth",
                          "All read input hashes unchanged after diagnostic"])
(OUT / "dynamic_fit_diagnostic_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
print(json.dumps({"status":"pass", "rows":validation["rows"]}))
print(tables["decomposition"].to_string(index=False))
