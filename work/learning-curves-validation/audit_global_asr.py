"""Independent no-training replay of the separately implemented global ASR run.

This audit imports only the original frozen model inference functions. It does
not import the global-ASR adapter, runner, or its source/scoring helpers.
"""

import os
for field in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[field] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from gbd_park.tcn import load_checkpoint, predict_changes as neural_predict, state_fingerprint
from gbd_park.pooled import predict_changes as nonneural_predict

AGE = "Full-age ASR"
KEYS = ["target", "outcome", "donor_scope", "family", "sex", "horizon"]
LOCAL = ["persistence", "log_trend", "damped_ets", "arima"]
LEARNED = ["pooled_ridge", "pooled_boosting", "donor_ridge_unadapted", "donor_ridge_adapted",
           "donor_boosting_unadapted", "donor_boosting_adapted", "tcn_unadapted", "tcn_adapted", "tcn_intercept"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root, hashes):
    for name, expected in hashes.items():
        if sha(root / name) != expected:
            raise ValueError(f"Changed artifact {root / name}")


def source(wide, outcome):
    frames = []
    for sex in ["Male", "Female"]:
        column = f"{outcome}_rate_age_std_{sex.lower()}"
        frame = wide[["location_name", "year", column]].rename(columns={column: "rate"}).copy()
        frame["sex"] = sex
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def examples(table, countries):
    xs, ys, metadata = [], [], []
    for country in countries:
        for sex in ["Male", "Female"]:
            history = table.loc[table.location_name.eq(country) & table.sex.eq(sex)
                                & table.year.between(1990, 2018)].sort_values("year")
            if history.year.tolist() != list(range(1990, 2019)) or history.rate.le(0).any():
                raise ValueError("Incomplete independently reconstructed source series")
            logs = np.log(history.rate.to_numpy())
            for origin_index in range(7, 24):
                lags = logs[origin_index-7:origin_index+1]
                xs.append(np.r_[lags-lags[-1], lags[-1], float(sex == "Male"), 1.])
                ys.append(logs[origin_index+1:origin_index+6]-logs[origin_index])
                origin = 1990+origin_index
                metadata.append(dict(country=country, sex=sex, age=AGE, window_origin=origin,
                                     input_start=origin-7, input_end=origin, label_start=origin+1, label_end=origin+5))
    meta = pd.DataFrame(metadata)
    # Independent hierarchy construction, explicit rather than imported weights.
    raw = []
    ncountry = meta.country.nunique()
    nsex = meta.groupby("country").sex.nunique().to_dict()
    nage = meta.groupby(["country", "sex"]).age.nunique().to_dict()
    nwindow = meta.groupby(["country", "sex", "age"]).size().to_dict()
    for row in meta.itertuples():
        raw.append(1/(ncountry*nsex[row.country]*nage[(row.country, row.sex)]*
                      nwindow[(row.country, row.sex, row.age)]))
    weights = np.asarray(raw)*len(meta)
    return np.asarray(xs), np.asarray(ys), meta, weights


def current(table, target):
    xs, levels = [], []
    for sex in ["Male", "Female"]:
        values = table.loc[table.location_name.eq(target) & table.sex.eq(sex)
                           & table.year.between(2011, 2018)].sort_values("year")
        if values.year.tolist() != list(range(2011, 2019)):
            raise ValueError("Incomplete origin input sequence")
        logs = np.log(values.rate.to_numpy())
        xs.append(np.r_[logs-logs[-1], logs[-1], float(sex == "Male"), 1.])
        levels.append(logs[-1])
    return np.asarray(xs), np.asarray(levels)


def location_optimum(residuals, penalty):
    """Independent convex solution by evaluating kinks and interval stationary roots."""
    values = np.sort(np.ravel(residuals))
    n = len(values)
    roots = (n-2*np.arange(n+1))/(2*penalty*n)
    left, right = np.r_[-np.inf, values], np.r_[values, np.inf]
    candidates = np.r_[values, roots[(roots >= left) & (roots <= right)]]
    objectives = np.mean(np.abs(values[:, None]-candidates[None, :]), axis=0)+penalty*candidates**2
    return float(candidates[np.argmin(objectives)])


def correction(observed, predicted, intercept_only=False):
    residual = (observed-predicted).ravel()
    h = np.tile(np.arange(1, 6)/5, len(observed))
    penalty = 1.
    if intercept_only:
        return location_optimum(residual, penalty), 0.
    bound = float(np.sqrt(np.mean(np.abs(residual))/penalty))
    if bound == 0:
        return 0., 0.

    def objective(b1):
        b0 = location_optimum(residual-h*b1, penalty)
        return np.mean(abs(residual-b0-h*b1))+penalty*(b0*b0+b1*b1)

    fitted = minimize_scalar(objective, bounds=(-bound, bound), method="bounded",
                             options={"xatol": 1e-12, "maxiter": 500})
    if not fitted.success:
        raise ValueError("Independent adaptation optimization failed")
    b1 = min([float(fitted.x), 0., -bound, bound], key=objective)
    return location_optimum(residual-h*b1, penalty), b1


def check_scaler(fitted, x, weights):
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x-mean)**2, axis=0, weights=weights)
    np.testing.assert_allclose(fitted["scaler"].mean_, mean, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(fitted["scaler"].var_, variance, atol=1e-11, rtol=1e-11)


def compare_forecast(frame, family, sex, expected):
    subset = frame.loc[frame.family.eq(family) & frame.sex.eq(sex)].sort_values("horizon")
    if subset.horizon.tolist() != [1, 2, 3, 4, 5]:
        raise ValueError("Missing forecast horizon")
    np.testing.assert_allclose(subset.log_prediction, expected, atol=3e-7, rtol=3e-7)
    np.testing.assert_allclose(subset.prediction, np.exp(expected), atol=3e-7, rtol=3e-7)


def run(out):
    manifest_path = out / "run_manifest.json"
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError("Global ASR production must complete before independent audit")
    identity = manifest["identity"]
    verify(ROOT, identity["source_sha256"])
    verify(ROOT, identity["code_sha256"])
    verify(out, manifest["output_sha256"])
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    wide = pd.read_csv(ROOT / "age_standard/parkinsons_ml_ready_wide.csv")
    registry = pd.read_csv(out / "location_registry.csv")
    if len(registry) != 203 or registry.iso3.nunique() != 203 or not registry.un_location_type.eq("Country/Area").all():
        raise ValueError("Invalid global geographic registry")
    if set(zip(registry.location_id, registry.location_name)) != set(wide[["location_id", "location_name"]].itertuples(index=False, name=None)):
        raise ValueError("Supplied country identities differ from registry")
    saved_source = pd.read_csv(out / "source_panel.csv.gz")
    for outcome in ["prevalence", "incidence"]:
        original = source(wide, outcome)
        saved = saved_source.loc[saved_source.outcome.eq(outcome)]
        joined = original.merge(saved, on=["location_name", "sex", "year"], validate="one_to_one", suffixes=("_raw", "_saved"))
        if len(joined) != 203*2*34:
            raise ValueError("Saved source panel incomplete")
        np.testing.assert_allclose(joined.rate_raw, joined.rate_saved, atol=1e-12, rtol=1e-12)
    points = pd.read_csv(out / "predictions.csv")
    if "observed_rate" in points or len(points) != 2640 or points.duplicated(KEYS).any():
        raise ValueError("Issued forecast ledger invalid")
    expected = {(c["name"], outcome, scope, family, sex, h)
                for c in config["countries"] if c["gcc"] for outcome in ["prevalence", "incidence"]
                for scope, family in [("local", f) for f in LOCAL]+[(s, f) for s in ["global", "regional"] for f in LEARNED]
                for sex in ["Male", "Female"] for h in range(1, 6)}
    if set(points[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("All twelve complete country/outcome cases are required")
    issue = json.loads((out / "issued_commit.json").read_text())
    score_commit = json.loads((out / "scoring_complete.json").read_text())
    if issue["final_period_scored"] or issue["issued_utc"] > score_commit["scored_utc"]:
        raise ValueError("Scoring preceded issuance")
    if score_commit["issued_commit_sha256"] != sha(out / "issued_commit.json"):
        raise ValueError("Scoring references a different issued ledger")
    verify(out, issue["output_sha256"])
    verify(out, score_commit["output_sha256"])
    indexed = {}
    for job in identity["jobs"]:
        code = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
        folder = out / "jobs" / job["id"] / code
        complete = json.loads((folder / "complete.json").read_text())
        if complete["job"] != job:
            raise ValueError("Mismatched saved fit job")
        verify(folder, complete["output_sha256"])
        indexed[(job["target"], job["outcome"], job["kind"], job["scope"], job.get("seed"))] = (job, folder, joblib.load(folder / "result.joblib"))
    checkpoint_count, correction_count = 0, 0
    cases = []
    for country in [c["name"] for c in config["countries"] if c["gcc"]]:
        for outcome in ["prevalence", "incidence"]:
            table = source(wide, outcome)
            cx, levels = current(table, country)
            tx, ty, tm, _ = examples(table, [country])
            local = indexed[(country, outcome, "local", "regional", None)][2]
            p = points.loc[points.target.eq(country) & points.outcome.eq(outcome)]
            for row in local["predictions"].itertuples():
                saved = p.loc[p.donor_scope.eq("local") & p.family.eq(row.family) & p.sex.eq(row.sex) & p.horizon.eq(row.horizon)]
                np.testing.assert_allclose(saved.prediction, row.prediction, rtol=1e-12, atol=1e-12)
            for scope in ["global", "regional"]:
                names = registry.location_name.tolist() if scope == "global" else [c["name"] for c in config["countries"]]
                donors = [name for name in names if name != country]
                x, y, meta, weights = examples(table, donors)
                subset = p.loc[p.donor_scope.eq(scope)]
                payloads = []
                for seed in [11, 23, 37, 53, 71]:
                    job, folder, payload = indexed[(country, outcome, "tcn", scope, seed)]
                    payloads.append(payload)
                    audit = payload["audit"]
                    if (audit["countries"] != donors or audit["target_in_base_fit"] or audit["parameter_count"] != 1732
                            or audit["maximum_label_year"] != 2018 or audit["last_target_label_year"] != 2018):
                        raise ValueError("Invalid source exclusion, count or chronology")
                    pd.testing.assert_frame_equal(payload["training_meta"].drop(columns=["sample_weight", "fit_origin"]), meta)
                    np.testing.assert_allclose(payload["training_meta"].sample_weight, weights, atol=1e-12, rtol=1e-12)
                    pd.testing.assert_frame_equal(payload["target_meta"], tm)
                    np.testing.assert_allclose(payload["target_y"], ty, atol=1e-13, rtol=1e-13)
                    np.testing.assert_allclose(payload["levels"], levels, atol=1e-13, rtol=1e-13)
                    if audit["status"] != "ok":
                        if audit["status"] != "fallback" or not audit["reason"]:
                            raise ValueError("Unexplained neural fit status")
                        continue
                    fitted = load_checkpoint(folder / "checkpoint.joblib")
                    if state_fingerprint(fitted) != audit["fingerprint_before"] or audit["fingerprint_before"] != audit["fingerprint_after"]:
                        raise ValueError("Neural state changed")
                    check_scaler(fitted, x, weights)
                    np.testing.assert_allclose(neural_predict(fitted, cx), payload["current_changes"], atol=2e-7, rtol=2e-6)
                    np.testing.assert_allclose(neural_predict(fitted, tx), payload["target_changes"], atol=2e-7, rtol=2e-6)
                    checkpoint_count += 1
                failed = any(item["audit"]["status"] != "ok" for item in payloads)
                current_mean = np.mean([item["current_changes"] for item in payloads], axis=0)
                historical_mean = np.mean([item["target_changes"] for item in payloads], axis=0)
                for si, sex in enumerate(["Male", "Female"]):
                    for family in ["tcn_unadapted", "tcn_adapted", "tcn_intercept"]:
                        predicted = np.repeat(levels[si], 5) if failed else levels[si]+current_mean[si]
                        if not failed and family != "tcn_unadapted":
                            take = tm.sex.eq(sex).to_numpy()
                            b0, b1 = correction(ty[take], historical_mean[take], family == "tcn_intercept")
                            predicted += b0+b1*np.arange(1, 6)/5
                            correction_count += 1
                        compare_forecast(subset, family, sex, predicted)
                job, folder, nonneural = indexed[(country, outcome, "nonneural", scope, None)]
                for fit in nonneural["fits"]:
                    mode, algorithm = fit["pool"], fit["base"]["algorithm"]
                    expected_names = names if mode == "pooled" else donors
                    if fit["countries"] != expected_names or fit["maximum_label_year"] != 2018:
                        raise ValueError("Invalid non-neural country or chronology")
                    family = f"{mode}_{algorithm}"+("_unadapted" if mode == "donor" else "")
                    if fit["status"] == "ok":
                        fitted = joblib.load(folder / f"{fit['model_id']}.joblib")
                        nx, ny, nm, nw = examples(table, expected_names)
                        check_scaler(fitted, nx, nw)
                        predicted = nonneural_predict(fitted, cx)
                        history_predicted = nonneural_predict(fitted, tx) if mode == "donor" else None
                        checkpoint_count += 1
                    else:
                        if fit["status"] != "fallback" or not fit["reason"]:
                            raise ValueError("Unexplained non-neural fit status")
                        predicted, history_predicted = np.zeros((2, 5)), None
                    for si, sex in enumerate(["Male", "Female"]):
                        compare_forecast(subset, family, sex, levels[si]+predicted[si])
                        if mode == "donor":
                            corrected = levels[si]+predicted[si]
                            saved_adapted = subset.loc[subset.family.eq(f"donor_{algorithm}_adapted") & subset.sex.eq(sex)]
                            if fit["status"] == "ok" and saved_adapted.status.eq("ok").all():
                                take = tm.sex.eq(sex).to_numpy()
                                b0, b1 = correction(ty[take], history_predicted[take])
                                corrected += b0+b1*np.arange(1, 6)/5
                                correction_count += 1
                            elif saved_adapted.status.eq("fallback").all():
                                corrected = np.repeat(levels[si], 5)
                            compare_forecast(subset, f"donor_{algorithm}_adapted", sex, corrected)
            cases.append({"target": country, "outcome": outcome, "forecast_rows": len(p)})
    truth_rows = []
    for outcome in ["prevalence", "incidence"]:
        table = source(wide, outcome)
        table["outcome"] = outcome
        truth_rows.append(table.rename(columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"}))
    expected_scores = points.merge(pd.concat(truth_rows), on=["target", "outcome", "sex", "forecast_year"], validate="many_to_one")
    expected_scores["absolute_log_error"] = abs(np.log(expected_scores.prediction/expected_scores.observed_rate))
    expected_scores["absolute_rate_error"] = abs(expected_scores.prediction-expected_scores.observed_rate)
    scored = pd.read_csv(out / "point_scores.csv")
    for field in ["observed_rate", "absolute_log_error", "absolute_rate_error"]:
        np.testing.assert_allclose(scored[field], expected_scores[field], atol=1e-11, rtol=1e-11)
    summary = pd.read_csv(out / "summary.csv")
    merged = summary.merge(expected_scores[KEYS+["absolute_log_error", "absolute_rate_error"]], on=KEYS, validate="one_to_one", suffixes=("_saved", "_replay"))
    for field in ["absolute_log_error", "absolute_rate_error"]:
        np.testing.assert_allclose(merged[field+"_saved"], merged[field+"_replay"], atol=1e-11, rtol=1e-11)
    means = expected_scores.groupby(KEYS[:-1], as_index=False).agg(mean_absolute_log_error=("absolute_log_error", "mean"), mean_absolute_rate_error=("absolute_rate_error", "mean"))
    pd.testing.assert_frame_equal(means, pd.read_csv(out / "five_horizon_means.csv"), check_dtype=False, atol=1e-11, rtol=1e-11)
    contrasts = pd.read_csv(out / "donor_scope_comparisons.csv")
    keys = ["target", "outcome", "family", "sex", "horizon"]
    paired = expected_scores.loc[expected_scores.donor_scope.eq("global"), keys+["absolute_log_error"]].merge(
        expected_scores.loc[expected_scores.donor_scope.eq("regional"), keys+["absolute_log_error"]], on=keys, validate="one_to_one", suffixes=("_global", "_regional"))
    joined = contrasts.merge(paired, on=keys, validate="one_to_one", suffixes=("_saved", "_replay"))
    difference = joined.absolute_log_error_global_replay-joined.absolute_log_error_regional_replay
    np.testing.assert_allclose(joined.absolute_log_error_change_global_minus_regional, difference, atol=1e-11, rtol=1e-11)
    improvement = np.where(joined.absolute_log_error_regional_replay.gt(0), -100*difference/joined.absolute_log_error_regional_replay, np.nan)
    np.testing.assert_allclose(joined.relative_improvement_percent, improvement, atol=1e-7, rtol=1e-9, equal_nan=True)
    adapted = pd.read_csv(out / "adaptation_comparisons.csv")
    np.testing.assert_allclose(adapted.absolute_log_error_change_adapted_minus_unadapted,
                              adapted.absolute_log_error_adapted-adapted.absolute_log_error_unadapted, atol=1e-12, rtol=1e-12)
    verify(out, manifest["output_sha256"])
    if sha(manifest_path) != manifest_hash:
        raise ValueError("Global run changed during audit")
    return {"passed": True, "audit_utc": datetime.now(timezone.utc).isoformat(), "run_manifest_sha256": manifest_hash,
            "audit_code_sha256": sha(Path(__file__)), "forecast_rows": len(points), "cases": cases,
            "source_checkpoint_replays": checkpoint_count, "independently_optimized_sex_corrections": correction_count,
            "manual_features_and_windows": True, "manual_hierarchy_weights": True,
            "scaler_mean_and_variance_checked": True, "score_and_contrast_arithmetic_checked": True,
            "local_source_models_refitted": False, "imports_new_global_implementation": False,
            "source_authenticity_independently_established": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/global_asr_v1")
    parser.add_argument("--output", default="work/learning-curves-validation/global_asr_independent_audit.json")
    args = parser.parse_args()
    result = run(ROOT / args.run)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ["passed", "forecast_rows", "source_checkpoint_replays", "independently_optimized_sex_corrections"]}))
