"""Independent audit of the bounded v1.3 reliability comparison.

Scientific results are read-only. This reporter independently reconstructs
stored transformations and scores; it never fits or selects a forecasting model.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
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
from scipy.linalg import block_diag

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_local_baselines import check_lock, sha
from report_demography import node_definitions, STAT
from report_calibration import rate_summary, burden_endpoints

RUN = ROOT / "results/reliability_v1_3"
OUT = ROOT / "reports/reliability_v1_3"
RATE_KEYS = ["target", "outcome", "origin", "family", "sex", "age", "horizon", "forecast_year", "scale"]
POINT_KEYS = [key for key in RATE_KEYS if key != "scale"]
METRICS = ["coverage", "width", "lower_miss", "upper_miss", "interval_score", "wis_50_80", "wis_50_80_95"]


def read(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def read_json(path):
    return json.loads(Path(path).read_text())


def compare(actual, expected, keys, values, exact=False):
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any()
    a, b = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    a_values, b_values = a[values].to_numpy(dtype=float), b[values].to_numpy(dtype=float)
    if exact:
        np.testing.assert_array_equal(a_values, b_values)
    else:
        np.testing.assert_allclose(a_values, b_values, atol=1e-9, rtol=2e-12)
    return float(np.max(abs(a_values-b_values)))


def verify_manifest(directory):
    manifest = read_json(directory / "run_manifest.json")
    assert manifest["status"] == "complete"
    for filename, digest in manifest["output_sha256"].items():
        assert sha(directory / filename) == digest, (directory, filename)
    for filename, digest in manifest["code_sha256"].items():
        assert sha(ROOT / filename) == digest, filename
    return manifest


def derived_arrays(rates, counts, config, ratios=False):
    """Independent coherent transformations in canonical sex/age order."""
    if ratios:
        male = config["sexes"].index("Male")*len(config["ages"])
        female = config["sexes"].index("Female")*len(config["ages"])
        values = rates[..., male:male+11]/rates[..., female:female+11]
        return values, ["Male_Female__"+age for age in config["ages"]]
    bottom, definitions, matrix = node_definitions(config)
    assert counts.shape[-1] == 22
    values = [counts @ matrix.T]
    names = [row[0] for row in definitions]
    for sex in config["sexes"]:
        same_sex = np.asarray([s == sex for s, age in bottom])
        denominator = counts[..., same_sex].sum(axis=-1)
        for threshold in [65, 80]:
            older = np.asarray([s == sex and int(age.split("-")[0].rstrip("+")) >= threshold for s, age in bottom])
            share = 100*counts[..., older].sum(axis=-1)/denominator
            assert np.isfinite(share).all() and ((share >= 0) & (share <= 100)).all()
            values.append(share[..., None])
            names.append(f"{sex}__{threshold}+_within_45+")
    return np.concatenate(values, axis=-1), names


def audit_array_ledgers(arrays, config, rate_points, rate_intervals, burden_points, burden_intervals):
    """Recompute all points and scale-specific empirical quantiles from one npz."""
    coords = pd.MultiIndex.from_product([range(1, 6), config["sexes"], config["ages"]], names=["horizon", "sex", "age"])
    points, draws = arrays["point_rates"], arrays["rate_draws"]
    assert points.shape == (5, 22) and draws.shape[1:] == (5, 22)
    assert np.isfinite(points).all() and np.isfinite(draws).all() and (points > 0).all() and (draws > 0).all()
    selected = rate_points.set_index(["horizon", "sex", "age"]).reindex(coords)
    assert len(rate_points) == 110 and not rate_points.duplicated(["horizon", "sex", "age"]).any()
    np.testing.assert_array_equal(selected.prediction, points.ravel())
    np.testing.assert_allclose(selected.log_prediction, np.log(points).ravel(), atol=1e-12, rtol=1e-12)
    assert len(rate_intervals) == 660
    for scale in ["rate", "log_rate"]:
        values = np.log(draws) if scale == "log_rate" else draws
        point_values = np.log(points) if scale == "log_rate" else points
        for level in [.5, .8, .95]:
            part = rate_intervals[rate_intervals.scale.eq(scale) & rate_intervals.level.eq(level)].set_index(["horizon", "sex", "age"]).reindex(coords)
            expected = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0, method="linear").reshape(3, -1).T
            np.testing.assert_allclose(part[["lower", "median", "upper"]], expected, atol=1e-10, rtol=2e-12)
            np.testing.assert_allclose(part.point_prediction, point_values.ravel(), atol=1e-12, rtol=1e-12)
            assert part.n_draws.eq(len(draws)).all()
    methods = [key.removeprefix("count_draws__") for key in arrays if key.startswith("count_draws__")]
    for method in methods+["not_applicable"]:
        ratio = method == "not_applicable"
        values, nodes = derived_arrays(draws, None if ratio else arrays["count_draws__"+method], config, ratio)
        point_values, point_nodes = derived_arrays(points, None if ratio else arrays["point_counts__"+method], config, ratio)
        assert nodes == point_nodes
        keys = pd.MultiIndex.from_product([range(1, 6), nodes], names=["horizon", "node"])
        part = burden_points[burden_points.population_method.eq(method)].set_index(["horizon", "node"]).reindex(keys)
        np.testing.assert_allclose(part.value, point_values.ravel(), atol=1e-9, rtol=2e-12)
        if not ratio:
            for prefix in ["point_", ""]:
                # Point/draw population arrays are checked when supplied by the runner.
                population_key = "point_populations__"+method if prefix else "population_draws__"+method
                if population_key in arrays:
                    count_values = arrays["point_counts__"+method] if prefix else arrays["count_draws__"+method]
                    rates_values = points if prefix else draws
                    np.testing.assert_allclose(count_values, rates_values*arrays[population_key]/100000, atol=1e-9, rtol=2e-12)
        for level in [.5, .8, .95]:
            part = burden_intervals[burden_intervals.population_method.eq(method) & burden_intervals.level.eq(level)].set_index(["horizon", "node"]).reindex(keys)
            expected = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0, method="linear").reshape(3, -1).T
            np.testing.assert_allclose(part[["lower", "median", "upper"]], expected, atol=1e-9, rtol=2e-12)
            assert part.n_draws.eq(len(values)).all()
    return dict(rate_point_cells=110, rate_interval_rows=660, n_draws=len(draws), burden_interval_rows=len(burden_intervals))


def audit_scores(cells, wis, keys, observed_name):
    """Independent interval-score and proper-score arithmetic on each scale."""
    observed = cells[observed_name].to_numpy()
    lower, upper, alpha = cells.lower.to_numpy(), cells.upper.to_numpy(), 1-cells.level.to_numpy()
    width = upper-lower
    scores = width+2/alpha*(np.maximum(lower-observed, 0)+np.maximum(observed-upper, 0))
    np.testing.assert_allclose(cells.width, width, atol=1e-9, rtol=2e-12)
    np.testing.assert_allclose(cells.interval_score, scores, atol=1e-8, rtol=2e-12)
    np.testing.assert_array_equal(cells.covered, (observed >= lower) & (observed <= upper))
    np.testing.assert_array_equal(cells.lower_miss, observed < lower)
    np.testing.assert_array_equal(cells.upper_miss, observed > upper)
    weighted = cells[keys+["level"]].copy()
    weighted["term"] = alpha*scores/2
    matrix = weighted.pivot(index=keys, columns="level", values="term")
    base = wis.set_index(keys).reindex(matrix.index)
    true = cells.drop_duplicates(keys).set_index(keys).reindex(matrix.index)
    np.testing.assert_array_equal(base[observed_name], true[observed_name])
    for column, levels in [("wis_50_80", [.5, .8]), ("wis_50_80_95", [.5, .8, .95])]:
        expected = (.5*abs(base[observed_name]-base["median"])+matrix[levels].sum(axis=1))/(len(levels)+.5)
        np.testing.assert_allclose(base[column], expected, atol=1e-8, rtol=2e-12)


def audit_structured(case, item, config, arrays, points, parameters, panel, history, source_points):
    """Replay the fixed marginal fits from source forecasts and cutoff truth."""
    origin = int(item["origin"])
    bank = joblib.load(ROOT / item["original_bank"])
    raw = np.asarray(bank["raw_residuals"])
    original = raw-raw.mean(axis=0)
    np.testing.assert_allclose(original, bank["centered_residuals"], atol=1e-12, rtol=1e-12)
    assert item["residual_origins"] == bank["origins"] == list(range(2003, origin-4))
    assert item["n_draws"] == item["n_blocks"] == len(bank["origins"])
    coords = pd.MultiIndex.from_product([config["sexes"], config["ages"], range(1, 6)], names=["sex", "age", "horizon"])
    assert bank["coords"].equals(coords.to_frame(index=False))
    source = pd.concat([history.drop(columns=["observed_rate"], errors="ignore"), source_points], ignore_index=True)
    source = source[source.origin.lt(origin) & source.forecast_year.le(origin)]
    truth = panel[panel.location_name.eq(case["target"]) & panel.outcome.eq(case["outcome"]) & panel.year.le(origin)]
    truth = truth[["year", "sex", "age", "rate"]].rename(columns={"year": "forecast_year", "rate": "observed_rate"})
    groups = {name: [age for age in config["ages"] if int(age.split("-")[0].rstrip("+")) in starts]
              for name, starts in config["age_groups"].items()}
    expected_records, transformed = [], original.copy()
    for sex in config["sexes"]:
        family = item["source_family_by_sex"][sex]
        part = source[source.family.eq(family) & source.sex.eq(sex)].merge(truth, on=["forecast_year", "sex", "age"], validate="many_to_one")
        observed_quantiles, reference_quantiles, counts = {}, {}, {}
        for horizon in range(1, 6):
            expected = pd.MultiIndex.from_product([range(2003, origin-horizon+1), config["ages"]], names=["origin", "age"])
            eligible = part[part.horizon.eq(horizon)]
            assert len(eligible) == len(expected) and not eligible.duplicated(["origin", "age"]).any()
            eligible = eligible.set_index(["origin", "age"]).reindex(expected).reset_index()
            assert eligible.forecast_year.eq(eligible.origin+horizon).all() and eligible.forecast_year.le(origin).all()
            eligible["residual"] = np.log(eligible.observed_rate)-eligible.log_prediction
            assert np.isfinite(eligible.residual).all()
            counts[horizon] = len(range(2003, origin-horizon+1))
            for group, ages in {"sex_pool": config["ages"], **groups}.items():
                observed_quantiles[group, horizon] = np.quantile(eligible[eligible.age.isin(ages)].residual, [.2, .5, .8], method="linear")
                columns = bank["coords"].sex.eq(sex) & bank["coords"].horizon.eq(horizon) & bank["coords"].age.isin(ages)
                reference_quantiles[group, horizon] = np.quantile(original[:, columns], [.2, .5, .8], method="linear")
        for horizon in range(1, 6):
            adjacent = [j for j in range(1, 6) if abs(j-horizon) <= 1]
            weights = np.asarray([2. if j == horizon else 1. for j in adjacent])
            weights /= weights.sum()
            n = float(np.dot(weights, [counts[j] for j in adjacent]))
            a, b = n/(n+20.), n/(n+8.)
            smoothed = {}
            ratios, zero = {}, {}
            for group in ["sex_pool", *groups]:
                obs = np.sum([weight*observed_quantiles[group, j] for weight, j in zip(weights, adjacent)], axis=0)
                ref = np.sum([weight*reference_quantiles[group, j] for weight, j in zip(weights, adjacent)], axis=0)
                obs_spread, ref_spread = np.diff(obs), np.diff(ref)
                mask = ref_spread <= 1e-6
                ratio = np.ones(2)
                ratio[~mask] = np.clip(obs_spread[~mask]/ref_spread[~mask], .5, 3.)
                ratios[group], zero[group], smoothed[group] = ratio, mask, (obs, ref)
            for group, ages in groups.items():
                location = a*a*smoothed[group][0][1]+a*(1-a)*smoothed["sex_pool"][0][1]
                scales = np.exp(b*(a*np.log(ratios[group])+(1-a)*np.log(ratios["sex_pool"])))
                columns = bank["coords"].sex.eq(sex) & bank["coords"].horizon.eq(horizon) & bank["coords"].age.isin(ages)
                z = original[:, columns]
                transformed[:, columns] = location+np.where(z < 0, scales[0]*z, scales[1]*z)
                record = dict(sex=sex, age_group=group, horizon=horizon, weighted_distinct_origin_count=n,
                              distinct_origins_at_horizon=counts[horizon], original_joint_blocks=len(bank["origins"]),
                              last_label_year=origin, location=location, left_scale=scales[0], right_scale=scales[1],
                              group_location_shrinkage=a, scale_shrinkage=b, location_group_weight=a*a,
                              location_pool_weight=a*(1-a), location_zero_weight=1-a,
                              original_mean_offset=float(z.mean()), transformed_mean_offset=float(transformed[:, columns].mean()))
                for label, group_name in [("group", group), ("pool", "sex_pool")]:
                    for q, value in zip([20, 50, 80], smoothed[group_name][0]):
                        record[f"{label}_observed_q{q}"] = value
                    for q, value in zip([20, 50, 80], smoothed[group_name][1]):
                        record[f"{label}_reference_q{q}"] = value
                    for side, index in [("left", 0), ("right", 1)]:
                        record[f"{label}_{side}_ratio"] = ratios[group_name][index]
                        record[f"{label}_{side}_zero_reference"] = zero[group_name][index]
                expected_records.append(record)
    expected = pd.DataFrame(expected_records)
    keys = ["sex", "age_group", "horizon"]
    actual = parameters[parameters.origin.eq(origin) & parameters.role.eq(item["family"].split("__")[0])]
    assert len(actual) == 30 and not actual.count_is_effective_sample_size.any()
    compare(actual, expected, keys, [key for key in expected if key not in keys])
    current = points[points.origin.eq(origin) & points.family.eq(item["family"])].set_index(["sex", "age", "horizon"]).reindex(coords)
    logs = current.log_prediction.to_numpy()[None, :]+transformed
    expected_draws = np.exp(logs).reshape(len(bank["origins"]), 2, 11, 5).transpose(0, 3, 1, 2).reshape(len(bank["origins"]), 5, 22)
    np.testing.assert_allclose(arrays["rate_draws"], expected_draws, atol=1e-10, rtol=2e-12)
    # Positive piecewise slopes retain each marginal historical-origin ordering.
    np.testing.assert_array_equal(np.argsort(original, axis=0, kind="stable"), np.argsort(transformed, axis=0, kind="stable"))
    demographic = ROOT / case["demography"]
    population = read(demographic / "population_forecasts.csv")
    errors = read(demographic / "population_residuals.csv")
    coordinate_order = pd.MultiIndex.from_product([range(1, 6), config["sexes"], config["ages"]], names=["horizon", "sex", "age"])
    for method in item["population_methods"]:
        selected = population[population.origin.eq(origin) & population.population_method.eq(method)].set_index(["horizon", "sex", "age"]).reindex(coordinate_order)
        np.testing.assert_array_equal(arrays["point_populations__"+method], selected.population.to_numpy().reshape(5, 22))
        selected_errors = errors[errors.fit_origin.eq(origin) & errors.population_method.eq(method)]
        values = selected_errors.pivot(index="residual_origin", columns=["horizon", "sex", "age"], values="centered_population_log_error").reindex(index=bank["origins"], columns=coordinate_order)
        pop_draws = np.exp(selected.log_population.to_numpy()[None, :]+values.to_numpy()).reshape(len(bank["origins"]), 5, 22)
        np.testing.assert_allclose(arrays["population_draws__"+method], pop_draws, atol=1e-8, rtol=2e-12)
    return dict(marginal_parameters_reconstructed=30, original_joint_blocks=len(bank["origins"]),
                point_predictions_unchanged=True, population_inputs_unchanged=True, marginal_ranks_preserved=True)


def decode_joint(latent, basis):
    """Independent stable inverse of the two identifiable sex-specific states."""
    counts, populations = [], []
    for sex in range(2):
        z = latent[..., sex*22:(sex+1)*22]
        clr = z[..., 1:11] @ basis.T
        unnormalized = np.exp(clr-clr.max(axis=-1, keepdims=True))
        shares = unnormalized/unnormalized.sum(axis=-1, keepdims=True)
        counts.append(np.exp(z[..., :1])*shares)
        populations.append(np.exp(z[..., 11:]))
    count, population = np.concatenate(counts, axis=-1), np.concatenate(populations, axis=-1)
    assert np.isfinite(count).all() and np.isfinite(population).all() and (count > 0).all() and (population > 0).all()
    return count/population*100000, count, population


def audit_dynamic(case, item, model, arrays, config, panel):
    """Independently replay saved state calculations and seeded simulation."""
    fitted = model["fitted_state"]
    origin = int(item["origin"])
    years = np.arange(config["calendar"]["history_start"], origin+1)
    np.testing.assert_array_equal(fitted["years"], years)
    np.testing.assert_array_equal(fitted["curvature_years"], years[2:])
    np.testing.assert_array_equal(fitted["filter_update_years"], years[1:])
    assert fitted["origin"] == origin and fitted["target"] == case["target"] and fitted["outcome"] == case["outcome"]
    cells = pd.MultiIndex.from_product([years, config["sexes"], config["ages"]], names=["year", "sex", "age"])
    history = panel[panel.location_name.eq(case["target"]) & panel.outcome.eq(case["outcome"]) & panel.year.isin(years)].set_index(["year", "sex", "age"]).reindex(cells)
    counts = history["count"].to_numpy().reshape(len(years), 2, 11)
    population = (history["count"]/history.rate*100000).to_numpy().reshape(len(years), 2, 11)
    basis = np.zeros((11, 10))
    for j in range(10):
        denominator = np.sqrt((j+1)*(j+2))
        basis[:j+1, j] = 1/denominator
        basis[j+1, j] = -(j+1)/denominator
    np.testing.assert_array_equal(fitted["composition_basis"], basis)
    np.testing.assert_allclose(basis.T @ basis, np.eye(10), atol=1e-15)
    np.testing.assert_allclose(basis.sum(axis=0), 0, atol=1e-15)
    expected_z = np.column_stack([np.column_stack([np.log(counts[:, sex].sum(axis=1)), np.log(counts[:, sex]) @ basis,
                                                   np.log(population[:, sex])]) for sex in range(2)])
    z = fitted["transformed_history"]
    np.testing.assert_allclose(z, expected_z, atol=1e-12, rtol=1e-12)
    # Replay downstream arithmetic from the saved, independently checked
    # transform to avoid conflating equivalent summation orders with fit changes.
    curvature = np.diff(z, n=2, axis=0)
    median = np.median(curvature, axis=0)
    scales = np.maximum(1.482602218505602*np.median(abs(curvature-median), axis=0), 1e-4)
    standardized = (curvature-median)/scales
    radial = np.sqrt(np.mean(standardized**2, axis=1))
    weights = np.minimum(1, 2.5/np.maximum(radial, np.finfo(float).tiny))
    clipped = standardized*weights[:, None]
    centered = clipped-clipped.mean(axis=0)
    empirical = centered.T @ centered/(len(centered)-1)
    variance = np.diag(empirical)
    active = variance > 1e-12
    correlation = np.zeros((44, 44))
    index = np.flatnonzero(active)
    if len(index):
        correlation[np.ix_(index, index)] = empirical[np.ix_(index, index)]/np.sqrt(np.outer(variance[index], variance[index]))
    np.fill_diagonal(correlation, 1)
    shrunk = .25*correlation+.75*np.eye(44)
    sigma = scales[:, None]*scales[None, :]*shrunk
    for name, expected in [("curvature", curvature), ("curvature_median", median), ("mad_scales", scales),
                           ("standardized_curvature", standardized), ("radial_rms", radial), ("radial_weights", weights),
                           ("clipped_standardized_curvature", clipped), ("empirical_correlation", correlation),
                           ("shrunk_correlation", shrunk), ("sigma", sigma)]:
        np.testing.assert_allclose(fitted[name], expected, atol=1e-12, rtol=2e-12)
    np.testing.assert_array_equal(fitted["numerically_constant_coordinates"], ~active)
    d = np.diff(np.eye(11), axis=0)
    smoothing = np.linalg.solve(np.eye(11)+d.T @ d, np.eye(11))
    one_sex = block_diag(np.ones((1, 1)), basis.T @ smoothing @ basis, smoothing)
    slope_smoothing = block_diag(one_sex, one_sex)
    transition = np.block([[np.eye(44), np.eye(44)], [np.zeros((44, 44)), .95*slope_smoothing]])
    process, observation = block_diag(.2*sigma, .1*sigma), sigma/12
    for name, expected in [("transition", transition), ("process_covariance", process), ("observation_covariance", observation),
                           ("slope_smoothing", slope_smoothing), ("age_smoothing", smoothing)]:
        np.testing.assert_allclose(fitted[name], expected, atol=1e-12, rtol=2e-12)
    mean, covariance = np.r_[z[0], np.zeros(44)], block_diag(observation, 25*sigma)
    np.testing.assert_allclose(fitted["initial_mean"], mean, atol=1e-12, rtol=2e-12)
    np.testing.assert_allclose(fitted["initial_covariance"], covariance, atol=1e-12, rtol=2e-12)
    means, residuals, distances, filter_weights = [mean.copy()], [], [], []
    for current in z[1:]:
        predicted = transition @ mean
        prior_covariance = transition @ covariance @ transition.T+process
        prior_covariance = (prior_covariance+prior_covariance.T)/2
        residual = current-predicted[:44]
        ordinary = prior_covariance[:44, :44]+observation
        distance = residual @ np.linalg.solve(ordinary, residual)
        weight = min(1., 49./(5.+distance))
        effective = observation/weight
        gain = np.linalg.solve(prior_covariance[:44, :44]+effective, prior_covariance[:44]).T
        mean = predicted+gain @ residual
        remainder = np.eye(88)
        remainder[:, :44] -= gain
        covariance = remainder @ prior_covariance @ remainder.T+gain @ effective @ gain.T
        covariance = (covariance+covariance.T)/2
        means.append(mean.copy())
        residuals.append(residual)
        distances.append(distance)
        filter_weights.append(weight)
    for name, expected in [("final_mean", mean), ("final_covariance", covariance), ("filtered_means", means),
                           ("filter_innovations", residuals), ("filter_weights", filter_weights), ("filter_mahalanobis_squared", distances)]:
        np.testing.assert_allclose(fitted[name], expected, atol=1e-9, rtol=1e-9)
    assert item["n_blocks"] == 0 and item["n_draws"] == 1024
    assert model["diagnostics"]["draws_are_independent_temporal_blocks"] is False
    # Simulate from saved fitted arrays; no model fitting or parameter changes.
    rng = np.random.Generator(np.random.PCG64(item["seed"]))
    mean = fitted["final_mean"].copy()
    def chol(matrix):
        return np.linalg.cholesky((matrix+matrix.T)/2)
    initial = rng.standard_normal((512, 88)) @ chol(fitted["final_covariance"]).T
    states = mean[None, :]+np.vstack([initial, -initial])
    latent_points, latent_draws = [], []
    for _ in range(5):
        innovations = rng.standard_normal((512, 88)) @ chol(fitted["process_covariance"]).T
        innovations *= np.sqrt(3/rng.chisquare(5., size=512))[:, None]
        states = states @ fitted["transition"].T+np.vstack([innovations, -innovations])
        noise = rng.standard_normal((512, 44)) @ chol(fitted["observation_covariance"]).T
        noise *= np.sqrt(3/rng.chisquare(5., size=512))[:, None]
        latent_draws.append(states[:, :44]+np.vstack([noise, -noise]))
        mean = fitted["transition"] @ mean
        latent_points.append(mean[:44].copy())
    latent_points, latent_draws = np.stack(latent_points), np.stack(latent_draws, axis=1)
    np.testing.assert_array_equal(model["point_latent"], latent_points)
    np.testing.assert_array_equal(model["draw_latent"], latent_draws)
    np.testing.assert_allclose((latent_draws[:512]+latent_draws[512:])/2, np.broadcast_to(latent_points, (512, 5, 44)), atol=1e-12, rtol=1e-12)
    rate, count, pop = decode_joint(latent_points, basis)
    rate_draws, count_draws, pop_draws = decode_joint(latent_draws, basis)
    for name, expected in [("point_rates", rate), ("point_counts__dynamic_joint", count), ("point_populations__dynamic_joint", pop),
                           ("rate_draws", rate_draws), ("count_draws__dynamic_joint", count_draws), ("population_draws__dynamic_joint", pop_draws)]:
        np.testing.assert_allclose(arrays[name], expected, atol=1e-8, rtol=2e-12)
    return dict(history_years=len(years), last_input_year=origin, state_dimensions=88, fit_arrays_reconstructed=True,
                monte_carlo_draws=1024, antithetic_pairs_replayed=512, joint_horizon_simulation_replayed=True,
                inverse_composition_reconstructed=True, empirical_residual_bank_added=False)


def verify_commits(manifest, spec):
    lock = ROOT / "study_design/reliability_v1_3.lock.json"
    assert sha(lock) == manifest["amendment_lock_sha256"]
    frozen = read_json(lock)
    for filename, digest in frozen["document_sha256"].items():
        assert sha(ROOT / filename) == digest
    for run in spec["prior_runs"]:
        assert sha(ROOT / "results" / run / "run_manifest.json") == frozen["prior_manifest_sha256"][run]
        verify_manifest(ROOT / "results" / run)
    assert read_json(ROOT / "results/primary_v1/primary_verdict.json")["joint_success"] is False
    events = read_json(RUN / "events.json")
    names = [event["event"] for event in events]
    phases = ["issuance_started", "all_cases_committed_before_scoring", "evaluation_scoring_started", "complete"]
    assert all(names.count(name) == 1 for name in phases)
    positions = [names.index(name) for name in phases]
    assert positions == sorted(positions)
    times = [datetime.fromisoformat(events[index]["time_utc"]) for index in positions]
    assert times == sorted(times)
    global_commit = read_json(RUN / "global_issued_commit.json")
    digest = sha(RUN / "global_issued_commit.json")
    assert digest == events[positions[1]]["sha256"]
    assert len(global_commit["case_commit_sha256"]) == 12
    global_time = datetime.fromisoformat(global_commit["committed_utc"])
    assert global_time <= times[2]
    for filename, expected in global_commit["case_commit_sha256"].items():
        path = RUN / filename
        assert sha(path) == expected
        own = read_json(path)
        assert not own["final_period_scored"] and datetime.fromisoformat(own["committed_utc"]) <= global_time
        for artifact, value in own["artifact_sha256"].items():
            assert sha(path.parent / artifact) == value
        scored = read_json(path.parent / "scoring_event.json")
        assert scored["global_commit_sha256"] == digest
        assert global_time <= datetime.fromisoformat(scored["scoring_started_utc"])


def audit_case(case, config, spec):
    directory, reference = RUN / case["id"], ROOT / "results/calibration_v1_2" / case["id"]
    for filename, digest in read_json(directory / "source_references.json").items():
        assert sha(ROOT / filename) == digest
    rate_points, rate_intervals = read(directory / "rate_predictions.csv"), read(directory / "rate_intervals.csv")
    burden_points, burden_intervals = read(directory / "burden_predictions.csv"), read(directory / "burden_intervals.csv")
    families = [role+"__"+variant for role in spec["roles"] for variant in [*spec["reference_variants"], "structured"]]+["robust_dynamic__joint"]
    assert set(rate_points.family) == set(families) and len(rate_points) == 7150 and len(rate_intervals) == 42900
    assert len(burden_points) == 25450 and len(burden_intervals) == 76350
    for frame in [rate_points, rate_intervals, burden_points, burden_intervals]:
        assert frame.family.eq(frame.role+"__"+frame.variant).all()
        assert set(frame.family) == set(families)
        assert frame.forecast_year.eq(frame.origin+frame.horizon).all()
    original = read(ROOT / case["source"] / "predictions.csv")
    for role in spec["roles"]:
        for variant in [*spec["reference_variants"], "structured"]:
            expected = original[original.family.eq(role)].copy()
            expected["family"] = role+"__"+variant
            selected = rate_points[rate_points.family.eq(role+"__"+variant)]
            compare(selected, expected, POINT_KEYS, ["prediction"], exact=True)
            compare(selected, expected, POINT_KEYS, ["log_prediction"])
            a, b = selected.set_index(POINT_KEYS).sort_index(), expected.set_index(POINT_KEYS).sort_index()
            np.testing.assert_array_equal(a.status, b.status)
    for filename, keys, fields in [("rate_intervals.csv", RATE_KEYS+["level"], ["lower", "median", "upper", "point_prediction"]),
                                    ("burden_intervals.csv", STAT+["level"], ["lower", "median", "upper"]),
                                    ("burden_predictions.csv", STAT, ["value"])]:
        previous = read(reference / filename)
        previous = previous[previous.variant.isin(spec["reference_variants"])]
        actual = {"rate_intervals.csv": rate_intervals, "burden_intervals.csv": burden_intervals,
                  "burden_predictions.csv": burden_points}[filename]
        actual = actual[actual.variant.isin(spec["reference_variants"])]
        compare(actual, previous, keys, fields, exact=True)
    mappings = read(ROOT / case["source"] / "champion_family_mappings.csv")
    parameters = read(directory / "structured_parameters.csv")
    assert len(parameters) == 450
    panel = read(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    history = read(ROOT / case["history"] / "prequential_scores.csv")
    index = read_json(directory / "draw_index.json")
    assert len(index) == 20
    assert {(row["origin"], row["family"]) for row in index} == {(origin, family) for origin in spec["evaluation_origins"]
                               for family in [*[role+"__structured" for role in spec["roles"]], "robust_dynamic__joint"]}
    array_checks, dynamics, structured, population_scores, decompositions = [], [], [], [], []
    for item in index:
        arrays = dict(np.load(directory / item["path"]))
        take = dict(origin=item["origin"], family=item["family"])
        p = rate_points[rate_points.origin.eq(take["origin"]) & rate_points.family.eq(take["family"])]
        i = rate_intervals[rate_intervals.origin.eq(take["origin"]) & rate_intervals.family.eq(take["family"])]
        bp = burden_points[burden_points.origin.eq(take["origin"]) & burden_points.family.eq(take["family"])]
        bi = burden_intervals[burden_intervals.origin.eq(take["origin"]) & burden_intervals.family.eq(take["family"])]
        assert i.n_blocks.eq(item["n_blocks"]).all() and bi.n_blocks.eq(item["n_blocks"]).all()
        array_checks.append({**take, **audit_array_ledgers(arrays, config, p, i, bp, bi)})
        if item["kind"] == "structured":
            role = item["family"].split("__")[0]
            for sex, family in item["source_family_by_sex"].items():
                expected = "tcn_adapted"
                if role != "tcn_adapted":
                    row = mappings[mappings.role.eq(role) & mappings.sex.eq(sex) & mappings.fit_origin.eq(item["origin"])]
                    assert len(row) == 1 and row.last_selection_target_year.le(item["origin"]).all()
                    expected = row.source_family.iloc[0]
                assert family == expected
            assert set(i.draw_kind) == {"transformed_historical_joint_blocks"}
            structured.append({**take, **audit_structured(case, item, config, arrays, rate_points, parameters, panel, history, original)})
            old_points = read(ROOT / case["demography"] / "predictions.csv")
            old_points = old_points[old_points.origin.eq(item["origin"]) & old_points.family.eq(role)].copy()
            old_points["family"] = item["family"]
            compare(bp, old_points, STAT, ["value"])
        else:
            assert item["kind"] == "dynamic_joint" and set(i.draw_kind) == {"model_monte_carlo"}
            expected_seed = spec["seed"]+1000*case["case_index"]+item["origin"]
            assert item["seed"] == expected_seed
            model = joblib.load(directory / item["fitted_state"])
            dynamics.append({**take, **audit_dynamic(case, item, model, arrays, config, panel)})
        coords = pd.MultiIndex.from_product([range(1, 6), config["sexes"], config["ages"]], names=["horizon", "sex", "age"])
        context = coords.to_frame(index=False)
        for key, value in dict(target=case["target"], outcome=case["outcome"], **take).items():
            context[key] = value
        context["forecast_year"] = context.origin+context.horizon
        truth = panel[panel.location_name.eq(case["target"]) & panel.outcome.eq(case["outcome"])][["year", "sex", "age", "rate", "count"]].rename(columns={"year": "forecast_year"})
        observed = context.merge(truth, on=["forecast_year", "sex", "age"], validate="many_to_one")
        observed_population = (observed["count"]/observed.rate*100000).to_numpy()
        for method in item["population_methods"]:
            population_points = arrays["point_populations__"+method].ravel()
            population_draws = arrays["population_draws__"+method].reshape(item["n_draws"], -1)
            row = context.copy()
            row["population_method"] = method
            row["rate_signed_log_error"] = np.log(arrays["point_rates"].ravel()/observed.rate)
            row["population_signed_log_error"] = np.log(population_points/observed_population)
            row["count_signed_log_error"] = np.log(arrays["point_counts__"+method].ravel()/observed["count"])
            np.testing.assert_allclose(row.count_signed_log_error, row.rate_signed_log_error+row.population_signed_log_error, atol=1e-12)
            decompositions.append(row)
            levels = [.5, .8, .95]
            quantiles = np.quantile(population_draws, [.25, .75, .1, .9, .025, .975, .5], axis=0, method="linear")
            lower, upper = quantiles[[0, 2, 4]], quantiles[[1, 3, 5]]
            alpha = 1-np.asarray(levels)
            scores = upper-lower+2/alpha[:, None]*(np.maximum(lower-observed_population, 0)+np.maximum(observed_population-upper, 0))
            median_error = .5*abs(observed_population-quantiles[6])
            pop = context.copy()
            pop["population_method"] = method
            pop["prediction"], pop["observed"] = population_points, observed_population
            pop["absolute_log_error"] = abs(row.population_signed_log_error)
            pop["coverage_80"] = (lower[1] <= observed_population) & (observed_population <= upper[1])
            pop["width_80"] = upper[1]-lower[1]
            pop["wis_50_80"] = (median_error+(.5*alpha[:2, None]*scores[:2]).sum(axis=0))/2.5
            pop["wis_50_80_95"] = (median_error+(.5*alpha[:, None]*scores).sum(axis=0))/3.5
            population_scores.append(pop)
    populations = pd.concat(population_scores, ignore_index=True)
    emitted_populations = read(directory / "dynamic_population_scores.csv")
    pop_keys = ["target", "outcome", "origin", "horizon", "sex", "age"]
    compare(populations[populations.family.eq("robust_dynamic__joint")], emitted_populations, pop_keys,
            ["prediction", "observed", "absolute_log_error", "coverage_80", "width_80"])
    point_scores = read(directory / "rate_point_scores.csv")
    truth = panel[panel.location_name.eq(case["target"]) & panel.outcome.eq(case["outcome"])][["year", "sex", "age", "rate", "count"]].rename(columns={"year": "forecast_year", "rate": "observed_rate", "count": "observed_count"})
    expected = rate_points.merge(truth, on=["forecast_year", "sex", "age"], validate="many_to_one")
    expected["absolute_log_error"] = abs(expected.log_prediction-np.log(expected.observed_rate))
    expected["signed_log_error"] = expected.log_prediction-np.log(expected.observed_rate)
    compare(point_scores, expected, POINT_KEYS, ["prediction", "log_prediction", "observed_rate", "absolute_log_error", "signed_log_error"])
    cells, wis = read(directory / "rate_interval_scores.csv.gz"), read(directory / "rate_wis_scores.csv")
    audit_scores(cells, wis, RATE_KEYS, "observed_value")
    joined = cells.merge(expected[POINT_KEYS+["observed_rate"]], on=POINT_KEYS, validate="many_to_one")
    np.testing.assert_allclose(joined.observed_value, np.where(joined.scale.eq("rate"), joined.observed_rate, np.log(joined.observed_rate)), atol=1e-12)
    rates = rate_summary(cells, wis, config)
    summary_keys = ["target", "outcome", "origin", "role", "variant", "sex", "horizon", "scale", "level", "age_scope"]
    emitted = read(directory / "rate_summary.csv").rename(columns={"mean_width": "width", "lower_miss_rate": "lower_miss", "upper_miss_rate": "upper_miss", "mean_interval_score": "interval_score"})
    compare(rates, emitted, summary_keys, ["coverage", "width", "lower_miss", "upper_miss", "interval_score"])
    b_cells, b_wis = read(directory / "burden_interval_scores.csv.gz"), read(directory / "burden_wis_scores.csv")
    audit_scores(b_cells, b_wis, STAT, "observed")
    observed_frames = []
    for (origin, family), block in expected.groupby(["origin", "family"]):
        coordinates = pd.MultiIndex.from_product([range(1, 6), config["sexes"], config["ages"]], names=["horizon", "sex", "age"])
        ordered = block.set_index(["horizon", "sex", "age"]).reindex(coordinates)
        actual_rates, actual_counts = ordered.observed_rate.to_numpy().reshape(5, 22), ordered.observed_count.to_numpy().reshape(5, 22)
        methods = ["dynamic_joint"] if family == "robust_dynamic__joint" else spec["population_methods"]
        for method in methods+["not_applicable"]:
            values, nodes = derived_arrays(actual_rates, actual_counts, config, method == "not_applicable")
            frame = pd.MultiIndex.from_product([range(1, 6), nodes], names=["horizon", "node"]).to_frame(index=False)
            frame["origin"], frame["family"], frame["population_method"] = origin, family, method
            frame["reference_observed"] = values.ravel()
            observed_frames.append(frame)
    actual = pd.concat(observed_frames, ignore_index=True)
    observed_keys = ["origin", "family", "population_method", "horizon", "node"]
    joined = b_cells.merge(actual, on=observed_keys, validate="many_to_one")
    np.testing.assert_allclose(joined.observed, joined.reference_observed, atol=1e-9, rtol=2e-12)
    b_points = read(directory / "burden_point_scores.csv")
    joined = b_points.merge(actual, on=observed_keys, validate="one_to_one")
    np.testing.assert_allclose(joined.observed, joined.reference_observed, atol=1e-9, rtol=2e-12)
    np.testing.assert_allclose(joined.absolute_error, abs(joined.value-joined.reference_observed), atol=1e-9, rtol=2e-12)
    np.testing.assert_allclose(joined.absolute_log_error, abs(np.log(joined.value/joined.reference_observed)), atol=1e-12)
    burden_keys = ["target", "outcome", "origin", "role", "variant", "horizon", "population_method", "measure", "node", "sex", "age_group", "unit"]
    burdens = b_cells.groupby(burden_keys+["level"], as_index=False).agg(coverage=("covered", "mean"), width=("width", "mean"), lower_miss=("lower_miss", "mean"), upper_miss=("upper_miss", "mean"), interval_score=("interval_score", "mean"))
    proper = b_wis.groupby(burden_keys, as_index=False)[["wis_50_80", "wis_50_80_95"]].mean()
    burdens = burdens.merge(proper, on=burden_keys, validate="many_to_one")
    points_summary = []
    scopes = {"45+": config["ages"], **{group: [age for age in config["ages"] if int(age.split("-")[0].rstrip("+")) in starts] for group, starts in config["age_groups"].items()}, **{"age:"+age: [age] for age in config["ages"]}}
    for scope, ages in scopes.items():
        selected = point_scores[point_scores.age.isin(ages)]
        result = selected.groupby(["target", "outcome", "origin", "family", "role", "variant", "sex", "horizon"], as_index=False)[["absolute_log_error", "signed_log_error"]].mean()
        result["age_scope"] = scope
        points_summary.append(result)
    validation = dict(passed=True, case=case["id"], procedures=13, original_reference_intervals_exact=True,
                      source_point_predictions_exact=True, rate_interval_rows=len(rate_intervals), burden_interval_rows=len(burden_intervals),
                      rate_point_rows=len(rate_points), burden_point_rows=len(burden_points),
                      array_checks=array_checks, structured_checks=structured, dynamic_checks=dynamics,
                      rate_wis_rows_recalculated=len(wis), burden_wis_rows_recalculated=len(b_wis))
    return validation, rates, burdens, pd.concat(points_summary, ignore_index=True), b_points, populations, pd.concat(decompositions, ignore_index=True), parameters


MAIN = ["tcn_adapted__original", "tcn_adapted__target_only", "tcn_adapted__gcc_assisted", "tcn_adapted__structured", "robust_dynamic__joint"]
SHORT = {"tcn_adapted__original": "TCN original", "tcn_adapted__target_only": "TCN target", "tcn_adapted__gcc_assisted": "TCN GCC",
         "tcn_adapted__structured": "TCN structured", "robust_dynamic__joint": "Joint dynamic"}
COLORS = ["#89949b", "#6499b0", "#a2a39a", "#d38c31", "#95495f"]


def procedure_label(family):
    if family in SHORT:
        return SHORT[family]
    role, variant = family.split("__")
    role = {"local_champion": "Local", "nonneural_champion": "Non-neural"}[role]
    return role+" "+{"original": "original", "target_only": "target", "gcc_assisted": "GCC", "structured": "structured"}[variant]


def mean_metrics(frame, groups, metrics=METRICS):
    return frame.groupby(groups, as_index=False)[metrics].mean()


def markdown(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n" + "\n".join(
        "| " + " | ".join(str(item) for item in row) + " |" for row in rows)


def point_endpoints(points):
    definitions = {"Both-sex 45+ count": points.measure.eq("count") & points.node.eq("Both__45+"),
                   "Male 65+ share": points.node.eq("Male__65+_within_45+"),
                   "Female 65+ share": points.node.eq("Female__65+_within_45+"),
                   "Male 80+ share": points.node.eq("Male__80+_within_45+"),
                   "Female 80+ share": points.node.eq("Female__80+_within_45+"),
                   "Age-specific M/F ratio": points.measure.eq("sex_rate_ratio")}
    frames = []
    keys = ["target", "outcome", "origin", "family", "role", "variant", "horizon", "population_method", "unit"]
    for endpoint, mask in definitions.items():
        part = points[mask].copy()
        part["signed_error"] = part.value-part.observed
        part["signed_log_error"] = np.log(part.value/part.observed)
        result = mean_metrics(part, keys, ["value", "observed", "absolute_error", "absolute_log_error", "signed_error", "signed_log_error"])
        result["endpoint"] = endpoint
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def rate_comparison(rates, points):
    groups = ["target", "outcome", "origin", "role", "variant", "sex", "horizon", "age_scope"]
    selected = rates[rates.level.eq(.8)].copy()
    rate = selected[selected.scale.eq("rate")][groups+METRICS]
    log = selected[selected.scale.eq("log_rate")][groups+["wis_50_80", "wis_50_80_95"]].rename(columns={"wis_50_80": "log_wis_50_80", "wis_50_80_95": "log_wis_50_80_95"})
    comparison = rate.merge(log, on=groups, validate="one_to_one")
    comparison = comparison.merge(points[groups+["absolute_log_error", "signed_log_error"]], on=groups, validate="one_to_one")
    comparison["family"] = comparison.role+"__"+comparison.variant
    return comparison


def rate_table(comparison, scope="45+", final=False, all_procedures=False):
    part = comparison[comparison.target.eq("Saudi Arabia") & comparison.horizon.eq(5) & comparison.age_scope.eq(scope)]
    if final:
        part = part[part.origin.eq(2018)]
    metrics = ["coverage", "width", "wis_50_80", "log_wis_50_80", "absolute_log_error"]
    if all_procedures:
        mean = mean_metrics(part, ["outcome", "family"], metrics)
        rows = [[row.outcome, procedure_label(row.family), f"{100*row.coverage:.1f}", f"{row.width:.3g}",
                 f"{row.wis_50_80:.3g}", f"{row.log_wis_50_80:.4f}", f"{row.absolute_log_error:.4f}"] for row in mean.itertuples()]
        return markdown(["Outcome", "Procedure", "Coverage (%)", "Width", "Rate WIS", "Log WIS", "Point ALE"], rows)
    mean = mean_metrics(part[part.family.isin(MAIN)], ["outcome", "sex", "family"], metrics).set_index(["outcome", "sex", "family"])
    rows = []
    for outcome in ["prevalence", "incidence"]:
        for sex in ["Male", "Female"]:
            for family in MAIN:
                row = mean.loc[(outcome, sex, family)]
                rows.append([outcome, sex, SHORT[family], f"{100*row.coverage:.1f}", f"{row.width:.3g}",
                             f"{row.wis_50_80:.3g}", f"{row.log_wis_50_80:.4f}", f"{row.absolute_log_error:.4f}"])
    return markdown(["Outcome", "Sex", "Procedure", "Coverage (%)", "Width", "Rate WIS", "Log WIS", "Point ALE"], rows)


def plot_rate_coverage(comparison, scope):
    selected = comparison[comparison.target.eq("Saudi Arabia") & comparison.horizon.eq(5) & comparison.age_scope.eq(scope) & comparison.family.isin(MAIN)]
    mean = mean_metrics(selected, ["outcome", "sex", "family"], ["coverage"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True, constrained_layout=True)
    for i, outcome in enumerate(["prevalence", "incidence"]):
        for j, sex in enumerate(["Male", "Female"]):
            part = mean[mean.outcome.eq(outcome) & mean.sex.eq(sex)].set_index("family").reindex(MAIN)
            ax = axes[i, j]
            bars = ax.bar(range(5), 100*part.coverage, color=COLORS)
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in part.coverage], padding=3, fontsize=9)
            ax.set_xticks(range(5), ["Original", "Target", "GCC", "Structured", "Dynamic"], rotation=20, ha="right")
            ax.axhline(80, linestyle="--", color="#222222", linewidth=1)
            ax.set_ylim(0, 111)
            ax.set_title(f"{outcome.capitalize()} · {sex.lower()}")
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("80% interval coverage (%)")
    cells = 55 if scope == "45+" else 20
    fig.suptitle(f"Saudi age-specific rates, ages {scope}: horizon 5, origins 2014–2018\n"
                 f"TCN reference/calibration procedures and separate joint dynamic model; {cells} dependent cells per sex", fontsize=11)
    stem = "saudi_"+("45plus" if scope == "45+" else "80plus")+"_coverage"
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=170)
    plt.close(fig)


def plot_derived(endpoints):
    selected = endpoints[endpoints.target.eq("Saudi Arabia") & endpoints.horizon.eq(5) & endpoints.level.eq(.8)
                         & endpoints.family.isin(MAIN) & endpoints.population_method.isin(["log_trend_last8", "dynamic_joint", "not_applicable"])]
    mean = mean_metrics(selected, ["outcome", "endpoint", "family"], ["coverage"])
    display = ["Both-sex 45+ count", "Male 80+ share", "Female 80+ share", "Age-specific M/F ratio"]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7), sharey=True, constrained_layout=True)
    for i, outcome in enumerate(["prevalence", "incidence"]):
        for j, endpoint in enumerate(display):
            part = mean[mean.outcome.eq(outcome) & mean.endpoint.eq(endpoint)].set_index("family").reindex(MAIN)
            ax = axes[i, j]
            bars = ax.bar(range(5), 100*part.coverage, color=COLORS)
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in part.coverage], padding=3, fontsize=8)
            ax.set_xticks(range(5), ["Orig", "Target", "GCC", "Struct", "Dynamic"], rotation=35, ha="right", fontsize=8)
            ax.axhline(80, linestyle="--", color="#222222", linewidth=1)
            ax.set_ylim(0, 112)
            ax.set_title(endpoint, fontsize=10)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(f"{outcome.capitalize()}\n80% interval coverage (%)")
    fig.suptitle("Saudi derived burden: horizon 5, origins 2014–2018\n"
                 "TCN references/structured and joint dynamic; five dependent totals/shares, 55 dependent ratio cells", fontsize=11)
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"saudi_derived_coverage.{suffix}", dpi=170)
    plt.close(fig)


def dynamic_diagnostics():
    """Describe saved pre-origin tracking without changing or refitting states."""
    rows = []
    selected = {}
    for case in read_json(RUN / "cases.json"):
        for item in read_json(RUN / case["id"] / "draw_index.json"):
            if item["kind"] != "dynamic_joint":
                continue
            model = joblib.load(RUN / case["id"] / item["fitted_state"])
            state = model["fitted_state"]
            weights = state["filter_weights"]
            row = dict(target=case["target"], outcome=case["outcome"], origin=item["origin"],
                       first_weight=float(weights[0]), final_weight=float(weights[-1]),
                       median_weight=float(np.median(weights)), minimum_weight=float(weights.min()),
                       final_effective_observation_variance_multiplier=float(1/weights[-1]),
                       fraction_updates_below_point_one=float(np.mean(weights < .1)))
            for index, sex in [(0, "male"), (22, "female")]:
                observed = state["transformed_history"][:, index]
                filtered = state["filtered_means"][:, index]
                row[sex+"_final_count_error_percent"] = float(100*np.expm1(filtered[-1]-observed[-1]))
                first_slope = state["filtered_means"][1, 44+index]
                observed_change = observed[1]-observed[0]
                row[sex+"_first_slope_fraction_of_observed_growth"] = float(first_slope/observed_change) if observed_change != 0 else None
                last_change = observed[-1]-observed[-2]
                row[sex+"_next_slope_fraction_of_latest_observed_growth"] = float(state["final_mean"][44+index]/last_change) if last_change != 0 else None
            rows.append(row)
            if case["target"] == "Saudi Arabia" and item["origin"] == 2018:
                selected[case["outcome"]] = state
    summary = pd.DataFrame(rows).sort_values(["target", "outcome", "origin"])
    summary.to_csv(OUT / "dynamic_preorigin_tracking.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, outcome in enumerate(["prevalence", "incidence"]):
        state = selected[outcome]
        years = state["years"]
        for index, sex, color in [(0, "Male", "#227ca7"), (22, "Female", "#b56b38")]:
            axes[row, 0].plot(years, np.exp(state["transformed_history"][:, index]), color=color, label=sex+" observed")
            axes[row, 0].plot(years, np.exp(state["filtered_means"][:, index]), color=color, linestyle="--", label=sex+" filtered")
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_ylabel(outcome.capitalize()+" count total\n(logarithmic axis)")
        axes[row, 0].set_title("Saved pre-origin count tracking")
        axes[row, 0].legend(fontsize=8, ncol=2, loc="upper left")
        weights = state["filter_weights"]
        axes[row, 1].plot(years[1:], weights, color="#95495f", marker=".")
        axes[row, 1].axhline(.1, color="#888888", linestyle=":", linewidth=1)
        axes[row, 1].set_ylim(0, 1.05)
        axes[row, 1].set_ylabel("Whole-vector robust update weight")
        axes[row, 1].set_title(f"Final weight {weights[-1]:.4f}; R multiplier {1/weights[-1]:.1f}×")
        for column in range(2):
            axes[row, column].set_xlabel("Historical observation year")
            axes[row, column].grid(alpha=.18)
    fig.suptitle("Saudi joint dynamic model: 2018-origin fits to 1990–2018 observations\n"
                 "Saved fits support horizons 1–5; low robust weights reduce the whole annual update", fontsize=11)
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"saudi_dynamic_preorigin_tracking.{suffix}", dpi=170)
    plt.close(fig)
    return summary


def build_report(comparison, endpoints, point_endpoints_frame, populations, validations, diagnostics):
    selected = endpoints[endpoints.target.eq("Saudi Arabia") & endpoints.horizon.eq(5) & endpoints.level.eq(.8)
                         & endpoints.family.isin(["tcn_adapted__original", "tcn_adapted__structured", "robust_dynamic__joint"])
                         & endpoints.population_method.isin(["log_trend_last8", "dynamic_joint", "not_applicable"])]
    mean = mean_metrics(selected, ["outcome", "endpoint", "unit", "family"])
    p = point_endpoints_frame[point_endpoints_frame.target.eq("Saudi Arabia") & point_endpoints_frame.horizon.eq(5)
                              & point_endpoints_frame.population_method.isin(["log_trend_last8", "dynamic_joint", "not_applicable"])]
    p = mean_metrics(p, ["outcome", "endpoint", "family"], ["absolute_error", "absolute_log_error"])
    mean = mean.merge(p, on=["outcome", "endpoint", "family"], validate="one_to_one")
    rows = [[row.outcome, row.endpoint, SHORT[row.family], row.unit, f"{100*row.coverage:.1f}", f"{row.width:.3g}",
             f"{row.wis_50_80:.3g}", f"{row.absolute_error:.3g}"] for row in mean.itertuples()]
    burden_table = markdown(["Outcome", "Endpoint", "Procedure", "Unit", "Coverage (%)", "Width", "WIS", "Point MAE"], rows)
    # Population marginals of the three structured rate roles are identical;
    # display each original population method once using the TCN copy.
    selected = populations[populations.target.eq("Saudi Arabia") & populations.horizon.eq(5)
                           & populations.family.isin(["tcn_adapted__structured", "robust_dynamic__joint"])]
    pop = mean_metrics(selected, ["outcome", "sex", "population_method"], ["absolute_log_error", "coverage_80", "width_80", "wis_50_80"])
    pop_rows = [[row.outcome, row.sex, row.population_method, f"{row.absolute_log_error:.4f}", f"{100*row.coverage_80:.1f}",
                f"{row.width_80:.3g}", f"{row.wis_50_80:.3g}"] for row in pop.itertuples()]
    population_table = markdown(["Outcome", "Sex", "Population method", "Point ALE", "Coverage (%)", "Width", "Population WIS"], pop_rows)
    gcc = comparison[comparison.horizon.eq(5) & comparison.age_scope.eq("45+")
                     & comparison.family.isin(["tcn_adapted__original", "tcn_adapted__structured", "robust_dynamic__joint"])]
    gcc = mean_metrics(gcc, ["target", "outcome", "family"], ["coverage", "width", "wis_50_80", "log_wis_50_80", "absolute_log_error"])
    gcc_rows = []
    order = ["tcn_adapted__original", "tcn_adapted__structured", "robust_dynamic__joint"]
    for (target, outcome), block in gcc.groupby(["target", "outcome"]):
        block = block.set_index("family")
        gcc_rows.append([target, outcome, " / ".join(f"{100*block.loc[f,'coverage']:.1f}" for f in order),
                         " / ".join(f"{block.loc[f,'width']:.3g}" for f in order),
                         " / ".join(f"{block.loc[f,'wis_50_80']:.3g}" for f in order),
                         " / ".join(f"{block.loc[f,'absolute_log_error']:.4f}" for f in order)])
    gcc_table = markdown(["Country", "Outcome", "Coverage O / S / D (%)", "Width O / S / D", "Rate WIS O / S / D", "Point ALE O / S / D"], gcc_rows)
    total_rates = sum(row["rate_interval_rows"] for row in validations)
    total_burdens = sum(row["burden_interval_rows"] for row in validations)
    overall = comparison[comparison.target.eq("Saudi Arabia") & comparison.horizon.eq(5) & comparison.age_scope.eq("45+")]
    overall = mean_metrics(overall, ["outcome", "sex", "family"], ["coverage", "width", "wis_50_80", "log_wis_50_80", "absolute_log_error"]).set_index(["outcome", "sex", "family"])
    original, structured, target, gcc_family, dynamic = MAIN[0], MAIN[3], MAIN[1], MAIN[2], MAIN[4]
    pm0, pms = overall.loc[("prevalence", "Male", original)], overall.loc[("prevalence", "Male", structured)]
    pf0, pfs = overall.loc[("prevalence", "Female", original)], overall.loc[("prevalence", "Female", structured)]
    im0, ims = overall.loc[("incidence", "Male", original)], overall.loc[("incidence", "Male", structured)]
    if0, ifs = overall.loc[("incidence", "Female", original)], overall.loc[("incidence", "Female", structured)]
    imt, img = overall.loc[("incidence", "Male", target)], overall.loc[("incidence", "Male", gcc_family)]
    final = comparison[comparison.target.eq("Saudi Arabia") & comparison.horizon.eq(5) & comparison.origin.eq(2018)].set_index(["outcome", "sex", "age_scope", "family"])
    regional = comparison[comparison.horizon.eq(5) & comparison.age_scope.eq("45+") & comparison.family.isin(MAIN)]
    regional = mean_metrics(regional, ["target", "outcome", "sex", "family"], ["coverage"])
    below = regional.assign(below=regional.coverage < .8-1e-10).groupby("family").below.sum()
    diag = diagnostics[diagnostics.target.eq("Saudi Arabia") & diagnostics.origin.eq(2018)].set_index("outcome")
    first_fractions = diag[["male_first_slope_fraction_of_observed_growth", "female_first_slope_fraction_of_observed_growth"]].to_numpy()
    next_fractions = diag[["male_next_slope_fraction_of_latest_observed_growth", "female_next_slope_fraction_of_latest_observed_growth"]].to_numpy()
    accounting = read(OUT / "rate_population_error_accounting.csv.gz")
    example = accounting[accounting.target.eq("Saudi Arabia") & accounting.outcome.eq("incidence") & accounting.origin.eq(2018)
                         & accounting.horizon.eq(5) & accounting.sex.eq("Female") & accounting.age.eq("85-89")
                         & accounting.family.eq("robust_dynamic__joint")].iloc[0]
    text = f"""# Exploratory reliability comparison v1.3

This bounded comparison evaluates a fixed structured postprocessor and a separate robust dynamic model after the original and v1.2 outcomes were known. It is exploratory. The original adapted-TCN joint male–female primary criterion remains unmet and unchanged. No new result selects a replacement primary winner.

All six GCC countries, prevalence/incidence, both sexes, eleven ages 45–49 through 95+, horizons 1–5 and origins 2014–2018 are retained. Each case contains thirteen procedures: original, target-only width calibration, GCC-assisted width calibration and structured calibration for each of the three frozen forecasting roles, plus the new joint dynamic model.

The structured TCN produces mixed, modest gains rather than a reliable replacement. Saudi male prevalence coverage increases from {100*pm0.coverage:.1f}% to {100*pms.coverage:.1f}%, with rate WIS improving {100*(1-pms.wis_50_80/pm0.wis_50_80):.2f}% and width increasing {100*(pms.width/pm0.width-1):.2f}%. Female prevalence coverage remains {100*pfs.coverage:.1f}%, despite a {100*(1-pfs.wis_50_80/pf0.wis_50_80):.2f}% rate-WIS improvement and {100*(pfs.width/pf0.width-1):.2f}% wider intervals. Male incidence coverage rises from {100*im0.coverage:.1f}% to {100*ims.coverage:.1f}%, but rate WIS worsens {100*(ims.wis_50_80/im0.wis_50_80-1):.2f}%; female incidence coverage falls from {100*if0.coverage:.1f}% to {100*ifs.coverage:.1f}%, with rate WIS worsening {100*(ifs.wis_50_80/if0.wis_50_80-1):.2f}%.

Against the v1.2 male-incidence references, structured calibration raises rolling coverage from {100*imt.coverage:.1f}% to {100*ims.coverage:.1f}% with {100*(1-ims.width/imt.width):.2f}% narrower intervals than target-only calibration and {100*(1-ims.width/img.width):.2f}% narrower intervals than GCC assistance. Its rate WIS is only {100*(1-ims.wis_50_80/imt.wis_50_80):.2f}% and {100*(1-ims.wis_50_80/img.wis_50_80):.2f}% better, respectively. The unchanged points mean that these changes concern distributions alone.

The new dynamic model performs poorly for Saudi aggregate burden and age composition: both-sex count intervals and both male and female 80+ share intervals cover zero of five origins for each outcome. Its five-origin Saudi rate errors are also larger than the TCN's for both sexes and outcomes. Some final female-rate and regional comparisons improve, as shown below; those isolated gains do not repair the joint burden failures.

## What each new procedure means

Structured calibration preserves each frozen point forecast and both original population procedures. It uses only matured historical errors to estimate strongly shrunk sex/broad-age/horizon location shifts and asymmetric tail scales. Adjacent-horizon quantiles are smoothed before scales are calculated. Short-horizon completed errors can inform a marginal fit, but the joint draws still contain the original 7–11 complete temporal blocks. Their age/sex/horizon/population pairing remains intact. This is an additive robust-location and scale heuristic, not exact quantile matching or a conformal coverage guarantee. Positive slopes preserve marginal ordering; means, medians and covariance may change.

The dynamic model is a distinct target-only forecast of native count totals, count age composition and implied populations. Its 44 identifiable coordinates have 88 level/slope states, fixed age smoothing and damping, robust covariance regularization and approximate robust state updates. Counts and rates are derived coherently from every joint state trajectory. Its point is the decoded zero-innovation path, not a marginal predictive median. The 1,024 simulations are 512 antithetic pairs from an assumed distribution; they are not observed residual blocks. Noise matrices and hyperparameters are plug-in estimates/fixed assumptions, and their estimation uncertainty is not fully propagated. Future process and observation discrepancy are separate from GBD source-estimate uncertainty.

## Saudi five-year rate forecasts

The tables average origins 2014–2018 equally. Coverage refers to central 80% rate intervals; width and rate WIS are on the rate scale per 100,000. Log WIS and point absolute log error (ALE) are shown separately. WIS combines 50% and 80% intervals and the predictive median. Lower WIS/ALE is better, while coverage must be read alongside width and directional misses. The first four TCN procedures share exactly the same point predictions.

### All ages 45+

Each sex has 55 dependent age–origin cells, not 55 independent replications.

{rate_table(comparison, '45+')}

![Saudi 45+ interval coverage](saudi_45plus_coverage.png)

### Explicit ages 80+

Each sex has 20 dependent age–origin cells spanning 80–84, 85–89, 90–94 and 95+. All four ages remain visible in the [complete summaries](rate_summary.csv.gz), regardless of performance.

{rate_table(comparison, '80+')}

![Saudi 80+ interval coverage](saudi_80plus_coverage.png)

### Separate 2018→2023 endpoint

This endpoint is also included once in the preceding rolling summaries. The two displays do not constitute independent evidence. Each sex's 45+ row contains eleven dependent age cells. Age-80+ and single-age final results are retained in [final endpoint comparisons](final_2018_rate_comparison.csv).

Rolling improvements do not persist uniformly at this endpoint. Structured TCN male coverage is {100*final.loc[('prevalence','Male','45+',structured),'coverage']:.1f}% for prevalence and {100*final.loc[('incidence','Male','45+',structured),'coverage']:.1f}% for incidence, below the corresponding v1.2 selected values of {100*final.loc[('prevalence','Male','45+',target),'coverage']:.1f}% and {100*final.loc[('incidence','Male','45+',target),'coverage']:.1f}%. The dynamic model covers {100*final.loc[('prevalence','Female','45+',dynamic),'coverage']:.1f}% of female age cells for both outcomes at this endpoint, despite much poorer rolling coverage. For female prevalence at age 80+, dynamic coverage is four of four versus zero of four for every TCN variant; dynamic male prevalence at age 80+ covers zero of four. These opposing results are retained explicitly.

{rate_table(comparison, '45+', final=True)}

## All frozen comparators and new procedures

The following Saudi table averages male and female strata equally, retaining all thirteen procedures. It does not discard adverse comparisons or promote a final-score winner. Local and non-neural champion families remain those selected in the original chronological analysis.

{rate_table(comparison, all_procedures=True)}

## Derived burden and age composition

The display below compares the original TCN, structured TCN and joint dynamic model. Existing target-only/GCC-assisted distributions and all comparator roles remain in the [derived endpoint ledger](burden_endpoint_summary.csv). Old/structured rate procedures use the original eight-year log-trend population method here; their persistence sensitivity remains in the ledger. The dynamic model supplies its own population forecasts. Because the dynamic model changes several components together, differences cannot be attributed causally to one component.

Each count or share endpoint has five dependent origin observations. Ratio summaries average eleven ages across those origins. Share values/widths are percentage points; count, share, rate and ratio WIS have different units and are never pooled. Point MAE describes the fixed or newly issued point, not the predictive median.

{burden_table}

Structured calibration leaves male 80+ share coverage at 20% for both Saudi outcomes. Its TCN sex-rate-ratio coverage falls from 50.9% to 49.1% for prevalence and from 49.1% to 47.3% for incidence. The dynamic model reaches 78.2% ratio coverage for both outcomes, while all five total-count observations exceed its upper bounds and all five observed male/female 80+ shares fall below its lower bounds. Improved ratio coverage therefore coexists with severe count and composition miscalibration.

![Saudi derived-burden interval coverage](saudi_derived_coverage.png)

## Population errors and rate/count error accounting

These summaries assess implied populations separately. The same two old population trajectories are shared by all three structured rate roles and are displayed once per method. Values reflect only the study's operational/estimated population assumptions; they are not independent validation against an external population release.

{population_table}

The [age-cell error ledger](rate_population_error_accounting.csv.gz) verifies the exact identity signed log count error = signed log rate error + signed log population error. This is an algebraic accounting identity, not a causal decomposition. Opposing component errors can cancel in a count total. The separate population-source sensitivity and projection scenarios remain outside this comparison.

One illustrative final-period cell also shows cancellation in a rate: for Saudi female incidence at age 85–89 in 2023, the dynamic model's count is {100*np.expm1(example.count_signed_log_error):+.2f}% and its implied population {100*np.expm1(example.population_signed_log_error):+.2f}% above the GBD counterparts, while the derived rate differs by only {100*np.expm1(example.rate_signed_log_error):+.2f}%. A relatively accurate rate therefore does not establish accurate demographic or count trajectories. This cell illustrates the accounting identity; it is not a selected performance endpoint or a new evaluation criterion.

## Saved dynamic-filter diagnostics

The pre-origin diagnostics show persistent later tracking lag, rather than an unlearned initial count slope: the first update captures {100*first_fractions.min():.1f}–{100*first_fractions.max():.1f}% of the initial observed log-count growth. In the 2018 Saudi fit, final robust update weights are {diag.loc['prevalence','final_weight']:.5f} for prevalence and {diag.loc['incidence','final_weight']:.5f} for incidence, corresponding to effective observation-discrepancy multipliers of {diag.loc['prevalence','final_effective_observation_variance_multiplier']:.1f}× and {diag.loc['incidence','final_effective_observation_variance_multiplier']:.1f}×. Signed differences between filtered 2018 totals and observed totals are {diag.loc['prevalence','male_final_count_error_percent']:+.2f}% / {diag.loc['prevalence','female_final_count_error_percent']:+.2f}% for male/female prevalence, and {diag.loc['incidence','male_final_count_error_percent']:+.2f}% / {diag.loc['incidence','female_final_count_error_percent']:+.2f}% for incidence.

The saved next-year log-total-count slopes are {100*next_fractions.min():.1f}–{100*next_fractions.max():.1f}% of the most recent observed annual growth across those four sex–outcome series. This comparison describes the issued state's responsiveness; it does not assume that the previous year's growth must persist.

These saved-state observations are consistent with reduced responsiveness under the fixed robust update and damped slopes. They do not isolate either mechanism causally. No alternative initialization, covariance, robustness setting or model was fitted after seeing outcomes. The [diagnostic ledger](dynamic_preorigin_tracking.csv) retains every country, outcome and fit origin.

![Saudi saved dynamic-filter diagnostics](saudi_dynamic_preorigin_tracking.png)

## Gulf benchmarking

O / S / D means original TCN / structured TCN / joint dynamic model. These five-origin, horizon-five summaries give equal weight to male and female age strata. Saudi Arabia remains the primary target. All thirteen procedures, age groups, individual ages, horizons and directional misses remain in the [rate comparison ledger](rate_comparison.csv.gz).

{gcc_table}

Across 24 country–outcome–sex rate summaries, coverage is below 80% in {int(below[original])} original-TCN summaries, {int(below[target])} target-calibrated summaries, {int(below[structured])} structured-TCN summaries and {int(below[dynamic])} dynamic-model summaries. Thus the Saudi dynamic failures should not be generalized to every GCC rate forecast, while the structured method also fails to deliver uniform regional improvement. These dependent descriptive counts are not significance tests or a new model-selection criterion.

## Independent checks and interpretation limits

The independent audit checked all twelve cases, including {total_rates:,} rate interval rows and {total_burdens:,} derived interval rows. It reconstructed every structured coefficient and historical transformed draw, replayed all 60 dynamic state calculations and seeded simulations, verified coherent counts/shares/ratios, recalculated quantiles, interval scores, WIS and point errors, and checked the unchanged original controls. All case artifacts were committed before any new final scoring. Source, scientific code, report and artifact hashes are recorded in [validation](validation.json).

New calibration gains must be weighed against width, WIS, point accuracy, lower/upper misses and stability across sexes, ages and origins. Repeated ages, overlapping horizons/origins, related GCC targets and antithetic simulations do not supply independent sample sizes. High aggregate coverage can coexist with poor age-share or sex-ratio reliability. These forecasts concern retrospectively modeled GBD estimates, not patient survival, clinical events, causal biological effects or demonstrated health-service needs. Stronger confirmation requires untouched future data. The [original primary report](../primary_v1/report.md) remains authoritative for the prespecified hypothesis.
"""
    (OUT / "report.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--render-only", action="store_true", help="Refresh narrative/figures from hash-verified, already audited summaries")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one through four independent audit workers")
    check_lock()
    config = read_json(ROOT / "study_design/locked_v1/design.json")
    spec = read_json(ROOT / "study_design/reliability_v1_3.json")
    if args.render_only:
        validation = read_json(OUT / "validation.json")
        assert validation["passed"] and validation["source_manifest_sha256"] == sha(RUN / "run_manifest.json")
        for filename, digest in validation["artifact_sha256"].items():
            assert sha(OUT / filename) == digest, filename
        for filename, digest in validation["helper_code_sha256"].items():
            assert sha(ROOT / filename) == digest, filename
        # The numeric summaries are exactly those verified in the completed
        # audit. This branch issues no forecasts and reruns no scientific fit.
        comparison = read(OUT / "rate_comparison.csv.gz")
        endpoints = read(OUT / "burden_endpoint_summary.csv")
        p_endpoints = read(OUT / "burden_point_endpoint_summary.csv")
        populations = read(OUT / "population_error_summary.csv.gz")
        for case in read_json(RUN / "cases.json"):
            committed = read_json(RUN / case["id"] / "issued_commit.json")["artifact_sha256"]
            for item in read_json(RUN / case["id"] / "draw_index.json"):
                if item["kind"] == "dynamic_joint":
                    assert sha(RUN / case["id"] / item["fitted_state"]) == committed[item["fitted_state"]]
        diagnostics = dynamic_diagnostics()
        plot_rate_coverage(comparison, "45+")
        plot_rate_coverage(comparison, "80+")
        plot_derived(endpoints)
        build_report(comparison, endpoints, p_endpoints, populations, validation["case_validations"], diagnostics)
        validation.setdefault("numerical_audit_report_code_sha256", validation["report_code_sha256"])
        validation["report_code_sha256"] = sha(Path(__file__))
        validation["render_refresh_utc"] = datetime.now(timezone.utc).isoformat()
        validation["render_refresh_scope"] = "Narrative and figures from hash-verified audited summaries; descriptive saved-state diagnostics; no scientific change or refit"
        validation["artifact_sha256"] = {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "validation.json"}
        (OUT / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
        print(json.dumps(dict(passed=True, render_only=True, report=str(OUT / "report.md")), indent=2))
        return
    manifest = verify_manifest(RUN)
    verify_commits(manifest, spec)
    cases = read_json(RUN / "cases.json")
    assert len(cases) == 12
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(audit_case, case, config, spec) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print("Independent v1.3 audit complete: "+result[0]["case"], flush=True)
    validations = sorted([result[0] for result in results], key=lambda row: row["case"])
    rates, burdens, points, burden_points, populations, accounting, parameters = [pd.concat([result[i] for result in results], ignore_index=True) for i in range(1, 8)]
    rate_keys = ["target", "outcome", "origin", "role", "variant", "sex", "horizon", "scale", "level", "age_scope"]
    rates = rates.sort_values(rate_keys)
    comparison = rate_comparison(rates, points)
    endpoints = burden_endpoints(burdens)
    endpoints["family"] = endpoints.role+"__"+endpoints.variant
    p_endpoints = point_endpoints(burden_points)
    OUT.mkdir(parents=True, exist_ok=True)
    rates.to_csv(OUT / "rate_summary.csv.gz", index=False, compression="gzip")
    comparison.to_csv(OUT / "rate_comparison.csv.gz", index=False, compression="gzip")
    comparison[comparison.origin.eq(2018) & comparison.horizon.eq(5)].to_csv(OUT / "final_2018_rate_comparison.csv", index=False)
    comparison[comparison.target.eq("Saudi Arabia") & comparison.horizon.eq(5)].to_csv(OUT / "saudi_all_procedures.csv", index=False)
    burdens.to_csv(OUT / "burden_node_summary.csv.gz", index=False, compression="gzip")
    endpoints.to_csv(OUT / "burden_endpoint_summary.csv", index=False)
    p_endpoints.to_csv(OUT / "burden_point_endpoint_summary.csv", index=False)
    points.to_csv(OUT / "point_rate_summary.csv.gz", index=False, compression="gzip")
    populations.to_csv(OUT / "population_error_summary.csv.gz", index=False, compression="gzip")
    accounting.to_csv(OUT / "rate_population_error_accounting.csv.gz", index=False, compression="gzip")
    parameters.to_csv(OUT / "structured_parameter_summary.csv", index=False)
    diagnostics = dynamic_diagnostics()
    plot_rate_coverage(comparison, "45+")
    plot_rate_coverage(comparison, "80+")
    plot_derived(endpoints)
    build_report(comparison, endpoints, p_endpoints, populations, validations, diagnostics)
    helpers = ["scripts/report_demography.py", "scripts/report_calibration.py", "scripts/run_local_baselines.py"]
    validation = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), role="independent_exploratory_reliability_v1_3_audit",
                      report_code_sha256=sha(Path(__file__)), helper_code_sha256={name: sha(ROOT / name) for name in helpers},
                      source_manifest_sha256=sha(RUN / "run_manifest.json"), original_primary_preserved=True,
                      all_cases_committed_before_scoring=True, case_validations=validations)
    validation["artifact_sha256"] = {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "validation.json"}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps(dict(passed=True, cases=len(validations), report=str(OUT / "report.md")), indent=2))


if __name__ == "__main__":
    main()
