"""Exploratory robust dynamic model of count composition and population.

This is an approximate robust linear-state filter with fixed hyperparameters,
not a posterior calibrated to GBD uncertainty draws. Count totals/compositions
and population cells are the only modeled coordinates; rates are derived.
"""

import copy

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.special import logsumexp


SETTINGS = {
    "model_id": "robust_dynamic_count_composition_population_v1_3",
    "minimum_history_years": 8,
    "slope_damping": .95,
    "age_laplacian_penalty": 1.,
    "mad_normal_consistency": 1.482602218505602,
    "minimum_curvature_scale": 1e-4,
    "radial_clipping_standardized_rms": 2.5,
    "diagonal_correlation_shrinkage": .75,
    "numerically_constant_standardized_variance": 1e-12,
    "level_innovation_variance_fraction": .2,
    "slope_innovation_variance_fraction": .1,
    "observation_variance_fraction": 1/12,
    "student_df": 5.,
    "initial_slope_variance_multiplier": 25.,
    "draws": 1024,
    "default_seed": 1301,
    "antithetic": True,
    "covariance_jitter_or_eigenvalue_clipping": False,
    "empirical_residual_bank_added": False,
    "extra_global_variance_mixture": False,
    "covariance_and_hyperparameter_uncertainty_fully_propagated": False,
}


def helmert_basis(n):
    """Columns are orthonormal contrasts orthogonal to the all-ones vector."""
    if n < 2:
        raise ValueError("At least two ages are required for composition coordinates")
    basis = np.zeros((n, n-1), dtype=float)
    for index in range(n-1):
        size = index+1
        denominator = np.sqrt(size*(size+1))
        basis[:size, index] = 1/denominator
        basis[size, index] = -size/denominator
    return basis


def _cholesky(matrix, label):
    """Symmetrization is allowed; changing variances/eigenvalues is not."""
    symmetric = (np.asarray(matrix, dtype=float)+np.asarray(matrix, dtype=float).T)/2
    if not np.isfinite(symmetric).all():
        raise ValueError(f"Nonfinite {label}")
    try:
        return np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{label} is not positive definite; no silent covariance repair") from exc


def _validated_panel(panel, config, origin, target, outcome):
    required = {"location_name", "outcome", "year", "sex", "age", "rate", "count", "implied_population"}
    if not required.issubset(panel.columns):
        raise ValueError("The regional panel lacks required count/rate/population columns")
    if isinstance(origin, (bool, np.bool_)) or not isinstance(origin, (int, np.integer)):
        raise ValueError("Origin must be an integer")
    allowed_targets = {country["name"] for country in config["countries"] if country["gcc"]}
    if target not in allowed_targets or outcome not in {"prevalence", "incidence"}:
        raise ValueError("This exploratory model is restricted to GCC prevalence/incidence")
    expected_ages = [f"{age}-{age+4}" for age in range(45, 95, 5)]+["95+"]
    if list(config["sexes"]) != ["Male", "Female"] or list(config["ages"]) != expected_ages:
        raise ValueError("Use both configured sexes and all eleven ages45–95+")
    first_year = int(config["calendar"]["history_start"])
    if origin > config["calendar"]["history_end"] or origin-first_year+1 < SETTINGS["minimum_history_years"]:
        raise ValueError("Origin lacks the required bounded complete history")
    # Future rows and other cases are removed before any value validation.
    history = panel.loc[panel.location_name.eq(target) & panel.outcome.eq(outcome)
                        & panel.year.between(first_year, origin)].copy()
    years = np.arange(first_year, origin+1, dtype=int)
    coords = pd.MultiIndex.from_product([config["sexes"], config["ages"]], names=["sex", "age"])
    expected = pd.MultiIndex.from_product([years, config["sexes"], config["ages"]], names=["year", "sex", "age"])
    keys = ["year", "sex", "age"]
    actual = pd.MultiIndex.from_frame(history[keys])
    if len(history) != len(expected) or history.duplicated(keys).any() or set(actual) != set(expected):
        raise ValueError("History must contain complete unique year/sex/age cells through origin")
    history = history.set_index(keys).reindex(expected).reset_index()
    values = history[["rate", "count", "implied_population"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("History requires finite positive rates, counts and populations")
    inferred = history["count"].to_numpy()/history.rate.to_numpy()*100000
    if (not np.isfinite(inferred).all()
            or not np.allclose(inferred, history.implied_population, rtol=1e-8, atol=1e-6)):
        raise ValueError("Implied population does not agree with count/rate×100000")
    return history, years, coords


def _transform(history, years, config, basis):
    ages = len(config["ages"])
    counts = history["count"].to_numpy().reshape(len(years), len(config["sexes"]), ages)
    population = (history["count"].to_numpy()/history.rate.to_numpy()*100000).reshape(counts.shape)
    pieces, names = [], []
    for sex_index, sex in enumerate(config["sexes"]):
        logs = np.log(counts[:, sex_index])
        pieces.append(np.column_stack([logsumexp(logs, axis=1), logs @ basis,
                                       np.log(population[:, sex_index])]))
        names += [f"{sex}:log_count_total"]
        names += [f"{sex}:count_ilr_{index+1}" for index in range(ages-1)]
        names += [f"{sex}:log_population:{age}" for age in config["ages"]]
    transformed = np.column_stack(pieces)
    if not np.isfinite(transformed).all():
        raise ValueError("Nonfinite transformed historical coordinates")
    return transformed, names


def _covariance(transformed):
    curvature = np.diff(transformed, n=2, axis=0)
    location = np.median(curvature, axis=0)
    scales = np.maximum(SETTINGS["mad_normal_consistency"]*np.median(np.abs(curvature-location), axis=0),
                        SETTINGS["minimum_curvature_scale"])
    standardized = (curvature-location)/scales
    rms = np.sqrt(np.mean(standardized**2, axis=1))
    weights = np.ones(len(rms))
    take = rms > SETTINGS["radial_clipping_standardized_rms"]
    weights[take] = SETTINGS["radial_clipping_standardized_rms"]/rms[take]
    clipped = standardized*weights[:, None]
    centered = clipped-clipped.mean(axis=0)
    covariance = centered.T @ centered/(len(centered)-1)
    variances = np.diag(covariance)
    active = variances > SETTINGS["numerically_constant_standardized_variance"]
    correlation = np.zeros_like(covariance)
    indices = np.flatnonzero(active)
    if len(indices):
        deviations = np.sqrt(variances[indices])
        correlation[np.ix_(indices, indices)] = covariance[np.ix_(indices, indices)]/np.outer(deviations, deviations)
    np.fill_diagonal(correlation, 1.)
    shrinkage = SETTINGS["diagonal_correlation_shrinkage"]
    shrunk = (1-shrinkage)*correlation+shrinkage*np.eye(len(scales))
    sigma = np.outer(scales, scales)*shrunk
    sigma = (sigma+sigma.T)/2
    _cholesky(sigma, "Regularized curvature covariance")
    return sigma, {"curvature": curvature, "curvature_median": location, "mad_scales": scales,
                   "standardized_curvature": standardized, "radial_rms": rms, "radial_weights": weights,
                   "clipped_standardized_curvature": clipped, "empirical_correlation": correlation,
                   "shrunk_correlation": shrunk, "numerically_constant_coordinates": ~active}


def _state_matrices(sigma, basis, sexes):
    ages = len(basis)
    difference = np.diff(np.eye(ages), axis=0)
    age_smoothing = np.linalg.solve(np.eye(ages)+SETTINGS["age_laplacian_penalty"]*(difference.T @ difference), np.eye(ages))
    one_sex_smoothing = block_diag(np.ones((1, 1)), basis.T @ age_smoothing @ basis, age_smoothing)
    smoothing = block_diag(*[one_sex_smoothing for _ in sexes])
    dimensions = len(sigma)
    transition = np.block([[np.eye(dimensions), np.eye(dimensions)],
                           [np.zeros((dimensions, dimensions)), SETTINGS["slope_damping"]*smoothing]])
    process = block_diag(SETTINGS["level_innovation_variance_fraction"]*sigma,
                         SETTINGS["slope_innovation_variance_fraction"]*sigma)
    observation = SETTINGS["observation_variance_fraction"]*sigma
    return transition, process, observation, smoothing, age_smoothing


def _filter(transformed, sigma, transition, process, observation):
    dimensions = transformed.shape[1]
    mean = np.r_[transformed[0], np.zeros(dimensions)]
    covariance = block_diag(observation, SETTINGS["initial_slope_variance_multiplier"]*sigma)
    initial_mean, initial_covariance = mean.copy(), covariance.copy()
    means, innovations, weights, distances = [mean.copy()], [], [], []
    identity = np.eye(2*dimensions)
    for values in transformed[1:]:
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T+process
        covariance = (covariance+covariance.T)/2
        residual = values-mean[:dimensions]
        ordinary_variance = covariance[:dimensions, :dimensions]+observation
        distance = float(residual @ np.linalg.solve(ordinary_variance, residual))
        weight = min(1., (SETTINGS["student_df"]+dimensions)/(SETTINGS["student_df"]+distance))
        effective_observation = observation/weight
        variance = covariance[:dimensions, :dimensions]+effective_observation
        gain = np.linalg.solve(variance, covariance[:dimensions, :]).T
        mean = mean+gain @ residual
        remainder = identity.copy()
        remainder[:, :dimensions] -= gain
        covariance = remainder @ covariance @ remainder.T+gain @ effective_observation @ gain.T
        covariance = (covariance+covariance.T)/2
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or weight <= 0:
            raise ValueError("Nonfinite robust-filter state or weight")
        means.append(mean.copy())
        innovations.append(residual)
        weights.append(weight)
        distances.append(distance)
    _cholesky(covariance, "Filtered state covariance")
    return {"initial_mean": initial_mean, "initial_covariance": initial_covariance,
            "final_mean": mean, "final_covariance": covariance, "filtered_means": np.asarray(means),
            "filter_innovations": np.asarray(innovations), "filter_weights": np.asarray(weights),
            "filter_mahalanobis_squared": np.asarray(distances)}


def _decode(latent, fitted):
    basis = fitted["composition_basis"]
    ages = len(fitted["ages"])
    count_parts, population_parts = [], []
    for sex_index, _ in enumerate(fitted["sexes"]):
        part = latent[..., sex_index*(2*ages):(sex_index+1)*(2*ages)]
        clr = part[..., 1:ages] @ basis.T
        log_shares = clr-logsumexp(clr, axis=-1, keepdims=True)
        log_counts = part[..., :1]+log_shares
        log_populations = part[..., ages:]
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                count_parts.append(np.exp(log_counts))
                population_parts.append(np.exp(log_populations))
            except FloatingPointError as exc:
                raise ValueError("Unsafe positive count/population reconstruction") from exc
    counts = np.concatenate(count_parts, axis=-1)
    populations = np.concatenate(population_parts, axis=-1)
    rates = counts/populations*100000
    if not all(np.isfinite(values).all() and (values > 0).all() for values in [counts, populations, rates]):
        raise ValueError("Draw reconstruction produced nonfinite or nonpositive values; no clipping applied")
    return rates, counts, populations


def simulate_fitted(fitted_state, seed=1301, draws=1024):
    """Replay coherent paths from saved fit arrays, without refitting any model.

    Each antithetic pair shares all chi-square scales and opposite standard
    normals through the full recursion. Future process and observation shocks
    use separate joint Student-t vectors. Their t5 standardization makes the
    supplied Q/R matrices covariances, not unstandardized scale matrices.
    """
    if (draws != SETTINGS["draws"] or isinstance(draws, (bool, np.bool_))
            or not isinstance(draws, (int, np.integer))):
        raise ValueError("This bounded model uses exactly1024 Monte Carlo paths")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    fitted = fitted_state
    if fitted["settings"] != SETTINGS:
        raise ValueError("Saved fit settings differ from this frozen model")
    dimensions = len(fitted["sigma"])
    mean = np.asarray(fitted["final_mean"], dtype=float).copy()
    covariance = np.asarray(fitted["final_covariance"], dtype=float)
    transition, process, observation = fitted["transition"], fitted["process_covariance"], fitted["observation_covariance"]
    initial_chol = _cholesky(covariance, "Filtered state covariance")
    process_chol = _cholesky(process, "Future process covariance")
    observation_chol = _cholesky(observation, "Future observation covariance")
    rng = np.random.default_rng(int(seed))
    half = draws//2
    initial = rng.standard_normal((half, 2*dimensions)) @ initial_chol.T
    states = mean[None, :]+np.concatenate([initial, -initial], axis=0)
    point_latent, draw_latent = [], []
    df = SETTINGS["student_df"]
    for _ in fitted["horizons"]:
        process_base = rng.standard_normal((half, 2*dimensions)) @ process_chol.T
        process_base *= np.sqrt((df-2)/rng.chisquare(df, size=half))[:, None]
        states = states @ transition.T+np.concatenate([process_base, -process_base], axis=0)
        observation_base = rng.standard_normal((half, dimensions)) @ observation_chol.T
        observation_base *= np.sqrt((df-2)/rng.chisquare(df, size=half))[:, None]
        observed = states[:, :dimensions]+np.concatenate([observation_base, -observation_base], axis=0)
        mean = transition @ mean
        point_latent.append(mean[:dimensions].copy())
        draw_latent.append(observed)
    point_latent = np.stack(point_latent)
    draw_latent = np.stack(draw_latent, axis=1)
    point_rates, point_counts, point_populations = _decode(point_latent, fitted)
    rate_draws, count_draws, population_draws = _decode(draw_latent, fitted)
    return {"point_rates": point_rates, "point_counts": point_counts, "point_populations": point_populations,
            "rate_draws": rate_draws, "count_draws": count_draws, "population_draws": population_draws,
            "point_latent": point_latent, "draw_latent": draw_latent}


def fit_dynamic_joint(panel, config, origin, target, outcome, seed=1301, draws=1024):
    """Fit one history-only GCC/outcome case and issue five coherent joint horizons."""
    history, years, coordinates = _validated_panel(panel, config, origin, target, outcome)
    if list(config["calendar"]["horizons"]) != [1, 2, 3, 4, 5]:
        raise ValueError("This experiment requires horizons1–5")
    basis = helmert_basis(len(config["ages"]))
    transformed, coordinate_names = _transform(history, years, config, basis)
    sigma, covariance_audit = _covariance(transformed)
    transition, process, observation, smoothing, age_smoothing = _state_matrices(sigma, basis, config["sexes"])
    fitted = _filter(transformed, sigma, transition, process, observation)
    fitted.update(settings=copy.deepcopy(SETTINGS), target=target, outcome=outcome, origin=int(origin),
                  years=years, curvature_years=years[2:], filter_update_years=years[1:],
                  sexes=list(config["sexes"]), ages=list(config["ages"]), horizons=list(config["calendar"]["horizons"]),
                  coordinate_names=coordinate_names, composition_basis=basis,
                  transformed_history=transformed, sigma=sigma, transition=transition,
                  process_covariance=process, observation_covariance=observation,
                  slope_smoothing=smoothing, age_smoothing=age_smoothing, **covariance_audit)
    output = simulate_fitted(fitted, seed=seed, draws=draws)
    eigenvalues = np.linalg.eigvalsh(sigma)
    output["coords"] = coordinates.to_frame(index=False)
    output["fitted_state"] = fitted
    output["diagnostics"] = {
        "target": target, "outcome": outcome, "origin": int(origin), "seed": int(seed),
        "settings": copy.deepcopy(SETTINGS), "history_years": years.tolist(), "last_input_year": int(years[-1]),
        "curvature_years": years[2:].tolist(), "filter_update_years": years[1:].tolist(),
        "annual_observations": int(len(years)), "annual_curvature_vectors": int(len(years)-2),
        "latent_coordinates": int(len(sigma)), "state_dimensions": int(2*len(sigma)),
        "coordinate_names": coordinate_names, "monte_carlo_draws": int(draws), "antithetic_pairs": int(draws//2),
        "draws_are_independent_temporal_blocks": False, "filter_approximation": "single_weight_robust_Kalman_update",
        "initialization": "m0=[first_observed_z,zero_slopes];P0=blockdiag(R,25Sigma);firstyear_not_updated_again",
        "transition_formula": "F=[[I,I],[0,0.95S]];first_increment_uses_current_slope",
        "observation_weight": "min(1,(5+44)/(5+innovation.T@ordinary_variance^-1@innovation));effective_R=R/weight",
        "covariance_update": "Joseph_form_then_symmetrize;no_jitter_or_eigenvalue_clipping",
        "curvature_covariance": "all_preorigin_second_differences;median_center;MAD_floor;RMS_radial_clip;recenter;correlation_shrink75%toI",
        "future_noise": "joint88D_process_t5_and_independent44D_observation_t5_eachyear;standardized_sqrt(3/chi2_5)",
        "initial_state_noise": "Gaussian_full_filtered_covariance",
        "uncertainty_scope": "estimated_level_and_slope_state_uncertainty_plus_future_process_and_observation_discrepancy",
        "uncertainty_omitted": "covariance_shape_and_hyperparameters_are_plugin;their_estimation_uncertainty_is_not_fully_propagated;not_GBD_source_uncertainty",
        "point_definition": "zero_innovation_state_path_decoded_to_counts_population_and_rates;not_predictive_median",
        "student_tail_interpretation": "fixed_heavy_tailed_model_assumption;no_exact_coverage_guarantee",
        "radial_weights": covariance_audit["radial_weights"].tolist(), "filter_weights": fitted["filter_weights"].tolist(),
        "sigma_min_eigenvalue": float(eigenvalues.min()), "sigma_max_eigenvalue": float(eigenvalues.max()),
        "sigma_condition_number": float(eigenvalues.max()/eigenvalues.min()),
        "minimum_mad_scale": float(covariance_audit["mad_scales"].min()),
        "maximum_mad_scale": float(covariance_audit["mad_scales"].max()),
        "population_consistency_rtol": 1e-8, "population_consistency_atol": 1e-6,
        "numeric_clipping_applied": False, "source_model_reused_or_modified": False,
    }
    return output
