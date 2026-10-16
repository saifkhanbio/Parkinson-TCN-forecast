"""Bayesian Gaussian age-time forecasting with a fixed discrete hyperprior.

The age covariance is diagonalized once per length scale. Each eigenmode is
a two-state local trend, giving the same posterior as the full multivariate
Kalman filter. Hyperparameters are integrated, never selected on test error.
These distributions predict GBD point estimates, not underlying true burden.
"""
from itertools import product

import numpy as np
from scipy.special import logsumexp, ndtr


def age_kernel(n_ages, lengthscale):
    distance = np.subtract.outer(np.arange(n_ages), np.arange(n_ages))
    return .8 * np.exp(-.5 * (distance / lengthscale) ** 2) + .2 * np.eye(n_ages)


def component_grid(spec):
    return np.array(list(product(spec['age_lengthscales'], spec['observation_sd'],
                                 spec['level_innovation_sd'], spec['slope_innovation_sd'],
                                 spec['damping'])), dtype=float)


def transition(mean, covariance, phi, level_variance, slope_variance):
    """Vectorized two-state prediction; leading dimensions are components/modes."""
    level, slope = mean[..., 0], mean[..., 1]
    a, b, c = covariance[..., 0, 0], covariance[..., 0, 1], covariance[..., 1, 1]
    predicted_mean = np.stack([level + phi * slope, phi * slope], axis=-1)
    predicted_covariance = np.empty_like(covariance)
    predicted_covariance[..., 0, 0] = a + 2*phi*b + phi**2*c + level_variance
    predicted_covariance[..., 0, 1] = phi*b + phi**2*c
    predicted_covariance[..., 1, 0] = predicted_covariance[..., 0, 1]
    predicted_covariance[..., 1, 1] = phi**2*c + slope_variance
    return predicted_mean, predicted_covariance


def fit(log_rates, spec, horizons=(1, 2, 3, 4, 5)):
    y = np.asarray(log_rates, dtype=float)
    if y.ndim != 2 or y.shape[0] < 8 or not np.isfinite(y).all():
        raise ValueError('Need at least eight complete finite year-by-age observations')
    if list(horizons) != list(range(1, max(horizons)+1)):
        raise ValueError('Forecast horizons must be consecutive positive integers')
    grid = component_grid(spec)
    n, ages = len(grid), y.shape[1]
    vectors, eigenvalues = np.empty((n, ages, ages)), np.empty((n, ages))
    for length in spec['age_lengthscales']:
        values, basis = np.linalg.eigh(age_kernel(ages, length))
        use = grid[:, 0] == length
        vectors[use], eigenvalues[use] = basis, values
    # U has original age on row and orthogonal mode on column.
    rotated = np.einsum('ta,kaj->ktj', y, vectors)
    obs_var = grid[:, 1, None] ** 2
    q_level = grid[:, 2, None] ** 2 * eigenvalues
    q_slope = grid[:, 3, None] ** 2 * eigenvalues
    phi = grid[:, 4, None]
    mean = np.zeros((n, ages, 2)); mean[..., 0] = rotated[:, 0, :]
    covariance = np.zeros((n, ages, 2, 2))
    covariance[..., 0, 0] = obs_var
    covariance[..., 1, 1] = .02**2 * eigenvalues
    log_likelihood = np.zeros(n)
    for t in range(1, y.shape[0]):
        mean, covariance = transition(mean, covariance, phi, q_level, q_slope)
        innovation = rotated[:, t, :] - mean[..., 0]
        variance = covariance[..., 0, 0] + obs_var
        log_likelihood -= .5*np.sum(np.log(2*np.pi*variance) + innovation**2/variance, axis=1)
        column = covariance[..., :, 0].copy()
        mean += column / variance[..., None] * innovation[..., None]
        covariance -= column[..., :, None]*column[..., None, :]/variance[..., None, None]
        covariance = .5*(covariance + covariance.swapaxes(-1, -2))
    weights = np.exp(log_likelihood - logsumexp(log_likelihood))
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid Bayesian component weights')
    forecast_means, forecast_variances = [], []
    current_mean, current_covariance = mean.copy(), covariance.copy()
    for h in horizons:
        current_mean, current_covariance = transition(current_mean, current_covariance, phi, q_level, q_slope)
        forecast_means.append(np.einsum('kaj,kj->ka', vectors, current_mean[..., 0]))
        forecast_variances.append(np.einsum('kaj,kj->ka', vectors**2, current_covariance[..., 0, 0]) + obs_var)
    return dict(grid=grid, weights=weights, log_likelihood=log_likelihood,
                vectors=vectors, eigenvalues=eigenvalues, filtered_mean=mean,
                filtered_covariance=covariance,
                means=np.stack(forecast_means, axis=1), variances=np.stack(forecast_variances, axis=1),
                horizons=list(horizons), training_years=y.shape[0], ages=ages)


def mixture_quantiles(result, probabilities):
    """Deterministic inverse mixture CDF; no Monte Carlo point/interval scores."""
    means, sd = result['means'], np.sqrt(result['variances'])
    weights = result['weights'][:, None, None]
    answers = []
    for p in probabilities:
        if not 0 < p < 1:
            raise ValueError('Quantiles must be strictly between zero and one')
        lo = np.min(means - 12*sd, axis=0)
        hi = np.max(means + 12*sd, axis=0)
        for _ in range(55):
            middle = (lo+hi)/2
            cdf = np.sum(weights*ndtr((middle[None, ...]-means)/sd), axis=0)
            lo = np.where(cdf < p, middle, lo)
            hi = np.where(cdf >= p, middle, hi)
        answers.append((lo+hi)/2)
    return np.stack(answers, axis=-1)


def joint_log_draws(result, n_draws, seed):
    """Preserve age/horizon dependence and shared component identity per path."""
    rng = np.random.default_rng(seed)
    choices = rng.choice(len(result['weights']), n_draws, p=result['weights'])
    mean = result['filtered_mean'][choices]
    covariance = result['filtered_covariance'][choices]
    cholesky = np.linalg.cholesky(covariance)
    state = mean + np.einsum('daij,daj->dai', cholesky, rng.normal(size=mean.shape))
    grid, eigenvalues = result['grid'][choices], result['eigenvalues'][choices]
    basis = result['vectors'][choices]
    phi = grid[:, 4, None]
    q_level = grid[:, 2, None] * np.sqrt(eigenvalues)
    q_slope = grid[:, 3, None] * np.sqrt(eigenvalues)
    paths = []
    for h in result['horizons']:
        old_slope = state[..., 1].copy()
        state[..., 0] += phi*old_slope + q_level*rng.normal(size=old_slope.shape)
        state[..., 1] = phi*old_slope + q_slope*rng.normal(size=old_slope.shape)
        y = np.einsum('daj,dj->da', basis, state[..., 0])
        y += grid[:, 1, None]*rng.normal(size=y.shape)
        paths.append(y)
    return np.stack(paths, axis=1)


def training_matrix(panel, target, outcome, sex, origin, ages, start=1990):
    """Discard future rows and unrelated strata before validating/using values."""
    keep = (panel.location_name.eq(target) & panel.outcome.eq(outcome)
            & panel.sex.eq(sex) & panel.year.between(start, origin))
    part = panel.loc[keep, ['year', 'age', 'rate']]
    if part.duplicated(['year', 'age']).any():
        raise ValueError('Duplicate training observations')
    matrix = part.pivot(index='year', columns='age', values='rate').reindex(
        index=range(start, origin+1), columns=ages).to_numpy()
    if len(part) != matrix.size or not np.isfinite(matrix).all() or (matrix <= 0).any():
        raise ValueError('Missing, invalid or unexpected training observations')
    return np.log(matrix)
