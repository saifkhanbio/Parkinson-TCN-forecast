"""Fixed-role pooling of complete conditional predictive trajectories."""
import numpy as np


def inverse_ecdf(values, probabilities):
    """Uniform discrete inverse CDF; replication invariant, no tail extension."""
    values = np.asarray(values, dtype=float)
    q = np.asarray(probabilities, dtype=float)
    if values.ndim < 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Need finite nonempty draws on axis zero')
    if q.ndim != 1 or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError('Quantile probabilities must be a vector in [0, 1]')
    rank = np.maximum(1, np.ceil(q * len(values) - 1e-12)).astype(int) - 1
    return np.sort(values, axis=0)[rank]


def validate_residual_origins(origins, origin):
    values = list(origins)
    if values != list(range(2003, origin - 4)) or len(values) < 5:
        raise ValueError('Need complete, distinct and matured historical origins')
    return len(values)


def pool_trajectories(components, roles, methods, residual_origins, origin):
    """Enumerate role/origin atoms; preserve shared population coupling."""
    if len(roles) != 3 or len(set(roles)) != 3 or set(components) != set(roles):
        raise ValueError('Exactly the three frozen roles are required')
    n = validate_residual_origins(residual_origins, origin)
    fields = {'point_rates', 'rate_draws'}
    fields.update(prefix + '__' + method for method in methods for prefix in
                  ['point_populations', 'population_draws', 'point_counts', 'count_draws'])
    shape = np.asarray(components[roles[0]]['point_rates']).shape
    if len(shape) != 2:
        raise ValueError('Point coordinates must be horizon by sex-age')
    reference = components[roles[0]]
    for role in roles:
        component = components[role]
        if set(component) != fields:
            raise ValueError('Component array fields differ from declared population methods')
        for name, value in component.items():
            value = np.asarray(value)
            expected = shape if name.startswith('point_') else (n,) + shape
            if value.shape != expected or not np.isfinite(value).all() or (value <= 0).any():
                raise ValueError('Invalid component grid: ' + role + '/' + name)
        for method in methods:
            for prefix in ['point_', '']:
                rate = component['point_rates' if prefix else 'rate_draws']
                pop = component[('point_populations' if prefix else 'population_draws') + '__' + method]
                counts = component[('point_counts' if prefix else 'count_draws') + '__' + method]
                if not np.allclose(counts, rate * pop / 100000, rtol=2e-12, atol=1e-10):
                    raise ValueError('Rate/population pairing does not reproduce counts')
                key = ('point_populations' if prefix else 'population_draws') + '__' + method
                if not np.allclose(pop, reference[key], rtol=2e-12, atol=1e-10):
                    raise ValueError('Component population trajectories do not share origins')
    result = {'point_rates': np.mean([components[r]['point_rates'] for r in roles], axis=0),
              'rate_draws': np.concatenate([components[r]['rate_draws'] for r in roles])}
    for method in methods:
        result['point_populations__' + method] = reference['point_populations__' + method].copy()
        result['point_counts__' + method] = result['point_rates'] * result['point_populations__' + method] / 100000
        for field in ['population_draws', 'count_draws']:
            key = field + '__' + method
            result[key] = np.concatenate([components[r][key] for r in roles])
    atoms = [{'atom_id': j * n + i, 'role': role, 'residual_origin': residual_origin,
              'mass': 1.0 / (3 * n)} for j, role in enumerate(roles)
             for i, residual_origin in enumerate(residual_origins)]
    return result, atoms
