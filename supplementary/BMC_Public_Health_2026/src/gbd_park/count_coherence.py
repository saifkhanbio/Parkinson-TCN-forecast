"""Native-count ETS and origin-restricted nonnegative reconciliation."""
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from gbd_park.demography import hierarchy
from gbd_park.local import forecast_setting


def node_history(panel, config, target, outcome, maximum_year):
    matrix, nodes, bottom = hierarchy(config)
    source = panel.loc[panel.location_name.eq(target) & panel.outcome.eq(outcome)
                       & panel.year.between(config['calendar']['history_start'], maximum_year)]
    if source.duplicated(['year', 'sex', 'age']).any():
        raise ValueError('Duplicate native-count history')
    pivot = source.pivot(index='year', columns=['sex', 'age'], values='count').reindex(
        index=range(config['calendar']['history_start'], maximum_year+1),
        columns=pd.MultiIndex.from_tuples(bottom))
    if not np.isfinite(pivot.to_numpy()).all() or (pivot <= 0).any().any():
        raise ValueError('Incomplete or nonpositive native-count history')
    values = pivot.to_numpy() @ matrix.T
    return pd.DataFrame(values, index=pivot.index, columns=nodes.node)


def independent_counts(panel, config, origin, target, outcome):
    history = node_history(panel, config, target, outcome, origin)
    working = history.rename_axis('year').reset_index().melt(id_vars='year', var_name='age', value_name='rate')
    working['location_name'], working['sex'], working['outcome'] = target, 'Nodes', 'prevalence'
    copied = deepcopy(config)
    copied['ages'] = list(history.columns)
    spec = {'family': 'damped_ets', 'setting_id': 'damped_ets', 'setting': {}, 'grid_order': 0}
    rows, audit, _ = forecast_setting(working, copied, origin, 'Nodes', spec, target=target)
    result = pd.DataFrame(rows).rename(columns={'age': 'node', 'prediction': 'count_prediction'})
    result = result.drop(columns=['sex', 'log_prediction'])
    result['outcome'], result['family'] = outcome, 'independent'
    result = result.merge(hierarchy(config)[1], on='node', validate='many_to_one')
    return result, audit


def reconciliation_weights(historical_scores, training_history, config, origin, horizon):
    nodes = list(hierarchy(config)[1].node)
    part = historical_scores.loc[historical_scores.horizon.eq(horizon)
                                 & (historical_scores.origin+5).le(origin)]
    if part.duplicated(['origin', 'node']).any():
        raise ValueError('Duplicate count residual coordinates')
    pivot = part.pivot(index='origin', columns='node', values='count_residual').reindex(columns=nodes)
    if len(pivot) and not np.isfinite(pivot.to_numpy()).all():
        raise ValueError('Count residual blocks must be jointly complete')
    if training_history.index.max() > origin:
        raise ValueError('Future training-node mean')
    floor = 1e-8 * training_history[nodes].mean().to_numpy()**2
    if len(pivot) < config['coherence']['minimum_residual_blocks']:
        variance = np.ones(len(nodes))
        status = 'equal_weight_fallback'
    else:
        errors = pivot.to_numpy()
        variance = np.maximum(np.mean((errors-errors.mean(axis=0))**2, axis=0), floor)
        status = 'ok'
    return variance, {'weight_status': status, 'residual_blocks': len(pivot),
                      'last_residual_label_year': int(pivot.index.max()+5) if len(pivot) else None,
                      'minimum_variance': float(variance.min())}


def reconcile_counts(independent, variance, config):
    matrix, _, _ = hierarchy(config)
    values, weights = np.asarray(independent, float), np.asarray(variance, float)
    if values.shape != (31,) or weights.shape != (31,) or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any() or (values < 0).any():
        raise ValueError('Need 31 finite nonnegative predictions and positive variances')
    bottom_up = matrix @ values[:22]
    root = np.sqrt(weights)
    bottom, _ = nnls(matrix/root[:, None], values/root, maxiter=1000)
    reconciled = matrix @ bottom
    return {'independent': values, 'bottom_up': bottom_up, 'nonnegative_diagonal_wls': reconciled}
