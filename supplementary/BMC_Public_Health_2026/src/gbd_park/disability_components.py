"""Prespecified accounting forecasts; no model fitting or final-period selection."""
import numpy as np
import pandas as pd

from .intervals import build_residual_bank, apply_bank
from .prequential import champion_history
from .secondary import actual_truth
from .scoring import score_forecasts

KEYS = ['target', 'origin', 'family', 'sex', 'age', 'horizon', 'forecast_year']
ROLES = ['tcn_adapted', 'local_champion', 'nonneural_champion']


def check_grid(frame, config, draw=False):
    keys = KEYS + (['residual_origin'] if draw else [])
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError('Empty or duplicate forecast coordinates')
    if not frame.forecast_year.eq(frame.origin + frame.horizon).all():
        raise ValueError('Forecast year disagrees with origin and horizon')
    expected = {(s, a, h) for s in config['sexes'] for a in config['ages'] for h in config['calendar']['horizons']}
    group = ['target', 'origin', 'family'] + (['residual_origin'] if draw else [])
    for _, part in frame.groupby(group):
        if len(part) != len(expected) or set(map(tuple, part[['sex', 'age', 'horizon']].to_numpy())) != expected:
            raise ValueError('Incomplete or unexpected age-sex-horizon grid')
    if draw:
        for (_, origin, _), part in frame.groupby(['target', 'origin', 'family']):
            wanted = set(range(config['intervals']['residual_first_origin'], int(origin)-max(config['calendar']['horizons'])+1))
            if set(part.residual_origin) != wanted:
                raise ValueError('Incomplete or future joint residual-block origins')


def training_ratios(panel, config, target, origin):
    """Fixed ratio of sums over eight complete years, separately by sex and age."""
    selected = panel.loc[panel.location_name.eq(target) & panel.outcome.isin(['prevalence', 'ylds'])
                         & panel.year.between(origin-7, origin)].copy()
    keys = ['sex', 'age', 'year', 'outcome']
    expected = {(s, a, y, o) for s in config['sexes'] for a in config['ages']
                for y in range(origin-7, origin+1) for o in ['prevalence', 'ylds']}
    if selected.duplicated(keys).any() or set(map(tuple, selected[keys].to_numpy())) != expected:
        raise ValueError('Ratio requires eight complete unique years for each age, sex and outcome')
    if not np.isfinite(selected.rate).all() or not selected.rate.gt(0).all():
        raise ValueError('Training ratios require finite positive rates')
    sums = selected.groupby(['sex', 'age', 'outcome']).rate.sum().unstack()
    result = (sums.ylds / sums.prevalence).rename('training_ratio').reset_index()
    result['ratio_history_start'], result['ratio_history_end'] = origin-7, origin
    result['target'], result['origin'] = target, origin
    return result


def prevalence_yld_predictions(frame, panel, config, target):
    if not frame.target.eq(target).all() or not frame.outcome.eq('prevalence').all():
        raise ValueError('YLD baseline needs the specified target prevalence forecasts')
    if 'observed_rate' in frame:
        raise ValueError('Forecast ledger must exclude verification outcomes')
    check_grid(frame, config)
    if not np.isfinite(frame.prediction).all() or not frame.prediction.gt(0).all():
        raise ValueError('Invalid prevalence forecasts')
    ratios = pd.concat([training_ratios(panel, config, target, int(origin)) for origin in sorted(frame.origin.unique())], ignore_index=True)
    result = frame.merge(ratios, on=['target', 'origin', 'sex', 'age'], how='left', validate='many_to_one')
    result['prevalence_prediction'] = result.prediction
    result['prediction'] = result.prevalence_prediction * result.training_ratio
    result['log_prediction'] = np.log(result.prediction)
    result['outcome'] = 'ylds'
    result['setting_id'] = 'prevalence_ratio8__' + result.setting_id.astype(str)
    result['procedure'] = 'prevalence_training_ratio'
    return result


def sum_daly(ylds, ylls, config, draw=False):
    """Add rates only after exact pairing; preserve component source families."""
    for frame, outcome in [(ylds, 'ylds'), (ylls, 'ylls')]:
        if not frame.outcome.eq(outcome).all():
            raise ValueError('Unexpected component outcome')
        check_grid(frame, config, draw=draw)
    keys = KEYS + (['residual_origin'] if draw else [])
    field = 'rate_draw' if draw else 'prediction'
    left, right = ylds.copy(), ylls.copy()
    for frame in [left, right]:
        if not np.isfinite(frame[field]).all() or not frame[field].gt(0).all():
            raise ValueError('Components require finite positive rates')
        if 'source_family' not in frame:
            frame['source_family'] = frame.family
        else:
            frame['source_family'] = frame.source_family.fillna(frame.family)
    joined = left.merge(right, on=keys, how='outer', suffixes=('_yld', '_yll'), validate='one_to_one', indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('Component forecast coordinates or block origins differ')
    result = joined[keys].copy()
    result['outcome'], result['procedure'] = 'dalys', 'component_sum'
    result['source_family_yld'] = joined.source_family_yld
    result['source_family_yll'] = joined.source_family_yll
    result['prediction'] = joined.prediction_yld + joined.prediction_yll
    result['log_prediction'] = np.log(result.prediction)
    result['setting_id'] = 'sum_independently_selected_components'
    if draw:
        result['rate_draw'] = joined.rate_draw_yld + joined.rate_draw_yll
        result['log_draw'] = np.log(result.rate_draw)
    else:
        status_left = joined.get('status_yld', pd.Series('ok', index=joined.index))
        status_right = joined.get('status_yll', pd.Series('ok', index=joined.index))
        result['status'] = np.where(status_left.eq('ok') & status_right.eq('ok'), 'ok', 'component_fallback')
    return result


def intervals_from_draws(points, draws, config):
    """Quantile after component summation, separately on rate and log scales."""
    check_grid(points, config)
    check_grid(draws, config, draw=True)
    if not np.isfinite(draws.rate_draw).all() or not draws.rate_draw.gt(0).all():
        raise ValueError('Invalid joint rate draws')
    keys = KEYS + ['outcome']
    point = points.set_index(keys)
    frames = []
    work = draws.copy()
    work['log_draw'] = np.log(work.rate_draw)
    for scale, field, point_field in [('rate', 'rate_draw', 'prediction'), ('log_rate', 'log_draw', 'log_prediction')]:
        group = work.groupby(keys)[field]
        q = group.quantile([.025, .1, .25, .5, .75, .9, .975], interpolation='linear').unstack()
        if set(q.index) != set(point.index):
            raise ValueError('Joint draws do not match issued points')
        n = group.size()
        if n.min() < config['intervals']['minimum_blocks_to_emit']:
            raise ValueError('Insufficient historical blocks for derived interval')
        for level, lo, hi in [(.5, .25, .75), (.8, .1, .9), (.95, .025, .975)]:
            frame = pd.DataFrame({'lower': q[lo], 'median': q[.5], 'upper': q[hi], 'n_blocks': n,
                                  'point_prediction': point[point_field].reindex(q.index)}).reset_index()
            frame['scale'], frame['level'], frame['status'] = scale, level, 'ok'
            frame['procedure'] = 'component_sum'
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def issue_ratio_forecasts(history_prevalence, points_prevalence, mappings, panel, config, target):
    """Own baseline residuals; historical ratios and outer prevalence-family mapping."""
    history = prevalence_yld_predictions(history_prevalence, panel, config, target)
    truth = actual_truth(panel, target, 'ylds', 2018)
    scored = score_forecasts(history, truth, config['ages'], config['calendar']['horizons'], 2018)
    selected = points_prevalence.loc[points_prevalence.family.isin(ROLES)].copy()
    points = prevalence_yld_predictions(selected, panel, config, target)
    interval_frames, draw_frames, banks = [], [], {}
    for origin in config['calendar']['reliability_origins']:
        for role in ROLES:
            if role == 'tcn_adapted':
                source = scored
                by_sex = {sex: role for sex in config['sexes']}
            else:
                selected_map = mappings.loc[mappings.fit_origin.eq(origin) & mappings.role.eq(role)]
                if len(selected_map) != len(config['sexes']) or selected_map.sex.duplicated().any():
                    raise ValueError('Missing or duplicated outer prevalence champion mapping')
                if selected_map.last_selection_target_year.gt(origin).any():
                    raise ValueError('Prevalence champion uses future selection outcomes')
                by_sex = selected_map.set_index('sex').source_family.to_dict()
                source = champion_history(scored, config, by_sex, role)
            bank = build_residual_bank(source, config, int(origin), role)
            bank['prevalence_source_family_by_sex'] = by_sex
            current = points.loc[points.origin.eq(origin) & points.family.eq(role)]
            intervals, draws = apply_bank(current, bank, config)
            for frame in [intervals, draws]:
                frame['procedure'] = 'prevalence_training_ratio'
                frame['prevalence_source_family'] = frame.sex.map(by_sex)
            interval_frames.append(intervals)
            draw_frames.append(draws)
            banks[(int(origin), role)] = bank
    return points, pd.concat(interval_frames, ignore_index=True), pd.concat(draw_frames, ignore_index=True), banks, history


def source_accounting(panel, config):
    """Check modeled point identities and common implied denominators, not bounds."""
    selected = panel.loc[panel.year.between(1990, 2023)].copy()
    keys = ['location_name', 'sex', 'age', 'year']
    if selected.duplicated(keys + ['outcome']).any():
        raise ValueError('Duplicate source outcome cells')
    result = selected[keys].drop_duplicates().copy()
    for field in ['rate', 'count']:
        table = selected.pivot(index=keys, columns='outcome', values=field)
        if not np.isfinite(table).all().all() or not table.gt(0).all().all():
            raise ValueError('Missing or invalid source point')
        difference = table.dalys - table.ylds - table.ylls
        np.testing.assert_allclose(table.dalys, table.ylds + table.ylls, rtol=1e-10, atol=1e-10)
        result = result.merge((difference / table.dalys).rename(field + '_relative_identity_error').reset_index(),
                              on=keys, validate='one_to_one')
    selected['population'] = selected['count'] / selected.rate * 100000
    table = selected.pivot(index=keys, columns='outcome', values='population')
    relative = table.div(table.prevalence, axis=0) - 1
    if relative.abs().to_numpy().max() > 1e-10:
        raise ValueError('Outcome-implied population denominators differ beyond rounding')
    result = result.merge(relative.abs().max(axis=1).rename('maximum_relative_population_difference').reset_index(),
                          on=keys, validate='one_to_one')
    return result
