"""Report independently audited mortality/disability analyses without model selection."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gbd_park_matplotlib')
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_local_baselines import sha, now
from run_secondary import write_json, verify_hashes, hash_files

ROLES = ['tcn_adapted', 'local_champion', 'nonneural_champion']
LABELS = {'tcn_adapted': 'Adapted TCN', 'local_champion': 'Local champion', 'nonneural_champion': 'Non-neural champion'}
COLORS = {'tcn_adapted': '#1479a6', 'local_champion': '#717980', 'nonneural_champion': '#bd7428'}
G = ['target', 'outcome', 'procedure', 'family', 'sex', 'origin', 'horizon', 'age_band']


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def table(frame, columns, decimals=4):
    lines = ['| ' + ' | '.join(columns.values()) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for _, row in frame.iterrows():
        cells = [f'{row[key]:.{decimals}f}' if isinstance(row[key], (float, np.floating)) else str(row[key]) for key in columns]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def verify_run(run, audit_name):
    manifest_path = run / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'complete'
    verify_hashes(run, manifest['output_sha256'])
    verify_hashes(ROOT, manifest['code_sha256'])
    audit_path = ROOT / 'work/supporting-validation' / (audit_name + '.json')
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] and audit['run_manifest_sha256'] == sha(manifest_path)
    assert audit['audit_code_sha256'] == sha(audit_path.with_suffix('.py'))
    for key in ['audit_helper_sha256', 'helper_sha256']:
        if key in audit:
            assert audit[key] == sha(ROOT / 'work/supporting-validation/audit_rates.py')
    config = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
    countries = [c['iso3'] for c in config['countries'] if c['gcc']]
    if audit_name == 'audit_rates':
        expected = {c + '_' + o for c in countries for o in ['deaths', 'ylds', 'ylls', 'dalys']}
        rows, key = audit['cases'], 'case'
    elif audit_name == 'audit_components':
        expected, rows, key = set(countries), audit['countries'], 'country'
    else:
        expected = {f'{c}_{o}_history{start}' for c in countries for o in ['deaths', 'ylls'] for start in [1980, 1990]}
        rows, key = audit['arms'], 'case'
    assert len(rows) == len(expected) and {row[key] for row in rows} == expected
    assert all(row['passed'] for row in rows)
    return manifest, {str(manifest_path.relative_to(ROOT)): sha(manifest_path), str(audit_path.relative_to(ROOT)): sha(audit_path)}


def summaries(points, intervals, wis):
    point_tables, interval_tables = [], []
    for band in ['45+', '80+']:
        selected = points.copy()
        bounds, weights = intervals.copy(), wis.copy()
        if band == '80+':
            ages = ['80-84', '85-89', '90-94', '95+']
            selected, bounds, weights = [x.loc[x.age.isin(ages)].copy() for x in [selected, bounds, weights]]
        for frame in [selected, bounds, weights]:
            frame['age_band'] = band
        p = selected.groupby(G, as_index=False).agg(mean_ale=('absolute_log_error', 'mean'),
            mean_mae=('absolute_rate_error', 'mean'), age_cells=('age', 'size'))
        assert p.age_cells.eq(11 if band == '45+' else 4).all()
        point_tables.append(p)
        bounds['below_lower'] = bounds.observed_value.lt(bounds.lower)
        bounds['above_upper'] = bounds.observed_value.gt(bounds.upper)
        q = bounds.groupby(G + ['scale', 'level'], as_index=False).agg(coverage=('covered', 'mean'),
            mean_width=('width', 'mean'), mean_interval_score=('interval_score', 'mean'),
            below_lower=('below_lower', 'mean'), above_upper=('above_upper', 'mean'),
            n_blocks=('n_blocks', 'first'), age_cells=('age', 'size'))
        w = weights.groupby(G + ['scale'], as_index=False).agg(mean_wis_50_80=('wis_50_80', 'mean'),
            mean_wis_50_80_95=('wis_50_80_95', 'mean'))
        q = q.merge(w, on=G + ['scale'], how='left', validate='many_to_one')
        assert q.mean_wis_50_80.notna().all()
        interval_tables.append(q)
    return pd.concat(point_tables, ignore_index=True), pd.concat(interval_tables, ignore_index=True)


def comparison(points, outcome, procedure):
    keys = ['target', 'outcome', 'family', 'sex', 'origin', 'horizon', 'age_band']
    direct = points.loc[points.outcome.eq(outcome) & points.procedure.eq('direct')]
    derived = points.loc[points.outcome.eq(outcome) & points.procedure.eq(procedure)]
    joined = direct.merge(derived, on=keys, how='inner', validate='one_to_one', suffixes=('_direct', '_derived'))
    joined['derived_minus_direct_ale'] = joined.mean_ale_derived - joined.mean_ale_direct
    joined['derived_relative_improvement_percent'] = np.where(joined.mean_ale_direct.gt(0),
        -100*joined.derived_minus_direct_ale / joined.mean_ale_direct, np.nan)
    joined['comparison'] = procedure
    return joined


def figures(out, points, component, ratio):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    for row, outcome in enumerate(['deaths', 'ylds', 'ylls', 'dalys']):
        for col, sex in enumerate(['Male', 'Female']):
            ax = axes[row, col]
            selected = points.loc[points.target.eq('Saudi Arabia') & points.outcome.eq(outcome) & points.sex.eq(sex)
                                  & points.procedure.eq('direct') & points.horizon.eq(5) & points.age_band.eq('45+')]
            for family in ROLES:
                part = selected.loc[selected.family.eq(family)].sort_values('origin')
                ax.plot(part.origin, part.mean_ale, marker='o', lw=1.4, ms=3, color=COLORS[family], label=LABELS[family])
            ax.set_title(outcome.upper() + ' — ' + sex)
            ax.set_ylabel('Mean absolute log error')
            ax.set_xticks(range(2014, 2019))
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Forecast origin (five-year horizon)')
    fig.suptitle('Saudi supporting outcomes: all ages 45+, identical rolling evaluation')
    fig.text(.5, .01, 'GBD modeled rates; five overlapping origins. Supporting results do not replace the primary prevalence result.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .03, 1, .96])
    for ext in ['png', 'svg']:
        fig.savefig(out / ('saudi_supporting_rate_errors.' + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, source, title in zip(axes, [component, ratio], ['DALYs: sum of YLD and YLL forecasts', 'YLDs: prevalence × training ratio']):
        selected = source.loc[source.origin.eq(2018) & source.horizon.eq(5) & source.age_band.eq('45+') & source.family.isin(ROLES)]
        bound = float(selected[['mean_ale_direct', 'mean_ale_derived']].max().max()) * 1.08
        for family in ROLES:
            part = selected.loc[selected.family.eq(family)]
            ax.scatter(part.mean_ale_direct, part.mean_ale_derived, color=COLORS[family], label=LABELS[family], s=30, alpha=.75)
        ax.plot([0, bound], [0, bound], '--', color='black', lw=.9)
        ax.set_xlim(0, bound)
        ax.set_ylim(0, bound)
        ax.set_xlabel('Direct-outcome absolute log error')
        ax.set_ylabel('Derived-method absolute log error')
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Six GCC countries, both sexes: 2018-origin, five-year endpoint')
    fig.text(.5, .015, 'Each point is one country × sex × method view. Below the diagonal favors the derived method.\n'
             'Champion views may use different outcome-specific source families; cells are dependent.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .09, 1, .94])
    for ext in ['png', 'svg']:
        fig.savefig(out / ('gcc_disability_accounting_comparison.' + ext), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', default='results/supporting_v1')
    parser.add_argument('--components', default='results/disability_components_v1')
    parser.add_argument('--history', default='results/mortality_history_v1')
    parser.add_argument('--output', default='reports/supporting_v1')
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError('Refusing existing supporting report')
    sources = {}
    for path, audit in [(args.core, 'audit_rates'), (args.components, 'audit_components'), (args.history, 'audit_history')]:
        _, hashes = verify_run(ROOT / path, audit)
        sources.update(hashes)
    config = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
    point_tables, interval_tables, burden_points, burden_bounds, burden_wis = [], [], [], [], []
    fallback_rows = []
    for country in config['countries']:
        if not country['gcc']:
            continue
        for outcome in ['deaths', 'ylds', 'ylls', 'dalys']:
            directory = ROOT / args.core / 'trials' / (country['iso3'] + '_' + outcome)
            points, bounds, wis = [read(directory / name) for name in ['point_scores.csv', 'interval_scores.csv', 'wis_scores.csv']]
            for frame in [points, bounds, wis]:
                frame['procedure'] = 'direct'
            a, b = summaries(points, bounds, wis)
            point_tables.append(a)
            interval_tables.append(b)
            fallback_rows.append(dict(target=country['name'], outcome=outcome, selected_fallback_cells=int(points.status.eq('fallback').sum())))
        directory = ROOT / args.components / country['iso3']
        a, b = summaries(read(directory / 'rate_point_scores.csv'), read(directory / 'rate_interval_scores.csv.gz'),
                         read(directory / 'rate_wis_scores.csv'))
        point_tables.append(a)
        interval_tables.append(b)
        burden_points.append(read(directory / 'burden_point_scores.csv'))
        burden_bounds.append(read(directory / 'burden_interval_scores.csv.gz'))
        burden_wis.append(read(directory / 'burden_wis_scores.csv'))
    points, intervals = pd.concat(point_tables, ignore_index=True), pd.concat(interval_tables, ignore_index=True)
    component, ratio = comparison(points, 'dalys', 'component_sum'), comparison(points, 'ylds', 'prevalence_training_ratio')
    mean_keys = [k for k in G if k != 'origin'] + ['scale', 'level']
    rolling = intervals.groupby(mean_keys, as_index=False).agg(coverage=('coverage', 'mean'), mean_width=('mean_width', 'mean'),
        mean_wis_50_80=('mean_wis_50_80', 'mean'), mean_wis_50_80_95=('mean_wis_50_80_95', 'mean'),
        below_lower=('below_lower', 'mean'), above_upper=('above_upper', 'mean'), n_origins=('origin', 'nunique'),
        dependent_age_origin_cells=('age_cells', 'sum'))
    assert rolling.n_origins.eq(5).all()
    bp, bb, bw = pd.concat(burden_points, ignore_index=True), pd.concat(burden_bounds, ignore_index=True), pd.concat(burden_wis, ignore_index=True)
    bg = ['target', 'outcome', 'family', 'procedure', 'population_method', 'horizon', 'measure', 'node', 'sex', 'age_group', 'unit']
    bs = bb.groupby(bg + ['level'], as_index=False).agg(coverage=('covered', 'mean'), mean_width=('width', 'mean'),
        below_lower=('below_lower', 'mean'), above_upper=('above_upper', 'mean'), n_origins=('origin', 'nunique'))
    bw = bw.groupby(bg, as_index=False).agg(mean_wis_50_80=('wis_50_80', 'mean'), mean_wis_50_80_95=('wis_50_80_95', 'mean'))
    bs = bs.merge(bw, on=bg, how='left', validate='many_to_one')
    assert bs.n_origins.eq(5).all() and bs.mean_wis_50_80.notna().all()
    endpoint = points.loc[points.origin.eq(2018) & points.horizon.eq(5) & points.age_band.eq('45+')]
    direct = endpoint.loc[endpoint.procedure.eq('direct') & endpoint.family.isin(ROLES)]
    contrasts = direct.pivot(index=['target', 'outcome', 'sex'], columns='family', values='mean_ale').reset_index()
    contrasts['tcn_better_than_both'] = contrasts.tcn_adapted.lt(contrasts.local_champion) & contrasts.tcn_adapted.lt(contrasts.nonneural_champion)
    joint = contrasts.groupby(['target', 'outcome']).tcn_better_than_both.all()
    saudi_joint = int(joint.loc['Saudi Arabia'].sum())
    saudi_table = contrasts.loc[contrasts.target.eq('Saudi Arabia')]
    coverage = rolling.loc[rolling.target.eq('Saudi Arabia') & rolling.procedure.eq('direct') & rolling.family.eq('tcn_adapted')
                           & rolling.horizon.eq(5) & rolling.level.eq(.8) & rolling.scale.eq('rate')].copy()
    coverage['coverage_percent'] = 100*coverage.coverage
    derived_coverage = rolling.loc[rolling.target.eq('Saudi Arabia') & rolling.outcome.isin(['dalys', 'ylds'])
        & rolling.family.eq('tcn_adapted') & rolling.horizon.eq(5) & rolling.level.eq(.8) & rolling.scale.eq('rate')].copy()
    derived_coverage['coverage_percent'] = 100*derived_coverage.coverage
    oldest_share_coverage = bs.loc[bs.target.eq('Saudi Arabia') & bs.family.eq('tcn_adapted') & bs.horizon.eq(5)
        & bs.level.eq(.8) & bs.age_group.eq('80+_within_45+') & bs.population_method.eq('log_trend_last8')].copy()
    oldest_share_coverage['coverage_percent'] = 100*oldest_share_coverage.coverage
    component_saudi = component.loc[component.target.eq('Saudi Arabia') & component.origin.eq(2018) & component.horizon.eq(5)
                                    & component.age_band.eq('45+') & component.family.isin(ROLES)]
    ratio_saudi = ratio.loc[ratio.target.eq('Saudi Arabia') & ratio.origin.eq(2018) & ratio.horizon.eq(5)
                            & ratio.age_band.eq('45+') & ratio.family.isin(ROLES)]
    burden_endpoint = bp.loc[bp.target.eq('Saudi Arabia') & bp.family.eq('tcn_adapted') & bp.origin.eq(2018)
                             & bp.horizon.eq(5) & bp.population_method.eq('log_trend_last8')
                             & (bp.node.eq('Both__45+') | bp.age_group.eq('80+_within_45+'))]
    histories = read(ROOT / args.history / 'history_comparisons.csv')
    history_saudi = histories.loc[histories.target.eq('Saudi Arabia') & histories.horizon.eq(5) & histories.age_group.eq('45+')]
    out.mkdir(parents=True)
    for name, frame in [('rate_error_by_origin.csv', points), ('rate_interval_by_origin.csv', intervals),
                        ('rate_interval_five_origin.csv', rolling), ('component_daly_comparison.csv', component),
                        ('prevalence_ratio_yld_comparison.csv', ratio), ('burden_interval_five_origin.csv', bs),
                        ('burden_endpoint_points.csv', bp.loc[bp.origin.eq(2018) & bp.horizon.eq(5)]),
                        ('endpoint_tcn_comparisons.csv', contrasts), ('mortality_history_comparison.csv', histories),
                        ('fallback_summary.csv', pd.DataFrame(fallback_rows))]:
        frame.to_csv(out / name, index=False)
    figures(out, points, component, ratio)
    body = f'''# Mortality and disability supporting analyses

Completed {now()[:10]}. Identical retrospective evaluation covers deaths, YLDs, YLLs and DALYs in all six GCC countries, both sexes and all eleven ages 45–49 through 95+. These 24 supporting comparisons do not replace the original Saudi prevalence primary result. The methods were specified after earlier prevalence/incidence results were inspected but before these supporting models were fitted.

At the 2018-origin/five-year endpoint, adapted TCN error was strictly lower than both comparator views for both sexes in {saudi_joint} of four Saudi supporting outcomes and {int(joint.sum())} of 24 GCC country–outcome settings. These are descriptive comparisons, not multiplicity-adjusted confirmatory tests, independent replications or clinical significance. All fourteen underlying families and both champion views remain in the numerical tables.

## Saudi direct-outcome forecasts

Mean absolute log error across eleven ages at the 2023 endpoint (lower is better):

{table(saudi_table, {'outcome':'Outcome','sex':'Sex','tcn_adapted':'Adapted TCN','local_champion':'Local champion','nonneural_champion':'Non-neural champion'})}

![Saudi supporting rate errors](saudi_supporting_rate_errors.png)

The five overlapping origins are retained individually; the final 2018 origin is counted once. Error at a single endpoint can differ from rolling performance. The [locked design](../../study_design/locked_v1/protocol.md) fixes the same grids, settings-selection cutoffs, five seeds, donor exclusions and limited adaptation as the primary analysis. New core fitting uses 1990 onward; the older mortality-history experiment below is separate. Selected fallback cells across the complete direct ledgers: {sum(x['selected_fallback_cells'] for x in fallback_rows)}. They are retained, including repeated champion views where applicable.

## Rate-interval reliability, including ages 80+

Adapted TCN nominal-80% rate intervals, horizon five, averaged over origins 2014–2018:

{table(coverage, {'outcome':'Outcome','sex':'Sex','age_band':'Ages','coverage_percent':'Coverage %','mean_width':'Mean width','mean_wis_50_80':'50/80 WIS','dependent_age_origin_cells':'Dependent cells'}, 3)}

Each 45+ summary contains 55 dependent age–origin cells; each 80+ summary contains 20. Width and WIS are in outcome-specific rate units: death events or modeled burden-years per 100,000. Do not pool those scales. Seven to eleven complete historical blocks underlie the intervals; block replication or five neural seeds does not increase epidemiological sample size. Source uncertainty bounds are not forecast intervals. Every horizon, both rate/log-rate scores, tail misses and supplementary 95% sparse-tail diagnostics are saved. No calibration method was changed in response to these results.

## DALYs: direct versus component-derived forecasts

Within each age–sex coordinate, the derived point is forecast YLD plus forecast YLL. Joint draws use the same historical residual origin across components, and quantiles are calculated after summation. Marginal medians or interval endpoints are never summed. Each component's champion can be a different independently selected family; the saved ledger retains that provenance.

Saudi five-year endpoint, equal-age absolute log error:

{table(component_saudi, {'sex':'Sex','family':'View','mean_ale_direct':'Direct DALY','mean_ale_derived':'YLD + YLL','derived_relative_improvement_percent':'Derived improvement %'})}

## YLDs: prevalence-times-training-ratio baseline

The baseline multiplies each frozen prevalence forecast by the ratio of summed YLD rates to summed prevalence rates over the previous eight years in that target/sex/age cell. Each historical baseline forecast uses its own historical ratio; its YLD errors form a separate residual bank. The outer prevalence champion's family is retained across its historical bank. This avoids using current ratios to retrospectively modify old forecasts. A baseline champion need not match the direct-YLD champion.

{table(ratio_saudi, {'sex':'Sex','family':'Prevalence view','mean_ale_direct':'Direct YLD','mean_ale_derived':'Prevalence × ratio','derived_relative_improvement_percent':'Baseline improvement %'})}

The ratio is a modeled accounting relationship, not patient severity, individual progression or survival. Improvement in this comparison would not provide independent mechanistic validation.

![GCC derived disability comparison](gcc_disability_accounting_comparison.png)

### Reliability of derived rates

Saudi adapted-TCN-view nominal-80% intervals at horizon five, over all five origins:

{table(derived_coverage, {'outcome':'Outcome','procedure':'Procedure','sex':'Sex','age_band':'Ages','coverage_percent':'Coverage %','mean_width':'Mean width','mean_wis_50_80':'50/80 WIS'}, 3)}

These comparisons retain direct and derived procedures regardless of which has smaller point error. A gain in point accuracy does not establish interval calibration; width and WIS must be assessed alongside coverage within each outcome.

## Operational burden and oldest-age shares

All direct outcomes and both derived procedures use the same preserved prevalence-implied population forecasts for burden conversion. Outcome-specific native Number/Rate denominators agree within rounding, and source DALYs equal YLDs plus YLLs on both rate and Number scales. The shared population procedure makes forecast count/burden-year component sums exact, including draw by draw. Common historical population errors are paired with the corresponding rate-error blocks. Persistence remains the population sensitivity; neither procedure uses realized future populations at issuance.

Saudi adapted-TCN-view 2023 endpoint under the operational last-eight-year population log trend:

{table(burden_endpoint, {'outcome':'Outcome','procedure':'Procedure','node':'Burden node','unit':'Unit','value':'Forecast','observed':'GBD point','signed_error':'Signed error'}, 3)}

Burden intervals and 65+/80+ shares are evaluated separately in `burden_interval_five_origin.csv`. A burden-node coverage percentage uses only five overlapping origins: one observation changes it by 20 percentage points. Coherent component totals do not by themselves establish accuracy or calibrated uncertainty. These populations and outcomes are modeled estimates; no count is presented as an independent patient enumeration or full propagation of GBD uncertainty.

Nominal-80% intervals for the fraction of each sex's 45+ burden occurring at ages 80+, adapted-TCN view under the operational population forecast, horizon five:

{table(oldest_share_coverage, {'outcome':'Outcome','procedure':'Procedure','sex':'Sex','coverage_percent':'Coverage %','mean_width':'Width (percentage points)','mean_wis_50_80':'50/80 WIS','n_origins':'Origins'}, 3)}

## Additional 1980–1989 mortality history

This fixed sensitivity expands both donor and target histories, comparing starts in 1980 and 1990 at origin 2018. Both TCN arms use the same GPU device, five seeds, fixed 16-channel/50-epoch settings and adaptation penalty 1. The same damped ETS procedure is also compared. It is an available-history comparison, not isolated target learning efficiency; it must not be compared as a matched accuracy gain against the tuned CPU core model. No intervals or new hyperparameter search were added.

Saudi five-year, equal-age absolute log errors:

{table(history_saudi, {'outcome':'Outcome','sex':'Sex','family':'Method','mean_absolute_log_error_1990':'1990 start','mean_absolute_log_error_1980':'1980 start','relative_improvement_percent':'Earlier-history improvement %'})}

Older modeled history does not necessarily represent additional independent observations. Full GCC, sex, horizon and 80+ results remain in the machine-readable history table. Negative effects remain reported.

## Verification, preservation and next stage

Core fitting, earlier-history fitting and derived accounting each have separate pre-run test gates and hashed manifests. All 24 core ledgers were committed before any core final scoring; all 24 history arms and all six derived country ledgers similarly preceded their own scoring. Independent audits reconstructed actual-outcome selections, residual banks, point/interval scores, component sums, own-ratio baselines, common population draws and representative neural/non-neural saved checkpoints. The three audit reports are linked in `validation.json`; original prevalence/incidence and calibration artifacts remain unchanged.

See the [implementation specification](../../study_design/supporting_outcomes_implementation.md) and [independent review](../../work/supporting-validation/review.md). Reproduction uses `scripts/run_supporting.py`, `scripts/run_mortality_history_sensitivity.py`, `scripts/run_disability_components.py`, the three independent audit scripts and `scripts/report_supporting.py`, all in `agpu`. Existing results require verified resumption; reports refuse overwriting.

The remaining planned work is the 15/20/29-year target-history learning curve, the separate global full-age standardized-rate benchmark, and 2023-origin disease projections for 2024–2028 under the two prepared population scenarios. No further tuning on the inspected evaluation years is introduced here.
'''
    (out / 'report.md').write_text(body)
    write_json(out / 'validation.json', dict(passed=True, created_utc=now(), audited_source_sha256=sources,
        reporter_sha256=sha(Path(__file__)), case_count=24, derived_countries=6, earlier_history_arms=24,
        artifact_sha256=hash_files(out, [p for p in out.iterdir() if p.is_file()])))
    print(str(out / 'report.md'), flush=True)


if __name__ == '__main__':
    main()
