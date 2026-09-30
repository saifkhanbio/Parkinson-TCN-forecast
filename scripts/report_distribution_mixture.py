"""Independent replay, fixed decision gates and mixture comparison report."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from run_local_baselines import sha, now
from run_distribution_mixture import verify_preflight, read, write_json
from run_demography import STAT
from run_reliability_v1_3 import RATE_KEYS


def derived(counts, rates, config, ratio=False):
    """Replay each nonlinear endpoint without the issuance transformation."""
    if ratio:
        return rates[..., :11] / rates[..., 11:], ['Male_Female__' + a for a in config['ages']]
    values = [counts[..., i] for i in range(22)]
    nodes = [s + '__' + a for s in config['sexes'] for a in config['ages']]
    starts = [int(a.split('-')[0].rstrip('+')) for a in config['ages']]
    for j, sex in enumerate(config['sexes']):
        for group, ages in config['age_groups'].items():
            values.append(counts[..., [11*j+i for i, age in enumerate(starts) if age in ages]].sum(-1))
            nodes.append(sex + '__' + group)
    for j, sex in enumerate(config['sexes']):
        values.append(counts[..., 11*j:11*(j+1)].sum(-1)); nodes.append(sex + '__45+')
    values.append(counts.sum(-1)); nodes.append('Both__45+')
    for j, sex in enumerate(config['sexes']):
        total = counts[..., 11*j:11*(j+1)].sum(-1)
        for threshold in [65, 80]:
            values.append(100 * counts[..., [11*j+i for i, age in enumerate(starts) if age >= threshold]].sum(-1) / total)
            nodes.append(sex + '__' + str(threshold) + '+_within_45+')
    return np.stack(values, axis=-1), nodes


def score_audit(cells, wis, keys, observed):
    y, lo, hi, alpha = [cells[x].to_numpy() for x in [observed, 'lower', 'upper', 'level']]
    alpha = 1-alpha
    expected = hi-lo + 2/alpha * (np.maximum(lo-y, 0) + np.maximum(y-hi, 0))
    np.testing.assert_allclose(cells.interval_score, expected, rtol=2e-12, atol=1e-8)
    np.testing.assert_allclose(cells.width, hi-lo, rtol=2e-12, atol=1e-9)
    np.testing.assert_array_equal(cells.covered, (y >= lo) & (y <= hi))
    np.testing.assert_array_equal(cells.lower_miss, y < lo)
    np.testing.assert_array_equal(cells.upper_miss, y > hi)
    terms = cells[keys + ['level']].copy()
    terms['term'] = alpha * expected / 2
    table = terms.pivot(index=keys, columns='level', values='term')
    w = wis.set_index(keys).reindex(table.index)
    for column, levels in [('wis_50_80', [.5, .8]), ('wis_50_80_95', [.5, .8, .95])]:
        value = (.5 * abs(w[observed] - w['median']) + table[levels].sum(axis=1)) / (len(levels) + .5)
        np.testing.assert_allclose(w[column], value, rtol=2e-12, atol=1e-8)


def audit_case(directory, config, spec):
    commit = json.loads((directory / 'issued_commit.json').read_text())
    assert all(sha(directory / p) == digest for p, digest in commit['artifact_sha256'].items())
    r, b = read(directory / 'rate_intervals.csv'), read(directory / 'burden_intervals.csv')
    rp, bp = read(directory / 'rate_predictions.csv'), read(directory / 'burden_predictions.csv')
    atoms = read(directory / 'mixture_atoms.csv')
    coords = pd.MultiIndex.from_product([range(1, 6), config['sexes'], config['ages']], names=['horizon','sex','age'])
    for origin in spec['evaluation_origins']:
        data = {role: dict(np.load(directory / f'draws/origin{origin}__{role}.npz')) for role in spec['roles'] + ['mixture']}
        n = origin - 2007
        assert n in range(7, 12)
        a = atoms[atoms.origin.eq(origin)].sort_values('atom_id')
        assert len(a) == 3*n and a.atom_id.tolist() == list(range(3*n))
        assert a.role.tolist() == [role for role in spec['roles'] for _ in range(n)]
        assert a.residual_origin.tolist() == list(range(2003, origin-4))*3
        np.testing.assert_allclose(a.mass, 1/(3*n), atol=1e-15, rtol=1e-14)
        for key in data['mixture']:
            if key.startswith('point_rates'):
                expected = sum(data[role][key] for role in spec['roles']) / 3
            elif key.startswith('point_populations'):
                expected = data[spec['roles'][0]][key]
            elif key.startswith('point_counts'):
                expected = sum(data[role][key] for role in spec['roles']) / 3
            else:
                expected = np.concatenate([data[role][key] for role in spec['roles']])
            np.testing.assert_allclose(data['mixture'][key], expected, rtol=2e-12, atol=1e-9)
        for role, arrays in data.items():
            for method in spec['population_methods']:
                np.testing.assert_allclose(arrays['count_draws__'+method], arrays['rate_draws']*arrays['population_draws__'+method]/100000, rtol=2e-12, atol=1e-9)
            variants = ['cdf'] if role == 'mixture' else spec['controls']
            for variant in variants:
                family = spec['candidate'] if role == 'mixture' else role + '__' + variant
                method_q = 'inverted_cdf' if variant == 'cdf' else 'linear'
                part = r[r.origin.eq(origin) & r.family.eq(family)]
                for scale in ['rate', 'log_rate']:
                    values = arrays['rate_draws'] if scale == 'rate' else np.log(arrays['rate_draws'])
                    for level in spec['levels']:
                        sub = part[part.scale.eq(scale) & part.level.eq(level)].set_index(coords.names).reindex(coords)
                        expected = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0, method=method_q).reshape(3,-1).T
                        np.testing.assert_allclose(sub[['lower','median','upper']], expected, rtol=2e-12, atol=1e-10)
                        assert sub.n_blocks.eq(n).all() and sub.n_draws.eq(len(values)).all()
                ps = rp[rp.origin.eq(origin) & rp.family.eq(family)].set_index(coords.names).reindex(coords)
                np.testing.assert_allclose(ps.prediction, arrays['point_rates'].ravel(), rtol=2e-12, atol=1e-10)
                for method in spec['population_methods'] + ['not_applicable']:
                    ratio = method == 'not_applicable'
                    values, nodes = derived(None if ratio else arrays['count_draws__'+method], arrays['rate_draws'], config, ratio)
                    pv, _ = derived(None if ratio else arrays['point_counts__'+method], arrays['point_rates'], config, ratio)
                    index = pd.MultiIndex.from_product([range(1,6),nodes], names=['horizon','node'])
                    points = bp[bp.origin.eq(origin) & bp.family.eq(family) & bp.population_method.eq(method)].set_index(index.names).reindex(index)
                    np.testing.assert_allclose(points.value, pv.ravel(), rtol=2e-12, atol=1e-9)
                    for level in spec['levels']:
                        sub = b[b.origin.eq(origin) & b.family.eq(family) & b.population_method.eq(method) & b.level.eq(level)].set_index(index.names).reindex(index)
                        expected = np.quantile(values, [(1-level)/2,.5,(1+level)/2], axis=0, method=method_q).reshape(3,-1).T
                        np.testing.assert_allclose(sub[['lower','median','upper']], expected, rtol=2e-12, atol=1e-9)
    rc, rw = read(directory / 'rate_interval_scores.csv.gz'), read(directory / 'rate_wis_scores.csv')
    bc, bw = read(directory / 'burden_interval_scores.csv.gz'), read(directory / 'burden_wis_scores.csv')
    score_audit(rc, rw, RATE_KEYS, 'observed_value')
    score_audit(bc, bw, STAT, 'observed')
    return dict(case=directory.name, rate_intervals_verified=len(rc), burden_intervals_verified=len(bc), passed=True)


def summarize(directory, config):
    rate = read(directory / 'rate_summary.csv')
    wis = read(directory / 'rate_wis_summary.csv')
    group = ['target','outcome','role','variant','sex','horizon','scale','age_scope']
    rate = rate[rate.level.eq(.8)].groupby(group,as_index=False).agg(coverage=('coverage','mean'),mean_width=('mean_width','mean'),lower_miss=('lower_miss_rate','mean'),upper_miss=('upper_miss_rate','mean'),origins=('origin','nunique'))
    wis = wis.groupby(group,as_index=False).agg(wis=('mean_wis_50_80','mean'),wis_with_95=('mean_wis_50_80_95','mean'))
    rate = rate.merge(wis,on=group,validate='one_to_one')
    point = read(directory / 'rate_point_scores.csv')
    scopes = {'45+':config['ages'],**{g:[a for a in config['ages'] if int(a.split('-')[0].rstrip('+')) in starts] for g,starts in config['age_groups'].items()},**{'age:'+a:[a] for a in config['ages']}}
    points=[]
    pg=['target','outcome','role','variant','sex','horizon']
    for scope,ages in scopes.items():
        part=point[point.age.isin(ages)].groupby(pg+['origin'],as_index=False).absolute_log_error.mean()
        part=part.groupby(pg,as_index=False).absolute_log_error.mean().rename(columns={'absolute_log_error':'point_ale'})
        part['age_scope']=scope;points.append(part)
    rate=rate.merge(pd.concat(points,ignore_index=True),on=pg+['age_scope'],validate='many_to_one')
    rate['family']=rate.role+'__'+rate.variant
    bc,bw,bp=[read(directory/name) for name in ['burden_interval_scores.csv.gz','burden_wis_scores.csv','burden_point_scores.csv']]
    bg=[k for k in STAT if k not in ['origin','forecast_year']]
    b=bc[bc.level.eq(.8)].groupby(bg,as_index=False).agg(coverage=('covered','mean'),mean_width=('width','mean'),lower_miss=('lower_miss','mean'),upper_miss=('upper_miss','mean'),origins=('origin','nunique'))
    b=b.merge(bw.groupby(bg,as_index=False).agg(wis=('wis_50_80','mean'),wis_with_95=('wis_50_80_95','mean')),on=bg,validate='one_to_one')
    b=b.merge(bp.groupby(bg,as_index=False).agg(point_mae=('absolute_error','mean')),on=bg,validate='one_to_one')
    return rate,b


def decision_gates(rates, burdens, spec):
    """Apply locked numerical guardrails; no score-driven selection or weights."""
    r=rates[rates.horizon.eq(5)];b=burdens[burdens.horizon.eq(5)]
    a=spec['assessment']; ref=a['primary_reference'];candidate=spec['candidate'];rows=[]
    def gate(name, endpoint, metric, actual, reference, limit=None, comparator=ref):
        if limit is None:
            passed=actual<=reference+1e-12;ratio=None
        else:
            passed=(actual==0) if reference==0 else actual<=limit*reference+1e-12
            ratio=actual/reference if reference else None
        rows.append(dict(gate=name,endpoint=endpoint,metric=metric,candidate_value=float(actual),reference_value=float(reference),maximum_ratio=limit,observed_ratio=ratio,comparator=comparator,passed=bool(passed)))
    for sex in ['Male','Female']:
        for scope in ['45+','80+']:
            part=r[r.sex.eq(sex)&r.age_scope.eq(scope)&r.scale.eq('rate')].set_index('family')
            mix,old=part.loc[candidate],part.loc[ref];endpoint=sex+' '+scope
            limit=a['rate_wis_ratio_max_each_sex_45plus' if scope=='45+' else 'rate_wis_ratio_max_each_sex_80plus']
            gate('rate_wis',endpoint,'WIS',mix.wis,old.wis,limit)
            gate('coverage_distance',endpoint,'abs(coverage80-0.8)',abs(mix.coverage-.8),abs(old.coverage-.8))
            gate('width',endpoint,'width80',mix.mean_width,old.mean_width,a['width_ratio_max_each_sex_45plus_and_80plus'])
            if scope=='45+':
                for family in ['local_champion__cdf','nonneural_champion__cdf']:
                    gate('comparator_wis',endpoint,'WIS',mix.wis,part.loc[family].wis,a['rate_wis_ratio_max_vs_each_other_component_each_sex_45plus'],family)
                gate('point_accuracy',endpoint,'ALE',mix.point_ale,old.point_ale,a['point_ale_ratio_max_each_sex_45plus'])
                log=r[r.sex.eq(sex)&r.age_scope.eq(scope)&r.scale.eq('log_rate')].set_index('family')
                gate('log_wis',endpoint,'log_WIS',log.loc[candidate].wis,log.loc[ref].wis,a['log_wis_ratio_max_each_sex_45plus'])
    for node in a['guarded_count_nodes']+a['guarded_share_nodes']+['sex_rate_ratio_mean_11_ages']:
        part=b[b.population_method.eq(spec['primary_population_method'])&b.node.eq(node)] if node!='sex_rate_ratio_mean_11_ages' else b[b.measure.eq('sex_rate_ratio')].groupby('family',as_index=False)[['wis','coverage','mean_width']].mean()
        part=part.set_index('family');mix,old=part.loc[candidate],part.loc[ref]
        gate('derived_wis',node,'WIS',mix.wis,old.wis,a['derived_wis_ratio_max_each_guarded_endpoint'])
        gate('derived_coverage_distance',node,'abs(coverage80-0.8)',abs(mix.coverage-.8),abs(old.coverage-.8))
        gate('derived_width',node,'width80',mix.mean_width,old.mean_width,a['derived_width_ratio_max_each_guarded_endpoint'])
    return pd.DataFrame(rows)


def markdown_table(frame, columns):
    def cell(value):
        if pd.isna(value): return '—'
        return f'{value:.3f}' if isinstance(value,(float,np.floating)) else str(value)
    lines=['| '+' | '.join(label for _,label in columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(cell(row[key]) for key,_ in columns)+' |')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='results/distribution_mixture_v1');parser.add_argument('--output',default='reports/distribution_mixture_v1');args=parser.parse_args()
    source,out=ROOT/args.input,ROOT/args.output
    if out.exists():raise SystemExit('Refusing to replace a mixture report')
    manifest=json.loads((source/'run_manifest.json').read_text());assert manifest['status']=='complete'
    assert all(sha(source/p)==digest for p,digest in manifest['output_sha256'].items())
    preflight=json.loads((source/'preflight.json').read_text());verify_preflight(preflight)
    assert sha(Path(__file__))==preflight['code_sha256'][str(Path(__file__).relative_to(ROOT))]
    config=json.loads((ROOT/'study_design/locked_v1/design.json').read_text());spec=json.loads((ROOT/'study_design/distribution_mixture_v1.json').read_text())
    tasks=json.loads((source/'cases.json').read_text());audits=[];rates=[];burdens=[];gates=[]
    for case in tasks:
        directory=source/case['id'];audits.append(audit_case(directory,config,spec))
        r,b=summarize(directory,config);rates.append(r);burdens.append(b)
        g=decision_gates(r,b,spec);g['target']=case['target'];g['outcome']=case['outcome'];gates.append(g)
        print('Audited: '+case['id'],flush=True)
    out.mkdir(parents=True)
    rates,burdens,gates=map(lambda x:pd.concat(x,ignore_index=True),(rates,burdens,gates))
    rates.to_csv(out/'rate_comparison.csv',index=False);burdens.to_csv(out/'burden_comparison.csv',index=False);gates.to_csv(out/'decision_gates.csv',index=False)
    decisions=gates.groupby(['target','outcome'],as_index=False).agg(passed=('passed','all'),gates_passed=('passed','sum'),gates_total=('passed','size'))
    decisions.to_csv(out/'decisions.csv',index=False)
    key=['target','outcome','role','sex','horizon','scale','age_scope']
    metrics=['coverage','mean_width','wis','point_ale']
    effects=rates[rates.variant.eq('original')][key+metrics].merge(rates[rates.variant.eq('cdf')][key+metrics],on=key,suffixes=('_linear','_cdf'),validate='one_to_one')
    for metric in metrics:effects[metric+'_change']=effects[metric+'_cdf']-effects[metric+'_linear']
    effects.to_csv(out/'quantile_convention_effect.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families=['tcn_adapted__cdf','local_champion__cdf','nonneural_champion__cdf',spec['candidate']]
    colors=['#3973ac','#829f43','#b87c45','#9f426e'];names=['TCN','Local','Pooled','Mixture']
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
    for i,outcome in enumerate(spec['outcomes']):
        for j,scope in enumerate(['45+','80+']):
            ax=axes[i,j];part=rates[rates.target.eq('Saudi Arabia')&rates.outcome.eq(outcome)&rates.age_scope.eq(scope)&rates.scale.eq('rate')&rates.horizon.eq(5)]
            for k,(family,color,label) in enumerate(zip(families,colors,names)):
                rows=part[part.family.eq(family)].set_index('sex').reindex(['Male','Female'])
                ax.bar(np.arange(2)+(k-1.5)*.18,100*rows.coverage,width=.17,color=color,label=label)
            ax.axhline(80,color='black',linestyle='--',linewidth=1);ax.set_xticks([0,1],['Male','Female']);ax.set_ylim(0,105)
            ax.set_title(outcome.title()+' | ages '+scope);ax.set_ylabel('80% interval coverage (%)')
    axes[0,0].legend(ncol=4,fontsize=8,loc='lower center',bbox_to_anchor=(1.05,1.12));fig.tight_layout()
    fig.savefig(out/'saudi_coverage.png',dpi=180);fig.savefig(out/'saudi_coverage.svg');plt.close(fig)
    primary=decisions[decisions.target.eq('Saudi Arabia')&decisions.outcome.eq('prevalence')].iloc[0]
    table=rates[rates.target.eq('Saudi Arabia')&rates.horizon.eq(5)&rates.scale.eq('rate')&rates.age_scope.eq('45+')&rates.family.isin(families)].copy()
    table['coverage_pct']=table.coverage*100
    older=rates[rates.target.eq('Saudi Arabia')&rates.horizon.eq(5)&rates.scale.eq('rate')&rates.age_scope.eq('80+')&rates.family.isin(['tcn_adapted__cdf',spec['candidate']])].copy();older['coverage_pct']=older.coverage*100
    endpoints=burdens[burdens.target.eq('Saudi Arabia')&burdens.horizon.eq(5)&burdens.family.isin(['tcn_adapted__cdf',spec['candidate']])&burdens.population_method.eq(spec['primary_population_method'])&burdens.node.isin(spec['assessment']['guarded_count_nodes']+spec['assessment']['guarded_share_nodes'])].copy();endpoints['coverage_pct']=100*endpoints.coverage
    failed=gates[gates.target.eq('Saudi Arabia')&gates.outcome.eq('prevalence')&~gates.passed]
    report=f'''# Equal-weight predictive distribution mixture

Completed {now()}. The Saudi prevalence candidate **{'passes' if primary.passed else 'does not pass'} the locked exploratory decision rule**: {int(primary.gates_passed)} of {int(primary.gates_total)} guards passed. This is a numerical research decision, not a significance test or a claim of adequate calibration. The original primary transfer-learning verdict is unchanged.

## Experiment and validation

One equal-weight mixture combines the three frozen original model roles: local statistical champion, pooled non-neural champion and adapted TCN. All six GCC countries, prevalence/incidence, both sexes, eleven ages 45–49 through 95+, origins 2014–2018 and horizons 1–5 were evaluated. No model was refitted. All twelve cases were committed before new scoring.

The mixture enumerates 21–33 complete role/origin trajectories, based on only 7–11 distinct overlapping historical origins. Role selection applies to the whole age–sex–horizon trajectory; paired population errors and coherent count transformations are preserved. An independent audit replayed every component/mixture quantile, count/share/ratio transformation and interval/WIS calculation. Source and frozen-result hashes remained unchanged.

The main comparison uses inverse empirical CDF quantiles for both the mixture and its component controls. Archived linear-quantile controls were also reproduced. Their difference is saved in [quantile_convention_effect.csv](quantile_convention_effect.csv); coverage changes from that numerical convention are not credited to model pooling. Marginal source bounds and UN scenarios were not added to forecast distributions.

## Saudi five-year rate results

Coverage is for nominal 80% intervals. WIS uses the 50% and 80% intervals; lower is better. Width and WIS are in rate units per 100,000. Age cells are weighted equally within an origin and the five origins equally. All rows below use the same quantile rule.

{markdown_table(table,[('outcome','Outcome'),('sex','Sex'),('family','Procedure'),('coverage_pct','Coverage %'),('mean_width','Width'),('wis','WIS'),('point_ale','Point ALE')])}

![Saudi rate coverage](saudi_coverage.png)

## Ages 80+ remain explicit

The four oldest groups (80–84, 85–89, 90–94 and 95+) remain included. Their individual results and all other ages/horizons are retained in [rate_comparison.csv](rate_comparison.csv).

{markdown_table(older,[('outcome','Outcome'),('sex','Sex'),('family','Procedure'),('coverage_pct','Coverage %'),('mean_width','Width'),('wis','WIS')])}

## Derived burden

These results use the unchanged eight-year population log-trend procedure. Counts concern ages 45+; shares concern the sex-specific 80+ burden within 45+. Count widths/WIS are numbers; share widths/WIS are percentage points. Each displayed endpoint has only five dependent origin observations. Persistence and all count nodes/sex-rate ratios are retained in [burden_comparison.csv](burden_comparison.csv), and the ratio guard is included in the decision ledger.

{markdown_table(endpoints,[('outcome','Outcome'),('node','Endpoint'),('family','Procedure'),('coverage_pct','Coverage %'),('mean_width','Width'),('wis','WIS')])}

## Fixed decision and GCC benchmarking

The [locked specification](../../study_design/distribution_mixture_v1.md) requires at least 5% lower 45+ rate WIS than matched TCN in each sex, comparator/point/log-score guards, no worsened coverage distance from 80%, explicit oldest-age/derived-burden guards, and a twofold width ceiling. Thresholds were fixed before execution and are exploratory decision criteria. No weights or procedures were changed after scoring.

{markdown_table(decisions,[('target','Country'),('outcome','Outcome'),('passed','All guards pass'),('gates_passed','Passed'),('gates_total','Total')])}

Failed Saudi prevalence guards are listed below. A failure does not imply every endpoint worsened; it prevents presenting an aggregate coverage gain as an unqualified improvement.

{markdown_table(failed,[('gate','Guard'),('endpoint','Endpoint'),('metric','Metric'),('candidate_value','Mixture'),('reference_value','Reference'),('maximum_ratio','Maximum ratio'),('comparator','Comparator')]) if len(failed) else 'No Saudi prevalence guard failed.'}

## Interpretation and stopping rule

{'The candidate is promising under the fixed rule, but prospective or otherwise unexamined validation is still needed.' if primary.passed else 'Retain the mixture as an exploratory sensitivity; it does not qualify for replacing the reference procedure under the fixed rule.'} Do not tune new weights, omit difficult ages, or start another calibration grid from these results. Assess mixed sex/outcome and derived-burden findings separately. All candidate evaluation years had already been inspected during previous work, so this is not an untouched test.

Population-source differences, absent joint GBD/UN draws and limited temporal replication remain. More mixture atoms do not create independent observations. These are conditional retrospective forecasts of modeled GBD age-specific outcomes, not total uncertainty about clinical disease burden or validated cross-release stability. The new ASR export verifies a different, full-age estimand and does not supply uncertainty draws for this primary analysis.

## Reproduction

Run `scripts/run_distribution_mixture.py` in `agpu_env` with `--workers 12` and a new output directory; run this report script with matching `--input` and a new `--output`. Existing output directories are protected against replacement. [Decision gates](decision_gates.csv), [run manifest](../../results/distribution_mixture_v1/run_manifest.json), [report validation](validation.json) and the per-case committed trajectory archives retain all evidence.
'''
    (out/'report.md').write_text(report)
    verify_preflight(preflight)
    write_json(out/'validation.json',{'created_utc':now(),'passed':True,'independent_audits':audits,'all_cases_committed_before_scoring':True,'protected_files_unchanged':True,'source_manifest_sha256':sha(source/'run_manifest.json'),'report_code_sha256':sha(Path(__file__)),'primary_guard_pass':bool(primary.passed),'models_fitted':0,'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps({'primary_guard_pass':bool(primary.passed),'primary_guards_passed':int(primary.gates_passed),'primary_guards_total':int(primary.gates_total),'cases':len(tasks)},indent=2))


if __name__=='__main__':
    if not __debug__:raise SystemExit('Run without -O.')
    main()
