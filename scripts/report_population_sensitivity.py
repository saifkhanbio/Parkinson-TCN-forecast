"""Report audited population-source diagnostics without selecting a new model."""
import os
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from run_population_sensitivity import sha, now, read, write_json

LABELS = {'log_trend_last8': 'Operational log trend', 'persistence': 'Operational persistence',
          'gbd_realized_oracle': 'Realized GBD population (oracle)', 'un_2024_unaligned': 'UN 2024 unaligned',
          'un_2024_origin_aligned': 'UN 2024 origin-aligned growth'}
ORDER = list(LABELS)


def md(frame, columns, digits=2):
    header = '| ' + ' | '.join(columns.values()) + ' |'
    lines = [header, '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        values = []
        for key in columns:
            value = row[key]
            values.append(f'{value:,.{digits}f}' if isinstance(value, (float, np.floating)) else str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def figures(out, national, interval_endpoint):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    groups = ['45-64', '65-79', '80+', '45+']
    for ax, sex in zip(axes, ['Male', 'Female']):
        table = national.loc[national.outcome.eq('prevalence') & national.sex.eq(sex)].set_index('age_group').loc[groups]
        x = np.arange(4)
        ax.bar(x-.18, table.gbd_minus_gastat_percent, .36, color='#1479a6', label='Implied GBD')
        ax.bar(x+.18, table.un_minus_gastat_percent, .36, color='#c58135', label='UN WPP 2024')
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks(x, groups)
        ax.set_title(sex)
    axes[0].set_ylabel('Population difference from GASTAT (%)')
    axes[1].legend(frameon=False)
    fig.suptitle('Saudi population-source comparison, 2023; residents aged 45+')
    fig.text(.5, .015, 'Descriptive source differences; GBD reference-date comparability unverified.\n'
             'National 80+ group retained. Prevalence-implied GBD denominator shown.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .1, 1, .94])
    for ext in ['png', 'svg']:
        fig.savefig(out / ('saudi_population_sources_2023.' + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    short = ['Operational\ntrend', 'Operational\npersistence', 'Realized GBD\n(oracle)', 'UN 2024\nunaligned', 'UN 2024\naligned at 2018']
    for ax, (outcome, sex) in zip(axes.flat, [(o, s) for o in ['prevalence', 'incidence'] for s in ['Male', 'Female']]):
        table = interval_endpoint.loc[interval_endpoint.outcome.eq(outcome) & interval_endpoint.sex.eq(sex)].set_index('scenario').loc[ORDER]
        x = np.arange(5)
        ax.vlines(x, table.lower, table.upper, color='#1479a6', lw=2)
        ax.scatter(x, table.value, color='#1479a6', s=28, zorder=3, label='Frozen point × population')
        ax.scatter(x, table['median'], color='#c58135', marker='_', s=80, zorder=4, label='Conditional median')
        ax.axhline(table.observed.iloc[0], color='black', ls='--', label='GBD modeled share')
        ax.set_xticks(x, short, fontsize=8)
        ax.set_title(outcome.title() + ' — ' + sex)
        ax.set_ylabel('80+ share of burden among 45+ (%)')
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('Saudi adapted TCN: 2018-origin, five-year burden-share sensitivity')
    fig.text(.5, .015, 'Bars: nominal 80% rate-conditional intervals with deterministic populations.\n'
             'Eleven paired historical rate blocks; oracle/UN inputs use future population information.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .075, 1, .95])
    for ext in ['png', 'svg']:
        fig.savefig(out / ('saudi_80plus_share_population_scenarios.' + ext), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='results/population_sensitivity_v1')
    parser.add_argument('--output', default='reports/population_sensitivity_v1')
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.output
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing report')
    manifest = json.loads((run / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, digest in manifest['output_sha256'].items():
        assert sha(run / name) == digest, name
    for section in ['source_sha256', 'code_sha256']:
        for name, digest in manifest[section].items():
            assert sha(ROOT / name) == digest, name
    audit_path = ROOT / 'work/population-sensitivity-validation/audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] and audit['run_manifest_sha256'] == sha(run / 'run_manifest.json')
    for name, digest in audit['audited_code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    out.mkdir(parents=True)
    group = ['target', 'outcome', 'family', 'horizon', 'scenario', 'measure', 'node', 'sex', 'age_group', 'unit', 'uncertainty_scope']
    summaries, endpoints, accounting = [], [], []
    interval_endpoints = []
    for case in manifest['cases']:
        folder = run / case['id']
        conditional = read(folder / 'interval_scores.csv.gz')
        joint = read(folder / 'original_joint_reference.csv.gz')
        all_intervals = pd.concat([conditional, joint], ignore_index=True)
        summary = all_intervals.groupby(group + ['level'], as_index=False).agg(
            coverage=('covered', 'mean'), mean_width=('width', 'mean'), below_lower=('below_lower', 'mean'),
            above_upper=('above_upper', 'mean'), n_origins=('origin', 'nunique'), n_cells=('origin', 'size'))
        assert summary.n_origins.eq(5).all() and summary.n_cells.eq(5).all()
        wis = pd.concat([read(folder / 'wis_scores.csv'), read(folder / 'original_joint_wis_reference.csv')], ignore_index=True)
        wis = wis.groupby(group, as_index=False).agg(mean_wis_50_80=('wis_50_80', 'mean'), mean_wis_50_80_95=('wis_50_80_95', 'mean'))
        summary = summary.merge(wis, on=group, how='left', validate='many_to_one')
        assert summary.mean_wis_50_80.notna().all()
        summaries.append(summary)
        points = read(folder / 'point_scores.csv')
        endpoints.append(points.loc[points.origin.eq(2018) & points.horizon.eq(5)])
        effects = read(folder / 'count_error_accounting.csv')
        accounting.append(effects.loc[effects.origin.eq(2018) & effects.horizon.eq(5)])
        if case['target'] == 'Saudi Arabia':
            interval = conditional.loc[conditional.origin.eq(2018) & conditional.horizon.eq(5) & conditional.family.eq('tcn_adapted')
                                       & conditional.level.eq(.8) & conditional.measure.eq('age_share') & conditional.age_group.eq('80+_within_45+')]
            interval = interval.merge(points[['target', 'outcome', 'origin', 'horizon', 'family', 'scenario', 'node', 'value']],
                                      on=['target', 'outcome', 'origin', 'horizon', 'family', 'scenario', 'node'], validate='many_to_one')
            interval_endpoints.append(interval)
    summary = pd.concat(summaries, ignore_index=True)
    endpoint = pd.concat(endpoints, ignore_index=True)
    effects = pd.concat(accounting, ignore_index=True)
    summary.to_csv(out / 'five_origin_interval_summary.csv', index=False)
    endpoint.to_csv(out / 'final_2018_point_comparison.csv', index=False)
    effects.to_csv(out / 'final_2018_error_accounting.csv', index=False)
    national = read(run / 'saudi_gastat_comparison_2023.csv')
    national.to_csv(out / 'saudi_population_source_comparison.csv', index=False)
    shares = pd.concat(interval_endpoints, ignore_index=True)
    shares.to_csv(out / 'saudi_80plus_share_endpoint.csv', index=False)
    source = national.loc[national.outcome.eq('prevalence')]
    saudi = endpoint.loc[endpoint.target.eq('Saudi Arabia') & endpoint.family.eq('tcn_adapted')].copy()
    counts = saudi.loc[saudi.measure.eq('count') & saudi.age_group.eq('45+')].copy()
    counts['scenario_label'] = counts.scenario.map(LABELS)
    count_pivot = counts.pivot(index=['outcome', 'sex', 'observed'], columns='scenario', values='value').reset_index()
    share_pivot = shares.pivot(index=['outcome', 'sex', 'observed'], columns='scenario', values='value').reset_index()
    selected = summary.loc[summary.target.eq('Saudi Arabia') & summary.family.eq('tcn_adapted') & summary.horizon.eq(5)
                           & summary.level.eq(.8) & (summary.node.eq('Both__45+') | summary.age_group.eq('80+_within_45+'))].copy()
    selected['coverage_percent'] = 100 * selected.coverage
    conditional_cov = selected.loc[selected.uncertainty_scope.eq('rate_conditional_fixed_population')].pivot(
        index=['outcome', 'node'], columns='scenario', values='coverage_percent').reset_index()
    joint_cov = selected.loc[selected.uncertainty_scope.eq('joint_rate_population_original')].copy()
    joint_cov['scenario_label'] = joint_cov.scenario.map(LABELS)
    accounting_saudi = effects.loc[effects.target.eq('Saudi Arabia') & effects.family.eq('tcn_adapted')
                                   & effects.scenario.eq('log_trend_last8') & effects.age_group.isin(['45+', '80+'])]
    projections = read(run / 'population_scenarios_2024_2028.csv')
    validation = json.loads((run / 'validation_report.json').read_text())
    tests = json.loads((ROOT / 'work/population-sensitivity-validation/tests.json').read_text())
    oldest_male = accounting_saudi.loc[accounting_saudi.sex.eq('Male') & accounting_saudi.age_group.eq('80+')].set_index('outcome')
    male_shares = share_pivot.loc[share_pivot.sex.eq('Male')].set_index('outcome')
    oldest_population_fraction = 100 * oldest_male.population_effect / oldest_male.signed_error
    text = f'''# Population-source sensitivity: frozen GCC disease forecasts

Completed {now()[:10]}. This is a retrospective sensitivity analysis specified after the existing forecast results were inspected. No disease model, original point forecast, calibration procedure or primary conclusion changed. Saudi Arabia remains the primary target; identical transformations cover all six GCC countries, prevalence/incidence, both sexes and every age from 45–49 through 95+.

## Principal interpretation

Population forecasting contributes substantially to the oldest-male burden failure. At the Saudi 2018-origin/2023 endpoint, the symmetric population component accounts for {oldest_population_fraction['prevalence']:.1f}% of the signed male 80+ prevalence-count deficit and {oldest_population_fraction['incidence']:.1f}% of the corresponding incidence-count deficit under the operational log trend. These percentages partition those particular signed errors; they are not proportions of total predictive variance, causal effects, or evidence across independent patients.

Holding disease-rate forecasts fixed and supplying realized populations changes the male 80+ prevalence share from {male_shares.loc['prevalence','log_trend_last8']:.2f}% to {male_shares.loc['prevalence','gbd_realized_oracle']:.2f}%, against a GBD share of {male_shares.loc['prevalence','observed']:.2f}%. The incidence share changes from {male_shares.loc['incidence','log_trend_last8']:.2f}% to {male_shares.loc['incidence','gbd_realized_oracle']:.2f}%, against {male_shares.loc['incidence','observed']:.2f}%. This oracle diagnoses sensitivity; those future populations were unavailable at issuance and do not constitute a newly improved forecast.

Point sensitivity does not resolve interval reliability. Even with realized populations, male 80+ burden-share coverage is only 20% for prevalence and 40% for incidence across five overlapping origins under the fixed-population, rate-conditional intervals. Those intervals differ in uncertainty scope from the original joint intervals. Neither a better oracle point nor a different demographic source justifies adopting it as a calibrated operational procedure.

## What this analysis answers

The same saved adapted TCN, local-champion and non-neural-champion forecasts are multiplied by five population inputs. The operational log trend and persistence were formed using pre-origin GBD-implied population history. The realized-GBD oracle and both UN WPP 2024 alternatives deliberately use future/later-vintage population information. Their apparent closeness to evaluated burden is a diagnostic, not evidence of an operational forecasting gain or independent validation. Sources are not selected by these results.

Implied GBD population is native Number/Rate ×100,000, retained separately by outcome. UN historical alignment uses each issuance origin, never a future 2023 baseline. GBD population reference-date comparability remains unverified. UN and GASTAT are described as mid-year sources in the preserved metadata. Differences can reflect population coverage, source revision, modeling and timing; they cannot establish which source is correct.

## Saudi population comparison, 2023

The table uses the prevalence-implied denominator and GASTAT total-resident populations. GASTAT's original 80+ resolution is preserved, with no allocation into older subgroups. GASTAT/GCC-Stat concordance does not provide an independent demographic replication. Outcome-specific rows and complete age-specific UN comparisons remain in the numerical outputs.

{md(source, {'sex':'Sex','age_group':'Age','gbd_population':'Implied GBD','un_population':'UN 2024','gastat_population':'GASTAT','gbd_minus_gastat_percent':'GBD vs GASTAT %','un_minus_gastat_percent':'UN vs GASTAT %'})}

![Saudi population source comparison](saudi_population_sources_2023.png)

## Frozen TCN five-year endpoint: burden totals

Origin 2018, verification year 2023; modeled burden at ages 45+. Observed means the GBD modeled point estimate, not an independent enumeration of patients. Changing a population input does not change the age-specific rate forecast or the original primary rate error. Tables retain every fixed scenario, without choosing a winner.

{md(count_pivot, {'outcome':'Outcome','sex':'Sex','observed':'GBD count','log_trend_last8':'Operational trend','persistence':'Persistence','gbd_realized_oracle':'Oracle','un_2024_unaligned':'UN raw','un_2024_origin_aligned':'UN origin-aligned'})}

## Oldest-age composition

These percentages are the share of prevalence or incidence at ages 80+ within all modeled burden at ages 45+, not population shares. They retain ages 80–84, 85–89, 90–94 and 95+ in the numerator. An accurate total can conceal incorrect composition.

{md(share_pivot, {'outcome':'Outcome','sex':'Sex','observed':'GBD share %','log_trend_last8':'Operational trend','persistence':'Persistence','gbd_realized_oracle':'Oracle','un_2024_unaligned':'UN raw','un_2024_origin_aligned':'UN origin-aligned'})}

![Oldest-age burden-share sensitivity](saudi_80plus_share_population_scenarios.png)

## Conditional interval diagnostics

For each of the five inputs, the population is deterministic and the same complete historical rate-error blocks are transformed. These are **rate-conditional scenario intervals**, excluding population uncertainty. Their coverage against native GBD burden describes compatibility under that condition; later-vintage and oracle inputs do not yield deployable historical prediction intervals. The original operational joint rate/population intervals are a separate reference below.

Nominal 80% coverage, horizon five, across origins 2014–2018:

{md(conditional_cov, {'outcome':'Outcome','node':'Burden node','log_trend_last8':'Operational trend','persistence':'Persistence','gbd_realized_oracle':'Oracle','un_2024_unaligned':'UN raw','un_2024_origin_aligned':'UN origin-aligned'}, 1)}

Each entry has only five overlapping-origin observations: one hit changes coverage by 20 percentage points. Eleven paired residual blocks at origin 2018 (seven at origin 2014) are not independent age-specific samples. No statistical-significance, full GBD-uncertainty or exact-calibration claim follows. Per-node coverage, width, directional misses and 50/80 WIS at every horizon, plus 95% sparse-tail sensitivities, are saved in `five_origin_interval_summary.csv`.

Original joint rate/population reference, same five origins and nominal 80% level:

{md(joint_cov, {'outcome':'Outcome','node':'Burden node','scenario_label':'Population forecast','coverage_percent':'Coverage %','mean_width':'Mean width','mean_wis_50_80':'50/80 WIS'}, 2)}

Width/WIS units are modeled numbers for counts and percentage points for shares. These scales must not be pooled. Conditional-versus-joint differences also change uncertainty scope, so they are not evidence that removing population uncertainty improves reliability.

## Signed error accounting

For the original operational log-trend TCN at the 2023 endpoint, the table splits count error symmetrically between rate and population differences. Contributions sum exactly to predicted minus native GBD count. Opposing signs demonstrate cancellation; they do not identify causes. At every age-sex cell, log-count error also equals log-rate error plus log-population error.

{md(accounting_saudi, {'outcome':'Outcome','sex':'Sex','age_group':'Age node','signed_error':'Count error','rate_effect':'Rate contribution','population_effect':'Population contribution'})}

## Projection handoff and limits

The run prepares {len(projections):,} population-scenario rows for 2024–2028 across six countries, both sexes, eleven ages, two outcomes and the two locked population choices: unaligned UN medium and GBD-2023-aligned UN growth. The unaligned UN numbers are repeated by outcome for explicit matching, not independent scenarios. These are population inputs only; future Parkinson's estimates still require the separately specified 2023-origin disease projections. The medium path is deterministic and no probabilistic UN draws were supplied.

This analysis supports demographic sensitivity reporting. It neither repairs the failed joint primary transfer criterion nor establishes reliable intervals. Mortality/disability, learning curves, global standardized-rate benchmarking and disease projections remain separate pending analyses.

## Reproduction and evidence

Run `/home/saif/agpu_env/bin/python scripts/run_population_sensitivity.py --workers 12` into a new output directory after the synthetic test gate; then run the independent audit and `/home/saif/agpu_env/bin/python scripts/report_population_sensitivity.py`. Existing run/report directories are refused. The [implementation specification](../../study_design/population_sensitivity_implementation.md) records fixed choices and prior result inspection.

Twelve cases passed production validation in {validation['elapsed_seconds']:.1f} seconds of analysis with twelve CPU workers and one numerical thread each. No GPU training was required. Synthetic tests passed, and an [independent audit](../../work/population-sensitivity-validation/review.md) reconstructed the new numerical transformations. All consumed source hashes and locked artifacts were preserved. `results/population_sensitivity_v1/run_manifest.json` records sources, code, environment and output hashes; this report's `validation.json` records its audited inputs and artifacts. The transformation commit precedes outcome scoring but is explicitly not a new blinded evaluation: future population data enter the diagnostic scenarios by design.

Metadata clarification: the case-validation key `whole_rate_blocks` counts transformed node-by-block statistic rows (118,125 per case), not distinct residual blocks. The actual historical block count is recorded correctly in each interval's `n_blocks` and ranges from seven to eleven. This naming clarification changes no numerical output.

Population provenance: [source register D1–D4](../../study_design/data_source_references.md), preserved [supporting-data metadata](../../supporting_data/2026-09-26/README.md), and the [locked protocol](../../study_design/locked_v1/protocol.md). No additional raw source was downloaded for this population analysis.
'''
    figures(out, national, shares)
    (out / 'report.md').write_text(text)
    write_json(out / 'validation.json', dict(passed=True, created_utc=now(), source_manifest_sha256=sha(run / 'run_manifest.json'),
        independent_audit_sha256=sha(audit_path), report_code_sha256=sha(Path(__file__)),
        artifact_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(str(out / 'report.md'))


if __name__ == '__main__':
    main()
