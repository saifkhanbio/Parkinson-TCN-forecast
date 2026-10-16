"""Integrate the independently audited CPU execution recovery without changing archived outputs."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gbd_park_matplotlib')
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/study_synthesis_v1_1'
RECOVERY_RUN = 'global_asr_cpu_recovery_v1'
RECOVERY_SNAPSHOT = 'work/global-asr-cpu-recovery-validation/preserved_sha256.json'
FAILURE_REVIEW = 'work/completion-validation/global_gpu_failure_review.json'
REVIEW_NOTICE = 'reports/study_synthesis_v1_review_notice.md'
AUDITS = {
    'learning_curves_v1': ('work/completion-validation/audit_learning.json', 'work/completion-validation/audit_learning.py'),
    'global_asr_cpu_recovery_v1': ('work/global-asr-cpu-recovery-validation/independent_audit.json', 'work/learning-curves-validation/audit_global_asr.py'),
    'projections_v1': ('work/global-asr-validation/projections_independent_audit.json', 'work/global-asr-validation/audit_projections.py'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def verify(root, hashes):
    for name, digest in hashes.items():
        assert sha(root / name) == digest, name


def table(frame, columns, decimals=4):
    lines = ['| ' + ' | '.join(columns.values()) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for _, row in frame.iterrows():
        cells = [f'{row[key]:.{decimals}f}' if isinstance(row[key], (float, np.floating)) else str(row[key]) for key in columns]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def validate_cpu_recovery():
    """A reproduced fallback is not a successfully executed neural comparison."""
    directory = ROOT / 'results' / RECOVERY_RUN
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['identity']['device'] != 'cpu':
        raise ValueError('The ASR recovery must be a completed matched CPU run')
    config = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
    targets = [country['name'] for country in config['countries'] if country['gcc']]
    outcomes = ['prevalence', 'incidence']
    scopes = ['global', 'regional']
    seeds = config['models']['tcn']['ensemble_seeds']
    if seeds != [11, 23, 37, 53, 71]:
        raise ValueError('The recovery must retain the complete fixed seed list')
    audit_path, _ = AUDITS[RECOVERY_RUN]
    audit = json.loads((ROOT / audit_path).read_text())
    if (not audit['passed'] or audit['run_manifest_sha256'] != sha(directory / 'run_manifest.json')
            or audit['source_checkpoint_replays'] != 216
            or audit['independently_optimized_sex_corrections'] != 192):
        raise ValueError('Recovery requires 216 learned-checkpoint and 192 sex-correction replays')
    seed_records = json.loads((directory / 'seed_audit.json').read_text())
    expected_seed_keys = {(target, outcome, scope, seed) for target in targets for outcome in outcomes
                          for scope in scopes for seed in seeds}
    actual_seed_keys = {(row['target'], row['outcome'], row['scope'], row['seed']) for row in seed_records}
    if len(seed_records) != 120 or actual_seed_keys != expected_seed_keys:
        raise ValueError('Recovery requires 120 distinct source fits and 24 complete five-seed ensembles')
    for row in seed_records:
        if (row['status'] != 'ok' or row['device'] != 'cpu'
                or row['base'] != {'channels': 16, 'epochs': 50, 'weight_decay': 0.001}
                or not row['fingerprint_before'] or row['fingerprint_before'] == 'unavailable'
                or row['fingerprint_before'] != row['fingerprint_after']):
            raise ValueError('A recovery neural source fit failed or changed its fixed procedure')
    points = read(directory / 'predictions.csv')
    neural_families = ['tcn_adapted', 'tcn_unadapted', 'tcn_intercept']
    local_families = ['persistence', 'log_trend', 'damped_ets', 'arima']
    learned_families = ['pooled_ridge', 'pooled_boosting', 'donor_ridge_unadapted',
                        'donor_ridge_adapted', 'donor_boosting_unadapted', 'donor_boosting_adapted']+neural_families
    keys = ['target', 'outcome', 'donor_scope', 'family', 'sex', 'horizon']
    expected_points = {(target, outcome, scope, family, sex, horizon)
                       for target in targets for outcome in outcomes
                       for scope, families in [('local', local_families), ('global', learned_families),
                                               ('regional', learned_families)]
                       for family in families for sex in config['sexes'] for horizon in range(1, 6)}
    if (len(points) != 2640 or points.duplicated(keys).any()
            or set(map(tuple, points[keys].to_numpy())) != expected_points
            or not points.status.eq('ok').all() or not points.origin.eq(2018).all()
            or not points.forecast_year.eq(points.origin+points.horizon).all()
            or not np.isfinite(points.prediction).all() or points.prediction.le(0).any()):
        raise ValueError('All 2640 complete recovery forecast cells must have successful status')
    neural = points.loc[points.family.isin(neural_families)]
    if len(neural) != 720 or not neural.status.eq('ok').all():
        raise ValueError('All 720 neural forecasts must come from successfully executed procedures')
    return {'run': RECOVERY_RUN, 'device': 'cpu', 'all_forecast_cells_ok': 2640,
            'neural_forecast_cells_ok': 720, 'successful_seed_fits': 120,
            'complete_five_seed_ensembles': 24, 'checkpoint_replays': 216,
            'neural_checkpoint_replays': 120, 'nonneural_checkpoint_replays': 96,
            'sex_correction_replays': 192, 'accuracy_based_recovery_selection': False}


def verify_archived_failure():
    """Keep failed execution and its earlier synthesis intact and explicitly qualified."""
    snapshot = ROOT / RECOVERY_SNAPSHOT
    verify(ROOT, json.loads(snapshot.read_text())['sha256'])
    evidence = json.loads((ROOT / FAILURE_REVIEW).read_text())
    directory = ROOT / 'results/global_asr_v1'
    manifest_path = directory / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (evidence['original_manifest_sha256'] != sha(manifest_path)
            or evidence['failed_seed_fits'] != 110 or evidence['successful_seed_fits'] != 10
            or evidence['complete_successful_ensembles'] != 0
            or evidence['total_ensembles'] != 24 or evidence['neural_fallback_cells'] != 720
            or not evidence['all_neural_points_exactly_equal_local_persistence']):
        raise ValueError('Archived GPU failure review does not match its preserved execution')
    for name in ['seed_audit.json', 'predictions.csv']:
        if sha(directory / name) != manifest['output_sha256'][name]:
            raise ValueError('Archived GPU artifact changed')
    seeds = json.loads((directory / 'seed_audit.json').read_text())
    if len(seeds) != 120 or sum(row['status'] != 'ok' for row in seeds) != 110:
        raise ValueError('Archived GPU source failure count changed')
    failed = [row for row in seeds if row['status'] != 'ok']
    if not all('CUDA' in row['reason'] for row in failed):
        raise ValueError('Archived source failures differ from the documented CUDA errors')
    points = read(directory / 'predictions.csv')
    neural = points.loc[points.family.str.startswith('tcn_')]
    if len(neural) != 720 or not neural.status.eq('fallback').all():
        raise ValueError('Archived GPU neural rows do not match the documented fallbacks')
    keys = ['target', 'outcome', 'sex', 'horizon']
    persistence = points.loc[points.family.eq('persistence'), keys+['prediction']]
    paired = neural.merge(persistence, on=keys, validate='many_to_one', suffixes=('', '_persistence'))
    np.testing.assert_array_equal(paired.prediction, paired.prediction_persistence)
    return {str(path.relative_to(ROOT)): sha(path) for path in [snapshot, ROOT / FAILURE_REVIEW,
            ROOT / REVIEW_NOTICE, manifest_path, directory / 'seed_audit.json', directory / 'predictions.csv']}


def validate_inputs():
    snapshot = ROOT / 'work/completion-validation/preserved_manifest_hashes.json'
    verify(ROOT, json.loads(snapshot.read_text())['sha256'])
    sources = {str(snapshot.relative_to(ROOT)): sha(snapshot)}
    sources.update(verify_archived_failure())
    config = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
    expected = {(c['name'], o) for c in config['countries'] if c['gcc'] for o in ['prevalence', 'incidence']}
    expected_ids = {c['iso3']+'_'+o for c in config['countries'] if c['gcc'] for o in ['prevalence', 'incidence']}
    for name, (audit_path, code_path) in AUDITS.items():
        directory = ROOT / 'results' / name
        manifest_path = directory / 'run_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['status'] == 'complete'
        verify(directory, manifest['output_sha256'])
        verify(ROOT, manifest['code_sha256'])
        audit = json.loads((ROOT / audit_path).read_text())
        assert audit['passed'] and audit['run_manifest_sha256'] == sha(manifest_path)
        assert audit['audit_code_sha256'] == sha(ROOT / code_path)
        for key in ['helper_sha256', 'frozen_helper_sha256']:
            if key in audit:
                verify(ROOT, audit[key])
        if name == 'learning_curves_v1':
            assert audit['prediction_rows'] == 9240 and audit['checkpoint_jobs'] == 26
        elif name == 'global_asr_cpu_recovery_v1':
            assert audit['forecast_rows'] == 2640 and len(audit['cases']) == 12
            assert {(c['target'], c['outcome']) for c in audit['cases']} == expected
        else:
            assert len(audit['cases']) == 12 and all(c['passed'] for c in audit['cases'])
            assert {c['case'] for c in audit['cases']} == expected_ids
        report_dir = ROOT / 'reports' / name
        validation_path = report_dir / 'validation.json'
        validation = json.loads(validation_path.read_text())
        assert validation['passed']
        manifest_ref = validation.get('run_manifest_sha256', validation.get('source_run_manifest_sha256'))
        assert manifest_ref == sha(manifest_path)
        for key in ['report_sha256', 'output_sha256', 'artifact_sha256']:
            if key in validation:
                verify(report_dir, validation[key])
        for path in [manifest_path, ROOT / audit_path, ROOT / code_path, validation_path]:
            sources[str(path.relative_to(ROOT))] = sha(path)
    for name, count in [('learning-curves', 10), ('global-asr', 12), ('projections', 12)]:
        path = ROOT / 'work' / (name+'-validation') / 'tests.json'
        tests = json.loads(path.read_text())
        assert tests['passed'] and tests['tests_run'] == count
        verify(ROOT, tests['tested_code_sha256'])
        sources[str(path.relative_to(ROOT))] = sha(path)
    draft_dir = ROOT / 'reports/study_synthesis_v1_draft'
    draft_validation = json.loads((draft_dir / 'validation.json').read_text())
    assert draft_validation['passed'] and draft_validation['draft_sha256'] == sha(draft_dir / 'results.md')
    verify(ROOT, draft_validation['source_sha256'])
    sources.update(draft_validation['source_sha256'])
    sources[str((draft_dir / 'validation.json').relative_to(ROOT))] = sha(draft_dir / 'validation.json')
    validate_cpu_recovery()
    return sources


def projection_figure(frame, out):
    colors = {'un_medium_unaligned': '#b66d2a', 'gbd_2023_aligned_un_growth': '#1479a6'}
    labels = {'un_medium_unaligned': 'UN medium', 'gbd_2023_aligned_un_growth': 'GBD baseline + UN growth'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ri, outcome in enumerate(['prevalence', 'incidence']):
        for ci, sex in enumerate(['Male', 'Female']):
            ax = axes[ri, ci]
            part = frame.loc[frame.target.eq('Saudi Arabia') & frame.family.eq('tcn_adapted') & frame.outcome.eq(outcome)
                             & frame.sex.eq(sex) & frame.age_group.eq('45+') & frame.measure.eq('count')]
            assert len(part) == 10
            for scenario, color in colors.items():
                rows = part.loc[part.scenario.eq(scenario)].sort_values('forecast_year')
                assert rows.forecast_year.tolist() == list(range(2024, 2029))
                ax.plot([2023, *rows.forecast_year], [rows.scenario_baseline_2023.iloc[0], *rows.value],
                        color=color, marker='o', markersize=3, label=labels[scenario])
                ax.fill_between(rows.forecast_year.to_numpy(), rows.lower.to_numpy(), rows.upper.to_numpy(),
                                color=color, alpha=.16)
            native = part.native_GBD_2023.unique()
            assert len(native) == 1
            ax.scatter([2023], native, color='black', marker='x', zorder=5, label='Native GBD 2023')
            ax.set_title(f'{outcome.title()} — {sex}')
            ax.set_ylabel('Modeled cases, ages 45+')
            ax.set_xticks(range(2023, 2029))
            ax.spines[['top', 'right']].set_visible(False)
            ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Year (disease data cutoff: 2023)')
    fig.suptitle('Saudi scenario projections: adapted TCN and fixed population paths')
    fig.text(.5, .015, 'Shading: nominal 80% intervals conditional on population. Previous undercoverage remains unresolved.\n'
             'Scenario differences are not a probability interval; 2024–2028 disease outcomes are unverified in the supplied data.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .07, 1, .96])
    for extension in ['png', 'svg']:
        fig.savefig(out / ('saudi_projection_scenarios.'+extension), dpi=180)
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError('Preserve the existing integrated study report')
    sources = validate_inputs()
    recovery = validate_cpu_recovery()
    primary = read(ROOT / 'results/primary_v1/primary_contrasts.csv')
    assert len(primary) == 4 and not primary.strictly_lower.all()
    curve = read(ROOT / 'results/learning_curves_v1/summary.csv')
    ct = curve.loc[curve.family.eq('tcn_adapted') & curve.horizon.eq(5) & curve.age_group.eq('45+')].pivot(
        index=['outcome', 'sex'], columns='history_years', values='mean_absolute_log_error').reset_index()
    assert len(ct) == 4
    global_summary = read(ROOT / 'results/global_asr_cpu_recovery_v1/summary.csv')
    gs = global_summary.loc[global_summary.target.eq('Saudi Arabia') & global_summary.horizon.eq(5)].copy()
    gs['view'] = gs.donor_scope+'__'+gs.family
    gt = gs.pivot(index=['outcome', 'sex'], columns='view', values='absolute_log_error').reset_index()
    contrasts = read(ROOT / 'results/global_asr_cpu_recovery_v1/donor_scope_comparisons.csv')
    tcn = contrasts.loc[contrasts.family.eq('tcn_adapted') & contrasts.horizon.eq(5)]
    assert len(tcn) == 24
    wins = int(tcn.absolute_log_error_change_global_minus_regional.lt(0).sum())
    projected = read(ROOT / 'reports/projections_v1/projection_summary_all_years.csv')
    totals = projected.loc[projected.target.eq('Saudi Arabia') & projected.family.eq('tcn_adapted')
        & projected.forecast_year.eq(2028) & projected.measure.eq('count') & projected.node.eq('Both__45+')].copy()
    assert len(totals) == 4 and totals[['lower','upper','scenario_baseline_2023','native_GBD_2023']].notna().all().all()
    totals['projected_with_interval'] = totals.apply(lambda r: f'{r.value:,.0f} [{r.lower:,.0f}, {r.upper:,.0f}]', axis=1)
    totals['population_scenario'] = totals.scenario.map({'un_medium_unaligned':'UN medium', 'gbd_2023_aligned_un_growth':'GBD baseline + UN growth'})
    global_text = f'''### Separate full-age standardized-rate benchmark

The fixed-setting global ASR benchmark completed 12 GCC prevalence/incidence cases using all 203 supplied countries/territories, with 202 non-target locations in each global donor arm. Global and regional TCN arms used matched CPU settings and five seeds. All 120 neural seed fits succeeded, all 24 five-seed ensembles were complete, and all 720 neural forecast cells came from successful fitted procedures. With these successfully executed CPU models, global donors reduced adapted-TCN five-year absolute log error in {wins}/24 country–outcome–sex comparisons. These dependent descriptive comparisons do not establish independent replication or replace the age-specific primary result.

This is an execution recovery of the archived GPU attempt: 110 of 120 source fits recorded CUDA launch failures, leaving all 24 neural ensembles and all 720 neural forecast cells as persistence fallbacks. That attempt cannot support a neural transfer comparison. The recovery repeats every case, donor scope and seed with unchanged model settings and data; it was triggered by execution failure, not by observed forecast accuracy. Device comparisons are not interpreted as method gains. See the [review notice](../study_synthesis_v1_review_notice.md), [failure evidence](../../work/completion-validation/global_gpu_failure_review.json) and [preservation snapshot](../../work/global-asr-cpu-recovery-validation/preserved_sha256.json). All original outputs remain archived unchanged.

Saudi five-year ASR errors (absolute log error; lower is better):

{table(gt, {'outcome':'Outcome','sex':'Sex','local__damped_ets':'Local ETS','regional__tcn_adapted':'Regional TCN','global__tcn_adapted':'Global TCN','global__pooled_ridge':'Global pooled ridge','global__pooled_boosting':'Global pooled boosting'})}

Every applicable family and both donor scopes remain in the [ASR report](../global_asr_cpu_recovery_v1/report.md). Geographic labels and workbook values passed internal checks; native extraction, precise standard weights and GBD boundary compatibility remain unverified. ASRs have no age-specific breakdown, were not converted to counts, and received no predictive-interval calibration claim.
'''
    projection_text = f'''### Projections from the 2023 data cutoff

All 12 GCC prevalence/incidence cases were projected for 2024–2028 using the original procedures, with settings selected from completed historical blocks through 2023 and comparator families selected using the unchanged 2009–2013 development-origin calendar. Sixteen original residual blocks supply the empirical intervals. No 2024–2028 disease outcome was available for scoring, and these are not forecasts issued in calendar 2023.

Saudi adapted-TCN projected totals at ages 45+, both sexes combined. Brackets are nominal 80% intervals conditional on the stated population path:

{table(totals, {'outcome':'Outcome','population_scenario':'Population scenario','native_GBD_2023':'Native 2023','scenario_baseline_2023':'Scenario 2023','projected_with_interval':'2028 [conditional 80%]','change_percent_from_scenario_baseline':'Change from scenario baseline %'}, 1)}

![Saudi sex-specific scenario projections](saudi_projection_scenarios.png)

Population-source differences at baseline must not be interpreted as disease growth. Scenario separation is not a probability interval. Previously demonstrated undercoverage remains relevant; these intervals omit demographic uncertainty and full GBD estimation uncertainty. All comparator, sex, age, age-share and rate-ratio outputs remain in the [projection report](../projections_v1/report.md).
'''
    draft = ROOT / 'reports/study_synthesis_v1_draft/results.md'
    manuscript = draft.read_text()
    for marker in ['<!-- GLOBAL_ASR_RESULTS -->', '<!-- PROJECTION_RESULTS -->']:
        assert manuscript.count(marker) == 1, 'Missing/duplicate manuscript placeholder: '+marker
    manuscript = re.sub(r'^### [^\n]+\n\n(?=<!-- (?:GLOBAL_ASR_RESULTS|PROJECTION_RESULTS) -->)', '', manuscript, flags=re.MULTILINE)
    manuscript = manuscript.replace('<!-- GLOBAL_ASR_RESULTS -->', global_text).replace('<!-- PROJECTION_RESULTS -->', projection_text)
    assert '<!-- GLOBAL_ASR_RESULTS -->' not in manuscript and '<!-- PROJECTION_RESULTS -->' not in manuscript
    manuscript = re.sub(r'^\*\*Draft status,.*$',
        '**Integration status, 30 September 2026:** All agreed modeling analyses and independent computational audits are complete. '
        'The successful CPU ASR execution recovery and conditional projections are incorporated below; the archived GPU neural comparisons were invalid because they used persistence fallbacks. This remains a manuscript draft; '
        'source comparability and external-validation limitations remain explicit.', manuscript, flags=re.MULTILINE)
    manuscript = manuscript.replace(
        '| Reserved Tables/Figures | Global ASR and future projection outputs pending final audited insertion | Preserve distinct estimands and uncertainty scope; fill after final handoff. |',
        '| Supplement Table S5 | [Global ASR comparisons](saudi_global_asr_comparisons.csv) | Separate full-age standardized-rate benchmark. |\n'
        '| Main Figure 5 / Supplement Table S6 | [Scenario projections](saudi_projection_scenarios.svg), [Saudi 2028 totals](saudi_2028_scenario_totals.csv) | Conditional projections; population scenarios and uncertainty limits. |')
    sources[str(draft.relative_to(ROOT))] = sha(draft)
    stage = Path(tempfile.mkdtemp(prefix='synthesis-recovery-', dir=ROOT / 'work/completion-validation'))
    projection_figure(projected, stage)
    ct.to_csv(stage / 'saudi_learning_curve_tcn.csv', index=False)
    gt.to_csv(stage / 'saudi_global_asr_comparisons.csv', index=False)
    totals.to_csv(stage / 'saudi_2028_scenario_totals.csv', index=False)
    (stage / 'manuscript_results.md').write_text(manuscript)
    text = f'''# Integrated Parkinson’s forecasting results — execution recovery

Prepared {datetime.now(timezone.utc).date()}. Version 1.1 uses the completely successful CPU ASR execution recovery and qualifies the archived GPU synthesis; it does not select models by observed accuracy. The primary, key secondary, supporting and explicitly exploratory findings remain distinct. The Saudi prevalence joint primary criterion was not met. Completion and computational audit do not establish reliable predictive intervals or clinical validity.

## Original primary result

{table(primary, {'sex':'Sex','comparator_source_family':'Comparator','tcn_error':'TCN log error','comparator_error':'Comparator log error','relative_improvement_percent':'TCN improvement %'})}

These frozen contrasts remain unchanged. See the [primary report](../primary_v1/report.md), [incidence/GCC evaluation](../secondary_v1/report.md), [donor comparisons](../donor_comparisons_gpu_v1/report.md), [population sensitivity](../population_sensitivity_v1/report.md), [mortality/disability report](../supporting_v1/report.md), [exploratory reliability comparison](../reliability_v1_3/report.md) and [limited release comparison](../release_sensitivity_v1/report.md).

## Target-history learning curves

The fixed donor checkpoints, settings and source histories are shared across target budgets. Saudi adapted-TCN five-year mean absolute log errors at ages 45+:

{table(ct, {'outcome':'Outcome','sex':'Sex',15:'15 years',20:'20 years',29:'29 years'}, 5)}

These comparisons have one common origin and verification period, with 3, 8 and 17 completed target windows per age–sex stratum. The fixed-default 29-year arm is a different procedure from the tuned primary model. All fourteen families and oldest-age results remain in the [learning-curve report](../learning_curves_v1/report.md); no new champion, interval method or primary replacement was selected.

{global_text}

{projection_text}

## Manuscript and remaining evidence limitations

The [integrated results draft](manuscript_results.md) assembles the findings and a proposed figure/table map. Modeling and reporting for the agreed remaining analyses are complete. Native export metadata, precise source-bound interpretation, common population reference dates and harmonized annual historical GBD vintages remain evidence limitations. Current older-release tables cannot validate sex/age forecast rankings or interval stability across releases. These gaps must remain explicit rather than being marked as completed validation.

The study describes GBD-modeled aggregate burden. Age cells, overlapping origins, countries sharing donors and neural seeds are not independent patients. Sex/age patterns and demographic accounting do not identify biological mechanisms. Further changes evaluated on the same inspected years would be exploratory; no new tuning is introduced by this report.

## Verification

The three scientific implementations passed 34 pre-run tests in total; tests alone do not establish successful production execution. The CPU recovery additionally requires all 2,640 forecast cells to have successful status, including all 720 neural cells, and all 120 neural source fits to succeed. Its independent audit replayed 216 learned checkpoints (120 neural and 96 non-neural) and 192 sex-specific corrections. Independent source-feature/checkpoint, adaptation, scoring and scenario audits cover the complete learning, recovered ASR and projection outputs. All earlier completed-run manifests, report-validation records, locked files and AGENTS.md match the saved preservation snapshot. Input and output hashes for this synthesis are recorded in `validation.json`.
'''
    (stage / 'report.md').write_text(text)
    links_checked = 0
    for document in stage.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            if link.startswith(('https://', 'http://', '#')):
                continue
            path = (OUT / link.split('#')[0]).resolve()
            actual = stage / path.name if path.parent == OUT else path
            assert actual.is_file(), 'Missing report link: '+link
            if path.parent != OUT:
                sources[str(path.relative_to(ROOT))] = sha(path)
            links_checked += 1
    verify(ROOT, sources)
    validation = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        reporter_sha256=sha(__file__), source_sha256=sources, original_primary_unchanged=True,
        archived_v1_preserved=True, execution_recovery=recovery,
        file_links_checked=links_checked,
        scope='agreed_remaining_analyses_and_integrated_report_not_external_validation',
        artifact_sha256={p.name: sha(p) for p in stage.iterdir() if p.is_file()})
    (stage / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    stage.rename(OUT)
    print(OUT / 'report.md', flush=True)


if __name__ == '__main__':
    main()
