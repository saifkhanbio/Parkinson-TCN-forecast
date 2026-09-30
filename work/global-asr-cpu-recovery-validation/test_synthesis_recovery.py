"""Read-only synthesis reporter review with isolated synthetic completion fixtures."""
from pathlib import Path
import copy
import hashlib
import json
import re
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTER = ROOT / 'scripts/report_study_synthesis_recovery.py'
SOURCE = REPORTER.read_bytes()
CONFIG = json.loads((ROOT/'study_design/locked_v1/design.json').read_text())
DRAFT = (ROOT/'reports/study_synthesis_v1_draft/results.md').read_text()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def dump(path, value):
    write(path,json.dumps(value,indent=2)+'\n')

def load_reporter(root):
    module=types.ModuleType('synthetic_reporter_review')
    module.__file__=str(REPORTER)
    exec(compile(SOURCE,str(REPORTER),'exec'),module.__dict__)
    module.ROOT=root
    module.OUT=root/'reports/study_synthesis_v1_1'
    return module

def projected_fixture():
    rows=[]
    for outcome, factor in [('prevalence',1.),('incidence',.15)]:
        for sex, sexfactor in [('Male',1.),('Female',.7),('Both',1.7)]:
            for scenario, popfactor in [('un_medium_unaligned',1.2),('gbd_2023_aligned_un_growth',1.)]:
                for year in range(2024,2029):
                    baseline=1000*factor*sexfactor*popfactor
                    value=baseline*(1+.03*(year-2023))
                    rows.append(dict(target='Saudi Arabia',outcome=outcome,sex=sex,family='tcn_adapted',
                                     age_group='45+',measure='count',node=sex+'__45+',forecast_year=year,
                                     scenario=scenario,scenario_baseline_2023=baseline,
                                     native_GBD_2023=1000*factor*sexfactor,value=value,lower=value*.9,upper=value*1.1,
                                     change_percent_from_scenario_baseline=100*(value/baseline-1)))
    return pd.DataFrame(rows)

def fixture(root,module):
    write(root/'study_design/locked_v1/design.json',json.dumps(CONFIG))
    protected=root/'preserved_artifact.txt';write(protected,'synthetic protected artifact\n')
    dump(root/'work/completion-validation/preserved_manifest_hashes.json',{'sha256':{'preserved_artifact.txt':sha(protected)}})
    # Every linked old artifact is a synthetic stub, except the real read-only input schemas copied below.
    out=root/'reports/study_synthesis_v1_1'
    for link in re.findall(r'\]\(([^)]+)\)',DRAFT):
        if not link.startswith(('https://','http://','#')):
            dest=(out/link.split('#')[0]).resolve()
            if dest!=root/'study_design/locked_v1/protocol.md':
                write(dest,'synthetic linked artifact\n')
            else:
                write(dest,'# Synthetic protocol\n')
    for name in ['primary_v1','secondary_v1','donor_comparisons_gpu_v1','population_sensitivity_v1','supporting_v1',
                 'reliability_v1_3','release_sensitivity_v1','learning_curves_v1','global_asr_cpu_recovery_v1','projections_v1']:
        write(root/'reports'/name/'report.md','# Synthetic review report\n')
    write(root/'results/primary_v1/primary_contrasts.csv',(ROOT/'results/primary_v1/primary_contrasts.csv').read_text())
    write(root/'results/learning_curves_v1/summary.csv',(ROOT/'results/learning_curves_v1/summary.csv').read_text())
    (root/'results/global_asr_cpu_recovery_v1').mkdir(parents=True,exist_ok=True)
    rows=[]
    families=[('local','damped_ets'),('regional','tcn_adapted'),('global','tcn_adapted'),
              ('global','pooled_ridge'),('global','pooled_boosting')]
    countries=[c for c in CONFIG['countries'] if c['gcc']]
    for country in countries:
        for outcome in ['prevalence','incidence']:
            for sex in ['Male','Female']:
                for scope,family in families:
                    rows.append(dict(target=country['name'],outcome=outcome,sex=sex,horizon=5,
                                     donor_scope=scope,family=family,absolute_log_error=.01+len(family)*.001))
    pd.DataFrame(rows).to_csv(root/'results/global_asr_cpu_recovery_v1/summary.csv',index=False)
    pd.DataFrame([dict(target=c['name'],outcome=o,sex=s,family='tcn_adapted',horizon=5,
                       absolute_log_error_change_global_minus_regional=(-.01 if s=='Male' else .01))
                  for c in countries for o in ['prevalence','incidence'] for s in ['Male','Female']]).to_csv(
        root/'results/global_asr_cpu_recovery_v1/donor_scope_comparisons.csv',index=False)
    projected_fixture().to_csv(root/'reports/projections_v1/projection_summary_all_years.csv',index=False)
    write(root/'reports/study_synthesis_v1_draft/results.md',DRAFT)
    linked=[root/'results/primary_v1/primary_contrasts.csv']
    for link in re.findall(r'\]\(([^)]+)\)',DRAFT):
        if not link.startswith(('https://','http://','#')):
            linked.append((out/link.split('#')[0]).resolve())
    draft_hashes={str(p.relative_to(root)):sha(p) for p in linked}
    dump(root/'reports/study_synthesis_v1_draft/validation.json',
         {'passed':True,'draft_sha256':sha(root/'reports/study_synthesis_v1_draft/results.md'),'source_sha256':draft_hashes})
    for name,(audit_path,code_path) in module.AUDITS.items():
        directory=root/'results'/name;directory.mkdir(parents=True,exist_ok=True)
        write(directory/'synthetic_output.txt','review output '+name)
        write(root/code_path,'# synthetic independent audit code '+name+'\n')
        helper=root/'src'/('helper_'+name+'.py');write(helper,'# synthetic frozen helper\n')
        code=root/'scripts'/('source_'+name+'.py');write(code,'# synthetic production code\n')
        manifest={'status':'complete','output_sha256':{p.name:sha(p) for p in directory.iterdir() if p.is_file()},
                  'code_sha256':{str(code.relative_to(root)):sha(code)}}
        dump(directory/'run_manifest.json',manifest)
        audit={'passed':True,'run_manifest_sha256':sha(directory/'run_manifest.json'),
               'audit_code_sha256':sha(root/code_path)}
        if name=='learning_curves_v1':
            audit.update(prediction_rows=9240,checkpoint_jobs=26)
        elif name=='global_asr_cpu_recovery_v1':
            audit.update(forecast_rows=2640,cases=[{'target':c['name'],'outcome':o} for c in countries for o in ['prevalence','incidence']])
        else:
            audit.update(cases=[{'case':c['iso3']+'_'+o,'passed':True} for c in countries for o in ['prevalence','incidence']],
                         frozen_helper_sha256={str(helper.relative_to(root)):sha(helper)})
        dump(root/audit_path,audit)
        report=root/'reports'/name
        key='report_sha256' if name=='learning_curves_v1' else ('output_sha256' if name=='global_asr_cpu_recovery_v1' else 'artifact_sha256')
        validation={'passed':True,'run_manifest_sha256':sha(directory/'run_manifest.json'),
                    key:{p.name:sha(p) for p in report.iterdir() if p.is_file() and p.name!='validation.json'}}
        dump(report/'validation.json',validation)
    for name,count in [('learning-curves',10),('global-asr',12),('projections',12)]:
        code=root/'scripts'/('test_'+name+'.py');write(code,'# synthetic tested script\n')
        dump(root/'work'/(name+'-validation')/'tests.json',
             {'passed':True,'tests_run':count,'tested_code_sha256':{str(code.relative_to(root)):sha(code)}})
    return projected_fixture()

def recovery_fixture(root,module):
    # Copy the original failed execution as an archived, internally hash-checked fixture.
    olddir=root/'results/global_asr_v1';olddir.mkdir(parents=True,exist_ok=True)
    for name in ['seed_audit.json','predictions.csv']:
        (olddir/name).write_bytes((ROOT/'results/global_asr_v1'/name).read_bytes())
    manifest={'output_sha256':{name:sha(olddir/name) for name in ['seed_audit.json','predictions.csv']}}
    dump(olddir/'run_manifest.json',manifest)
    evidence=json.loads((ROOT/module.FAILURE_REVIEW).read_text())
    evidence['original_manifest_sha256']=sha(olddir/'run_manifest.json')
    dump(root/module.FAILURE_REVIEW,evidence)
    write(root/module.REVIEW_NOTICE,'# Review notice\nArchived GPU neural comparisons are invalid because of CUDA execution failure.\n')
    write(root/'reports/study_synthesis_v1/report.md','Archived synthesis; preserve unchanged.\n')
    protected=['results/global_asr_v1/run_manifest.json','reports/study_synthesis_v1/report.md']
    dump(root/module.RECOVERY_SNAPSHOT,{'sha256':{name:sha(root/name) for name in protected}})
    # All production-shaped coordinates are represented; values are synthetic, not evidence of model performance.
    directory=root/'results'/module.RECOVERY_RUN
    points=pd.read_csv(olddir/'predictions.csv',float_precision='round_trip')
    points['status']='ok';points['prediction']=points.prediction.astype(float)+1
    points.to_csv(directory/'predictions.csv',index=False)
    seeds=json.loads((olddir/'seed_audit.json').read_text())
    for row in seeds:
        row.update(status='ok',device='cpu',reason='',fingerprint_before='synthetic_checkpoint',fingerprint_after='synthetic_checkpoint')
    dump(directory/'seed_audit.json',seeds)
    path=directory/'run_manifest.json';manifest=json.loads(path.read_text())
    manifest['identity']={'device':'cpu'}
    manifest['output_sha256']={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='run_manifest.json'}
    dump(path,manifest)
    audit_path=root/module.AUDITS[module.RECOVERY_RUN][0]
    audit=json.loads(audit_path.read_text());audit.update(run_manifest_sha256=sha(path),source_checkpoint_replays=216,
        independently_optimized_sex_corrections=192)
    dump(audit_path,audit)
    validation_path=root/'reports'/module.RECOVERY_RUN/'validation.json'
    validation=json.loads(validation_path.read_text());validation['run_manifest_sha256']=sha(path);dump(validation_path,validation)

class ReporterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.module=load_reporter(self.root)
        self.frame=fixture(self.root,self.module)
        recovery_fixture(self.root,self.module)
    def tearDown(self):
        self.module.plt.close('all')
        self.temp.cleanup()
    def test_complete_three_stage_twelve_case_gates_accept(self):
        sources=self.module.validate_inputs()
        self.assertGreater(len(sources),45)
        self.assertIn('results/primary_v1/primary_contrasts.csv',sources)
    def test_corrupt_run_artifact_is_rejected(self):
        path=self.root/'results/projections_v1/synthetic_output.txt'
        path.write_text('corrupt')
        with self.assertRaises(AssertionError): self.module.validate_inputs()
    def test_missing_case_is_rejected(self):
        path=self.root/self.module.AUDITS['projections_v1'][0]
        audit=json.loads(path.read_text());audit['cases'].pop();dump(path,audit)
        with self.assertRaises(AssertionError):self.module.validate_inputs()
    def test_duplicate_country_case_is_rejected(self):
        path=self.root/self.module.AUDITS['global_asr_cpu_recovery_v1'][0]
        audit=json.loads(path.read_text());audit['cases'][-1]=audit['cases'][0];dump(path,audit)
        with self.assertRaises(AssertionError):self.module.validate_inputs()
    def test_failed_case_and_unfinished_manifest_are_rejected(self):
        path=self.root/self.module.AUDITS['projections_v1'][0]
        audit=json.loads(path.read_text());audit['cases'][0]['passed']=False;dump(path,audit)
        with self.assertRaises(AssertionError):self.module.validate_inputs()
        audit['cases'][0]['passed']=True;dump(path,audit)
        path=self.root/'results/global_asr_cpu_recovery_v1/run_manifest.json'
        manifest=json.loads(path.read_text());manifest['status']='running';dump(path,manifest)
        with self.assertRaises(AssertionError):self.module.validate_inputs()
    def test_changed_frozen_helper_is_rejected(self):
        (self.root/'src/helper_projections_v1.py').write_text('# changed helper\n')
        with self.assertRaises(AssertionError):self.module.validate_inputs()
    def test_changed_old_primary_draft_or_test_code_is_rejected(self):
        path=self.root/'results/primary_v1/primary_contrasts.csv';saved=path.read_bytes();path.write_text('corrupt')
        with self.assertRaises(AssertionError):self.module.validate_inputs()
        path.write_bytes(saved)
        path=self.root/'reports/study_synthesis_v1_draft/results.md';saved=path.read_bytes();path.write_text('changed draft')
        with self.assertRaises(AssertionError):self.module.validate_inputs()
        path.write_bytes(saved)
        (self.root/'scripts/test_projections.py').write_text('# changed code\n')
        with self.assertRaises(AssertionError):self.module.validate_inputs()
    def test_changed_report_artifact_is_rejected(self):
        path=self.root/'reports/projections_v1/projection_summary_all_years.csv';path.write_text('corrupt')
        with self.assertRaises(AssertionError):self.module.validate_inputs()
    def test_figure_exact_series_geometry_and_render(self):
        destination=self.root/'figure';destination.mkdir()
        with patch.object(self.module.plt,'close'):
            self.module.projection_figure(self.frame,destination)
            figure=self.module.plt.gcf()
            self.assertEqual(len(figure.axes),4)
            for ax in figure.axes:
                self.assertEqual(len(ax.lines),2)
                for line in ax.lines:
                    np.testing.assert_array_equal(line.get_xdata(),range(2023,2029))
                    self.assertEqual(len(line.get_ydata()),6)
                self.assertEqual(len(ax.collections),3)
            for ext in ['png','svg']:
                self.assertGreater((destination/('saudi_projection_scenarios.'+ext)).stat().st_size,5000)
            for ext in ['png','svg']:
                (ROOT/'work/global-asr-cpu-recovery-validation'/('synthetic_reporter_projection.'+ext)).write_bytes(
                    (destination/('saudi_projection_scenarios.'+ext)).read_bytes())
    def test_figure_missing_or_duplicate_coordinates_fail(self):
        destination=self.root/'figure';destination.mkdir()
        with self.assertRaises(AssertionError):self.module.projection_figure(self.frame.iloc[1:],destination)
        altered=pd.concat([self.frame.iloc[1:],self.frame.iloc[[1]]],ignore_index=True)
        with self.assertRaises(AssertionError):self.module.projection_figure(altered,destination)
    def test_complete_integration_atomic_output_links_and_preserved_input(self):
        before=sha(self.root/'reports/study_synthesis_v1_draft/results.md')
        self.module.main()
        output=self.module.OUT
        report=(output/'report.md').read_text()
        manuscript=(output/'manuscript_results.md').read_text()
        self.assertNotIn('<!-- GLOBAL_ASR_RESULTS -->',manuscript)
        self.assertNotIn('<!-- PROJECTION_RESULTS -->',manuscript)
        self.assertNotIn('pending final audited insertion',manuscript)
        self.assertNotIn('**Draft status,',manuscript)
        self.assertEqual(manuscript.count('### Separate full-age standardized-rate benchmark'),1)
        self.assertEqual(manuscript.count('### Projections from the 2023 data cutoff'),1)
        self.assertIn('12/24 country–outcome–sex comparisons',report)
        self.assertNotIn('0/24 country',report)
        self.assertIn('110 of 120 source fits recorded CUDA launch failures',report)
        self.assertIn('not by observed forecast accuracy',report)
        self.assertIn('matched CPU settings',report)
        self.assertIn('216 learned checkpoints',report)
        self.assertIn('192 sex-specific corrections',report)
        self.assertIn('../global_asr_cpu_recovery_v1/report.md',report)
        for name in [self.module.RECOVERY_SNAPSHOT,self.module.REVIEW_NOTICE,self.module.FAILURE_REVIEW]:
            self.assertIn(name,json.loads((output/'validation.json').read_text())['source_sha256'])
        self.assertEqual((self.root/'reports/study_synthesis_v1/report.md').read_text(),'Archived synthesis; preserve unchanged.\n')
        self.assertIn('joint primary criterion was not met',report)
        self.assertEqual(sha(self.root/'reports/study_synthesis_v1_draft/results.md'),before)
        validation=json.loads((output/'validation.json').read_text())
        self.assertTrue(validation['passed'])
        self.assertTrue(validation['archived_v1_preserved'])
        self.assertEqual(validation['execution_recovery']['all_forecast_cells_ok'],2640)
        self.assertGreater(validation['file_links_checked'],50)
        self.module.verify(output,validation['artifact_sha256'])
        with self.assertRaises(FileExistsError):self.module.main()
    def test_failed_link_check_does_not_publish_partial_report(self):
        path=self.root/'reports/secondary_v1/report.md';path.unlink()
        # The verified old draft references this file, so the input gate fails before staging.
        with self.assertRaises(FileNotFoundError):self.module.main()
        self.assertFalse(self.module.OUT.exists())

    def update_recovery_hashes(self):
        directory=self.root/'results'/self.module.RECOVERY_RUN
        path=directory/'run_manifest.json';m=json.loads(path.read_text())
        m['output_sha256']={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='run_manifest.json'}
        dump(path,m)
        a=self.root/self.module.AUDITS[self.module.RECOVERY_RUN][0];audit=json.loads(a.read_text())
        audit['run_manifest_sha256']=sha(path);dump(a,audit)
        v=self.root/'reports'/self.module.RECOVERY_RUN/'validation.json';validation=json.loads(v.read_text())
        validation['run_manifest_sha256']=sha(path);dump(v,validation)
    def test_successful_cpu_production_shape_accepts(self):
        result=self.module.validate_cpu_recovery()
        self.assertEqual(result['successful_seed_fits'],120)
        self.assertEqual(result['neural_forecast_cells_ok'],720)
        self.assertEqual(result['checkpoint_replays'],216)
        self.assertFalse(result['accuracy_based_recovery_selection'])
    def test_hash_consistent_fallback_rejected_for_neural_and_baseline(self):
        path=self.root/'results'/self.module.RECOVERY_RUN/'predictions.csv'
        original=pd.read_csv(path,float_precision='round_trip')
        for family in ['tcn_adapted','damped_ets']:
            with self.subTest(family=family):
                points=original.copy();points.loc[points.family.eq(family).idxmax(),'status']='fallback'
                points.to_csv(path,index=False);self.update_recovery_hashes()
                with self.assertRaisesRegex(ValueError,'2640'):self.module.validate_inputs()
    def test_failed_seed_hidden_by_successful_forecast_rows_rejected(self):
        path=self.root/'results'/self.module.RECOVERY_RUN/'seed_audit.json'
        rows=json.loads(path.read_text());rows[0]['status']='fallback';dump(path,rows);self.update_recovery_hashes()
        with self.assertRaisesRegex(ValueError,'source fit failed'):self.module.validate_inputs()
    def test_missing_or_duplicate_seed_rejected(self):
        path=self.root/'results'/self.module.RECOVERY_RUN/'seed_audit.json'
        original=json.loads(path.read_text())
        for rows in [original[:-1],original[:-1]+[original[0]]]:
            dump(path,rows);self.update_recovery_hashes()
            with self.assertRaisesRegex(ValueError,'120 distinct'):self.module.validate_cpu_recovery()
    def test_device_settings_and_checkpoint_mutation_rejected(self):
        path=self.root/'results'/self.module.RECOVERY_RUN/'seed_audit.json'
        original=json.loads(path.read_text())
        for changed in [{'device':'cuda:0'},{'base':{'channels':16,'epochs':49,'weight_decay':.001}},
                        {'fingerprint_after':'changed'},{'fingerprint_before':'unavailable','fingerprint_after':'unavailable'}]:
            with self.subTest(changed=changed):
                rows=copy.deepcopy(original);rows[0].update(changed);dump(path,rows);self.update_recovery_hashes()
                with self.assertRaisesRegex(ValueError,'source fit failed'):self.module.validate_cpu_recovery()
        dump(path,original)
        mpath=path.parent/'run_manifest.json';m=json.loads(mpath.read_text());m['identity']['device']='cuda:0';dump(mpath,m)
        self.update_recovery_hashes()
        with self.assertRaisesRegex(ValueError,'matched CPU'):self.module.validate_cpu_recovery()
    def test_old_fallback_checkpoint_audit_counts_rejected(self):
        path=self.root/self.module.AUDITS[self.module.RECOVERY_RUN][0]
        original=json.loads(path.read_text())
        for key,value in [('source_checkpoint_replays',106),('independently_optimized_sex_corrections',96)]:
            audit=copy.deepcopy(original);audit[key]=value;dump(path,audit)
            with self.assertRaisesRegex(ValueError,'216 learned-checkpoint'):self.module.validate_cpu_recovery()
    def test_prediction_coordinates_and_cutoff_rejected(self):
        path=self.root/'results'/self.module.RECOVERY_RUN/'predictions.csv'
        original=pd.read_csv(path,float_precision='round_trip')
        bad_year=original.copy();bad_year.loc[0,'forecast_year']=2030
        bad_cutoff=original.copy();bad_cutoff.loc[0,'origin']=2023
        bad_value=original.copy();bad_value.loc[0,'prediction']=float('inf')
        wrong_scope=original.copy();wrong_scope.loc[0,'donor_scope']='other'
        for points in [original.iloc[:-1],pd.concat([original.iloc[:-1],original.iloc[[0]]]),bad_year,bad_cutoff,bad_value,wrong_scope]:
            points.to_csv(path,index=False);self.update_recovery_hashes()
            with self.assertRaisesRegex(ValueError,'2640'):self.module.validate_cpu_recovery()
    def test_archived_failure_evidence_and_preservation_guard(self):
        evidence_path=self.root/self.module.FAILURE_REVIEW
        evidence=json.loads(evidence_path.read_text());evidence['failed_seed_fits']=109;dump(evidence_path,evidence)
        with self.assertRaisesRegex(ValueError,'Archived GPU failure'):self.module.verify_archived_failure()
        evidence['failed_seed_fits']=110;dump(evidence_path,evidence)
        (self.root/'reports/study_synthesis_v1/report.md').write_text('changed archived report')
        with self.assertRaises(AssertionError):self.module.verify_archived_failure()

if __name__=='__main__':
    start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReporterTests))
    evidence={'passed':result.wasSuccessful(),'reporter_sha256':hashlib.sha256(SOURCE).hexdigest(),
              'tested_code_sha256':{'scripts/report_study_synthesis_recovery.py':hashlib.sha256(SOURCE).hexdigest(),str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__)},
              'review_test_sha256':sha(__file__),'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
              'elapsed_seconds':time.perf_counter()-start,'production_outputs_modified':False,'scientific_files_modified':False,
              'fixture_type':'isolated_temporary_synthetic_completion_fixtures_with_read_only_actual_primary_learning_schema',
              'covered':['complete_audit_gates','corrupt_artifact_rejection','missing_duplicate_failed_cases',
                         'frozen_helper_identity','old_primary_and_draft_identity','test_code_identity',
                         'projection_figure_series_and_missing_coordinates','atomic_report_integration',
                         'placeholder_status_headings','all_markdown_links','existing_output_refusal','all_2640_success_gate','all_120_seed_success_gate','216_checkpoint_192_correction_replay_gate','archived_CUDA_failure_disclosure','no_accuracy_based_recovery_selection','new_report_atomic_publish_without_archive_changes']}
    dump(ROOT/'work/global-asr-cpu-recovery-validation/reporter_tests.json',evidence)
    raise SystemExit(0 if result.wasSuccessful() else 1)
