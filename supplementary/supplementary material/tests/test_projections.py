"""Synthetic projection chronology, pairing, validation and checkpoint tests."""

import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import run_projections as runner
from gbd_park import projections as p
from gbd_park.local import settings_grid as local_grid
from gbd_park.pooled import settings_grid as nonneural_grid
from gbd_park.secondary import baseline_choices, family_mappings, job_fingerprint, score_actual
from gbd_park.tcn import base_grid
from gbd_park.tcn_forecasting import select_tcn_settings
from gbd_park.intervals import build_residual_bank, apply_bank
from gbd_park.population_sensitivity import projection_populations

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())


def synthetic_panel():
    rows = []
    for ci, country in enumerate(CONFIG["countries"]):
        for oi, outcome in enumerate(["prevalence", "incidence", "deaths"]):
            for year in range(1980, 2030):
                for si, sex in enumerate(CONFIG["sexes"]):
                    for ai, age in enumerate(CONFIG["ages"]):
                        rate = np.exp(3+.1*ci+.2*oi+.08*ai+.11*si+.012*(year-1990))
                        population = (1000+100*ci+15*ai)*(1+.003*(year-1990))
                        rows.append(dict(location_name=country["name"], outcome=outcome, year=year,
                                         sex=sex, age=age, rate=rate, count=rate*population/100000,
                                         implied_population=population))
    return pd.DataFrame(rows)


def synthetic_un():
    return pd.DataFrame([dict(location_name=country["name"], sex=sex, age=age, year=year,
                              population_persons=(1500+100*ci+40*ai)*(1+.006*(year-2023)))
                         for ci, country in enumerate(CONFIG["countries"])
                         for sex in CONFIG["sexes"] for ai, age in enumerate(CONFIG["ages"])
                         for year in range(2023, 2029)])


def historical_fixture():
    rows = []
    for fi, family in enumerate(p.underlying_families(CONFIG)):
        for origin in range(2003, 2019):
            for si, sex in enumerate(CONFIG["sexes"]):
                for ai, age in enumerate(CONFIG["ages"]):
                    for h in CONFIG["calendar"]["horizons"]:
                        log_rate = 3+.08*ai+.11*si+.012*(origin+h-1990)
                        error = .03*fi+.01*(origin-2010)*(-1 if si else 1)+.001*h*ai
                        rows.append(dict(target="Saudi Arabia", outcome="prevalence", origin=origin,
                                         forecast_year=origin+h, family=family, sex=sex, age=age, horizon=h,
                                         prediction=np.exp(log_rate+error), log_prediction=log_rate+error,
                                         setting_id=family+"_historical_"+str(origin), parameter_count=fi+1,
                                         status="ok"))
    return pd.DataFrame(rows)


def current_fixture(history, panel):
    rows = history.loc[history.origin.eq(2018)].copy()
    rows["origin"] = 2023
    rows["forecast_year"] = 2023+rows.horizon
    rows["setting_id"] = rows.family+"_2023"
    scored = score_actual(history, panel, CONFIG, "Saudi Arabia", "prevalence", 2023)
    mapping = family_mappings(scored, CONFIG, [2023])
    rows["source_family"] = rows.family
    points = pd.concat([rows, p.champion_views(rows, mapping, CONFIG, "Saudi Arabia", "prevalence")], ignore_index=True)
    frames = []
    from gbd_park.prequential import champion_history
    for family in p.ROLES:
        source = scored
        if family in p.ROLES[1:]:
            source = champion_history(scored, CONFIG, mapping.loc[mapping.role.eq(family)].set_index("sex").source_family.to_dict(), family)
        bank = build_residual_bank(source, CONFIG, 2023, family)
        frames.append(apply_bank(points.loc[points.family.eq(family)], bank, CONFIG)[1])
    return points, pd.concat(frames, ignore_index=True), mapping


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel()
        cls.history = historical_fixture()
        cls.points, cls.draws, cls.mapping = current_fixture(cls.history, cls.panel)
        cls.handoff = projection_populations(cls.panel, synthetic_un(), CONFIG)
        cls.populations = p.population_scenarios(cls.handoff, CONFIG, "Saudi Arabia", "prevalence")

    def test_scope_and_extended_candidate_jobs(self):
        tasks = p.projection_tasks(CONFIG)
        self.assertEqual(len(tasks), 12)
        self.assertEqual({task["outcome"] for task in tasks}, {"prevalence", "incidence"})
        jobs = [job for task in tasks for job in runner.candidate_jobs(task, CONFIG)]
        self.assertEqual(len(jobs), 660)
        self.assertEqual(sum(job["kind"] == "tcn" for job in jobs), 480)
        self.assertEqual({job["origin"] for job in jobs}, set(range(2014, 2019)))
        self.assertEqual(len({job_fingerprint(job) for job in jobs}), 660)
        self.assertTrue(all(job["device"] == "cpu" and job["seed"] == 11 for job in jobs if job["kind"] == "tcn"))

    def test_actual_outcome_cutoff_and_irrelevant_future_rows(self):
        before = self.panel.copy(deep=True)
        for outcome in p.OUTCOMES:
            work, cfg = p.working_context(self.panel, CONFIG, "Oman", outcome, 2023)
            altered = self.panel.copy(deep=True)
            irrelevant = altered.outcome.ne(outcome) | ~altered.year.between(1990, 2023)
            altered.loc[irrelevant, ["rate", "count"]] = np.nan
            again, _ = p.working_context(altered, CONFIG, "Oman", outcome, 2023)
            pd.testing.assert_frame_equal(work, again)
            self.assertTrue(work.source_outcome.eq(outcome).all())
            self.assertEqual((work.year.min(), work.year.max()), (1990, 2023))
            self.assertEqual(cfg["calendar"]["history_end"], 2023)
        pd.testing.assert_frame_equal(self.panel, before)
        with self.assertRaises(ValueError):
            p.working_context(self.panel, CONFIG, "Oman", "deaths", 2023)
        with self.assertRaises(ValueError):
            p.working_context(self.panel, CONFIG, "Oman", "prevalence", 2024)

    def test_settings_extend_to_2018_but_family_origins_remain_2009_2013(self):
        rows = []
        for spec in local_grid(CONFIG):
            for origin in range(2003, 2020):
                for sex in CONFIG["sexes"]:
                    for age in CONFIG["ages"]:
                        rows.append(dict(**{k: spec[k] for k in ["family", "setting_id", "grid_order"]},
                                         origin=origin, sex=sex, age=age, horizon=5, parameter_count=1,
                                         absolute_log_error=.01*(spec["grid_order"]+1) if origin <= 2018 else np.nan))
        decisions = baseline_choices(pd.DataFrame(rows), CONFIG, [2023], "local")
        self.assertTrue(decisions.last_inner_label_year.eq(2023).all())
        self.assertTrue(decisions.inner_origins.eq("|".join(map(str, range(2003, 2019)))).all())
        scored = score_actual(self.history, self.panel, CONFIG, "Saudi Arabia", "prevalence", 2023)
        original = family_mappings(scored, CONFIG, [2023])
        changed = scored.copy()
        changed.loc[changed.origin.ge(2014), "absolute_log_error"] = np.nan
        replay = family_mappings(changed, CONFIG, [2023])
        pd.testing.assert_frame_equal(original, replay)
        self.assertTrue(replay.last_selection_target_year.eq(2018).all())
        self.assertTrue(replay.selection_origins.eq("2009|2010|2011|2012|2013").all())
        with self.assertRaises(ValueError):
            baseline_choices(pd.DataFrame(rows).loc[lambda d: d.origin.ne(2018)], CONFIG, [2023], "local")

    def test_original_history_and_sixteen_complete_blocks(self):
        early = self.history.loc[self.history.origin.le(2013)]
        issued = self.history.loc[self.history.origin.ge(2014)]
        actual = p.historical_ledger(early, issued, CONFIG, "Saudi Arabia", "prevalence")
        self.assertEqual(len(actual), 24640)
        np.testing.assert_array_equal(actual.prediction, pd.concat([early, issued]).prediction)
        scored = score_actual(actual, self.panel, CONFIG, "Saudi Arabia", "prevalence", 2023)
        bank = build_residual_bank(scored, CONFIG, 2023, "tcn_adapted")
        self.assertEqual(bank["origins"], list(range(2003, 2019)))
        self.assertEqual(bank["centered_residuals"].shape, (16, 110))
        np.testing.assert_allclose(bank["centered_residuals"].mean(axis=0), 0, atol=1e-15)
        with self.assertRaises(ValueError):
            p.historical_ledger(early, issued.iloc[1:], CONFIG, "Saudi Arabia", "prevalence")
        future = scored.iloc[:110].copy()
        future["origin"], future["observed_rate"] = 2019, np.nan
        same = build_residual_bank(pd.concat([scored, future]), CONFIG, 2023, "tcn_adapted")
        np.testing.assert_array_equal(bank["raw_residuals"], same["raw_residuals"])

    def test_projection_champions_do_not_require_or_admit_future_truth(self):
        current = self.points.loc[~self.points.family.isin(p.ROLES[1:])]
        champions = p.champion_views(current, self.mapping, CONFIG, "Saudi Arabia", "prevalence")
        self.assertEqual(len(champions), 220)
        self.assertEqual(set(champions.forecast_year), set(range(2024, 2029)))
        for row in self.mapping.itertuples():
            expected = current.loc[current.family.eq(row.source_family) & current.sex.eq(row.sex)].prediction.to_numpy()
            actual = champions.loc[champions.family.eq(row.role) & champions.sex.eq(row.sex)].prediction.to_numpy()
            np.testing.assert_array_equal(expected, actual)
        with self.assertRaises(ValueError):
            p.champion_views(current.assign(observed_rate=1.), self.mapping, CONFIG, "Saudi Arabia", "prevalence")
        changed = self.mapping.copy()
        changed["last_selection_target_year"] = 2023
        with self.assertRaises(ValueError):
            p.champion_views(current, changed, CONFIG, "Saudi Arabia", "prevalence")

    def test_population_alignment_and_rejection(self):
        populations = self.populations
        aligned = populations[populations.scenario.eq(p.SCENARIOS[1])]
        np.testing.assert_allclose(aligned.population/aligned.gbd_baseline, aligned.un_future/aligned.un_baseline)
        self.assertIn("95+", set(populations.age))
        for invalid in [self.handoff.iloc[1:], pd.concat([self.handoff, self.handoff.iloc[:1]])]:
            # First row is Bahrain: verify the affected target and outcome.
            row = self.handoff.iloc[0]
            with self.assertRaises(ValueError):
                p.population_scenarios(invalid, CONFIG, row.location_name, row.outcome)
        invalid = self.handoff.copy()
        invalid.loc[invalid.location_name.eq("Saudi Arabia"), "population"] *= 2
        with self.assertRaises(AssertionError):
            p.population_scenarios(invalid, CONFIG, "Saudi Arabia", "prevalence")

    def test_whole_block_conditional_counts_shares_and_ratios(self):
        bp, bi, bd = p.conditional_statistics(self.points, self.draws, self.populations, CONFIG)
        self.assertEqual((len(bp), len(bi), len(bd)), (1215, 3645, 19440))
        self.assertTrue(bi.n_blocks.eq(16).all())
        self.assertTrue(bi[bi.measure.eq("count")].uncertainty_scope.eq("rate_conditional_fixed_population").all())
        future = bp[bp.family.eq("tcn_adapted") & bp.horizon.eq(5) & bp.scenario.eq(p.SCENARIOS[1])]
        points = self.points[self.points.family.eq("tcn_adapted") & self.points.horizon.eq(5)]
        populations = self.populations[self.populations.scenario.eq(p.SCENARIOS[1]) & self.populations.horizon.eq(5)]
        expected = points.merge(populations[["sex", "age", "population"]], on=["sex", "age"])
        expected["count"] = expected.prediction*expected.population/100000
        np.testing.assert_allclose(future[future.node.eq("Both__45+")].value.iloc[0], expected["count"].sum())
        male = expected[expected.sex.eq("Male")]
        share = 100*male[male.age.isin(CONFIG["ages"][-4:])]["count"].sum()/male["count"].sum()
        np.testing.assert_allclose(future[future.node.eq("Male__80+_within_45+")].value.iloc[0], share)
        selected = bd[(bd.family == "tcn_adapted") & (bd.horizon == 5) & (bd.node == "Both__45+") & (bd.scenario == p.SCENARIOS[1])]
        bounds = bi[(bi.family == "tcn_adapted") & (bi.horizon == 5) & (bi.node == "Both__45+")
                    & (bi.scenario == p.SCENARIOS[1]) & bi.level.eq(.8)].iloc[0]
        np.testing.assert_allclose([bounds.lower, bounds['median'], bounds.upper], np.quantile(selected.value, [.1,.5,.9]))
        ratio = bd[bd.measure.eq("sex_rate_ratio") & bd.family.eq("tcn_adapted") & bd.horizon.eq(5) & bd.age_group.eq("95+")]
        block = self.draws[self.draws.family.eq("tcn_adapted") & self.draws.horizon.eq(5) & self.draws.age.eq("95+")].pivot(index="residual_origin", columns="sex", values="rate_draw")
        np.testing.assert_allclose(ratio.set_index("residual_origin").value.sort_index(), (block.Male/block.Female).sort_index())

    def test_conditional_strict_role_context_and_complete_block_guards(self):
        cases = [(self.points[~self.points.family.eq("local_champion")], self.draws, self.populations),
                 (self.points, self.draws[~self.draws.family.eq("local_champion")], self.populations),
                 (self.points, self.draws.assign(target="Oman"), self.populations),
                 (self.points, self.draws.assign(origin=2022), self.populations),
                 (self.points, pd.concat([self.draws.iloc[1:], self.draws.iloc[[1]]]), self.populations),
                 (self.points, self.draws, self.populations.iloc[1:]),
                 (self.points, self.draws, self.populations.assign(outcome="incidence"))]
        for points, draws, populations in cases:
            with self.subTest(rows=(len(points),len(draws),len(populations))), self.assertRaises(ValueError):
                p.conditional_statistics(points, draws, populations, CONFIG)

    def test_scenario_consistent_and_native_baselines_are_distinct(self):
        result = p.baseline_statistics(self.panel, self.populations, CONFIG, "Saudi Arabia", "prevalence")
        self.assertEqual(len(result), 105)
        native = result[(result.scenario == "native_gbd_2023") & (result.node == "Both__45+")].value.iloc[0]
        aligned = result[(result.scenario == p.SCENARIOS[1]) & (result.node == "Both__45+")].value.iloc[0]
        unaligned = result[(result.scenario == p.SCENARIOS[0]) & (result.node == "Both__45+")].value.iloc[0]
        np.testing.assert_allclose(native, aligned, rtol=1e-12)
        self.assertGreater(abs(native-unaligned), .1)

    def test_actual_cpu_tiny_tcn_future_invariance_and_cache_integrity(self):
        task = p.projection_tasks(CONFIG)[0]
        job = runner.job_spec(task, "tcn", 2023, base={"channels":16,"weight_decay":.001,"epochs":2}, seed=11, device="cpu")
        altered = self.panel.copy()
        altered.loc[altered.year.gt(2023) | altered.year.lt(1990) | altered.outcome.ne("prevalence"), "rate"] = np.nan
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            with patch.object(runner, "source_panel", return_value=self.panel):
                runner.fit_job(CONFIG, job, Path(one))
            with patch.object(runner, "source_panel", return_value=altered):
                runner.fit_job(CONFIG, job, Path(two))
            a,b = runner.load_result(Path(one), job), runner.load_result(Path(two), job)
            self.assertEqual(a["audit"]["status"], "ok")
            self.assertEqual(a["audit"]["fingerprint_before"], b["audit"]["fingerprint_before"])
            np.testing.assert_array_equal(a["current_changes"], b["current_changes"])
            self.assertEqual(a["audit"]["maximum_label_year"], 2023)
            self.assertNotIn(task["target"], a["audit"]["countries"])
            with patch.object(runner, "fit_seed", side_effect=AssertionError("No refit")):
                runner.fit_job(CONFIG, job, Path(one))
            changed = deepcopy(CONFIG)
            changed["version"] = "changed"
            with self.assertRaises(ValueError):
                runner.fit_job(changed, job, Path(one))
            runner.result_path(Path(one), job).write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                runner.fit_job(CONFIG, job, Path(one))

    def test_no_uncommitted_selected_jobs_or_verification(self):
        task = p.projection_tasks(CONFIG)[0]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            with self.assertRaises(ValueError):
                runner.selected_jobs(task, CONFIG, out)
            with self.assertRaises(ValueError):
                runner.verify_case(task, CONFIG, out)

    def test_complete_synthetic_case_assembly_and_immutable_reuse(self):
        """Exercise the whole 16-family issuance/transform phase without production fits."""
        task = p.projection_tasks(CONFIG)[0]
        decisions = []
        for specs in [local_grid(CONFIG), nonneural_grid(CONFIG)]:
            for family in dict.fromkeys(spec["family"] for spec in specs):
                spec = next(spec for spec in specs if spec["family"] == family)
                for sex in CONFIG["sexes"]:
                    decisions.append(dict(origin=2023, sex=sex, family=family,
                                          setting_id=spec["setting_id"], inner_loss=.1,
                                          inner_origins="|".join(map(str,range(2003,2019))),
                                          last_inner_label_year=2023, selection_status="frozen_completed_blocks"))
        decisions = pd.DataFrame(decisions)
        current = self.points.loc[~self.points.family.isin(p.ROLES[1:])].copy()
        for row in decisions.itertuples():
            current.loc[current.family.eq(row.family) & current.sex.eq(row.sex), "setting_id"] = row.setting_id
        history_scores = score_actual(self.history, self.panel, CONFIG, "Saudi Arabia", "prevalence", 2023)
        choice = dict(fit_origin=2023, base={"channels":16,"weight_decay":.001,"epochs":50},
                      penalties={sex:1 for sex in CONFIG["sexes"]}, last_inner_label_year=2023, status="tuned")
        def fake_result(out, job):
            if job["kind"] == "tcn":
                return {"seed": job["seed"], "audit": {"status":"synthetic_assembly_fixture"}}
            if job["kind"] == "local":
                selected = current.loc[current.sex.eq(job["sex"]) & current.family.isin(CONFIG["models"]["local_order"])]
            else:
                selected = current.loc[current.family.isin(CONFIG["models"]["nonneural_order"])]
            return {"forecasts": selected.to_dict("records"), "fits": [], "calibrations": []}
        original_read = runner.read_csv
        def synthetic_read(path):
            return self.handoff.copy() if Path(path) == ROOT/runner.HANDOFF else original_read(path)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            directory = out/"trials"/task["id"]
            directory.mkdir(parents=True)
            decisions.to_csv(directory/"settings_decisions.csv",index=False)
            runner.write_json(directory/"tcn_choice.json",choice)
            self.mapping.to_csv(directory/"champion_family_mappings.csv",index=False)
            history_scores.to_csv(directory/"original_historical_scores.csv",index=False)
            runner.commit_phase(directory,"choices_frozen",list(directory.iterdir()))
            neural = current.loc[current.family.str.startswith("tcn_")].to_dict("records")
            with patch.object(runner,"source_panel",return_value=self.panel), \
                 patch.object(runner,"read_csv",side_effect=synthetic_read), \
                 patch.object(runner,"load_result",side_effect=fake_result), \
                 patch.object(runner,"make_forecasts",return_value=(neural, [])):
                runner.assemble_case(task,CONFIG,out)
            validation=json.loads((directory/"validation_report.json").read_text())
            self.assertEqual((validation["rate_points"],validation["rate_intervals"],validation["rate_draws"]),
                             (1760,10560,28160))
            self.assertFalse(validation["future_verification_used"])
            self.assertEqual(validation["burden_points"],1215)
            frame=original_read(directory/"predictions.csv")
            self.assertNotIn("observed_rate",frame)
            self.assertEqual(set(frame.forecast_year),set(range(2024,2029)))
            with patch.object(runner,"load_result",side_effect=AssertionError("Completed case must not reassemble")):
                runner.assemble_case(task,CONFIG,out)
            (directory/"predictions.csv").write_text("corrupt\n")
            with self.assertRaises(ValueError):
                runner.assemble_case(task,CONFIG,out)


if __name__ == "__main__":
    started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProjectionTests))
    report={"passed":result.wasSuccessful(),"tests_run":result.testsRun,"errors":len(result.errors),
            "failures":len(result.failures),"elapsed_seconds":time.perf_counter()-started,
            "fixture_type":"synthetic_no_projection_production_fits","actual_tiny_tcn_fits":2,
            "tested_code_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in runner.REQUIRED}}
    out=ROOT/"work/projections-validation"
    out.mkdir(parents=True,exist_ok=True)
    (out/"tests.json").write_text(json.dumps(report,indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
