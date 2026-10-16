"""Synthetic full-grid issuance/scoring test; never uses real evaluation scores."""
import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import joblib
import numpy as np
import pandas as pd
import run_calibration as runner
from test_calibration import CONFIG, AMENDMENT, history_fixture
from gbd_park.intervals import apply_bank, build_residual_bank
from gbd_park.demography import joint_count_draws
from run_demography import STAT, burden_statistics, ratio_statistics, distribution_summary


class RunnerIntegrationTests(unittest.TestCase):
    def test_full_case_preserves_originals_and_requires_global_commit_before_truth(self):
        with tempfile.TemporaryDirectory(prefix="calibration-synthetic-") as temporary:
            root = Path(temporary)
            source, demography, output = root / "source", root / "demography", root / "output"
            for folder in [source / "banks", demography, output]:
                folder.mkdir(parents=True)
            case = dict(id="SYNTHETIC", target="Saudi Arabia", outcome="prevalence", source="source",
                        banks="source", history="source", demography="demography")
            history = history_fixture()
            family_map = dict(tcn_adapted="tcn_adapted", local_champion="persistence", nonneural_champion="pooled_ridge")
            points, intervals, original_draws, mappings, populations, errors = [], [], [], [], [], []
            burden_points, burden_draws = [], []
            for origin in AMENDMENT["evaluation_origins"]:
                template = history[history.family.eq("tcn_adapted") & history.origin.eq(2013)].drop(columns="observed_rate").copy()
                template["origin"], template["forecast_year"] = origin, origin + template.horizon
                for method in AMENDMENT["population_methods"]:
                    pop = template[["target", "outcome", "origin", "forecast_year", "sex", "age", "horizon"]].copy()
                    pop["population_method"], pop["population"] = method, 10000.
                    pop["log_population"] = np.log(pop.population)
                    populations.append(pop)
                    for residual_origin in range(2003, origin - 4):
                        error = pop.copy()
                        error["origin"], error["residual_origin"], error["fit_origin"] = residual_origin, residual_origin, origin
                        error["centered_population_log_error"] = .01 * (residual_origin - np.mean(range(2003, origin - 4)))
                        errors.append(error)
                current_pop = pd.concat(populations, ignore_index=True)
                current_error = pd.concat(errors, ignore_index=True)
                for role, family in family_map.items():
                    current = template.copy()
                    current["family"], current["source_family"] = role, family
                    points.append(current)
                    if role != "tcn_adapted":
                        mappings.extend(dict(fit_origin=origin, role=role, sex=sex, source_family=family,
                                             last_selection_target_year=origin) for sex in CONFIG["sexes"])
                    bank = build_residual_bank(history, CONFIG, origin, family)
                    bank["family"] = role
                    bank["source_family_by_sex"] = dict.fromkeys(CONFIG["sexes"], family)
                    joblib.dump(bank, source / "banks" / f"origin{origin}__{role}.joblib")
                    interval, draws = apply_bank(current, bank, CONFIG)
                    intervals.append(interval); original_draws.append(draws)
                    burden_points.append(ratio_statistics(current, CONFIG))
                    burden_draws.append(ratio_statistics(draws, CONFIG, True))
                    for method in AMENDMENT["population_methods"]:
                        pop = current_pop[current_pop.origin.eq(origin) & current_pop.population_method.eq(method)]
                        err = current_error[current_error.fit_origin.eq(origin) & current_error.population_method.eq(method)]
                        cells = current.copy()
                        cells["count"], cells["population_method"] = cells.prediction / 10, method
                        burden_points.append(burden_statistics(cells, CONFIG))
                        burden_draws.append(burden_statistics(joint_count_draws(draws, pop, err), CONFIG, True))
            points = pd.concat(points, ignore_index=True)
            points.to_csv(source / "predictions.csv", index=False)
            pd.concat(intervals, ignore_index=True).to_csv(source / "intervals.csv", index=False)
            pd.DataFrame(mappings).to_csv(source / "champion_family_mappings.csv", index=False)
            pd.concat(populations, ignore_index=True).to_csv(demography / "population_forecasts.csv", index=False)
            pd.concat(errors, ignore_index=True).to_csv(demography / "population_residuals.csv", index=False)
            burden_points = pd.concat(burden_points, ignore_index=True)
            burden_points.to_csv(demography / "predictions.csv", index=False)
            distribution_summary(pd.concat(burden_draws, ignore_index=True)).to_csv(demography / "intervals.csv", index=False)
            rows = []
            for country in AMENDMENT["countries"]:
                for family in family_map.values():
                    for sex in CONFIG["sexes"]:
                        for u in AMENDMENT["development_origins"]:
                            for factor in AMENDMENT["factor_grid"]:
                                loss = abs(factor - (2 if country == case["target"] else 1.25)) + .1
                                rows.append(dict(country=country, outcome="prevalence", family=family, sex=sex,
                                    development_origin=u, factor=factor, normalizer=.2, mean_log_wis=.2*loss,
                                    normalized_wis=loss, last_label_year=u+5))
            loss_path = root / "losses.csv"
            pd.DataFrame(rows).to_csv(loss_path, index=False)
            with patch.object(runner, "ROOT", root):
                issued = runner.issue_case(case, CONFIG, AMENDMENT, loss_path, output)
                self.assertTrue(issued["passed"])
                self.assertEqual(issued["factor_choices"], 60)
                directory = output / case["id"]
                choices = json.loads((directory / "factor_choices.json").read_text())
                for choice in choices:
                    if choice["fit_origin"] <= 2016:
                        self.assertEqual(choice["factor"], 1.)
                    elif choice["policy"] == "target_only":
                        self.assertEqual(choice["factor"], 2.)
                    else:
                        self.assertEqual(choice["factor"], 1.25)
                self.assertFalse((root / "data").exists())
                with self.assertRaises(FileNotFoundError):
                    runner.score_case(case, CONFIG, AMENDMENT, output)
                commit = json.loads((directory / "issued_commit.json").read_text())
                before = dict(commit["artifact_sha256"])
                # Fabricate the twelve global commit entries only for this isolated fixture.
                commits = {}
                for index in range(12):
                    name = f"synthetic_commit_{index}.json"
                    runner.write_json(output / name, dict(index=index))
                    commits[name] = runner.sha(output / name)
                runner.write_json(output / "global_issued_commit.json", dict(case_commit_sha256=commits))
                panel = points[["target", "outcome", "sex", "age", "forecast_year", "prediction"]].drop_duplicates()
                panel = panel.rename(columns={"target": "location_name", "forecast_year": "year", "prediction": "rate"})
                panel["rate"] *= 1.02
                panel["count"] = panel.rate / 10
                panel_path = root / "data/processed/design_v1/regional_outcomes.csv"
                panel_path.parent.mkdir(parents=True)
                panel.to_csv(panel_path, index=False)
                burden_truth = burden_points.copy()
                burden_truth["observed"] = burden_truth.value
                burden_truth.to_csv(demography / "point_scores.csv", index=False)
                scored = runner.score_case(case, CONFIG, AMENDMENT, output)
                self.assertTrue(scored["passed"])
                self.assertEqual(scored["rate_score_rows"], 69300)
                self.assertEqual(scored["burden_score_rows"], 54675)
                self.assertTrue(all(runner.sha(directory / p) == digest for p, digest in before.items()))
                summary = runner.read_csv(directory / "rate_summary.csv")
                self.assertEqual(set(summary[summary.age_scope.eq("80+")].age_cells), {4})
                for age in ["80-84", "85-89", "90-94", "95+"]:
                    self.assertIn("age:" + age, set(summary.age_scope))


if __name__ == "__main__":
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if result.wasSuccessful():
        paths = ["scripts/run_calibration.py", "tests/test_calibration_runner.py", "tests/test_calibration.py",
                 "src/gbd_park/calibration.py", "scripts/run_demography.py"]
        report = dict(passed=True, tests_run=result.testsRun, synthetic_only=True,
                      elapsed_seconds=time.perf_counter()-started,
                      tested_code_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
        (ROOT / "work/calibration-validation/integration_tests.json").write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
