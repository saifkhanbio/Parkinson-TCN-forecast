"""Synthetic chronology, nonlinear aggregation and complete issuance checks."""
import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
import joblib
import numpy as np
import pandas as pd
import run_reliability_v1_3 as runner
from gbd_park.intervals import apply_bank, build_residual_bank, score_intervals
from run_demography import burden_statistics, ratio_statistics, distribution_summary

CONFIG = json.loads((ROOT/"study_design/locked_v1/design.json").read_text())
SPEC = dict(roles=["tcn_adapted", "local_champion", "nonneural_champion"],
            evaluation_origins=list(range(2014, 2019)), reference_variants=["original", "target_only", "gcc_assisted"],
            population_methods=["log_trend_last8", "persistence"], seed=1301, dynamic_draws=1024)


def sample_arrays(seed=5, draws=64):
    rng = np.random.default_rng(seed)
    points = np.exp(np.linspace(3, 5, 110).reshape(5, 22))
    rates = points[None]*np.exp(rng.normal(size=(draws, 5, 22))*.12)
    pops = np.broadcast_to(np.linspace(1000, 10000, 22)[None, :], (5, 22)).copy()
    pop_draws = pops[None]*np.exp(rng.normal(size=(draws, 5, 22))*.05)
    return dict(point_rates=points, rate_draws=rates, point_populations=pops, population_draws=pop_draws,
                point_counts=points*pops/100000, count_draws=rates*pop_draws/100000)


class ArrayTests(unittest.TestCase):
    def test_vectorized_derived_quantiles_match_independent_legacy_transform(self):
        arrays = sample_arrays()
        context = dict(target="Saudi Arabia", outcome="prevalence", origin=2018, family="synthetic__joint")
        coords = pd.MultiIndex.from_product([range(64), range(1, 6), CONFIG["sexes"], CONFIG["ages"]],
                   names=["residual_origin", "horizon", "sex", "age"]).to_frame(index=False)
        for key, value in context.items():
            coords[key] = value
        coords["forecast_year"] = coords.origin+coords.horizon
        coords["count"] = arrays["count_draws"].ravel()
        coords["rate_draw"] = arrays["rate_draws"].ravel()
        coords["population_method"] = "synthetic"
        expected = distribution_summary(pd.concat([burden_statistics(coords, CONFIG, True),
                                                   ratio_statistics(coords, CONFIG, True)], ignore_index=True))
        stored = {k: arrays[k] for k in ["point_rates", "rate_draws"]}
        stored.update({k+"__synthetic": arrays[k] for k in ["point_counts", "count_draws"]})
        points, actual = runner.burden_ledgers(stored, ["synthetic"], CONFIG, context, 0, "model_monte_carlo")
        runner.match_numeric(actual, expected, runner.STAT+["level"], ["lower", "median", "upper"])
        self.assertEqual(len(points), 230)
        self.assertTrue(actual.n_blocks.eq(0).all())
        self.assertTrue(actual.n_draws.eq(64).all())
        shares = points[points.measure.eq("age_share")]
        self.assertTrue(shares.value.between(0, 100).all())
        # A ratio of marginal medians is not substituted for a ratio quantile.
        direct, _ = runner.derived_values(arrays["rate_draws"], arrays["count_draws"], CONFIG)
        self.assertEqual(direct.shape, (64, 5, 35))

    def test_new_scoring_reproduces_legacy_rate_scoring(self):
        arrays = sample_arrays()
        context = dict(target="Saudi Arabia", outcome="prevalence", origin=2018, family="synthetic__joint")
        points, intervals = runner.rate_ledgers(arrays, CONFIG, context, 7, "historical_joint_blocks")
        truth = points[["target", "outcome", "sex", "age", "forecast_year", "prediction"]].rename(columns={"prediction": "observed_rate"})
        truth["observed_rate"] *= 1.03
        old_cells, old_wis = score_intervals(intervals, truth, CONFIG, 2023)
        expanded = intervals[runner.RATE_KEYS].drop_duplicates().merge(truth, on=["target", "outcome", "sex", "age", "forecast_year"], validate="many_to_one")
        expanded["observed_value"] = np.where(expanded.scale.eq("rate"), expanded.observed_rate, np.log(expanded.observed_rate))
        cells, wis = runner.score_bounds(intervals, expanded, runner.RATE_KEYS, "observed_value")
        runner.match_numeric(cells, old_cells, runner.RATE_KEYS+["level"], ["covered", "width", "interval_score"])
        runner.match_numeric(wis, old_wis, runner.RATE_KEYS, ["wis_50_80", "wis_50_80_95"])

    def test_tensor_rejects_missing_cells_and_nonpositive_values(self):
        frame = pd.MultiIndex.from_product([range(1, 6), CONFIG["sexes"], CONFIG["ages"]], names=["horizon", "sex", "age"]).to_frame(index=False)
        frame["rate"] = 1.
        self.assertEqual(runner.tensor(frame, CONFIG, "rate").shape, (5, 22))
        with self.assertRaises(ValueError):
            runner.tensor(frame.iloc[:-1], CONFIG, "rate")
        frame.loc[0, "rate"] = 0.
        with self.assertRaises(ValueError):
            runner.tensor(frame, CONFIG, "rate")


def synthetic_sources(root):
    case = dict(id="SAU_prevalence", target="Saudi Arabia", outcome="prevalence", source="source",
                history="history", banks="history", demography="demography", case_index=0)
    source, history, demographic = root/"source", root/"history", root/"demography"
    prior = root/"results/calibration_v1_2/SAU_prevalence"
    for folder in [source, history/"banks", demographic, prior, root/"data/processed/design_v1"]:
        folder.mkdir(parents=True)
    panel = []
    for y in range(1990, 2024):
        for s, sex in enumerate(CONFIG["sexes"]):
            for a, age in enumerate(CONFIG["ages"]):
                rate = np.exp(3+.1*a+.08*s+.01*(y-1990)+.003*np.sin(y*.4+a))
                panel.append(dict(location_name=case["target"], outcome=case["outcome"], year=y,
                    sex=sex, age=age, rate=rate, count=rate/10, implied_population=10000.))
    panel = pd.DataFrame(panel)
    truth = panel.set_index(["year", "sex", "age"]).rate
    rows = []
    family_map = {"tcn_adapted": "tcn_adapted", "local_champion": "persistence", "nonneural_champion": "pooled_ridge"}
    for family in family_map.values():
        for origin in range(2003, 2019):
            for sex in CONFIG["sexes"]:
                for a, age in enumerate(CONFIG["ages"]):
                    for h in range(1, 6):
                        observed = truth.loc[(origin+h, sex, age)]
                        predicted = observed*np.exp(.008*h*np.sin(origin*.8+a*.2)+.001*h)
                        rows.append(dict(target=case["target"], outcome=case["outcome"], origin=origin, family=family,
                            sex=sex, age=age, horizon=h, forecast_year=origin+h, prediction=predicted,
                            log_prediction=np.log(predicted), observed_rate=observed, status="ok"))
    scored = pd.DataFrame(rows)
    scored[scored.origin.le(2013)].to_csv(history/"prequential_scores.csv", index=False)
    points = scored[scored.origin.ge(2014)].drop(columns="observed_rate").copy()
    point_frames, mappings = [points], []
    for role, family in family_map.items():
        if role == family:
            continue
        part = points[points.family.eq(family)].copy()
        part["family"], part["source_family"] = role, family
        point_frames.append(part)
        mappings.extend(dict(fit_origin=o, role=role, sex=s, source_family=family, last_selection_target_year=o)
                        for o in range(2014, 2019) for s in CONFIG["sexes"])
    points = pd.concat(point_frames, ignore_index=True)
    points.to_csv(source/"predictions.csv", index=False)
    pd.DataFrame(mappings).to_csv(source/"champion_family_mappings.csv", index=False)
    populations, errors, all_intervals, b_intervals, b_points, originals = [], [], [], [], [], []
    for origin in range(2014, 2019):
        template = points[points.origin.eq(origin) & points.family.eq("tcn_adapted")]
        for method in SPEC["population_methods"]:
            pop = template[["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year"]].copy()
            pop["population_method"], pop["population"], pop["log_population"] = method, 10000., np.log(10000.)
            populations.append(pop)
            for u in range(2003, origin-4):
                error = pop.copy()
                error["fit_origin"], error["origin"], error["residual_origin"] = origin, u, u
                error["centered_population_log_error"] = 0.
                errors.append(error)
        for role, family in family_map.items():
            bank = build_residual_bank(scored, CONFIG, origin, family)
            bank["family"], bank["source_family_by_sex"] = role, dict.fromkeys(CONFIG["sexes"], family)
            joblib.dump(bank, history/"banks"/f"origin{origin}__{role}.joblib")
            current = points[points.origin.eq(origin) & points.family.eq(role)].copy()
            _, original_draws = apply_bank(current, bank, CONFIG)
            old_points = [ratio_statistics(current, CONFIG)]
            for method in SPEC["population_methods"]:
                cells = current.copy()
                cells["population_method"], cells["count"] = method, cells.prediction/10
                old_points.append(burden_statistics(cells, CONFIG))
            original_b = pd.concat(old_points, ignore_index=True)
            originals.append(original_b)
            for variant in SPEC["reference_variants"]:
                current_variant = current.copy()
                current_variant["family"] = role+"__"+variant
                bank_variant = copy.deepcopy(bank)
                bank_variant["family"] = role+"__"+variant
                bounds, draws = apply_bank(current_variant, bank_variant, CONFIG)
                all_intervals.append(runner.labels(bounds))
                values = [ratio_statistics(draws, CONFIG, True)]
                for method in SPEC["population_methods"]:
                    cells = draws.copy()
                    cells["population_method"], cells["count"] = method, cells.rate_draw/10
                    values.append(burden_statistics(cells, CONFIG, True))
                b_intervals.append(runner.labels(distribution_summary(pd.concat(values, ignore_index=True))))
                point = original_b.copy()
                point["family"] = role+"__"+variant
                b_points.append(runner.labels(point))
    pd.concat(populations, ignore_index=True).to_csv(demographic/"population_forecasts.csv", index=False)
    pd.concat(errors, ignore_index=True).to_csv(demographic/"population_residuals.csv", index=False)
    pd.concat(originals, ignore_index=True).to_csv(demographic/"predictions.csv", index=False)
    pd.concat(all_intervals, ignore_index=True).to_csv(prior/"rate_intervals.csv", index=False)
    pd.concat(b_intervals, ignore_index=True).to_csv(prior/"burden_intervals.csv", index=False)
    pd.concat(b_points, ignore_index=True).to_csv(prior/"burden_predictions.csv", index=False)
    panel.to_csv(root/"data/processed/design_v1/regional_outcomes.csv", index=False)
    return case


class IntegrationTests(unittest.TestCase):
    def test_full_case_chronology_controls_and_commit_gate(self):
        from gbd_park.dynamic_joint import fit_dynamic_joint
        def fake_dynamic(panel, config, origin, target, outcome, seed, draws):
            self.assertLessEqual(panel.year.max(), origin)
            return fit_dynamic_joint(panel, config, origin, target, outcome, seed=seed, draws=draws)
        with tempfile.TemporaryDirectory(prefix="reliability-v13-synthetic-") as temp:
            root = Path(temp)
            case = synthetic_sources(root)
            output = root/"output"
            output.mkdir()
            with patch.object(runner, "ROOT", root), patch("gbd_park.dynamic_joint.fit_dynamic_joint", side_effect=fake_dynamic):
                issued = runner.issue_case(case, CONFIG, SPEC, output)
                self.assertTrue(issued["passed"])
                self.assertEqual(issued["issued_draw_files"], 20)
                with self.assertRaises(FileNotFoundError):
                    runner.score_case(case, CONFIG, SPEC, output)
                commits = {}
                for i in range(12):
                    path = output/f"synthetic{i}.json"
                    runner.write_json(path, dict(synthetic_case=i))
                    commits[path.name] = runner.sha(path)
                runner.write_json(output/"global_issued_commit.json", dict(case_commit_sha256=commits))
                result = runner.score_case(case, CONFIG, SPEC, output)
                self.assertTrue(result["passed"])
                self.assertEqual(result["procedures"], 13)
                self.assertEqual(result["point_rows"], 7150)
                self.assertEqual(result["rate_interval_rows"], 42900)
                parameters = runner.read(output/case["id"]/"structured_parameters.csv")
                self.assertTrue((parameters.last_label_year <= parameters.fit_origin).all())
                rows = parameters[(parameters.fit_origin == 2018) & (parameters.horizon == 1)]
                self.assertEqual(set(rows.distinct_origins_at_horizon), {15})
                summary = runner.read(output/case["id"]/"rate_summary.csv")
                self.assertEqual(set(summary[summary.age_scope.eq("80+")].age_cells), {4})


if __name__ == "__main__":
    unittest.main(verbosity=2)
