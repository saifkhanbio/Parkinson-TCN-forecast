"""Schema/phase smoke run using synthetic values only; no estimator fits."""
import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import pandas as pd
from test_secondary import panel_fixture, CONFIG
from gbd_park.secondary import working_context, context_config
from gbd_park.local import settings_grid as local_grid
from gbd_park.pooled import settings_grid as nn_grid, target_inputs, build_examples
from run_secondary import prepare_choices, selected_jobs, assemble_issued, score_trial
panel = panel_fixture()
task = {"id": "OMN_incidence", "target": "Oman", "outcome": "incidence"}
cfg = context_config(CONFIG, "Oman")

def fake_result(out, job):
    origin = job["origin"]
    work, _ = working_context(panel, CONFIG, "Oman", "incidence", origin=origin)
    if job["kind"] == "tcn":
        x, levels, meta = target_inputs(work, cfg, origin, "Oman")
        tx, ty, tm = build_examples(work, cfg, origin, ["Oman"])
        return {"origin": origin, "base": job["base"], "seed": job["seed"],
                "current_changes": np.ones((22, 5))*.001*np.arange(1, 6),
                "target_changes": np.zeros_like(ty), "target_y": ty,
                "levels": levels, "current_meta": meta, "target_meta": tm,
                "audit": {"status": "ok", "fingerprint_before": str(job["seed"]), "reason": ""}}
    specs = job["specs"] if job["kind"] == "local" else nn_grid(CONFIG)
    sexes = [job["sex"]] if job["kind"] == "local" else CONFIG["sexes"]
    rows = []
    for spec in specs:
        for sex in sexes:
            for ai, age in enumerate(CONFIG["ages"]):
                last = work.loc[work.location_name.eq("Oman") & work.sex.eq(sex) & work.age.eq(age) & work.year.eq(origin), "rate"].iloc[0]
                for h in range(1, 6):
                    pred = last*np.exp(.001*h)
                    row = {"target": "Oman", "outcome": "incidence", "origin": origin,
                           "sex": sex, "age": age, "horizon": h, "forecast_year": origin+h,
                           "family": spec["family"], "setting_id": spec["setting_id"], "prediction": pred,
                           "log_prediction": np.log(pred), "status": "ok", "fallback_reason": "",
                           "parameter_count": 1, "grid_order": spec["grid_order"]}
                    rows.append(row)
    return {"forecasts": rows, "fits": [], "arima": [], "calibrations": []}

with tempfile.TemporaryDirectory(prefix="synthetic-pipeline-", dir=ROOT / "work/secondary-validation") as temporary:
    out = Path(temporary)
    with patch("run_secondary.source_panel", return_value=panel), patch("run_secondary.load_result", side_effect=fake_result):
        prepare_choices(task, CONFIG, out, "cpu")
        jobs = selected_jobs(task, CONFIG, out, "cpu")
        assert len(jobs) == 95
        assemble_issued(task, CONFIG, out, "cpu")
        score_trial(task, CONFIG, out)
        directory = out / "trials" / task["id"]
        report = json.loads((directory / "validation_report.json").read_text())
        report.update(synthetic=True, no_estimators_fitted=True, complete_adapter_pipeline=True)
        (ROOT / "work/secondary-validation/synthetic_pipeline.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
