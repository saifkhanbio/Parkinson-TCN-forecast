"""Timing-only CPU4/GPU4 benchmark; no verification outcomes are scored."""
import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import pandas as pd
import torch
from gbd_park import global_asr as asr
from run_global_asr import write_json


def fit(job):
    started = time.perf_counter()
    panel = pd.read_csv(ROOT / "work/global-asr-validation/source_panel.csv.gz")
    registry = pd.read_csv(ROOT / "work/global-asr-validation/location_registry.csv")
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    work, cfg = asr.context(panel, config, registry, "Saudi Arabia", "prevalence", job["scope"])
    value = asr.fit_seed(work, cfg, job["seed"], job["device"])
    return {**job, "elapsed_seconds": time.perf_counter()-started,
            "fit_elapsed_seconds": value["audit"]["elapsed_seconds"],
            "source_windows": value["audit"]["training_windows"],
            "parameters": value["audit"]["parameter_count"],
            "status": value["audit"]["status"], "reason": value["audit"]["reason"],
            "settings": asr.fixed_base(cfg)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requires CUDA; no fallback")
    destination = ROOT / "work/global-asr-validation" / ("benchmark_gpu.json" if args.device.startswith("cuda") else "benchmark_cpu.json")
    if destination.exists():
        raise FileExistsError("Preserve the prior timing record")
    fixed = ["study_design/global_asr_implementation.md", "src/gbd_park/global_asr.py", "work/global-asr-validation/benchmark.py"]
    identity = {name: asr.digest(ROOT / name) for name in fixed}
    specs = [{"device": args.device, "scope": scope, "seed": seed} for scope in asr.SCOPES for seed in [11, 23]]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as executor:
        for future in as_completed([executor.submit(fit, spec) for spec in specs]):
            value = future.result(); results.append(value)
            print(json.dumps(value), flush=True)
    if identity != {name: asr.digest(ROOT / name) for name in fixed}:
        raise ValueError("Benchmark code/specification changed while running")
    report = {"purpose": "timing_only_no_evaluation_scores_no_predictive_setting_selection",
              "device": args.device, "workers": 4, "jobs": results,
              "elapsed_seconds": time.perf_counter()-started, "source_and_code_sha256": identity,
              "passed": all(item["status"] == "ok" for item in results)}
    write_json(destination, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
