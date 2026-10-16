"""Benchmark fixed real-data TCN fits without scoring any forecast outcomes."""

import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from gbd_park.pooled import build_examples, country_pool
from gbd_park.tcn import configure_torch, fit_tcn, predict_changes, state_fingerprint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def gpu_snapshot():
    """Sample total device use, including process contexts and unrelated workloads."""
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, check=True, timeout=10)
        rows = []
        for values in csv.reader(io.StringIO(result.stdout)):
            index, name, total, used, utilization, driver = [item.strip() for item in values]
            rows.append({"index": int(index), "name": name, "total_memory_mib": float(total),
                         "used_memory_mib": float(used), "utilization_percent": float(utilization),
                         "driver": driver})
        return {"time_utc": now(), "gpus": rows, "status": "ok"}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"time_utc": now(), "gpus": [], "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}"}


def hardware_snapshot():
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[name + "_kib"] = int(value.split()[0])
    except (OSError, ValueError):
        pass
    cpu_model = platform.processor()
    try:
        cpu_model = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"time_utc": now(), "platform": platform.platform(), "python": sys.version,
            "interpreter": sys.executable, "cpu_model": cpu_model, "logical_cpus": os.cpu_count(),
            "memory": memory, "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "numpy": np.__version__,
            "pandas": pd.__version__, "cuda_available": torch.cuda.is_available(),
            "gpu": gpu_snapshot()}


def fit_job(config, seed, device, epochs=50, include_predictions=False):
    """Worker sees only completed donor training rows through origin 2013."""
    started = time.perf_counter()
    resolved = configure_torch(seed, device)
    if resolved.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(resolved)
    origin = 2013
    donors = country_pool(config, config["primary_target"], "donor")
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(origin) & panel.outcome.eq("prevalence")
                      & panel.location_name.isin(donors)].copy()
    x, y, meta = build_examples(panel, config, origin, donors)
    if config["primary_target"] in set(meta.country) or int(meta.label_end.max()) > origin:
        raise AssertionError("Invalid benchmark source exclusion or cutoff")
    base = {"channels": 32, "weight_decay": 0.001, "epochs": epochs}
    fitted = fit_tcn(x, y, meta, base, config, seed=seed, device=device)
    before = state_fingerprint(fitted)
    prediction = predict_changes(fitted, x[:32])
    after = state_fingerprint(fitted)
    if before != after or not np.isfinite(prediction).all():
        raise AssertionError("Benchmark inference changed source state or was nonfinite")
    frozen = not fitted["model"].training and all(not p.requires_grad for p in fitted["model"].parameters())
    if not frozen:
        raise AssertionError("Benchmark model is not frozen")
    gpu_memory = {}
    if resolved.type == "cuda":
        gpu_memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(resolved),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(resolved),
                      "device_name": torch.cuda.get_device_name(resolved),
                      "total_device_bytes": torch.cuda.get_device_properties(resolved).total_memory}
    result = {"status": "ok", "seed": seed, "device": device, "pid": os.getpid(),
              "origin": origin, "base": base, "training_windows": len(meta), "donors": donors,
              "maximum_label_year": int(meta.label_end.max()), "parameter_count": fitted["parameter_count"],
              "fit_seconds": fitted["elapsed_seconds"], "worker_seconds": time.perf_counter() - started,
              "prediction_finite": bool(np.isfinite(prediction).all()), "frozen": frozen,
              "state_fingerprint_before": before, "state_fingerprint_after": after,
              "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
              "gpu_memory": gpu_memory}
    if include_predictions:
        result["predictions"] = prediction
    return result


def reproducibility_job(config):
    runs = [fit_job(config, 11, "cuda:0", epochs=2, include_predictions=True) for _ in range(2)]
    first, second = [run.pop("predictions") for run in runs]
    predictions_exact = bool(np.allclose(first, second, atol=0, rtol=0))
    fingerprints_exact = runs[0]["state_fingerprint_after"] == runs[1]["state_fingerprint_after"]
    return {"status": "ok" if predictions_exact and fingerprints_exact else "failed",
            "same_seed": 11, "epochs": 2, "device": "cuda:0", "runs": runs,
            "predictions_allclose_atol_0_rtol_0": predictions_exact,
            "state_fingerprints_identical": fingerprints_exact}


def benchmark_mode(config, device, workers):
    seeds = [11, 23, 37, 53]
    started = time.perf_counter()
    results, samples = [], [gpu_snapshot()]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(fit_job, config, seed, device): seed for seed in seeds}
        pending = set(futures)
        while pending:
            completed, pending = wait(pending, timeout=0.75, return_when=FIRST_COMPLETED)
            samples.append(gpu_snapshot())
            for future in completed:
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({"status": "failed", "seed": futures[future], "device": device,
                                    "reason": f"{type(exc).__name__}: {exc}"})
    elapsed = time.perf_counter() - started
    samples.append(gpu_snapshot())
    device_samples = [gpu for sample in samples for gpu in sample["gpus"] if gpu["index"] == 0]
    peak_fraction = max((row["used_memory_mib"] / row["total_memory_mib"] for row in device_samples), default=None)
    success = len(results) == len(seeds) and all(row["status"] == "ok" for row in results)
    return {"mode": f"{device}_workers{workers}", "device": device, "workers": workers,
            "status": "ok" if success else "failed", "seeds": seeds,
            "whole_batch_seconds_including_spawn_shutdown_and_monitoring": elapsed,
            "successful_fits_per_second": sum(row["status"] == "ok" for row in results) / elapsed,
            "fits_per_second": len(seeds) / elapsed if success else None,
            "sampled_peak_total_gpu_memory_fraction": peak_fraction,
            "gpu_memory_sampling_seconds": 0.75, "gpu_samples": samples,
            "jobs": sorted(results, key=lambda row: row["seed"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="work/tcn-benchmark/report.json")
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing benchmark report")
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    paths = [Path(__file__).resolve(), ROOT / "src/gbd_park/tcn.py", ROOT / "src/gbd_park/pooled.py",
             config_path, ROOT / "study_design/locked_v1/protocol.md",
             ROOT / "data/processed/design_v1/regional_outcomes.csv"]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    lock = json.loads((ROOT / "study_design/locked_v1/lock_manifest.json").read_text())
    for section in ["design_sha256", "output_sha256"]:
        for name, expected in lock[section].items():
            if name in hashes and hashes[name] != expected:
                raise ValueError(f"Benchmark input does not match locked artifact: {name}")
    report = {"created_utc": now(), "status": "running", "scope": "TCN training throughput only; no forecast error scoring",
              "origin": 2013, "base": {"channels": 32, "weight_decay": 0.001, "epochs": 50},
              "hardware_before": hardware_snapshot(), "input_sha256": hashes, "modes": [],
              "recommendation": None, "cuda_reproducibility": None}

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    # Exclusive first creation prevents an accidentally concurrent benchmark
    # from taking ownership of the same output path.
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    cuda_available = report["hardware_before"]["cuda_available"]
    for device, workers in [("cpu", 4), ("cuda:0", 1), ("cuda:0", 2)]:
        print(f"Benchmarking {device} with {workers} worker(s)", flush=True)
        if device.startswith("cuda") and not cuda_available:
            mode = {"mode": f"{device}_workers{workers}", "device": device, "workers": workers,
                    "status": "failed", "reason": "CUDA requested but unavailable; no CPU substitution"}
        else:
            mode = benchmark_mode(config, device, workers)
        report["modes"].append(mode)
        save()
        print(json.dumps({key: value for key, value in mode.items() if key not in {"jobs", "gpu_samples"}}), flush=True)
    if cuda_available:
        print("Checking repeatability of two real-data two-epoch CUDA fits", flush=True)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            try:
                report["cuda_reproducibility"] = pool.submit(reproducibility_job, config).result()
            except Exception as exc:
                report["cuda_reproducibility"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    else:
        report["cuda_reproducibility"] = {"status": "failed", "reason": "CUDA unavailable"}
    eligible = [mode for mode in report["modes"] if mode["status"] == "ok" and (
        mode["device"] == "cpu" or (mode["sampled_peak_total_gpu_memory_fraction"] is not None
                                   and mode["sampled_peak_total_gpu_memory_fraction"] < 0.60))]
    if eligible:
        best = max(eligible, key=lambda mode: mode["fits_per_second"])
        report["recommendation"] = {"device": best["device"], "workers": best["workers"], "mode": best["mode"],
                                    "fits_per_second": best["fits_per_second"],
                                    "criterion": "Highest measured whole-batch throughput; GPU candidates require sampled total memory below 60%",
                                    "memory_sampling_limitation": "Samples may miss brief peaks; per-job allocator peaks are recorded separately"}
    report["hardware_after"] = hardware_snapshot()
    report["finished_utc"] = now()
    unchanged = all(sha(ROOT / name) == expected for name, expected in hashes.items())
    report["inputs_unchanged"] = unchanged
    passed = unchanged and all(mode["status"] == "ok" for mode in report["modes"]) and report["cuda_reproducibility"]["status"] == "ok"
    report["status"] = "complete" if passed else "failed"
    save()
    print(json.dumps({"status": report["status"], "output": str(output), "recommendation": report["recommendation"],
                      "cuda_reproducibility": report["cuda_reproducibility"]["status"]}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
