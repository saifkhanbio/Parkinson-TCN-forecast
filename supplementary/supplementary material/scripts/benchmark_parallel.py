"""Measure larger CPU/GPU queues on fixed real-data fits, without outcome scoring."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from benchmark_tcn import ROOT, fit_job, hardware_snapshot, gpu_snapshot, sha, now


def benchmark(config, cpu_workers, gpu_workers, fits_per_device):
    started = time.perf_counter()
    pools, futures, jobs, samples = [], {}, [], []
    psutil.cpu_percent(interval=None, percpu=True)
    context = multiprocessing.get_context("spawn")
    try:
        for device, workers in [("cpu", cpu_workers), ("cuda:0", gpu_workers)]:
            if not workers:
                continue
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
            pools.append(pool)
            for index in range(fits_per_device):
                seed = [11, 23, 37, 53, 71, 89][index % 6]
                future = pool.submit(fit_job, config, seed, device, 50)
                futures[future] = {"device": device, "seed": seed, "job": index}
        pending = set(futures)
        while pending:
            completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                try:
                    jobs.append({**futures[future], **future.result()})
                except Exception as exc:
                    jobs.append({**futures[future], "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            if not samples or time.perf_counter() - samples[-1]["elapsed_seconds"] - started >= 2:
                samples.append({"elapsed_seconds": time.perf_counter()-started,
                                "cpu_percent_per_core": psutil.cpu_percent(interval=None, percpu=True),
                                "memory_available_gib": psutil.virtual_memory().available/2**30,
                                "gpu": gpu_snapshot()})
        for pool in pools:
            pool.shutdown()
        pools.clear()
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)
    elapsed = time.perf_counter() - started
    success = all(job["status"] == "ok" for job in jobs)
    gpu_rows = [gpu for sample in samples for gpu in sample["gpu"]["gpus"]]
    repeatable = True
    for device in {job["device"] for job in jobs}:
        for seed in {job["seed"] for job in jobs}:
            group = [job for job in jobs if job["device"] == device and job["seed"] == seed and job["status"] == "ok"]
            repeatable &= len({job["state_fingerprint_after"] for job in group}) <= 1
            repeatable &= len({job["prediction_sha256"] for job in group}) <= 1
    return {"mode": f"cpu{cpu_workers}_gpu{gpu_workers}", "cpu_workers": cpu_workers,
            "gpu_workers": gpu_workers, "fits": len(jobs), "status": "ok" if success and repeatable else "failed",
            "elapsed_seconds": elapsed, "fits_per_second": len(jobs)/elapsed if success else None,
            "same_device_seed_repeatable": repeatable,
            "maximum_gpu_memory_mib": max((g["used_memory_mib"] for g in gpu_rows), default=0),
            "maximum_gpu_utilization_percent": max((g["utilization_percent"] for g in gpu_rows), default=0),
            "mean_sampled_cpu_percent": float(np.mean([np.mean(s["cpu_percent_per_core"]) for s in samples])),
            "minimum_available_ram_gib": min(s["memory_available_gib"] for s in samples),
            "jobs": jobs, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="work/parallel-benchmark/report.json")
    parser.add_argument("--fits", type=int, default=12)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError("Refusing to overwrite a benchmark")
    if args.fits < 12:
        raise ValueError("At least twelve fits per queue are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this execution environment; request GPU access")
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    paths = [config_path, Path(__file__), ROOT / "scripts/benchmark_tcn.py",
             ROOT / "src/gbd_park/tcn.py", ROOT / "src/gbd_park/pooled.py",
             ROOT / "data/processed/design_v1/regional_outcomes.csv"]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    report = {"status": "running", "created_utc": now(), "scope": "Real origin-2013 source training throughput; no forecast error scoring",
              "hardware": hardware_snapshot(), "input_sha256": hashes, "modes": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    for cpu, gpu in [(4, 0), (8, 0), (12, 0), (0, 2), (0, 4), (0, 8), (8, 4)]:
        print(f"Benchmarking CPU workers={cpu}, GPU workers={gpu}", flush=True)
        result = benchmark(config, cpu, gpu, args.fits)
        report["modes"].append(result)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in {"jobs", "samples"}}), flush=True)
    assert all(sha(ROOT / path) == value for path, value in hashes.items())
    passed = [mode for mode in report["modes"] if mode["status"] == "ok"]
    cpu = max((m for m in passed if m["gpu_workers"] == 0), key=lambda m: m["fits_per_second"])
    gpu = max((m for m in passed if m["cpu_workers"] == 0), key=lambda m: m["fits_per_second"])
    report["recommendation"] = {"cpu_workers": cpu["cpu_workers"], "gpu_workers": gpu["gpu_workers"],
                                "cpu_fits_per_second": cpu["fits_per_second"], "gpu_fits_per_second": gpu["fits_per_second"],
                                "mixed_queue_fits_per_second": report["modes"][-1]["fits_per_second"],
                                "note": "Device-specific models can differ numerically; do not mix CPU and GPU seeds within one ensemble or silently replace v1."}
    report["status"] = "complete" if len(passed) == len(report["modes"]) else "complete_with_failed_modes"
    report["finished_utc"] = now()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["recommendation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
