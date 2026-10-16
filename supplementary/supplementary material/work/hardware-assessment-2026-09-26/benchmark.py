"""Synthetic hardware benchmark; does not fit or score study outcomes."""

from pathlib import Path
from datetime import datetime, timezone
import csv
import gc
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import time
import warnings

import numpy as np
import psutil
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
REPORT_PATH = ROOT / "benchmark_results.json"
REPORT = {"started_utc": datetime.now(timezone.utc).isoformat(), "synthetic_only": True}


def emit(section, record):
    REPORT.setdefault(section, []).append(record)
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    print(json.dumps({"section": section, **record}), flush=True)


def nvidia_query():
    query = "name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw"
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    return {"fields": query, "output": result.stdout.strip(), "error": result.stderr.strip()}


def inventory():
    versions = {}
    for name in ["torch", "numpy", "pandas", "scikit-learn", "statsmodels", "scipy", "openpyxl", "xgboost", "lightgbm"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu_text = Path("/proc/cpuinfo").read_text()
    cpu_model = next(line.split(":", 1)[1].strip() for line in cpu_text.splitlines() if line.startswith("model name"))
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(REPO)
    cgroup = {}
    for name in ["memory.max", "cpu.max"]:
        path = Path("/sys/fs/cgroup") / name
        cgroup[name] = path.read_text().strip() if path.exists() else None
    data_file = REPO / "age_standard/parkinsons_ml_ready_wide.csv"
    # Read keys only; no real outcome values enter model fitting or scoring.
    with data_file.open(encoding="utf-8-sig", newline="") as handle:
        keys = [(row["location_id"], int(row["year"])) for row in csv.DictReader(handle)]
    locations = len({key[0] for key in keys})
    years = sorted({key[1] for key in keys})
    assert torch.cuda.is_available(), "CUDA inaccessible; do not report a GPU benchmark as completed."
    p = torch.cuda.get_device_properties(0)
    emit("inventory", {
        "python": platform.python_version(), "executable": os.sys.executable,
        "platform": platform.platform(), "cpu_model": cpu_model,
        "logical_cpus_available": len(os.sched_getaffinity(0)),
        "ram_total_gib": vm.total / 2**30, "ram_available_gib": vm.available / 2**30,
        "swap_gib": psutil.swap_memory().total / 2**30,
        "disk_free_gib": disk.free / 2**30, "cgroup": cgroup,
        "versions": versions, "torch_cuda_build": torch.version.cuda,
        "gpu_name": p.name, "gpu_vram_gib": p.total_memory / 2**30,
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "gpu_snapshot": nvidia_query(), "input_csv_mib": data_file.stat().st_size / 2**20,
        "locations": locations, "rows": len(keys), "years": [min(years), max(years)],
        "donors_before_eligibility_filter": locations - 1,
    })
    return locations - 1


class SmallTCN(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(1 if i == 0 else channels, channels, kernel_size=3, dilation=d)
            for i, d in enumerate([1, 2, 4])
        ])
        self.dropout = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(channels + 2, 5)

    def forward(self, x, static):
        for conv, dilation in zip(self.convs, [1, 2, 4]):
            x = torch.nn.functional.pad(x, (2 * dilation, 0))
            x = self.dropout(torch.relu(conv(x)))
        return self.head(torch.cat([x[:, :, -1], static], dim=1))


def synthetic_data(n):
    rng = np.random.default_rng(1729)
    slopes = rng.normal(0.005, 0.01, size=(n, 1)).astype(np.float32)
    x = np.cumsum(slopes + rng.normal(0, 0.008, size=(n, 8)).astype(np.float32), axis=1)
    x -= x[:, -1:]
    static = np.column_stack([rng.uniform(3, 6, n), np.arange(n) % 2]).astype(np.float32)
    y = slopes * np.arange(1, 6, dtype=np.float32) + rng.normal(0, 0.005, size=(n, 5)).astype(np.float32)
    return x, static, y


def benchmark_tcn(device, channels, threads, x_np, static_np, y_np, epochs=5, label="2018_origin"):
    torch.set_num_threads(threads)
    torch.manual_seed(11)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    x = torch.as_tensor(x_np[:, None, :], device=device)
    static = torch.as_tensor(static_np, device=device)
    y = torch.as_tensor(y_np, device=device)
    model = SmallTCN(channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    batch_size = 64

    def step(ids):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[ids], static[ids])
        loss = torch.nn.functional.l1_loss(prediction, y[ids])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss

    for _ in range(20):
        step(torch.arange(min(batch_size, len(x)), device=device))
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    epoch_seconds = []
    for _ in range(epochs):
        start = time.perf_counter()
        order = torch.randperm(len(x), device=device)
        for offset in range(0, len(x), batch_size):
            loss = step(order[offset:offset + batch_size])
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - start)
    model.eval()
    with torch.no_grad():
        out = model(x[:12], static[:12])
    assert torch.isfinite(out).all().item()
    assert torch.isfinite(loss).item()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            model(x[:12], static[:12])
    if device == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - start) * 10
    seconds_per_epoch = float(np.median(epoch_seconds))
    result = {
        "device": device, "channels": channels, "cpu_threads": threads,
        "label": label, "samples": len(x), "batch_size": batch_size,
        "steps_per_epoch": math.ceil(len(x) / batch_size), "measured_epochs": epochs,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "epoch_seconds": epoch_seconds, "median_epoch_seconds": seconds_per_epoch,
        "projected_50_epoch_seconds": seconds_per_epoch * 50,
        "projected_100_epoch_seconds": seconds_per_epoch * 100,
        "inference_12_series_ms": inference_ms,
        "checkpoint_weights_kib": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024,
        "process_peak_rss_mib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "synthetic_last_batch_loss": float(loss.item()), "finite_output": True,
    }
    if device == "cuda":
        result.update(
            gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            gpu_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            gpu_snapshot=nvidia_query(),
        )
    emit("tcn", result)
    del model, optimizer, x, static, y, out, loss
    gc.collect()


def benchmark_cpu_baselines(x, static, y):
    features = np.column_stack([x + static[:, :1], x, static])
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        model = Ridge(alpha=1)
        model.fit(features, y)
        prediction = model.predict(features[:12])
        assert np.isfinite(prediction).all()
        emit("cpu_baselines", {"model": "ridge", "five_horizons_fit_and_predict_seconds": time.perf_counter() - start,
                               "samples": len(x), "features": features.shape[1], "threads": 1})
        start = time.perf_counter()
        for horizon in range(5):
            model = GradientBoostingRegressor(loss="absolute_error", learning_rate=0.03,
                                             max_depth=2, n_estimators=300, min_samples_leaf=5, random_state=11)
            fit_start = time.perf_counter()
            model.fit(features, y[:, horizon])
            assert np.isfinite(model.predict(features[:12])).all()
            emit("boosting_horizons", {"horizon": horizon + 1, "fit_and_predict_seconds": time.perf_counter() - fit_start})
        emit("cpu_baselines", {"model": "gradient_boosting", "five_horizons_fit_and_predict_seconds": time.perf_counter() - start,
                               "samples": len(x), "features": features.shape[1], "trees_per_horizon": 300,
                               "depth": 2, "threads": 1})
        rng = np.random.default_rng(1729)
        z = 5 + np.arange(29) * 0.005 + np.cumsum(rng.normal(0, 0.004, 29))
        start = time.perf_counter()
        ets = ExponentialSmoothing(z, trend="add", damped_trend=True, initialization_method="estimated").fit()
        assert np.isfinite(ets.forecast(5)).all()
        emit("cpu_baselines", {"model": "damped_ets", "fit_and_predict_seconds": time.perf_counter() - start,
                               "annual_observations": len(z), "threads": 1})
        start = time.perf_counter()
        failures = []
        warning_count = 0
        models = 0
        for p in range(3):
            for q in range(3):
                if p + q > 3:
                    continue
                for d in [0, 1]:
                    try:
                        with warnings.catch_warnings(record=True) as seen:
                            warnings.simplefilter("always")
                            fit = ARIMA(z, order=(p, d, q), trend="t" if d else "c").fit()
                            prediction = fit.forecast(5)
                        warning_count += len(seen)
                        assert np.isfinite(prediction).all()
                        models += 1
                    except Exception as exc:
                        failures.append({"order": [p, d, q], "error": str(exc)})
        emit("cpu_baselines", {"model": "arima_grid", "grid_fit_and_predict_seconds": time.perf_counter() - start,
                               "annual_observations": len(z), "successful_models": models,
                               "warnings": warning_count, "failures": failures, "threads": 1})


def main():
    donors = inventory()
    n_2018 = donors * 2 * 17
    n_2023 = donors * 2 * 22
    x, static, y = synthetic_data(n_2018)
    emit("data_size", {"training_windows_2018": n_2018, "training_windows_2023": n_2023,
                       "2018_float32_input_and_label_mib": (x.nbytes + static.nbytes + y.nbytes) / 2**20})
    for device, channels, threads in [("cpu", 16, 1), ("cpu", 32, 1), ("cpu", 32, 2),
                                      ("cpu", 32, 4), ("cuda", 16, 1), ("cuda", 32, 1)]:
        benchmark_tcn(device, channels, threads, x, static, y)
    x_final, static_final, y_final = synthetic_data(n_2023)
    benchmark_tcn("cuda", 32, 1, x_final, static_final, y_final, label="2023_origin")
    benchmark_cpu_baselines(x, static, y)
    emit("completed", {"finished_utc": datetime.now(timezone.utc).isoformat(),
                       "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       "gpu_snapshot": nvidia_query()})


if __name__ == "__main__":
    main()
