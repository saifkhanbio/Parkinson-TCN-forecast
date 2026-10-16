"""Compact causal TCN fitted to completed, hierarchically weighted windows.

Checkpoint loading is intended only for files produced by this repository;
joblib, like pickle, must never load an untrusted downloaded checkpoint.
"""

import hashlib
import itertools
import json
import os
import random
import time
from pathlib import Path

# cuBLAS reads this before creating its workspace. Set it before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as functional

from gbd_park.pooled import balanced_weights


def configure_torch(seed, device):
    """Set reproducible numerical choices; never substitute CPU for requested CUDA."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("The random seed must be an integer")
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized without deterministic cuBLAS configuration")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
        torch.cuda.get_device_properties(resolved)
        torch.cuda.manual_seed_all(int(seed))
    return resolved


class CompactTCN(nn.Module):
    """Three causal convolutions, final temporal state, and a static linear head."""

    def __init__(self, channels, config):
        super().__init__()
        spec = config["models"]["tcn"]
        self.channels = int(channels)
        self.window = int(config["calendar"]["window"])
        self.static_features = 2 + len(config["ages"])
        self.horizons = len(config["calendar"]["horizons"])
        self.kernel = int(spec["kernel"])
        self.dilations = tuple(int(d) for d in spec["dilations"])
        self.dropout_probability = float(spec["dropout"])
        if self.channels <= 0 or self.window != 8 or self.horizons != 5:
            raise ValueError("TCN requires positive channels, eight inputs, and five horizons")
        if self.kernel != 3 or self.dilations != (1, 2, 4):
            raise ValueError("TCN requires the locked kernel and dilation architecture")
        self.convolutions = nn.ModuleList([
            nn.Conv1d(1 if index == 0 else self.channels, self.channels,
                      kernel_size=self.kernel, dilation=dilation, padding=0)
            for index, dilation in enumerate(self.dilations)
        ])
        self.dropout = nn.Dropout(self.dropout_probability)
        self.head = nn.Linear(self.channels + self.static_features, self.horizons)
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if self.parameter_count >= int(spec["maximum_parameters"]):
            raise ValueError("TCN violates the strict parameter cap")

    def temporal_features(self, sequence):
        """Return all temporal states; no state accesses a later input position."""
        if sequence.ndim != 3 or sequence.shape[1:] != (1, self.window):
            raise ValueError("Temporal input must have shape N x 1 x 8")
        hidden = sequence
        for convolution, dilation in zip(self.convolutions, self.dilations):
            hidden = functional.pad(hidden, ((self.kernel - 1) * dilation, 0))
            hidden = self.dropout(functional.relu(convolution(hidden)))
        return hidden

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.window + self.static_features:
            raise ValueError("TCN input must contain eight lags and the static features")
        temporal = self.temporal_features(x[:, :self.window].unsqueeze(1))[:, :, -1]
        return self.head(torch.cat((temporal, x[:, self.window:]), dim=1))


def base_grid(config):
    """The locked eight settings, in channels/weight-decay/epochs order."""
    spec = config["models"]["tcn"]
    return [{"channels": channels, "weight_decay": decay, "epochs": epochs}
            for channels, decay, epochs in itertools.product(
                spec["channels"], spec["weight_decay"], spec["epochs"])]


def fit_tcn(x, y, meta, base, config, seed, device="cpu"):
    """Fit once on source rows; return frozen CPU weights for target calibration."""
    started = time.perf_counter()
    resolved = configure_torch(seed, device)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    expected_features = int(config["calendar"]["window"]) + 2 + len(config["ages"])
    if (x.ndim != 2 or x.shape[1] != expected_features or not len(x)
            or y.shape != (len(x), 5) or len(meta) != len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Training needs aligned, nonempty, finite features and five-horizon labels")
    if not {"country", "sex", "age"}.issubset(meta.columns):
        raise ValueError("Training metadata must identify country, sex, and age")
    if meta[["country", "sex", "age"]].isna().any().any():
        raise ValueError("Training metadata contains missing country, sex, or age")
    epochs = int(base["epochs"])
    if epochs <= 0 or epochs != base["epochs"] or float(base["weight_decay"]) < 0:
        raise ValueError("Training epochs must be positive integers and weight decay nonnegative")
    weights = balanced_weights(meta)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Training weights must be positive and finite")
    scaler = StandardScaler().fit(x, sample_weight=weights)
    source_x = torch.as_tensor(scaler.transform(x), dtype=torch.float32, device=resolved)
    source_y = torch.as_tensor(y, dtype=torch.float32, device=resolved)
    source_weight = torch.as_tensor(weights, dtype=torch.float32, device=resolved)
    model = CompactTCN(base["channels"], config).to(resolved)
    spec = config["models"]["tcn"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(spec["learning_rate"]),
        betas=(0.9, 0.999), eps=1e-8, weight_decay=float(base["weight_decay"]),
        amsgrad=False, foreach=False, fused=False,
    )
    batch_size = int(spec["batch_size"])
    shuffle = torch.Generator(device="cpu").manual_seed(int(seed))
    losses = []
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(x), generator=shuffle)
        cumulative_loss = 0.0
        for start in range(0, len(x), batch_size):
            index = permutation[start:start + batch_size].to(resolved)
            optimizer.zero_grad(set_to_none=True)
            error = (model(source_x[index]) - source_y[index]).abs().mean(dim=1)
            # Mean-one weights retain the global hierarchical objective under
            # uniformly shuffled minibatches, including the final short batch.
            loss = (error * source_weight[index]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite TCN training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip"]),
                                     error_if_nonfinite=True, foreach=False)
            optimizer.step()
            cumulative_loss += float(loss.detach().cpu()) * len(index)
        losses.append(cumulative_loss / len(x))
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    model = model.to("cpu").eval()
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
        if not torch.isfinite(parameter).all():
            raise FloatingPointError("Nonfinite fitted TCN parameter")
    return {"model": model, "scaler": scaler, "parameter_count": model.parameter_count,
            "base": dict(base), "seed": int(seed), "training_losses": losses,
            "epochs": epochs, "device": str(resolved),
            "elapsed_seconds": time.perf_counter() - started}


def predict_changes(fitted, x):
    """Predict with the frozen CPU model, without changing any fitted state."""
    model = fitted["model"]
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Prediction requires a frozen model in evaluation mode")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Persisted inference models must be on CPU")
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Prediction requires a nonempty finite feature matrix")
    transformed = torch.as_tensor(fitted["scaler"].transform(x), dtype=torch.float32)
    with torch.inference_mode():
        prediction = model(transformed).numpy().astype(float)
    if prediction.shape != (len(x), 5) or not np.isfinite(prediction).all():
        raise ValueError("Invalid TCN predictions")
    return prediction


def state_fingerprint(fitted):
    """Stable SHA-256 of architecture, tensor values, and learned scaler fields."""
    digest = hashlib.sha256()
    model = fitted["model"]
    architecture = {"class": type(model).__name__, "channels": model.channels,
                    "window": model.window, "static_features": model.static_features,
                    "horizons": model.horizons, "kernel": model.kernel,
                    "dilations": model.dilations, "dropout": model.dropout_probability}
    digest.update(json.dumps(architecture, sort_keys=True, separators=(",", ":")).encode())

    def add_array(name, value):
        array = np.ascontiguousarray(np.asarray(value))
        metadata = {"name": name, "dtype": array.dtype.str, "shape": array.shape}
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))

    for name, tensor in sorted(model.state_dict().items()):
        add_array("model." + name, tensor.detach().cpu().numpy())
    scaler = fitted["scaler"]
    digest.update(json.dumps({"with_mean": scaler.with_mean, "with_std": scaler.with_std},
                             sort_keys=True, separators=(",", ":")).encode())
    for name in ["mean_", "var_", "scale_", "n_features_in_", "n_samples_seen_"]:
        add_array("scaler." + name, getattr(scaler, name))
    return digest.hexdigest()


def save_checkpoint(fitted, path):
    """Save trusted local fitted state and its fingerprint without GPU tensors."""
    if fitted["model"].training or any(p.requires_grad for p in fitted["model"].parameters()):
        raise ValueError("Only frozen evaluation models can be saved")
    if any(p.device.type != "cpu" for p in fitted["model"].parameters()):
        raise ValueError("Checkpoint model must be on CPU")
    payload = {"format": "gbd_park_tcn_v1", "fingerprint": state_fingerprint(fitted),
               "fitted": fitted}
    joblib.dump(payload, Path(path), compress=3)


def load_checkpoint(path):
    """Load only a trusted local checkpoint, verifying its saved state fingerprint."""
    payload = joblib.load(Path(path))
    if payload.get("format") != "gbd_park_tcn_v1":
        raise ValueError("Unrecognized TCN checkpoint format")
    fitted = payload["fitted"]
    if state_fingerprint(fitted) != payload["fingerprint"]:
        raise ValueError("TCN checkpoint state fingerprint does not match")
    if fitted["model"].training or any(p.requires_grad for p in fitted["model"].parameters()):
        raise ValueError("Loaded TCN checkpoint is not frozen for inference")
    if any(p.device.type != "cpu" for p in fitted["model"].parameters()):
        raise ValueError("Loaded TCN checkpoint must be on CPU")
    return fitted
