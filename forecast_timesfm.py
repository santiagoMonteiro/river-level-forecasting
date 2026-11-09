"""TimesFM 3.0 zero-shot forecasts; select context length on validation only."""

import time

import numpy as np

import experiment_config as cfg
from experiment_common import run_experiment
from terminal_progress import progress, status

BACKEND = "torch"  # "mlx" is also supported on Apple Silicon with MLX installed.
DEVICE = "cpu"  # Used by the torch backend; set explicitly to "cuda" if available.
CHECKPOINT = "google/timesfm-3.0-pytorch"
CHECKPOINT_REVISION = "main"  # Set to the recorded commit hash for an exact rerun.
CACHE_DIR = cfg.PROJECT_DIR / ".model_cache"
BATCH_SIZE = 8
CANDIDATES = [{"context_length": length} for length in (365, 730, 1460)]


class TimesFMForecaster:
    def __init__(self, config, seed, pretrained, checkpoint_metadata):
        self.config = config
        self.pretrained = pretrained
        self.metadata = {**checkpoint_metadata, "context_length": config["context_length"],
                         "fine_tuned": False, "point_forecast": "median"}

    def fit(self, train, validation=None, fixed_epochs=None):
        if len(train) < self.config["context_length"]:
            raise ValueError("Training history is shorter than the TimesFM context")

    def forecast(self, histories, horizon):
        context = self.config["context_length"]
        if any(len(history) < context for history in histories):
            raise ValueError("A forecast history is shorter than the selected TimesFM context")
        forecasts = []
        for start in progress(range(0, len(histories), BATCH_SIZE),
                              desc="TimesFM forecast batches", unit="batch", leave=False):
            batch = [np.asarray(history[-context:], dtype=np.float32)
                     for history in histories[start:start + BATCH_SIZE]]
            outputs = list(self.pretrained.predict_batch(batch, horizon=horizon,
                           return_quantiles=False, use_symmetric_averaging=False))
            if len(outputs) != len(batch):
                raise ValueError("TimesFM returned a different number of forecasts than requested")
            for output in outputs:
                forecast = np.asarray(output.forecast, dtype=float).reshape(-1)
                if forecast.size != horizon:
                    raise ValueError(f"Unexpected TimesFM forecast shape: {np.shape(output.forecast)}")
                forecasts.append(forecast)
        return np.asarray(forecasts)


def main():
    # Delay imports and downloads until this model is explicitly executed.
    from huggingface_hub import snapshot_download
    start = time.perf_counter()
    if BACKEND == "torch":
        from timesfm3.torch import TimesFM3Forecaster
        options = {"device": DEVICE}
    elif BACKEND == "mlx":
        from timesfm3.mlx import TimesFM3Forecaster
        options = {}
    else:
        raise ValueError("BACKEND must be 'torch' or 'mlx'")
    status(f"Downloading/checking TimesFM checkpoint: {CHECKPOINT} ({CHECKPOINT_REVISION})")
    snapshot = snapshot_download(CHECKPOINT, revision=CHECKPOINT_REVISION, cache_dir=str(CACHE_DIR),
                                 allow_patterns=["*.json", "*.safetensors"])
    status(f"Loading TimesFM weights: backend={BACKEND}, device={DEVICE if BACKEND == 'torch' else 'mlx'}")
    pretrained = TimesFM3Forecaster.from_pretrained(snapshot, **options)
    from pathlib import Path
    checkpoint_metadata = {"checkpoint": CHECKPOINT, "resolved_revision": Path(snapshot).name,
                           "backend": BACKEND, "device": DEVICE if BACKEND == "torch" else "mlx",
                           "checkpoint_load_seconds": time.perf_counter() - start}
    print(f"Loaded TimesFM: {checkpoint_metadata}", flush=True)

    def factory(config, seed):
        return TimesFMForecaster(config, seed, pretrained, checkpoint_metadata)

    run_experiment("timesfm", CANDIDATES, [0], factory,
                   {**checkpoint_metadata, "batch_size": BATCH_SIZE, "symmetric_averaging": False})


if __name__ == "__main__":
    main()
