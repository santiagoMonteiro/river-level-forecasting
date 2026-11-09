"""Shared temporal splits, model selection, output schema and audit records."""

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import experiment_config as cfg


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(path)


def load_inputs():
    """Reject missing days, duplicate dates, invalid values and imputed targets."""
    dates = pd.date_range(cfg.DATA_START, cfg.TEST_END, freq="D", name="date")
    if not (pd.Timestamp(cfg.DATA_START) < pd.Timestamp(cfg.TRAIN_END)
            < pd.Timestamp(cfg.VALIDATION_END) < pd.Timestamp(cfg.TEST_END)):
        raise ValueError("Expected DATA_START < TRAIN_END < VALIDATION_END < TEST_END")
    if cfg.ORIGIN_STEP_DAYS < 1 or min(cfg.HORIZONS) < 1 or max(cfg.HORIZONS) != cfg.MAX_HORIZON:
        raise ValueError("Invalid horizons or origin step")
    series_by_station, hashes = {}, {}
    for station, code in cfg.STATIONS.items():
        path = cfg.INPUT_DIR / f"{station}_river_stage_complete.csv"
        frame = pd.read_csv(path, parse_dates=["date"])
        if frame["date"].isna().any() or frame["date"].duplicated().any():
            raise ValueError(f"{station}: invalid or duplicate dates")
        series = frame.set_index("date")["river_stage"].sort_index().reindex(dates)
        series = pd.to_numeric(series, errors="raise").astype(float)
        if not np.isfinite(series.to_numpy()).all():
            raise ValueError(f"{station}: input must cover every day from {dates[0]} to {dates[-1]}")
        if cfg.VERIFY_OBSERVED_EVALUATION:
            from prepare_training_series import load_series
            matches = list((cfg.PROJECT_DIR / "data").rglob(f"{code}_Cotas.csv"))
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one raw level-2 source for {station}")
            evaluation = series.loc[pd.Timestamp(cfg.TRAIN_END) + pd.Timedelta(days=1):]
            original = load_series(matches[0]).reindex(evaluation.index)
            if not np.isfinite(original.to_numpy()).all() or not np.allclose(original, evaluation):
                raise ValueError(f"{station}: validation/test must contain original level-2 observations")
            hashes[f"raw_{station}"] = digest(matches[0])
        series_by_station[station] = series
        hashes[station] = digest(path)
    protocol = {
        "data_start": cfg.DATA_START, "train_end": cfg.TRAIN_END,
        "validation_end": cfg.VALIDATION_END, "test_end": cfg.TEST_END,
        "horizons": list(cfg.HORIZONS), "max_horizon": cfg.MAX_HORIZON,
        "origin_step_days": cfg.ORIGIN_STEP_DAYS, "stations": cfg.STATIONS,
        "lstm_seeds": list(cfg.LSTM_SEEDS), "input_sha256": hashes,
        "observed_evaluation_verified": cfg.VERIFY_OBSERVED_EVALUATION,
        "selection": "mean validation MAE over selected horizons and seeds",
        "test_update": "new observations allowed; fitted parameters frozen",
        "common_code_sha256": digest(__file__),
    }
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return series_by_station, protocol, protocol_id


def origins_for(split):
    if split == "validation":
        cutoff, end = cfg.TRAIN_END, cfg.VALIDATION_END
    elif split == "test":
        cutoff, end = cfg.VALIDATION_END, cfg.TEST_END
    else:
        raise ValueError(f"Unknown split: {split}")
    origins = pd.date_range(cutoff, pd.Timestamp(end) - pd.Timedelta(days=cfg.MAX_HORIZON),
                            freq=f"{cfg.ORIGIN_STEP_DAYS}D")
    if origins.empty:
        raise ValueError(f"No complete {cfg.MAX_HORIZON}-day forecasts in {split}")
    return origins


def forecast_inputs(series, origins):
    """Histories end at the origin; targets start strictly after it."""
    values = series.to_numpy(dtype=float)
    positions = series.index.get_indexer(origins)
    if (positions < 0).any() or (positions + cfg.MAX_HORIZON >= len(values)).any():
        raise ValueError("Origins or complete forecast targets are outside the series")
    histories = [values[:position + 1].copy() for position in positions]
    targets = np.stack([values[position + 1:position + 1 + cfg.MAX_HORIZON] for position in positions])
    return histories, targets


def checked_forecast(model, histories):
    forecasts = np.asarray(model.forecast(histories, cfg.MAX_HORIZON), dtype=float)
    if forecasts.shape != (len(histories), cfg.MAX_HORIZON) or not np.isfinite(forecasts).all():
        raise ValueError(f"Invalid forecast matrix: {forecasts.shape}")
    return forecasts


def validation_mae(actual, predicted):
    indices = np.asarray(cfg.HORIZONS) - 1
    return float(np.mean(np.abs(actual[:, indices] - predicted[:, indices])))


def prediction_frame(station, model_name, seed, split, origins, actual, predicted, protocol_id):
    horizons = np.tile(np.arange(1, cfg.MAX_HORIZON + 1), len(origins))
    repeated_origins = np.repeat(origins.to_numpy(), cfg.MAX_HORIZON)
    return pd.DataFrame({
        "protocol_id": protocol_id, "station": station, "model": model_name, "seed": seed,
        "split": split, "origin": repeated_origins, "horizon": horizons,
        "target_date": pd.to_datetime(repeated_origins) + pd.to_timedelta(horizons, unit="D"),
        "y_true": actual.ravel(), "y_pred": predicted.ravel(),
    })


def environment():
    versions = {}
    for package in ("numpy", "pandas", "statsmodels", "torch", "timesfm", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {"python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "packages": versions}


def run_experiment(name, candidates, seeds, factory, model_settings):
    """Select each station's configuration on validation, then evaluate test once.

    factory(config, seed) returns an object implementing fit(train, validation,
    fixed_epochs), forecast(histories, horizon), metadata and optional save(path).
    Model code never receives test targets during fitting or model selection.
    """
    series_by_station, protocol, protocol_id = load_inputs()
    directory = cfg.RESULTS_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["protocol_id"] != protocol_id:
            raise ValueError("Protocol or inputs changed. Use a new RESULTS_DIR to preserve the previous experiment.")
    script = cfg.PROJECT_DIR / f"forecast_{name}.py"
    manifest = {"status": "running", "model": name, "protocol_id": protocol_id,
                "protocol": protocol, "environment": environment(), "seeds": list(seeds),
                "candidates": candidates, "model_settings": model_settings,
                "model_code_sha256": digest(script), "selected": {},
                "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(manifest_path, manifest)
    validation_rows, test_rows, search_rows, timing_rows = [], [], [], []
    try:
        for station, series in series_by_station.items():
            train = series.loc[:cfg.TRAIN_END].to_numpy()
            validation_origins = origins_for("validation")
            histories, actual = forecast_inputs(series.loc[:cfg.VALIDATION_END], validation_origins)
            best_score, selected, selected_predictions, selected_metadata = float("inf"), None, None, None
            for candidate_id, candidate in enumerate(candidates):
                predictions, details, scores = {}, {}, []
                for seed in seeds:
                    print(f"{name} | {station} | validation | candidate {candidate_id} | seed {seed}", flush=True)
                    model = factory(candidate, seed)
                    start = time.perf_counter()
                    model.fit(train.copy(), validation=(histories, actual))
                    fit_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    prediction = checked_forecast(model, histories)
                    forecast_seconds = time.perf_counter() - start
                    score = validation_mae(actual, prediction)
                    predictions[seed], details[seed] = prediction, dict(model.metadata)
                    scores.append(score)
                    search_rows.append({"station": station, "candidate_id": candidate_id, "seed": seed,
                                        "validation_mae": score, "configuration": json.dumps(candidate),
                                        "selected_epochs": model.metadata.get("epochs", 0)})
                    timing_rows.append({"station": station, "split": "validation", "candidate_id": candidate_id,
                                        "seed": seed, "fit_seconds": fit_seconds,
                                        "forecast_seconds": forecast_seconds, "n_origins": len(histories)})
                    write_csv(directory / "validation_search.csv", pd.DataFrame(search_rows))
                mean_score = float(np.mean(scores))
                if mean_score < best_score:
                    best_score, selected = mean_score, dict(candidate)
                    selected_predictions, selected_metadata = predictions, details
            if selected is None:
                raise ValueError("At least one candidate and seed are required")
            manifest["selected"][station] = {"configuration": selected, "validation_mae": best_score,
                                              "validation_models": selected_metadata, "test_models": {}}
            write_json(manifest_path, manifest)
            final_train = series.loc[:cfg.VALIDATION_END].to_numpy()
            test_origins = origins_for("test")
            test_histories, test_actual = forecast_inputs(series, test_origins)
            for seed in seeds:
                validation_rows.append(prediction_frame(station, name, seed, "validation", validation_origins,
                                                         actual, selected_predictions[seed], protocol_id))
                print(f"{name} | {station} | test | seed {seed} | selected {selected}", flush=True)
                model = factory(selected, seed)
                start = time.perf_counter()
                model.fit(final_train.copy(), fixed_epochs=selected_metadata[seed].get("epochs"))
                fit_seconds = time.perf_counter() - start
                if hasattr(model, "save"):
                    model.save(directory / f"{station}_seed_{seed}")
                start = time.perf_counter()
                prediction = checked_forecast(model, test_histories)
                forecast_seconds = time.perf_counter() - start
                test_rows.append(prediction_frame(station, name, seed, "test", test_origins,
                                                  test_actual, prediction, protocol_id))
                timing_rows.append({"station": station, "split": "test", "candidate_id": "selected",
                                    "seed": seed, "fit_seconds": fit_seconds,
                                    "forecast_seconds": forecast_seconds, "n_origins": len(test_origins)})
                manifest["selected"][station]["test_models"][seed] = dict(model.metadata)
            write_json(manifest_path, manifest)
        write_csv(directory / "validation_predictions.csv", pd.concat(validation_rows, ignore_index=True))
        write_csv(directory / "test_predictions.csv", pd.concat(test_rows, ignore_index=True))
        write_csv(directory / "timings.csv", pd.DataFrame(timing_rows))
        manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                        predictions_sha256=digest(directory / "test_predictions.csv"))
        write_json(manifest_path, manifest)
        print(f"Completed {name}. Results: {directory}", flush=True)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, manifest)
        raise
