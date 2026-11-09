"""Compare complete, matched test forecasts without silently dropping failures."""

import json

import numpy as np
import pandas as pd

import experiment_config as cfg
from experiment_common import digest, load_inputs, origins_for, forecast_inputs, write_csv, write_json

MODELS = ("persistence", "sarima", "lstm", "timesfm")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026
# A block spans at least the overlapping 90-day forecast window.
BLOCK_LENGTH_ORIGINS = int(np.ceil(cfg.MAX_HORIZON / cfg.ORIGIN_STEP_DAYS))


def validate_predictions(frame, model, manifest, series_by_station, protocol_id):
    """Reject missing forecasts, wrong observations, different protocols or seeds."""
    if manifest["status"] != "complete" or manifest["protocol_id"] != protocol_id:
        raise ValueError(f"{model}: incomplete run or incompatible experimental protocol")
    expected_seeds = set(cfg.LSTM_SEEDS if model == "lstm" else (0,))
    if set(manifest["seeds"]) != expected_seeds:
        raise ValueError(f"{model}: unexpected seed configuration")
    required = {"protocol_id", "station", "model", "seed", "split", "origin", "horizon", "target_date", "y_true", "y_pred"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"{model}: invalid prediction schema")
    if frame.isna().any().any() or not np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError(f"{model}: missing or non-finite predictions")
    if set(frame["protocol_id"]) != {protocol_id} or set(frame["model"]) != {model} or set(frame["split"]) != {"test"}:
        raise ValueError(f"{model}: unexpected protocol, model or split")
    if set(frame["station"]) != set(series_by_station):
        raise ValueError(f"{model}: station coverage differs")
    origins = origins_for("test")
    expected_keys = pd.MultiIndex.from_product([origins, range(1, cfg.MAX_HORIZON + 1)], names=["origin", "horizon"])
    for station, series in series_by_station.items():
        station_frame = frame[frame["station"] == station]
        if set(station_frame["seed"]) != expected_seeds:
            raise ValueError(f"{model}/{station}: some seeds are missing")
        _, targets = forecast_inputs(series, origins)
        for seed, run in station_frame.groupby("seed"):
            run = run.sort_values(["origin", "horizon"])
            if not pd.MultiIndex.from_frame(run[["origin", "horizon"]]).equals(expected_keys):
                raise ValueError(f"{model}/{station}/{seed}: forecast origins or horizons differ")
            dates = run["origin"] + pd.to_timedelta(run["horizon"], unit="D")
            if not (run["target_date"] == dates).all() or not np.allclose(run["y_true"], targets.ravel()):
                raise ValueError(f"{model}/{station}/{seed}: target dates or observed values differ")


def paired_skill_interval(model_errors, baseline_errors):
    """Circular moving-block bootstrap over origins, not over LSTM seeds."""
    n = len(model_errors)
    block = min(BLOCK_LENGTH_ORIGINS, n)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    starts = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, int(np.ceil(n / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(BOOTSTRAP_REPLICATES, -1)[:, :n]
    baseline_mae = baseline_errors[indices].mean(axis=1)
    model_mae = model_errors[indices].mean(axis=1)
    valid = baseline_mae > 0
    if not valid.any():
        return np.nan, np.nan
    return tuple(np.quantile(1 - model_mae[valid] / baseline_mae[valid], [0.025, 0.975]))


def metric_tables(predictions):
    rows, error_by_group = [], {}
    for (station, model, seed), run in predictions.groupby(["station", "model", "seed"]):
        for horizon in cfg.HORIZONS:
            for scope in ("point", "trajectory"):
                subset = run[run["horizon"] == horizon] if scope == "point" else run[run["horizon"] <= horizon]
                subset = subset.sort_values(["origin", "horizon"])
                errors = subset["y_pred"].to_numpy() - subset["y_true"].to_numpy()
                values = subset["y_true"].to_numpy()
                denominator = float(np.sum((values - values.mean()) ** 2))
                rows.append({"station": station, "model": model, "seed": seed, "horizon": horizon,
                             "scope": scope, "n_origins": subset["origin"].nunique(), "n_predictions": len(subset),
                             "mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))),
                             "bias": float(errors.mean()),
                             "r2": 1 - float(np.sum(errors ** 2)) / denominator if denominator else np.nan})
                origin_errors = subset.assign(absolute_error=np.abs(errors)).groupby("origin")["absolute_error"].mean()
                error_by_group.setdefault((station, model, horizon, scope), []).append(origin_errors.to_numpy())
    by_seed = pd.DataFrame(rows)
    baseline = by_seed[by_seed["model"] == "persistence"].set_index(["station", "horizon", "scope"])["mae"]
    by_seed["mae_skill"] = [1 - row.mae / baseline.loc[(row.station, row.horizon, row.scope)]
                            if baseline.loc[(row.station, row.horizon, row.scope)] > 0 else np.nan
                            for row in by_seed.itertuples()]
    summary = []
    for key, group in by_seed.groupby(["station", "model", "horizon", "scope"]):
        station, model, horizon, scope = key
        row = dict(zip(["station", "model", "horizon", "scope"], key))
        row.update(n_seeds=len(group), n_origins=int(group["n_origins"].iloc[0]))
        for metric in ("mae", "rmse", "bias", "r2", "mae_skill"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_seed_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        errors = np.mean(error_by_group[key], axis=0)
        baseline_errors = np.mean(error_by_group[(station, "persistence", horizon, scope)], axis=0)
        low, high = paired_skill_interval(errors, baseline_errors)
        row.update(mae_skill_ci_low=low, mae_skill_ci_high=high)
        summary.append(row)
    return by_seed, pd.DataFrame(summary)


def main():
    series_by_station, protocol, protocol_id = load_inputs()
    frames, timings, manifests = [], [], {}
    for model in MODELS:
        directory = cfg.RESULTS_DIR / model
        if not (directory / "manifest.json").exists():
            raise ValueError(f"Missing model: run forecast_{model}.py before comparing all four models")
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"{model}: the run is not complete")
        path = directory / "test_predictions.csv"
        if digest(path) != manifest["predictions_sha256"]:
            raise ValueError(f"{model}: prediction file changed after the run completed")
        frame = pd.read_csv(path, parse_dates=["origin", "target_date"])
        validate_predictions(frame, model, manifest, series_by_station, protocol_id)
        frames.append(frame)
        timings.append(pd.read_csv(directory / "timings.csv").assign(model=model))
        manifests[model] = manifest
    by_seed, summary = metric_tables(pd.concat(frames, ignore_index=True))
    destination = cfg.RESULTS_DIR / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "metrics_by_seed.csv", by_seed)
    write_csv(destination / "metrics_summary.csv", summary)
    write_csv(destination / "timings.csv", pd.concat(timings, ignore_index=True))
    write_json(destination / "comparison_manifest.json",
               {"protocol_id": protocol_id, "protocol": protocol, "models": manifests,
                "bootstrap": {"method": "paired circular moving blocks over origins",
                              "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                              "block_length_origins": BLOCK_LENGTH_ORIGINS,
                              "coverage": 0.95, "multiple_comparisons_adjusted": False}})
    print(summary[summary["scope"] == "point"][["station", "model", "horizon", "mae_mean", "rmse_mean", "mae_skill_mean"]].to_string(index=False))
    print(f"Comparison saved to {destination}")


if __name__ == "__main__":
    main()
