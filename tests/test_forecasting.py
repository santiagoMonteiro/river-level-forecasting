"""Tests for temporal integrity and comparable forecast outputs."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import experiment_config as cfg
import experiment_common as common
import compare_forecasts
import forecast_lstm
from compare_forecasts import metric_tables, validate_predictions
from forecast_persistence import Persistence
from forecast_sarima import SeasonalARIMA
from forecast_lstm import LSTMForecaster, TrainingWindows
from forecast_timesfm import TimesFMForecaster


class TemporalProtocolTests(unittest.TestCase):
    def test_origins_have_complete_targets_within_each_split(self):
        for split, start, end, count in [
            ("validation", "2016-01-01", cfg.VALIDATION_END, 249),
            ("test", "2021-01-01", cfg.TEST_END, 144),
        ]:
            origins = common.origins_for(split)
            self.assertEqual(len(origins), count)
            self.assertEqual(origins[0] + pd.Timedelta(days=1), pd.Timestamp(start))
            self.assertLessEqual(origins[-1] + pd.Timedelta(days=cfg.MAX_HORIZON), pd.Timestamp(end))

    def test_future_changes_cannot_change_history_or_persistence(self):
        index = pd.date_range("2000-01-01", periods=150)
        series = pd.Series(np.arange(150, dtype=float), index=index)
        origin = pd.DatetimeIndex([index[50]])
        histories, targets = common.forecast_inputs(series, origin)
        changed = series.copy()
        changed.iloc[51:] = -999
        new_histories, _ = common.forecast_inputs(changed, origin)
        np.testing.assert_array_equal(histories[0], new_histories[0])
        self.assertEqual(targets[0, 0], 51)
        self.assertEqual(targets[0, -1], 140)
        prediction = Persistence({}, 0).forecast(histories, 90)
        np.testing.assert_array_equal(prediction, np.full((1, 90), 50))

    def test_daily_training_windows_do_not_cross_supplied_boundary(self):
        windows = TrainingWindows(np.arange(50), context=10, horizon=7)
        self.assertEqual(len(windows), 34)
        x, y = windows[-1]
        self.assertEqual(x[-1].item(), 42)
        self.assertEqual(y[0].item(), 43)
        self.assertEqual(y[-1].item(), 49)


class ModelTests(unittest.TestCase):
    def test_sarima_restores_original_scale_and_updates_only_new_history(self):
        # An exact annual cycle has zero annual differences. A zero-difference
        # forecaster must reconstruct the cycle using observations only.
        train = np.tile(np.arange(365, dtype=float), 3)
        model = SeasonalARIMA({"order": [1, 0, 0]}, 0)
        with patch("forecast_sarima.SARIMAX") as constructor:
            state = constructor.return_value.fit.return_value
            state.mle_retvals = {"converged": True}
            state.params = np.array([0.0, 1.0])
            state.forecast.return_value = np.zeros(90)
            state.extend.return_value = state
            model.fit(train)
            history = np.concatenate([train, np.arange(7)])
            predicted = model.forecast([train, history], 90)
            np.testing.assert_array_equal(predicted[0], np.arange(90))
            np.testing.assert_array_equal(predicted[1], np.arange(7, 97))
            np.testing.assert_array_equal(state.extend.call_args.args[0], np.zeros(7))

    def test_sarima_real_fit_is_finite(self):
        rng = np.random.default_rng(9)
        history = 500 + np.sin(np.arange(900) * 2 * np.pi / 365) * 50 + rng.normal(size=900)
        model = SeasonalARIMA({"order": [1, 0, 0]}, 0)
        model.fit(history[:850])
        predictions = model.forecast([history[:850], history[:857]], 30)
        self.assertEqual(predictions.shape, (2, 30))
        self.assertTrue(np.isfinite(predictions).all())

    def test_lstm_training_only_scaling_and_checkpoint(self):
        train = np.arange(100, dtype=float)
        context = 14
        with patch.multiple(cfg, MAX_HORIZON=7, HORIZONS=(3, 7)):
            model = LSTMForecaster({"context_length": context, "hidden_size": 4, "layers": 1}, 17)
            model.fit(train, fixed_epochs=1)
            self.assertAlmostEqual(model.mean, train.mean())
            self.assertAlmostEqual(model.scale, train.std())
            result = model.forecast([train, np.append(train, 1000)], 7)
            self.assertEqual(result.shape, (2, 7))
            self.assertTrue(np.isfinite(result).all())
            self.assertAlmostEqual(model.mean, train.mean())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint"
                model.save(path)
                self.assertTrue(path.with_suffix(".pt").exists())

    def test_timesfm_adapter_passes_only_selected_context(self):
        calls = []

        class FakePretrained:
            def predict_batch(self, contexts, horizon, **kwargs):
                calls.extend(contexts)
                return [SimpleNamespace(forecast=np.repeat(context[-1], horizon)) for context in contexts]

        model = TimesFMForecaster({"context_length": 10}, 0, FakePretrained(), {})
        histories = [np.arange(30), np.arange(37)]
        model.fit(histories[0])
        forecasts = model.forecast(histories, 90)
        np.testing.assert_array_equal(calls[0], np.arange(20, 30))
        np.testing.assert_array_equal(calls[1], np.arange(27, 37))
        np.testing.assert_array_equal(forecasts[:, -1], [29, 36])

    def test_lstm_selects_epochs_then_refits_without_test_targets(self):
        index = pd.date_range("2000-01-01", "2000-12-31")
        series = pd.Series(np.sin(np.arange(len(index)) / 10) + 100, index=index)
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            cfg, DATA_START="2000-01-01", TRAIN_END="2000-04-30", VALIDATION_END="2000-08-31",
            TEST_END="2000-12-31", MAX_HORIZON=7, HORIZONS=(3, 7), RESULTS_DIR=Path(directory)
        ), patch.multiple(forecast_lstm, MAX_EPOCHS=2, PATIENCE=1), patch.object(
            common, "load_inputs", return_value=({"station": series}, {}, "p")
        ):
            common.run_experiment("lstm", [{"context_length": 14, "hidden_size": 4, "layers": 1}],
                                  [17], LSTMForecaster, {})
            manifest = json.loads((Path(directory) / "lstm/manifest.json").read_text())
            selection = manifest["selected"]["station"]
            validation_model = selection["validation_models"]["17"]
            test_model = selection["test_models"]["17"]
            self.assertEqual(test_model["epochs"], validation_model["epochs"])
            self.assertAlmostEqual(test_model["normalization_mean"], series.loc[:"2000-08-31"].mean())
            self.assertAlmostEqual(validation_model["normalization_mean"], series.loc[:"2000-04-30"].mean())


class ComparisonTests(unittest.TestCase):
    def test_comparison_writes_all_model_station_horizon_rows(self):
        index = pd.date_range("2000-01-01", "2000-12-31")
        series = pd.Series(np.arange(len(index), dtype=float), index=index)
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            cfg, DATA_START="2000-01-01", TRAIN_END="2000-04-30", VALIDATION_END="2000-08-31",
            TEST_END="2000-12-31", MAX_HORIZON=7, HORIZONS=(3, 7), RESULTS_DIR=Path(directory)
        ), patch.object(compare_forecasts, "load_inputs", return_value=({"station": series}, {}, "p")):
            origins = common.origins_for("test")
            _, truth = common.forecast_inputs(series, origins)
            for model in compare_forecasts.MODELS:
                path = Path(directory) / model
                path.mkdir()
                seeds = cfg.LSTM_SEEDS if model == "lstm" else (0,)
                frames = [common.prediction_frame("station", model, seed, "test", origins, truth, truth + 2, "p")
                          for seed in seeds]
                common.write_csv(path / "test_predictions.csv", pd.concat(frames))
                common.write_csv(path / "timings.csv", pd.DataFrame({"fit_seconds": [0]}))
                common.write_json(path / "manifest.json", {"status": "complete", "protocol_id": "p", "seeds": seeds,
                                                          "predictions_sha256": common.digest(path / "test_predictions.csv")})
            compare_forecasts.main()
            summary = pd.read_csv(Path(directory) / "comparison/metrics_summary.csv")
            self.assertEqual(len(summary), 4 * 2 * 2)
            self.assertTrue((summary.loc[summary.model == "lstm", "n_seeds"] == 3).all())

    def test_validator_rejects_dropped_forecast_or_modified_truth(self):
        series = pd.Series(np.arange(12053, dtype=float), index=pd.date_range(cfg.DATA_START, cfg.TEST_END))
        origins = common.origins_for("test")
        _, truth = common.forecast_inputs(series, origins)
        frame = common.prediction_frame("station", "persistence", 0, "test", origins, truth, truth, "protocol")
        manifest = {"status": "complete", "protocol_id": "protocol", "seeds": [0]}
        validate_predictions(frame, "persistence", manifest, {"station": series}, "protocol")
        with self.assertRaisesRegex(ValueError, "origins or horizons"):
            validate_predictions(frame.iloc[1:], "persistence", manifest, {"station": series}, "protocol")
        altered = frame.copy()
        altered.loc[0, "y_true"] = -999
        with self.assertRaisesRegex(ValueError, "observed values"):
            validate_predictions(altered, "persistence", manifest, {"station": series}, "protocol")

    def test_metrics_average_seed_errors_not_ensemble_predictions(self):
        origins = pd.date_range("2021-01-01", periods=4, freq="7D")
        truth = np.tile(np.arange(1, 91), (4, 1))
        frames = [common.prediction_frame("station", "persistence", 0, "test", origins, truth, truth + 2, "p")]
        # Opposite errors would cancel in an ensemble, but each seed MAE is 1.
        for seed, error in [(17, 1), (42, -1)]:
            frames.append(common.prediction_frame("station", "lstm", seed, "test", origins, truth, truth + error, "p"))
        by_seed, summary = metric_tables(pd.concat(frames))
        self.assertTrue((by_seed.loc[by_seed.model == "lstm", "mae"] == 1).all())
        self.assertTrue((summary.loc[summary.model == "lstm", "mae_skill_mean"] == 0.5).all())
        self.assertTrue((summary.loc[summary.model == "lstm", "mae_mean"] == 1).all())

    def test_end_to_end_baseline_manifest_and_refit_boundaries(self):
        index = pd.date_range("2000-01-01", "2000-12-31")
        series = pd.Series(np.arange(len(index), dtype=float), index=index)
        seen_fit_lengths = []

        class SpyPersistence(Persistence):
            def fit(self, train, validation=None, fixed_epochs=None):
                seen_fit_lengths.append(len(train))

        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            cfg, DATA_START="2000-01-01", TRAIN_END="2000-04-30", VALIDATION_END="2000-08-31",
            TEST_END="2000-12-31", MAX_HORIZON=7, HORIZONS=(3, 7), RESULTS_DIR=Path(directory)
        ), patch.object(common, "load_inputs", return_value=({"station": series}, {}, "p")):
            common.run_experiment("persistence", [{}], [0], SpyPersistence, {})
            manifest = json.loads((Path(directory) / "persistence/manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(seen_fit_lengths, [121, 244])
            predictions = pd.read_csv(Path(directory) / "persistence/test_predictions.csv", parse_dates=["origin", "target_date"])
            validate_predictions(predictions, "persistence", manifest, {"station": series}, "p")


if __name__ == "__main__":
    unittest.main()
