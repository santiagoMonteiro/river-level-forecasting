"""Shared, predeclared experimental protocol. Edit settings here, not in a CLI."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = PROJECT_DIR / "experiments" / "river_stage_v1"
STATIONS = {"humaita": "15630000", "manicore": "15700000", "porto_velho": "15400000"}
DATA_START = "1991-01-01"
TRAIN_END = "2015-12-31"
VALIDATION_END = "2020-12-31"
TEST_END = "2023-12-31"
HORIZONS = (7, 15, 30, 60, 90)
MAX_HORIZON = max(HORIZONS)
ORIGIN_STEP_DAYS = 7
LSTM_SEEDS = (17, 42, 123)
# Fail if validation/test contain reconstructed rather than original level-2 data.
VERIFY_OBSERVED_EVALUATION = True
# Display-only settings: bars are automatically hidden in redirected/non-TTY logs.
SHOW_PROGRESS = True
PROGRESS_REFRESH_SECONDS = 0.5
# Preserve previous results by choosing a new RESULTS_DIR for a new protocol.
# Forecast units remain identical to the input CSVs; no conversion is applied.
