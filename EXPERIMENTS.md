# River-stage forecasting experiment

All settings are Python constants. No command-line parameters are required.
Edit shared dates, station files, horizons and output location in
`experiment_config.py`; edit candidate grids and computing settings in each
`forecast_*.py` file **before** inspecting test results.

## Run

Use the same Python environment for all scripts. Required packages are listed
in `requirements-experiments.txt`. The TimesFM runner uses the TimesFM 3.0 API
(`timesfm3.torch`) and downloads its checkpoint on first execution. MLX is an
explicit alternative in `forecast_timesfm.py` for a compatible Apple Silicon
environment. The default PyTorch backend uses CPU. LSTM training and TimesFM
inference can take substantial time; scripts print progress during selection.

```bash
python3 forecast_persistence.py
python3 forecast_sarima.py
python3 forecast_lstm.py
python3 forecast_timesfm.py
python3 compare_forecasts.py
```

## Terminal feedback

Each model reports the current station, validation/test phase, candidate and
seed. Interactive terminals also display progress bars with elapsed time and,
when the total is known, estimated remaining time:

- LSTM: epochs, training batches, running training MAE in the original units,
  validation MAE, best epoch, early-stopping patience and forecast batches.
- SARIMA: actual optimizer iterations, convergence status and forecast origins.
  Iterations have no completion percentage because convergence can occur before
  the configured limit.
- TimesFM: checkpoint download/loading messages and forecast batches.
- Persistence: forecast origins (usually completes almost immediately).

Set `SHOW_PROGRESS = False` in `experiment_config.py` to hide the bars. Adjust
`PROGRESS_REFRESH_SECONDS` to change their refresh interval. Bars are automatically
hidden when output is redirected or the terminal is non-interactive; persistent
status messages remain available. LSTM training MAE covers all forecast days,
while validation MAE uses the five selected horizons. Display settings do not
change the scientific protocol or invalidate existing forecast results.

## Protocol

| Stage | Dates | Purpose |
| --- | --- | --- |
| Training | 1991-01-01 to 2015-12-31 | Fit candidate models and normalization |
| Validation | 2016-01-01 to 2020-12-31 | Select configurations and LSTM epochs |
| Final fitting | 1991-01-01 to 2020-12-31 | Refit selected models before test |
| Test | 2021-01-01 to 2023-12-31 | Evaluate once with no further tuning |

Each station has its own fitted model/configuration. Santo Antônio is only a
source for reconstruction in preprocessing; it is not a forecast target or
forecast input. The completed CSVs must have `date,river_stage` columns. Units
are preserved; confirm the source unit before labeling article figures.

Origins advance by seven days, starting at the last training day for validation
and at the last final-fitting day for test. Every origin must have all 90 future
days inside its evaluation split. This produces 249 validation origins and 144
test origins with the default dates. All models predict all 90 days, with main
metrics at days 7, 15, 30, 60 and 90. This common-origin design intentionally
does not include late-2023 origins whose 90-day target would fall in 2024.

Model selection minimizes validation MAE equally averaged over the five main
horizons, all origins and, for LSTM, all seeds. Early stopping also uses this
criterion. Every candidate uses the same validation origins. Failed fits or
non-finite predictions stop a run; they are never silently dropped from test.
The candidate grids are modest predefined searches, not equal computational
budgets or claims of optimal tuning for any model. Record the grids in the paper.

During evaluation, newly observed history is available up to each origin.
Parameters are fixed within the entire validation/test period. SARIMA updates
its filtered state, LSTM receives its latest context and TimesFM receives its
latest context. A forecast never consumes observations after its own origin.
LSTM training windows and their complete 90-day targets stay inside training;
its mean/std are fitted on training only. Shuffling complete training windows
does not mix training, validation or test partitions.

## Models

- **Persistence:** repeat the last observed value; no fitted parameters.
- **SARIMA:** restricted family `(p,d,q)(0,1,0)[365]`, with six nonseasonal order
  candidates. Annual differencing is applied explicitly, followed by ARIMA.
  Adding observed annual lags back gives forecasts on the original cota scale.
  This avoids hundreds of seasonal states and is valid for the configured
  horizons (at most 365 days). There is no seasonal AR/MA grid, and the fixed
  365-day period approximates the annual calendar including leap years. Do not
  describe this as an exhaustive SARIMA search. Fitting tries L-BFGS first and
  Powell if necessary, using training data only and recording each attempt.
  Nonconvergence after both optimizers stops the run.
- **LSTM:** one layer, 32 or 64 units, and contexts of 90, 365 or 730 days;
  direct 90-output head; MAE loss across the whole trajectory. Each candidate
  runs with seeds 17, 42 and 123. Validation determines the best epoch for each
  seed. After configuration selection by mean seed MAE, each seed is refitted
  on training + validation for its selected number of epochs. No best-test-seed
  selection and no implicit ensemble. CPU and deterministic algorithms are the
  defaults; hardware changes must be reported.
- **TimesFM 3.0:** externally pretrained, no local weight fitting; validation
  selects context length 365, 730 or 1460. Forecasts use the model's median
  output. The resolved checkpoint commit is saved; set `CHECKPOINT_REVISION`
  to that commit for an exact rerun. Zero-shot weights do not imply that context
  selection is untuned. External pretraining is a methodological difference:
  investigate overlap with these station series before claiming a strictly
  unseen-data benchmark. Our chronological split alone cannot establish that.

Available history is shared, but model context lengths can differ after
validation selection. Equal input lengths are not imposed as a requirement for
the main comparison. For a context-controlled ablation, run a new experiment
with predefined matching contexts and describe it separately.

## Outputs and checks

Results are stored in `experiments/river_stage_v1/<model>/`:

- `test_predictions.csv`: station, model, seed, split, origin, horizon,
  target_date, y_true, y_pred, and protocol_id. Every day from 1 to 90 is saved.
- `validation_predictions.csv`: predictions of the selected configuration.
- `validation_search.csv`: all candidate/seed validation scores and epochs.
- `timings.csv`: selection/refit and forecast wall times. LSTM selection fit
  time includes early-stopping validation; TimesFM checkpoint loading is
  recorded separately in the manifest. External pretraining time is excluded.
- `manifest.json`: completion status, splits, settings, selected configurations,
  input/code hashes, package versions, system information and model metadata.
- LSTM `.pt` checkpoints and SARIMA parameter `.json` files for final fits.

The comparison requires all four completed models and exactly matching station,
seed, origin, horizon, observed-value and protocol coverage. It refuses edited
prediction files and incompatible inputs/protocols. It does not take an inner
join that silently loses difficult forecast dates. Change `RESULTS_DIR` when
changing the scientific protocol. Rerunning the same model/protocol replaces
its outputs, so also use a new directory when preserving alternative runs.

By default, raw ANA consistency-level-2 data are checked against every validation
and test observation. Humaitá's reconstruction in 2014 is restricted to the
training interval under this protocol. If the dates are changed, keep imputed
targets out of the evaluation or explicitly redesign and document the study.
This is retrospective evaluation on quality-controlled data; it does not model
the operational delay with which consistency-level-2 observations became available.

`comparison/metrics_by_seed.csv` contains MAE, RMSE, signed bias (`predicted -
observed`), R² and MAE skill against persistence for each seed. The `point` scope
uses exactly day h; `trajectory` averages errors over days 1 through h. These
are different questions and must be reported separately.

`comparison/metrics_summary.csv` gives mean metrics and standard deviation
across LSTM seeds, not metrics of an averaged-prediction ensemble. MAE skill is
`1 - MAE_model / MAE_persistence`, with positive values indicating improvement.
A zero persistence MAE makes the skill undefined. Keep station-specific tables;
raw cota errors from different stations should not be pooled without scaling.

Approximate 95% skill intervals use paired circular moving-block bootstrap over
origins (13 weekly origins per block, 2,000 resamples). For LSTM, losses are
averaged across seeds before temporal resampling: seeds are not treated as
independent dates. These intervals acknowledge overlapping forecasts but do not
guarantee that all dependence is captured, do not incorporate training-data
uncertainty and are not adjusted for multiple comparisons. Assess block-length
sensitivity and event/season-specific errors before scientific claims of
significance. Report initialization variability separately.

## References

- [Forecasting: Principles and Practice — temporal validation and accuracy](https://otexts.com/fpp3/toolbox.html)
- [Statsmodels — state updates without refitting](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.extend.html)
- [Official TimesFM repository](https://github.com/google-research/timesfm)

## Verification

```bash
python3 -m unittest discover -s tests -v
```

Tests use synthetic data and tiny LSTM fits; they do not download TimesFM
weights. They exercise temporal boundaries, multi-step alignment, seasonal
inversion, training-only scaling, the TimesFM adapter and comparison coverage.
