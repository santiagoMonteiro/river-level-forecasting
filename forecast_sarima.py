"""Annual SARIMA(p,d,q)(0,1,0)[365] with fixed parameters during evaluation.

The annual difference is computed explicitly to avoid a large state vector.
An ARIMA(p,d,q) is fitted to z[t] = y[t] - y[t-365]. Forecasts are restored to
the original scale using observed annual lags (all horizons must be <= 365).
This is a restricted SARIMA family, not a search over all seasonal P/Q orders.
"""

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from experiment_common import run_experiment, write_json
from terminal_progress import progress, status

SEASONAL_PERIOD = 365
MAX_ITERATIONS = 300
OPTIMIZERS = ("lbfgs", "powell")
CANDIDATES = [{"order": order} for order in [(1, 0, 0), (2, 0, 0), (1, 0, 1),
                                            (1, 1, 0), (2, 1, 0), (1, 1, 1)]]


class SeasonalARIMA:
    def __init__(self, config, seed):
        self.order = tuple(config["order"])
        self.metadata = {"order": self.order, "seasonal_order": [0, 1, 0, SEASONAL_PERIOD],
                         "device": "cpu", "trend": "none"}

    def fit(self, train, validation=None, fixed_epochs=None):
        if len(train) <= SEASONAL_PERIOD + 10:
            raise ValueError("Not enough observations for annual differencing")
        self.consumed = len(train)
        differences = train[SEASONAL_PERIOD:] - train[:-SEASONAL_PERIOD]
        # Store the scale fitted on training only to improve optimizer conditioning.
        self.scale = max(float(np.std(differences)), 1.0)
        model = SARIMAX(differences / self.scale, order=self.order, trend="n",
                        enforce_stationarity=True, enforce_invertibility=True)
        # State predictions must remain available for extend(refit=False).
        attempts = []
        start_params = None
        for optimizer in OPTIMIZERS:
            status(f"  Fitting SARIMA {self.order}: optimizer={optimizer}, max_iterations={MAX_ITERATIONS}")
            # The optimizer can converge before its limit; show actual iterations,
            # not a misleading percentage toward MAX_ITERATIONS.
            with progress(desc=f"SARIMA {optimizer}", unit="iteration", leave=False) as iterations:
                self.result = model.fit(disp=False, maxiter=MAX_ITERATIONS, cov_type="none",
                                        method=optimizer, start_params=start_params,
                                        callback=lambda parameters: iterations.update(1))
            converged = bool(self.result.mle_retvals.get("converged", False))
            attempts.append({"optimizer": optimizer, "converged": converged})
            status(f"  SARIMA {optimizer}: {'converged' if converged else 'not converged'}")
            if converged and np.isfinite(self.result.params).all():
                break
            if np.isfinite(self.result.params).all():
                start_params = self.result.params
        else:
            raise ValueError(f"SARIMA {self.order} failed to converge; inspect the candidate before continuing")
        self.metadata.update(scale=self.scale, parameters=self.result.params.tolist(),
                             converged=True, optimization_attempts=attempts)

    def forecast(self, histories, horizon):
        if horizon > SEASONAL_PERIOD:
            raise ValueError("This seasonal inversion requires horizon <= SEASONAL_PERIOD")
        forecasts = []
        for history in progress(histories, desc="SARIMA forecasts", unit="origin", leave=False):
            if len(history) < self.consumed:
                raise ValueError("SARIMA origins must be chronological and cannot precede fitting")
            if len(history) > self.consumed:
                new = history[self.consumed:] - history[self.consumed - SEASONAL_PERIOD:-SEASONAL_PERIOD]
                self.result = self.result.extend(new / self.scale)
                self.consumed = len(history)
            predicted_difference = np.asarray(self.result.forecast(horizon)) * self.scale
            annual_lags = history[len(history) - SEASONAL_PERIOD:len(history) - SEASONAL_PERIOD + horizon]
            forecasts.append(predicted_difference + annual_lags)
        return np.asarray(forecasts)

    def save(self, path):
        write_json(path.with_suffix(".json"), self.metadata)


if __name__ == "__main__":
    run_experiment("sarima", CANDIDATES, [0], SeasonalARIMA,
                   {"seasonal_period": SEASONAL_PERIOD, "max_iterations": MAX_ITERATIONS,
                    "optimizers": OPTIMIZERS})
