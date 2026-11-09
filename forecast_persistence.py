"""Persistence baseline: repeat the last observed stage for all forecast days."""

import numpy as np

from experiment_common import run_experiment
from terminal_progress import progress


class Persistence:
    def __init__(self, config, seed):
        self.metadata = {"method": "last observed value", "device": "cpu"}

    def fit(self, train, validation=None, fixed_epochs=None):
        pass

    def forecast(self, histories, horizon):
        return np.array([[history[-1]] * horizon for history in
                         progress(histories, desc="Persistence forecasts", unit="origin", leave=False)])


if __name__ == "__main__":
    run_experiment("persistence", [{}], [0], Persistence, {})
