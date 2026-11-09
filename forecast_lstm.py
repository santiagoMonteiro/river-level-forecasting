"""Direct 90-day LSTM forecasts with training-only normalization and three seeds."""

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import experiment_config as cfg
from experiment_common import run_experiment, validation_mae
from terminal_progress import progress, status

# CPU is the reproducible default. Change explicitly if an accelerator is desired.
DEVICE = "cpu"
TORCH_THREADS = 4
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 0.001
TRAINING_STRIDE = 1
CANDIDATES = [{"context_length": context, "hidden_size": units, "layers": 1}
              for context in (90, 365, 730) for units in (32, 64)]


class TrainingWindows(Dataset):
    """All inputs and all targets lie inside the supplied training array."""
    def __init__(self, values, context, horizon, stride=1):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.context, self.horizon = context, horizon
        self.starts = range(0, len(values) - context - horizon + 1, stride)
        if not len(self.starts):
            raise ValueError("Training data are too short for the context and horizon")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        boundary = start + self.context
        return self.values[start:boundary, None], self.values[boundary:boundary + self.horizon]


class Network(nn.Module):
    def __init__(self, hidden_size, layers, horizon):
        super().__init__()
        self.recurrent = nn.LSTM(1, hidden_size, num_layers=layers, batch_first=True)
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, values):
        _, (hidden, _) = self.recurrent(values)
        return self.output(hidden[-1])


class LSTMForecaster:
    def __init__(self, config, seed):
        self.config, self.seed = dict(config), seed
        self.metadata = {"device": DEVICE, "seed": seed, "configuration": config,
                         "training_loss": "MAE across all 90 output days"}

    def fit(self, train, validation=None, fixed_epochs=None):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(TORCH_THREADS)
        torch.use_deterministic_algorithms(True)
        self.mean = float(np.mean(train))
        self.scale = max(float(np.std(train)), 1e-8)
        self.network = Network(self.config["hidden_size"], self.config["layers"], cfg.MAX_HORIZON).to(DEVICE)
        dataset = TrainingWindows((train - self.mean) / self.scale, self.config["context_length"],
                                  cfg.MAX_HORIZON, TRAINING_STRIDE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(self.seed))
        optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        loss_function = nn.L1Loss()
        if validation is None and fixed_epochs is None:
            raise ValueError("Provide validation for early stopping or fixed_epochs for final training")
        epochs = fixed_epochs if fixed_epochs is not None else MAX_EPOCHS
        best_score, best_state, best_epoch, stale = float("inf"), None, 0, 0
        with progress(total=epochs, desc="LSTM epochs", unit="epoch", position=0) as epoch_bar:
            for epoch in range(1, epochs + 1):
                self.network.train()
                loss_sum, sample_count = 0.0, 0
                with progress(loader, desc=f"Training epoch {epoch}/{epochs}", unit="batch",
                              position=1, leave=False) as batches:
                    for inputs, targets in batches:
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_function(self.network(inputs.to(DEVICE)), targets.to(DEVICE))
                        if not torch.isfinite(loss):
                            raise ValueError("LSTM training produced a non-finite loss")
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                        optimizer.step()
                        loss_sum += loss.item() * len(inputs)
                        sample_count += len(inputs)
                        batches.set_postfix(train_mae=f"{loss_sum / sample_count * self.scale:.4f}", refresh=False)
                training_mae = loss_sum / sample_count * self.scale
                if validation is not None:
                    histories, actual = validation
                    score = validation_mae(actual, self.forecast(histories, cfg.MAX_HORIZON))
                    if score < best_score:
                        best_score, best_epoch, stale = score, epoch, 0
                        best_state = copy.deepcopy(self.network.state_dict())
                    else:
                        stale += 1
                    epoch_bar.set_postfix(train_mae=f"{training_mae:.4f}", val_mae=f"{score:.4f}",
                                          best_epoch=best_epoch, patience=f"{stale}/{PATIENCE}", refresh=False)
                    status(f"  Epoch {epoch}/{epochs}: training MAE={training_mae:.4f}; "
                           f"validation MAE={score:.4f}; best epoch={best_epoch}; patience={stale}/{PATIENCE}")
                else:
                    epoch_bar.set_postfix(train_mae=f"{training_mae:.4f}", refresh=False)
                    status(f"  Final training epoch {epoch}/{epochs}: training MAE={training_mae:.4f}")
                epoch_bar.update(1)
                if validation is not None and stale >= PATIENCE:
                    status(f"  Early stopping at epoch {epoch}; restoring epoch {best_epoch}.")
                    break
        if validation is not None:
            if best_state is None:
                raise ValueError("No finite validation score was obtained")
            self.network.load_state_dict(best_state)
        else:
            best_epoch = epochs
        self.metadata.update(epochs=best_epoch, normalization_mean=self.mean, normalization_scale=self.scale)

    def forecast(self, histories, horizon):
        if horizon != cfg.MAX_HORIZON:
            raise ValueError("The direct LSTM head uses the configured MAX_HORIZON")
        context = self.config["context_length"]
        if any(len(history) < context for history in histories):
            raise ValueError("A forecast history is shorter than the selected LSTM context")
        values = np.stack([(history[-context:] - self.mean) / self.scale for history in histories])
        self.network.eval()
        forecasts = []
        with torch.inference_mode():
            for start in progress(range(0, len(values), BATCH_SIZE), desc="LSTM forecast batches",
                                  unit="batch", position=1, leave=False):
                inputs = torch.tensor(values[start:start + BATCH_SIZE, :, None], dtype=torch.float32, device=DEVICE)
                forecasts.append(self.network(inputs).cpu().numpy())
        return np.concatenate(forecasts) * self.scale + self.mean

    def save(self, path):
        torch.save({"state_dict": self.network.state_dict(), "metadata": self.metadata,
                    "max_horizon": cfg.MAX_HORIZON}, path.with_suffix(".pt"))


if __name__ == "__main__":
    run_experiment("lstm", CANDIDATES, cfg.LSTM_SEEDS, LSTMForecaster,
                   {"device": DEVICE, "threads": TORCH_THREADS, "batch_size": BATCH_SIZE,
                    "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "learning_rate": LEARNING_RATE,
                    "training_stride": TRAINING_STRIDE, "deterministic_algorithms": True})
