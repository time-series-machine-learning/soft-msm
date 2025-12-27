from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from aeon.forecasting import BaseForecaster
from torch.utils.data import DataLoader, TensorDataset


def _to_1d_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        if 1 in y.shape:
            y = y.reshape(-1)
        else:
            raise ValueError(
                "This example forecaster is univariate only. "
                f"Got y with shape {y.shape}."
            )
    elif y.ndim != 1:
        raise ValueError(
            f"Expected y to be 1D (or 2D with one channel). Got {y.ndim}D."
        )
    return y.astype(np.float32, copy=False)


def make_sliding_windows(
    y_1d: np.ndarray, window: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    n = y_1d.shape[0]
    n_samples = n - window - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough data for window={window} and horizon={horizon}. "
            f"Need at least {window + horizon} points, got {n}."
        )

    X = np.empty((n_samples, window), dtype=np.float32)
    Y = np.empty((n_samples, horizon), dtype=np.float32)
    for i in range(n_samples):
        X[i] = y_1d[i : i + window]
        Y[i] = y_1d[i + window : i + window + horizon]
    return X, Y


def build_mlp(
    in_features: int, out_features: int, hidden: tuple[int, ...] = (64, 64)
) -> nn.Module:
    layers = []
    prev = in_features
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


class TorchMLPForecaster(BaseForecaster):
    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "capability:horizon": True,
        "capability:exogenous": False,
        "fit_is_empty": False,
        "y_inner_type": "np.ndarray",
    }

    def __init__(
        self,
        horizon: int = 1,
        axis: int = 1,
        window: int = 24,
        net: nn.Module | None = None,
        max_epochs: int = 20,
        batch_size: int = 32,
        device: str | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.SGD,
        optimizer_kwargs: dict[str, Any] | None = None,
        loss_fn: nn.Module | None = None,  # <-- NEW
    ):
        super().__init__(horizon=horizon, axis=axis)
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")

        self.window = window
        self.net = net

        self.max_epochs = max_epochs
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

        self.optimizer_: torch.optim.Optimizer | None = None

    def _fit(self, y, exog):
        if exog is not None:
            raise ValueError(
                "This forecaster does not support exogenous variables (exog)."
            )

        y_1d = _to_1d_series(y)
        X, Y = make_sliding_windows(y_1d, window=self.window, horizon=self.horizon)

        if self.net is None:
            self.net = build_mlp(in_features=self.window, out_features=self.horizon)

        self.net.to(self.device)
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.to(self.device)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        if self.optimizer_kwargs is None:
            if self.optimizer_cls is torch.optim.SGD:
                opt_kwargs = {"lr": 1e-2, "momentum": 0.9}
            else:
                opt_kwargs = {"lr": 1e-3}
        else:
            opt_kwargs = dict(self.optimizer_kwargs)

        self.optimizer_ = self.optimizer_cls(self.net.parameters(), **opt_kwargs)

        self.net.train()
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer_.zero_grad(set_to_none=True)
                pred = self.net(xb)
                loss = self.loss_fn(pred, yb)
                if loss.ndim != 0:
                    loss = loss.mean()

                loss.backward()
                self.optimizer_.step()

        return self

    def _predict(self, y, exog):
        if exog is not None:
            raise ValueError(
                "This forecaster does not support exogenous variables (exog)."
            )
        if self.net is None:
            raise RuntimeError("Model is not fitted (net is None).")

        y_1d = _to_1d_series(y)
        if y_1d.shape[0] < self.window:
            raise ValueError(f"Need at least window={self.window} points to predict.")

        x_last = y_1d[-self.window :].astype(np.float32, copy=False)
        xb = torch.from_numpy(x_last[None, :]).to(self.device)

        self.net.eval()
        with torch.no_grad():
            pred = self.net(xb).detach().cpu().numpy().reshape(-1)

        return float(pred[0]) if self.horizon == 1 else pred


class L1PlusMSE(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target) + self.alpha * self.l1(pred, target)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=200)).astype(np.float32)

    f_sgd = TorchMLPForecaster(
        horizon=3,
        window=24,
        max_epochs=50,
        batch_size=16,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 1e-3, "momentum": 0.0},
        loss_fn=L1PlusMSE(alpha=0.05),
    )
    f_sgd.fit(y)
    print("SGD forecast:", f_sgd.predict(y))

    f_adam = TorchMLPForecaster(
        horizon=3,
        window=24,
        max_epochs=50,
        batch_size=16,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
    )
    f_adam.fit(y)
    print("Adam forecast:", f_adam.predict(y))
