from typing import Any, Dict

import torch
import torch.nn as nn

from soft_msm.custom_models._mlp_forecaster import TorchMLPForecaster, build_mlp

OPTIMIZERS = {
    "SGD": (torch.optim.SGD, {"lr": 1e-3, "momentum": 0.0}),
    "Adam": (torch.optim.Adam, {"lr": 1e-3}),
}


def mse_loss_factory(**kwargs) -> nn.Module:
    return nn.MSELoss()


def soft_msm_loss_factory(**kwargs) -> nn.Module:
    # TODO
    return nn.MSELoss()


def soft_dtw_loss_factory(**kwargs) -> nn.Module:
    # TODO
    return nn.MSELoss()


LOSSES = {
    "MSE": (mse_loss_factory, {}),
    "soft-DTW": (soft_dtw_loss_factory, {"gamma": 1.0}),
    "soft-MSM": (soft_msm_loss_factory, {"gamma": 1.0, "c": 1.0}),
}

FORECASTING_EXPERIMENT_MODELS = {}


def mlp_forecaster(
    horizon: int,
    window: int,
    axis: int,
    random_state: int,
    device: str,
    optimizer_cls,
    optimizer_kwargs: dict[str, Any],
    loss_fn: nn.Module,
    max_epochs: int = 50,
    batch_size: int = 32,
    hidden=(64, 64),
):
    net = build_mlp(in_features=window, out_features=horizon, hidden=hidden)

    torch.manual_seed(random_state)

    return TorchMLPForecaster(
        horizon=horizon,
        axis=axis,
        window=window,
        net=net,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        loss_fn=loss_fn,
    )


for opt_name, (opt_cls, opt_kwargs) in OPTIMIZERS.items():
    for loss_name, (loss_factory, loss_kwargs) in LOSSES.items():
        key = f"MLP-{opt_name}-{loss_name}"

        def _make_builder(
            opt_cls=opt_cls,
            opt_kwargs=opt_kwargs,
            loss_factory=loss_factory,
            loss_kwargs=loss_kwargs,
        ):
            def builder(
                horizon: int,
                window: int,
                axis: int,
                random_state: int,
                device: str,
                max_epochs: int = 50,
                batch_size: int = 32,
            ):
                loss_fn = loss_factory(**loss_kwargs)
                return mlp_forecaster(
                    horizon=horizon,
                    window=window,
                    axis=axis,
                    random_state=random_state,
                    device=device,
                    optimizer_cls=opt_cls,
                    optimizer_kwargs=opt_kwargs,
                    loss_fn=loss_fn,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                )

            return builder

        FORECASTING_EXPERIMENT_MODELS[key] = _make_builder()
