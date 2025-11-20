# # TorchMLPForecaster: a PyTorch forecaster compatible with aeon's BaseForecaster
#
# from typing import Optional
#
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
#
# from aeon.forecasting import BaseForecaster, DirectForecastingMixin, IterativeForecastingMixin
#
#
# class _MLP(nn.Module):
#     def __init__(self, window: int, hidden: int, activation: str = "tanh", dropout: float = 0.0):
#         super().__init__()
#         act = {
#             "tanh": nn.Tanh(),
#             "relu": nn.ReLU(),
#             "gelu": nn.GELU(),
#             "elu": nn.ELU(),
#             "leaky_relu": nn.LeakyReLU(),
#         }.get(activation, nn.Tanh())
#         layers = [
#             nn.Linear(window, hidden),
#             act,
#         ]
#         if dropout and dropout > 0:
#             layers.append(nn.Dropout(dropout))
#         layers.append(nn.Linear(hidden, 1))
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, x):
#         # x: (B, window)
#         return self.net(x).squeeze(-1)  # -> (B,)
#
#
# class TorchMLPForecaster(BaseForecaster, DirectForecastingMixin, IterativeForecastingMixin):
#     """
#     PyTorch MLP forecaster (univariate), trained on sliding windows from a single series.
#
#     Parameters
#     ----------
#     horizon : int
#         Forecast horizon used for training/prediction (used by BaseForecaster + mixins).
#     window : int, default=32
#         Number of most-recent time points the MLP sees as input.
#     hidden : int, default=64
#         Hidden layer width.
#     activation : {"tanh","relu","gelu","elu","leaky_relu"}, default="tanh"
#         Activation function.
#     epochs : int, default=50
#         Training epochs for each fit call.
#     batch_size : int, default=64
#         Mini-batch size.
#     lr : float, default=1e-3
#         Adam learning rate.
#     weight_decay : float, default=0.0
#         Adam weight decay (L2).
#     dropout : float, default=0.0
#         Dropout after the hidden layer.
#     normalize : bool, default=True
#         If True, z-score normalize inputs (and de-normalize targets implicitly unnecessary,
#         since target is a scalar value; we train directly on raw values).
#     seed : Optional[int], default=None
#         Random seed for PyTorch/Numpy (for reproducibility).
#     device : {"cpu","cuda"}, default="cpu"
#         Training/inference device.
#
#     Notes
#     -----
#     - Works with `DirectForecastingMixin` (clones, sets `horizon=i+1`, refits per step).
#     - Works with `IterativeForecastingMixin` (fits once, then feeds back own predictions).
#     - Expects **univariate** series; aeon handles axis via BaseForecaster.
#
#     """
#
#     _tags = {
#         "capability:univariate": True,
#         "capability:multivariate": False,
#         "capability:missing_values": False,
#         "capability:horizon": True,       # enables DirectForecastingMixin
#         "capability:exogenous": False,
#         "fit_is_empty": False,
#         "y_inner_type": "np.ndarray",
#     }
#
#     def __init__(
#         self,
#         horizon: int = 1,
#         window: int = 32,
#         hidden: int = 64,
#         activation: str = "tanh",
#         epochs: int = 50,
#         batch_size: int = 64,
#         lr: float = 1e-3,
#         weight_decay: float = 0.0,
#         dropout: float = 0.0,
#         normalize: bool = True,
#         seed: Optional[int] = None,
#         device: str = "cpu",
#         axis: int = 1,
#     ):
#         self.window = window
#         self.hidden = hidden
#         self.activation = activation
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.dropout = dropout
#         self.normalize = normalize
#         self.seed = seed
#         self.device = device
#
#         super().__init__(horizon=horizon, axis=axis)
#
#     # -------------------------
#     # aeon hooks to implement
#     # -------------------------
#     def _fit(self, y, exog):
#         """
#         y: np.ndarray shaped as (n_channels, n_timepoints) by aeon preprocessing.
#            This class is univariate => y.shape == (1, T)
#         """
#         if self.seed is not None:
#             np.random.seed(self.seed)
#             torch.manual_seed(self.seed)
#
#         y1d = y.squeeze()  # -> (T,)
#         T = y1d.shape[0]
#         w = self.window
#         h = self.horizon
#
#         if T < w + h:
#             raise ValueError(
#                 f"Series too short: length={T}, need at least window+horizon={w+h}"
#             )
#
#         # Build sliding-window dataset: x_t = y[t-w:t], target = y[t + h - 1]
#         X_list, t_list = [], []
#         # we form inputs ending at index t-1, predict y[t + h - 1]
#         for t in range(w, T - h + 1):
#             X_list.append(y1d[t - w : t])
#             t_list.append(y1d[t + h - 1])
#
#         X = np.stack(X_list, axis=0).astype(np.float32)   # (N, w)
#         targets = np.array(t_list, dtype=np.float32)      # (N,)
#
#         # Optional normalization (z-score per window feature dimension over training set)
#         if self.normalize:
#             self._x_mean_ = X.mean(axis=0, keepdims=True)
#             self._x_std_ = X.std(axis=0, keepdims=True) + 1e-8
#             Xn = (X - self._x_mean_) / self._x_std_
#         else:
#             self._x_mean_ = None
#             self._x_std_ = None
#             Xn = X
#
#         # Torch tensors & dataloader
#         X_t = torch.from_numpy(Xn)
#         y_t = torch.from_numpy(targets)
#         ds = TensorDataset(X_t, y_t)
#         loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
#
#         # Model/optim
#         self.model_ = _MLP(window=w, hidden=self.hidden, activation=self.activation, dropout=self.dropout).to(self.device)
#         self.model_.train()
#         opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         loss_fn = nn.MSELoss()
#
#         # Train
#         for _ in range(self.epochs):
#             for xb, yb in loader:
#                 xb = xb.to(self.device)
#                 yb = yb.to(self.device)
#                 opt.zero_grad()
#                 pred = self.model_(xb)   # (B,)
#                 loss = loss_fn(pred, yb)
#                 loss.backward()
#                 opt.step()
#
#         # Save last seen training series tail for convenience (optional)
#         # Used to compute default window at prediction time if needed.
#         self._last_train_y_ = y1d
#         return self
#
#     def _predict(self, y, exog):
#         """
#         Return a single scalar prediction `horizon` steps ahead of the end of y.
#         """
#         self.model_.eval()
#
#         y1d = y.squeeze()  # (T_pred,)
#         T = y1d.shape[0]
#
#         if T < self.window:
#             raise ValueError(
#                 f"Need at least `window={self.window}` points for prediction, got {T}."
#             )
#
#         x_win = y1d[-self.window :]  # last window
#         x_win = x_win.astype(np.float32)[None, :]  # (1, window)
#
#         if self.normalize and (self._x_mean_ is not None):
#             x_win = (x_win - self._x_mean_) / self._x_std_
#
#         xb = torch.from_numpy(x_win).to(self.device)
#         with torch.no_grad():
#             pred = self.model_(xb).cpu().numpy().ravel()[0]
#         return float(pred)