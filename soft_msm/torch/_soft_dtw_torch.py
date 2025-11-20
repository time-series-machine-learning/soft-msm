from typing import Tuple

import torch
from torch import nn

from soft_msm.torch._utils import _softmin3, _pairwise_sq_dists


def _soft_dtw_from_D(D: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute the soft-DTW cost from a distance matrix.

    Parameters
    ----------
    D : torch.Tensor
        Local squared distance matrix of shape (n_cases, n_timepoints_x, n_timepoints_y).
    gamma : float
        Smoothness parameter for the soft minimum.

    Returns
    -------
    torch.Tensor
        Soft-DTW costs of shape (n_cases,).
    """
    B, T, U = D.shape
    R = torch.full((B, 2, U + 1), float("inf"), dtype=D.dtype, device=D.device)
    R[:, 0, 0] = 0.0

    for i in range(1, T + 1):
        prev = (i - 1) & 1
        curr = i & 1
        R[:, curr, 0] = float("inf")
        for j in range(1, U + 1):
            up   = R[:, prev, j]
            diag = R[:, prev, j - 1]
            left = R[:, curr, j - 1]
            R[:, curr, j] = D[:, i - 1, j - 1] + _softmin3(up, diag, left, gamma)

    return R[:, T & 1, U]


class SoftDTWLoss(nn.Module):
    """
    Soft-DTW loss module without window constraints.

    Parameters
    ----------
    gamma : float, default=1.0
        Smoothness parameter for the soft minimum (must be > 0).
    reduction : {"mean", "sum", "none"}, default="mean"
        Reduction applied across the batch.
    """

    def __init__(self, gamma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of {'mean','sum','none'}")
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Soft-DTW loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_cases, n_channels, n_timepoints_x).
        y : torch.Tensor
            Target tensor of shape (n_cases, n_channels, n_timepoints_y).

        Returns
        -------
        torch.Tensor
            Scalar loss (if reduced) or batch of losses of shape (n_cases,).
        """
        if x.shape[:2] != y.shape[:2]:
            raise ValueError("x and y must have the same number of cases and channels")

        D = _pairwise_sq_dists(x, y)
        soft_dtw_costs = _soft_dtw_from_D(D, self.gamma)

        if self.reduction == "mean":
            return soft_dtw_costs.mean()
        if self.reduction == "sum":
            return soft_dtw_costs.sum()
        return soft_dtw_costs


def soft_dtw_alignment_matrix(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the expected alignment matrix and Soft-DTW cost.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (n_cases, n_channels, n_timepoints_x).
    y : torch.Tensor
        Target tensor of shape (n_cases, n_channels, n_timepoints_y).
    gamma : float, default=1.0
        Smoothness parameter for the soft minimum.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        E : Expected alignment matrix of shape (n_cases, n_timepoints_x, n_timepoints_y).
        s : Soft-DTW costs of shape (n_cases,).
    """
    with torch.enable_grad():
        D_leaf = _pairwise_sq_dists(x.detach(), y.detach()).requires_grad_(True)
        s = _soft_dtw_from_D(D_leaf, gamma)
        (E,) = torch.autograd.grad(s.sum(), D_leaf)
    return E.detach(), s.detach()


@torch.no_grad()
def soft_dtw_grad_x(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the gradient of the Soft-DTW cost with respect to x.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (n_cases, n_channels, n_timepoints_x).
    y : torch.Tensor
        Target tensor of shape (n_cases, n_channels, n_timepoints_y).
    gamma : float, default=1.0
        Smoothness parameter for the soft minimum.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        dx : Gradient with respect to x, shape (n_cases, n_channels, n_timepoints_x).
        s : Soft-DTW costs of shape (n_cases,).
    """
    E, s = soft_dtw_alignment_matrix(x, y, gamma=gamma)
    Wxt = E.sum(dim=2)
    Y_w = torch.bmm(y, E.transpose(1, 2))
    dx = 2.0 * (x * Wxt.unsqueeze(1) - Y_w)
    return dx, s