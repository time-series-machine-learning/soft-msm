from __future__ import annotations

from typing import Tuple

import torch

from soft_msm.torch._utils import _pairwise_sq_dists, _softmin3, _softmin3_weights


@torch.no_grad()
def soft_dtw_forward_full(D: torch.Tensor, gamma: float) -> torch.Tensor:
    D = D.contiguous()
    B, T, U = D.shape
    inf = torch.tensor(float("inf"), device=D.device, dtype=D.dtype)

    R = torch.empty((B, T + 1, U + 1), device=D.device, dtype=D.dtype)
    R.fill_(inf)
    R[:, 0, 0] = 0.0

    for i in range(1, T + 1):
        for j in range(1, U + 1):
            up = R[:, i - 1, j]
            diag = R[:, i - 1, j - 1]
            left = R[:, i, j - 1]
            R[:, i, j] = D[:, i - 1, j - 1] + _softmin3(up, diag, left, gamma)

    return R


@torch.no_grad()
def soft_dtw_alignment_from_D(
    D: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    D = D.contiguous()
    B, T, U = D.shape

    R = soft_dtw_forward_full(D, gamma)
    s = R[:, T, U].clone()

    G = torch.zeros_like(R)
    G[:, T, U] = 1.0

    E = torch.empty((B, T, U), device=D.device, dtype=D.dtype)

    for i in range(T, 0, -1):
        for j in range(U, 0, -1):
            g = G[:, i, j]

            E[:, i - 1, j - 1] = g

            up = R[:, i - 1, j]
            diag = R[:, i - 1, j - 1]
            left = R[:, i, j - 1]

            w_up, w_diag, w_left = _softmin3_weights(up, diag, left, gamma)

            G[:, i - 1, j] += g * w_up
            G[:, i - 1, j - 1] += g * w_diag
            G[:, i, j - 1] += g * w_left

    return E, s


@torch.no_grad()
def soft_dtw_cost(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    D = _pairwise_sq_dists(x, y)
    R = soft_dtw_forward_full(D, gamma)
    return R[:, D.shape[1], D.shape[2]].clone()


@torch.no_grad()
def soft_dtw_alignment_matrix(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    D = _pairwise_sq_dists(x, y)
    return soft_dtw_alignment_from_D(D, gamma)


@torch.no_grad()
def soft_dtw_grad_x(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    D = _pairwise_sq_dists(x, y)
    E, s = soft_dtw_alignment_from_D(D, gamma=gamma)

    Wxt = E.sum(dim=2)

    Y_w = torch.bmm(y.contiguous(), E.transpose(1, 2).contiguous())

    dx = 2.0 * (x * Wxt.unsqueeze(1) - Y_w)
    return dx, s


@torch.no_grad()
def soft_dtw_grad_y(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    D = _pairwise_sq_dists(x, y)
    E, s = soft_dtw_alignment_from_D(D, gamma=gamma)

    Wyu = E.sum(dim=1)

    X_w = torch.bmm(x.contiguous(), E.contiguous())

    dy = 2.0 * (y * Wyu.unsqueeze(1) - X_w)
    return dy, s


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    B, C, T, U = 4, 3, 16, 20
    gamma = 0.1

    x = torch.randn(B, C, T, device=device, dtype=torch.float32)
    y = torch.randn(B, C, U, device=device, dtype=torch.float32)

    E, s = soft_dtw_alignment_matrix(x, y, gamma)
    dx, _ = soft_dtw_grad_x(x, y, gamma)
    dy, _ = soft_dtw_grad_y(x, y, gamma)

    print("E:", E.shape, "s:", s.shape, "dx:", dx.shape, "dy:", dy.shape)

    from aeon.distances.elastic.soft._soft_dtw import (
        soft_dtw_alignment_matrix,
        soft_dtw_cost_matrix,
        soft_dtw_distance,
        soft_dtw_grad_x,
    )
