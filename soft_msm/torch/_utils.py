from typing import Tuple

import torch


def _softmin2(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Smooth minimum of two tensors.

    Parameters
    ----------
    a, b : torch.Tensor
        Candidate values, broadcast-compatible and of identical shape.
    gamma : float
        Smoothness parameter (> 0).

    Returns
    -------
    torch.Tensor
        Soft-min over (a, b) with the same shape as the inputs.
    """
    v = torch.stack((a, b), dim=0)
    return -gamma * torch.logsumexp(-v / gamma, dim=0)


@torch.jit.ignore
def _softmin3(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float
) -> torch.Tensor:
    """
    Smooth minimum of three tensors (allocation-free), using logaddexp chaining.

    softmin(a,b,c) = -gamma * log( exp(-a/g) + exp(-b/g) + exp(-c/g) )

    Parameters
    ----------
    a, b, c : torch.Tensor
        Same shape.
    gamma : float
        Smoothness parameter (> 0).

    Returns
    -------
    torch.Tensor
        Same shape as inputs.
    """
    inv_g = 1.0 / gamma
    x = -a * inv_g
    y = -b * inv_g
    z = -c * inv_g
    s = torch.logaddexp(torch.logaddexp(x, y), z)
    return -gamma * s


def _softmin3_weights(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Derivatives of softmin(a,b,c) w.r.t. (a,b,c).

    For softmin = -g*logsumexp([-a/g, -b/g, -c/g]),
    the partial derivatives are a softmax over [-a/g, -b/g, -c/g].

    Returns
    -------
    (wa, wb, wc) each same shape as inputs, summing to 1 elementwise.
    """
    inv_g = 1.0 / gamma
    xa = -a * inv_g
    xb = -b * inv_g
    xc = -c * inv_g

    m = torch.maximum(torch.maximum(xa, xb), xc)
    ea = torch.exp(xa - m)
    eb = torch.exp(xb - m)
    ec = torch.exp(xc - m)
    denom = ea + eb + ec

    return ea / denom, eb / denom, ec / denom


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared distances between time series.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (n_cases, n_channels, n_timepoints_x).
    y : torch.Tensor
        Input tensor of shape (n_cases, n_channels, n_timepoints_y).

    Returns
    -------
    torch.Tensor
        Pairwise squared distances of shape (n_cases, n_timepoints_x, n_timepoints_y).
    """
    x = x.contiguous()
    y = y.contiguous()
    x2 = (x * x).sum(dim=1)
    y2 = (y * y).sum(dim=1)
    xy = torch.bmm(x.transpose(1, 2), y)
    return x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2.0 * xy


def choose_device(prefer: str = "auto") -> torch.device:
    """
    prefer:
      - "auto": mps -> cuda -> cpu
      - "mps" / "cuda" / "cpu": force
    """
    prefer = prefer.lower()
    if prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError(
            "Requested MPS but torch.backends.mps.is_available() is False."
        )
    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
    if prefer == "cpu":
        return torch.device("cpu")

    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
