"""JAX implementation of Soft-MSM distance."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from soft_msm.jax._utils import _softmin3, _trans_cost


def _soft_msm_1d(x, y, c=1.0, gamma=1.0):
    """
    Soft-MSM distance between 1D series.

    Parameters
    ----------
    x : jnp.ndarray, shape (T,)
    y : jnp.ndarray, shape (U,)
    c : float
    gamma : float

    Returns
    -------
    jnp.ndarray, scalar
    """
    n = x.shape[0]
    m = y.shape[0]

    prev_row = [None] * m
    prev_row[0] = (x[0] - y[0]) ** 2

    # First row
    for j in range(1, m):
        trans = _trans_cost(y[j], y[j - 1], x[0], c, gamma)
        prev_row[j] = prev_row[j - 1] + trans

    # Rows 1..n-1
    for i in range(1, n):
        curr_row = [None] * m
        # First column
        trans = _trans_cost(x[i], x[i - 1], y[0], c, gamma)
        curr_row[0] = prev_row[0] + trans

        for j in range(1, m):
            match = (x[i] - y[j]) ** 2
            d_diag = prev_row[j - 1] + match
            d_up = prev_row[j] + _trans_cost(x[i], x[i - 1], y[j], c, gamma)
            d_left = curr_row[j - 1] + _trans_cost(y[j], y[j - 1], x[i], c, gamma)
            curr_row[j] = _softmin3(d_diag, d_up, d_left, gamma)

        prev_row = curr_row

    return prev_row[m - 1]


def _soft_msm_costs_batched(x, y, c, gamma):
    """
    Run DP on channel 0 per batch (matching Aeon's univariate MSM convention).

    Parameters
    ----------
    x : jnp.ndarray, shape (B, C, T)
    y : jnp.ndarray, shape (B, C, U)

    Returns
    -------
    jnp.ndarray, shape (B,)
    """
    B = x.shape[0]
    costs = jnp.zeros(B, dtype=x.dtype)
    for b in range(B):
        cost = _soft_msm_1d(x[b, 0], y[b, 0], c=c, gamma=gamma)
        costs = costs.at[b].set(costs[b] + cost)
    return costs


def _soft_msm_1d_from_M(M_slice, x_ch, y_ch, c, gamma):
    """
    Soft-MSM DP using provided match matrix M for diagonal costs.

    Parameters
    ----------
    M_slice : jnp.ndarray, shape (T, U) — match costs (leaf for grad)
    x_ch : jnp.ndarray, shape (T,) — channel 0, for transition costs
    y_ch : jnp.ndarray, shape (U,) — channel 0, for transition costs

    Returns
    -------
    jnp.ndarray, scalar
    """
    T = x_ch.shape[0]
    U = y_ch.shape[0]

    prev_row = [None] * U
    prev_row[0] = M_slice[0, 0]

    for j in range(1, U):
        trans = _trans_cost(y_ch[j], y_ch[j - 1], x_ch[0], c, gamma)
        prev_row[j] = prev_row[j - 1] + trans

    for i in range(1, T):
        curr_row = [None] * U
        trans = _trans_cost(x_ch[i], x_ch[i - 1], y_ch[0], c, gamma)
        curr_row[0] = prev_row[0] + trans

        for j in range(1, U):
            d_diag = prev_row[j - 1] + M_slice[i, j]
            d_up = prev_row[j] + _trans_cost(x_ch[i], x_ch[i - 1], y_ch[j], c, gamma)
            d_left = curr_row[j - 1] + _trans_cost(
                y_ch[j], y_ch[j - 1], x_ch[i], c, gamma
            )
            curr_row[j] = _softmin3(d_diag, d_up, d_left, gamma)

        prev_row = curr_row

    return prev_row[U - 1]


def _soft_msm_costs_from_M_batched(M, x, y, c, gamma):
    """
    DP with provided match matrix M (channel 0 only).

    Parameters
    ----------
    M : jnp.ndarray, shape (B, T, U)
    x : jnp.ndarray, shape (B, C, T)
    y : jnp.ndarray, shape (B, C, U)

    Returns
    -------
    jnp.ndarray, shape (B,)
    """
    B = M.shape[0]
    costs = jnp.zeros(B, dtype=M.dtype)
    for b in range(B):
        cost = _soft_msm_1d_from_M(M[b], x[b, 0], y[b, 0], c, gamma)
        costs = costs.at[b].set(costs[b] + cost)
    return costs


def soft_msm_loss(x, y, c=1.0, gamma=1.0, reduction="mean"):
    """
    Compute Soft-MSM loss.

    Parameters
    ----------
    x : jnp.ndarray, shape (B, C, T)
    y : jnp.ndarray, shape (B, C, U)
    c : float
        Transition cost parameter.
    gamma : float
        Smoothness parameter (> 0).
    reduction : {"mean", "sum", "none"}

    Returns
    -------
    jnp.ndarray
        Scalar (if reduced) or shape (B,).
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    costs = _soft_msm_costs_batched(x, y, c=c, gamma=gamma)

    if reduction == "mean":
        return jnp.mean(costs)
    if reduction == "sum":
        return jnp.sum(costs)
    return costs


def soft_msm_alignment_matrix(x, y, c=1.0, gamma=1.0):
    """
    Compute expected diagonal-match occupancy E and Soft-MSM cost.

    Parameters
    ----------
    x : jnp.ndarray, shape (B, C, T)
    y : jnp.ndarray, shape (B, C, U)
    c : float
    gamma : float

    Returns
    -------
    E : jnp.ndarray, shape (B, T, U) — channel 0 only (matches Aeon)
    s : jnp.ndarray, shape (B,) float64
    """
    x64 = x.astype(jnp.float64)
    y64 = y.astype(jnp.float64)
    # Channel 0 match matrix
    M = (x64[:, 0, :, None] - y64[:, 0, None, :]) ** 2  # (B, T, U)

    def cost_from_M(M_leaf):
        return _soft_msm_costs_from_M_batched(M_leaf, x64, y64, c, gamma).sum()

    E = jax.grad(cost_from_M)(M)
    s = _soft_msm_costs_batched(x64, y64, c=c, gamma=gamma)

    return E.astype(x.dtype), s


def soft_msm_grad_x(x, y, c=1.0, gamma=1.0):
    """
    Gradient of Soft-MSM cost w.r.t. x.

    Parameters
    ----------
    x : jnp.ndarray, shape (B, C, T)
    y : jnp.ndarray, shape (B, C, U)
    c : float
    gamma : float

    Returns
    -------
    dx : jnp.ndarray, shape (B, C, T) — only channel 0 non-zero
    s : jnp.ndarray, shape (B,) float64
    """
    x64 = x.astype(jnp.float64)
    y64 = y.astype(jnp.float64)

    def cost_from_x(x_leaf):
        return _soft_msm_costs_batched(x_leaf, y64, c=c, gamma=gamma).sum()

    dx = jax.grad(cost_from_x)(x64)
    s = _soft_msm_costs_batched(x64, y64, c=c, gamma=gamma)

    return dx.astype(x.dtype), s
