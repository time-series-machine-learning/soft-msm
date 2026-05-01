"""JAX implementation of Soft-DTW distance."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from soft_msm.jax._utils import _pairwise_sq_dists, _softmin3


def _soft_dtw_from_D(D, gamma):
    """
    Compute Soft-DTW cost from a pairwise distance matrix.

    Parameters
    ----------
    D : jnp.ndarray, shape (B, T, U)
    gamma : float

    Returns
    -------
    jnp.ndarray, shape (B,)
    """
    B, T, U = D.shape

    prev_row = [jnp.full(B, float("inf"), dtype=D.dtype) for _ in range(U + 1)]
    prev_row[0] = jnp.zeros(B, dtype=D.dtype)

    for i in range(1, T + 1):
        curr_row = [jnp.full(B, float("inf"), dtype=D.dtype)]
        for j in range(1, U + 1):
            up = prev_row[j]
            diag = prev_row[j - 1]
            left = curr_row[j - 1]
            val = D[:, i - 1, j - 1] + _softmin3(up, diag, left, gamma)
            curr_row.append(val)
        prev_row = curr_row

    return prev_row[U]


def soft_dtw_loss(x, y, gamma=1.0, reduction="mean"):
    """
    Compute Soft-DTW loss.

    Parameters
    ----------
    x : jnp.ndarray, shape (B, C, T)
    y : jnp.ndarray, shape (B, C, U)
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

    D = _pairwise_sq_dists(x, y)
    costs = _soft_dtw_from_D(D, gamma)

    if reduction == "mean":
        return jnp.mean(costs)
    if reduction == "sum":
        return jnp.sum(costs)
    return costs


def soft_dtw_alignment_matrix(x, y, gamma=1.0):
    """
    Compute the expected alignment matrix E and Soft-DTW cost.

    Parameters
    ----------
    x : jnp.ndarray, shape (B, C, T)
    y : jnp.ndarray, shape (B, C, U)
    gamma : float

    Returns
    -------
    E : jnp.ndarray, shape (B, T, U)
    s : jnp.ndarray, shape (B,)
    """
    x64 = x.astype(jnp.float64)
    y64 = y.astype(jnp.float64)
    D = _pairwise_sq_dists(x64, y64)

    def cost_from_D(D_leaf):
        return _soft_dtw_from_D(D_leaf, gamma).sum()

    E = jax.grad(cost_from_D)(D)
    s = _soft_dtw_from_D(D, gamma)
    return E.astype(x.dtype), s


def soft_dtw_grad_x(x, y, gamma=1.0):
    """
    Gradient of Soft-DTW cost w.r.t. x.

    Parameters
    ----------
    x : jnp.ndarray, shape (B, C, T)
    y : jnp.ndarray, shape (B, C, U)
    gamma : float

    Returns
    -------
    dx : jnp.ndarray, shape (B, C, T)
    s : jnp.ndarray, shape (B,)
    """
    E, s = soft_dtw_alignment_matrix(x, y, gamma=gamma)
    E64 = E.astype(jnp.float64)
    x64 = x.astype(jnp.float64)
    y64 = y.astype(jnp.float64)

    Wxt = E64.sum(axis=2)
    Y_w = jnp.einsum("bcu,btu->bct", y64, E64)
    dx = 2.0 * (x64 * Wxt[:, None, :] - Y_w)
    return dx.astype(x.dtype), s
