"""TensorFlow implementation of Soft-MSM distance."""

from __future__ import annotations

import tensorflow as tf

from soft_msm.tensorflow._utils import _softmin3, _trans_cost


def _soft_msm_1d(x, y, c=1.0, gamma=1.0):
    """
    Soft-MSM distance between 1D series.

    Parameters
    ----------
    x : tf.Tensor, shape (T,)
    y : tf.Tensor, shape (U,)
    c : float
    gamma : float

    Returns
    -------
    tf.Tensor, scalar
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
    x : tf.Tensor, shape (B, C, T)
    y : tf.Tensor, shape (B, C, U)

    Returns
    -------
    tf.Tensor, shape (B,)
    """
    B = x.shape[0]
    costs = [tf.constant(0.0, dtype=x.dtype)] * B
    for b in range(B):
        cost = _soft_msm_1d(x[b, 0], y[b, 0], c=c, gamma=gamma)
        costs[b] = costs[b] + cost
    return tf.stack(costs)


def _soft_msm_1d_from_M(M_slice, x_ch, y_ch, c, gamma):
    """
    Soft-MSM DP using provided match matrix M for diagonal costs.

    Parameters
    ----------
    M_slice : tf.Tensor, shape (T, U) — match costs (leaf for grad)
    x_ch : tf.Tensor, shape (T,) — channel 0, for transition costs
    y_ch : tf.Tensor, shape (U,) — channel 0, for transition costs

    Returns
    -------
    tf.Tensor, scalar
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
    M : tf.Tensor, shape (B, T, U)
    x : tf.Tensor, shape (B, C, T)
    y : tf.Tensor, shape (B, C, U)

    Returns
    -------
    tf.Tensor, shape (B,)
    """
    B = M.shape[0]
    costs = [tf.constant(0.0, dtype=M.dtype)] * B
    for b in range(B):
        cost = _soft_msm_1d_from_M(M[b], x[b, 0], y[b, 0], c, gamma)
        costs[b] = costs[b] + cost
    return tf.stack(costs)


class SoftMSMLoss:
    """
    Soft-MSM loss.

    Parameters
    ----------
    c : float, default=1.0
        Transition cost parameter.
    gamma : float, default=1.0
        Smoothness parameter (> 0).
    reduction : {"mean", "sum", "none"}, default="mean"
    """

    def __init__(self, c: float = 1.0, gamma: float = 1.0, reduction: str = "mean"):
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of {'mean','sum','none'}")
        self.c = float(c)
        self.gamma = float(gamma)
        self.reduction = reduction

    def __call__(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Compute Soft-MSM loss.

        Parameters
        ----------
        x : tf.Tensor, shape (B, C, T)
        y : tf.Tensor, shape (B, C, U)

        Returns
        -------
        tf.Tensor
            Scalar (if reduced) or shape (B,).
        """
        costs = _soft_msm_costs_batched(x, y, c=self.c, gamma=self.gamma)

        if self.reduction == "mean":
            return tf.reduce_mean(costs)
        if self.reduction == "sum":
            return tf.reduce_sum(costs)
        return costs


def soft_msm_alignment_matrix(x, y, c=1.0, gamma=1.0):
    """
    Compute expected diagonal-match occupancy E and Soft-MSM cost.

    Parameters
    ----------
    x : tf.Tensor, shape (B, C, T)
    y : tf.Tensor, shape (B, C, U)
    c : float
    gamma : float

    Returns
    -------
    E : tf.Tensor, shape (B, T, U) — channel 0 only (matches Aeon)
    s : tf.Tensor, shape (B,) float64
    """
    x64 = tf.cast(x, tf.float64)
    y64 = tf.cast(y, tf.float64)
    # Channel 0 match matrix
    M = (x64[:, 0, :, None] - y64[:, 0, None, :]) ** 2  # (B, T, U)

    with tf.GradientTape() as tape:
        tape.watch(M)
        costs = _soft_msm_costs_from_M_batched(M, x64, y64, c, gamma)
        total = tf.reduce_sum(costs)

    E = tape.gradient(total, M)
    s = _soft_msm_costs_batched(x64, y64, c=c, gamma=gamma)

    return tf.cast(E, x.dtype), s


def soft_msm_grad_x(x, y, c=1.0, gamma=1.0):
    """
    Gradient of Soft-MSM cost w.r.t. x.

    Parameters
    ----------
    x : tf.Tensor, shape (B, C, T)
    y : tf.Tensor, shape (B, C, U)
    c : float
    gamma : float

    Returns
    -------
    dx : tf.Tensor, shape (B, C, T) — only channel 0 non-zero
    s : tf.Tensor, shape (B,) float64
    """
    x64 = tf.cast(x, tf.float64)
    y64 = tf.cast(y, tf.float64)

    with tf.GradientTape() as tape:
        tape.watch(x64)
        costs = _soft_msm_costs_batched(x64, y64, c=c, gamma=gamma)
        total = tf.reduce_sum(costs)

    dx = tape.gradient(total, x64)
    s = costs

    return tf.cast(dx, x.dtype), s
