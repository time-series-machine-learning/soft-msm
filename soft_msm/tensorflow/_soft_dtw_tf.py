"""TensorFlow implementation of Soft-DTW distance."""

from __future__ import annotations

import tensorflow as tf

from soft_msm.tensorflow._utils import _pairwise_sq_dists, _softmin3


def _soft_dtw_from_D(D, gamma):
    """
    Compute Soft-DTW cost from a pairwise distance matrix.

    Parameters
    ----------
    D : tf.Tensor, shape (B, T, U)
    gamma : float

    Returns
    -------
    tf.Tensor, shape (B,)
    """
    B = tf.shape(D)[0]
    T = D.shape[1]
    U = D.shape[2]

    inf_val = tf.constant(float("inf"), dtype=D.dtype)
    prev_row = [tf.fill([B], inf_val) for _ in range(U + 1)]
    prev_row[0] = tf.zeros([B], dtype=D.dtype)

    for i in range(1, T + 1):
        curr_row = [tf.fill([B], inf_val)]
        for j in range(1, U + 1):
            up = prev_row[j]
            diag = prev_row[j - 1]
            left = curr_row[j - 1]
            val = D[:, i - 1, j - 1] + _softmin3(up, diag, left, gamma)
            curr_row.append(val)
        prev_row = curr_row

    return prev_row[U]


class SoftDTWLoss:
    """
    Soft-DTW loss.

    Parameters
    ----------
    gamma : float, default=1.0
        Smoothness parameter (> 0).
    reduction : {"mean", "sum", "none"}, default="mean"
    """

    def __init__(self, gamma: float = 1.0, reduction: str = "mean"):
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of {'mean','sum','none'}")
        self.gamma = float(gamma)
        self.reduction = reduction

    def __call__(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Compute Soft-DTW loss.

        Parameters
        ----------
        x : tf.Tensor, shape (B, C, T)
        y : tf.Tensor, shape (B, C, U)

        Returns
        -------
        tf.Tensor
            Scalar (if reduced) or shape (B,).
        """
        D = _pairwise_sq_dists(x, y)
        costs = _soft_dtw_from_D(D, self.gamma)

        if self.reduction == "mean":
            return tf.reduce_mean(costs)
        if self.reduction == "sum":
            return tf.reduce_sum(costs)
        return costs


def soft_dtw_alignment_matrix(x, y, gamma=1.0):
    """
    Compute the expected alignment matrix E and Soft-DTW cost.

    Parameters
    ----------
    x : tf.Tensor, shape (B, C, T)
    y : tf.Tensor, shape (B, C, U)
    gamma : float

    Returns
    -------
    E : tf.Tensor, shape (B, T, U)
    s : tf.Tensor, shape (B,)
    """
    x64 = tf.cast(x, tf.float64)
    y64 = tf.cast(y, tf.float64)
    D = _pairwise_sq_dists(x64, y64)
    D_watched = tf.identity(D)

    with tf.GradientTape() as tape:
        tape.watch(D_watched)
        s = _soft_dtw_from_D(D_watched, gamma)
        s_sum = tf.reduce_sum(s)

    E = tape.gradient(s_sum, D_watched)
    return tf.cast(E, x.dtype), s


def soft_dtw_grad_x(x, y, gamma=1.0):
    """
    Gradient of Soft-DTW cost w.r.t. x.

    Parameters
    ----------
    x : tf.Tensor, shape (B, C, T)
    y : tf.Tensor, shape (B, C, U)
    gamma : float

    Returns
    -------
    dx : tf.Tensor, shape (B, C, T)
    s : tf.Tensor, shape (B,)
    """
    E, s = soft_dtw_alignment_matrix(x, y, gamma=gamma)
    E64 = tf.cast(E, tf.float64)
    x64 = tf.cast(x, tf.float64)
    y64 = tf.cast(y, tf.float64)

    Wxt = tf.reduce_sum(E64, axis=2)
    Y_w = tf.einsum("bcu,btu->bct", y64, E64)
    dx = 2.0 * (x64 * Wxt[:, None, :] - Y_w)
    return tf.cast(dx, x.dtype), s
