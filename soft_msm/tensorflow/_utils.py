"""TensorFlow utility functions for soft distance computations."""

import tensorflow as tf


def _softmin3(a, b, c, gamma):
    """Smooth minimum of three values."""
    inv_g = 1.0 / gamma
    x = -a * inv_g
    y = -b * inv_g
    z = -c * inv_g
    m = tf.maximum(tf.maximum(x, y), z)
    s = m + tf.math.log(tf.exp(x - m) + tf.exp(y - m) + tf.exp(z - m))
    return -gamma * s


def _softmin2(a, b, gamma):
    """Smooth minimum of two values."""
    inv_g = 1.0 / gamma
    x = -a * inv_g
    y = -b * inv_g
    m = tf.maximum(x, y)
    s = m + tf.math.log(tf.exp(x - m) + tf.exp(y - m))
    return -gamma * s


def _pairwise_sq_dists(x, y):
    """
    Pairwise squared distances between time series.

    Parameters
    ----------
    x : tf.Tensor, shape (B, C, T)
    y : tf.Tensor, shape (B, C, U)

    Returns
    -------
    tf.Tensor, shape (B, T, U)
    """
    x2 = tf.reduce_sum(x * x, axis=1)
    y2 = tf.reduce_sum(y * y, axis=1)
    xy = tf.einsum("bct,bcu->btu", x, y)
    return x2[:, :, None] + y2[:, None, :] - 2.0 * xy


def _between_gate(a, b, eps=1e-9):
    """Smooth parameter-free gate: ~1 when a*b < 0 (between), ~0 otherwise."""
    u = a * b
    return 0.5 * (1.0 - u / tf.sqrt(u * u + eps))


def _trans_cost(x_val, y_prev, z_other, c, gamma):
    """MSM transition cost."""
    a = x_val - y_prev
    b = x_val - z_other
    g = _between_gate(a, b)
    base = _softmin2(a * a, b * b, gamma)
    return c + (1.0 - g) * base
