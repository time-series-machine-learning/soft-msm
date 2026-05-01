"""JAX utility functions for soft distance computations."""

import jax.numpy as jnp


def _softmin3(a, b, c, gamma):
    """Smooth minimum of three values."""
    inv_g = 1.0 / gamma
    x = -a * inv_g
    y = -b * inv_g
    z = -c * inv_g
    return -gamma * jnp.logaddexp(jnp.logaddexp(x, y), z)


def _softmin2(a, b, gamma):
    """Smooth minimum of two values."""
    inv_g = 1.0 / gamma
    return -gamma * jnp.logaddexp(-a * inv_g, -b * inv_g)


def _pairwise_sq_dists(x, y):
    """
    Pairwise squared distances between time series.

    Parameters
    ----------
    x : jnp.ndarray, shape (B, C, T)
    y : jnp.ndarray, shape (B, C, U)

    Returns
    -------
    jnp.ndarray, shape (B, T, U)
    """
    x2 = (x * x).sum(axis=1)
    y2 = (y * y).sum(axis=1)
    xy = jnp.einsum("bct,bcu->btu", x, y)
    return x2[:, :, None] + y2[:, None, :] - 2.0 * xy


def _between_gate(a, b, eps=1e-9):
    """Smooth parameter-free gate: ~1 when a*b < 0 (between), ~0 otherwise."""
    u = a * b
    return 0.5 * (1.0 - u / jnp.sqrt(u * u + eps))


def _trans_cost(x_val, y_prev, z_other, c, gamma):
    """MSM transition cost."""
    a = x_val - y_prev
    b = x_val - z_other
    g = _between_gate(a, b)
    base = _softmin2(a * a, b * b, gamma)
    return c + (1.0 - g) * base
