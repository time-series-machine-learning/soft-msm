import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from aeon.distances.elastic.soft import (
    soft_dtw_alignment_matrix as aeon_soft_dtw_alignment_matrix,
)
from aeon.distances.elastic.soft import (
    soft_dtw_distance,
)
from aeon.distances.elastic.soft import soft_dtw_grad_x as aeon_soft_dtw_grad_x

from soft_msm.jax import soft_dtw_alignment_matrix, soft_dtw_grad_x, soft_dtw_loss
from soft_msm.jax.tests._utils import check_arrays_close, check_values_close

GAMMAS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize(
    "C,T,U",
    [
        (1, 16, 16),
        (1, 20, 15),
        (3, 10, 12),
    ],
)
def test_soft_dtw_loss_equivalence(gamma, C, T, U):
    rng = np.random.RandomState(0)
    x_np = rng.randn(1, C, T).astype(np.float32)
    y_np = rng.randn(1, C, U).astype(np.float32)

    x = jnp.array(x_np)
    y = jnp.array(y_np)

    s_jax = float(soft_dtw_loss(x, y, gamma=gamma, reduction="none")[0])
    s_aeon = soft_dtw_distance(x_np.squeeze(0), y_np.squeeze(0), gamma=gamma)

    assert check_values_close(s_jax, s_aeon)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_soft_dtw_alignment_matrix_equivalence(gamma):
    rng = np.random.RandomState(1)
    C, T, U = 2, 8, 6
    x_np = rng.randn(1, C, T).astype(np.float32)
    y_np = rng.randn(1, C, U).astype(np.float32)

    x = jnp.array(x_np)
    y = jnp.array(y_np)

    E_jax, s_jax = soft_dtw_alignment_matrix(x, y, gamma=gamma)
    E_jax = np.asarray(E_jax.squeeze(0))
    s_jax = float(s_jax.squeeze(0))

    E_aeon, s_aeon = aeon_soft_dtw_alignment_matrix(
        x_np.squeeze(0), y_np.squeeze(0), gamma=gamma
    )

    assert check_arrays_close(E_jax, E_aeon)
    assert check_values_close(s_jax, s_aeon)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_soft_dtw_grad_x_equivalence(gamma):
    rng = np.random.RandomState(2)
    C, T, U = 3, 9, 11
    x_np = rng.randn(1, C, T).astype(np.float32)
    y_np = rng.randn(1, C, U).astype(np.float32)

    x = jnp.array(x_np)
    y = jnp.array(y_np)

    dx_jax, s_jax = soft_dtw_grad_x(x, y, gamma=gamma)
    dx_jax = np.asarray(dx_jax.squeeze(0))
    s_jax = float(s_jax.squeeze(0))

    dx_aeon, s_aeon = aeon_soft_dtw_grad_x(
        x_np.squeeze(0), y_np.squeeze(0), gamma=gamma
    )

    assert check_arrays_close(dx_jax, dx_aeon)
    assert check_values_close(s_jax, s_aeon)


def test_soft_dtw_autograd_smoke():
    rng = np.random.RandomState(3)
    B, C, T, U = 4, 2, 12, 10
    x = jnp.array(rng.randn(B, C, T).astype(np.float32))
    y = jnp.array(rng.randn(B, C, U).astype(np.float32))

    def loss_fn(x_in):
        return soft_dtw_loss(x_in, y, gamma=0.1, reduction="mean")

    grad = jax.grad(loss_fn)(x)
    assert jnp.isfinite(grad).all()
