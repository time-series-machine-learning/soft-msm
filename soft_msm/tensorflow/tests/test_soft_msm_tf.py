import numpy as np
import pytest
import tensorflow as tf
from aeon.distances.elastic.soft import (
    soft_msm_alignment_matrix as aeon_soft_msm_alignment_matrix,
)
from aeon.distances.elastic.soft import (
    soft_msm_distance,
)
from aeon.distances.elastic.soft import soft_msm_grad_x as aeon_soft_msm_grad_x

from soft_msm.tensorflow import SoftMSMLoss, soft_msm_alignment_matrix, soft_msm_grad_x
from soft_msm.tensorflow.tests._utils import check_arrays_close, check_values_close

GAMMAS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
CS = [0.25, 0.5, 1.0, 2.0]


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("c", CS)
@pytest.mark.parametrize(
    "C,T,U",
    [
        (1, 16, 16),
        (1, 20, 15),
        (3, 10, 12),
    ],
)
def test_soft_msm_loss_equivalence(gamma, c, C, T, U):
    rng = np.random.RandomState(0)
    x_np = rng.randn(1, C, T).astype(np.float32)
    y_np = rng.randn(1, C, U).astype(np.float32)

    x = tf.constant(x_np)
    y = tf.constant(y_np)

    loss_fn = SoftMSMLoss(c=c, gamma=gamma, reduction="none")
    s_tf = float(loss_fn(x, y)[0])
    s_aeon = soft_msm_distance(x_np.squeeze(0), y_np.squeeze(0), c=c, gamma=gamma)

    assert check_values_close(s_tf, s_aeon)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("c", CS)
def test_soft_msm_alignment_matrix_equivalence(gamma, c):
    rng = np.random.RandomState(1)
    C, T, U = 2, 8, 6
    x_np = rng.randn(1, C, T).astype(np.float32)
    y_np = rng.randn(1, C, U).astype(np.float32)

    x = tf.constant(x_np)
    y = tf.constant(y_np)

    E_tf, s_tf = soft_msm_alignment_matrix(x, y, c=c, gamma=gamma)
    E_tf = E_tf.numpy().squeeze(0)
    s_tf = float(s_tf.numpy().squeeze(0))

    E_aeon, s_aeon = aeon_soft_msm_alignment_matrix(
        x_np.squeeze(0), y_np.squeeze(0), c=c, gamma=gamma
    )

    assert check_arrays_close(E_tf, E_aeon)
    assert check_values_close(s_tf, s_aeon)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("c", CS)
def test_soft_msm_grad_x_equivalence(gamma, c):
    rng = np.random.RandomState(2)
    C, T, U = 3, 9, 11
    x_np = rng.randn(1, C, T).astype(np.float32)
    y_np = rng.randn(1, C, U).astype(np.float32)

    x = tf.constant(x_np)
    y = tf.constant(y_np)

    dx_tf, s_tf = soft_msm_grad_x(x, y, c=c, gamma=gamma)
    # Aeon MSM uses channel 0 only; compare channel 0 of gradient
    dx_tf = dx_tf.numpy()[0, 0]
    s_tf = float(s_tf.numpy()[0])

    dx_aeon, s_aeon = aeon_soft_msm_grad_x(
        x_np.squeeze(0), y_np.squeeze(0), c=c, gamma=gamma
    )

    assert check_arrays_close(dx_tf, dx_aeon)
    assert check_values_close(s_tf, s_aeon)


def test_soft_msm_autograd_smoke():
    rng = np.random.RandomState(3)
    B, C, T, U = 4, 2, 12, 10
    x = tf.Variable(rng.randn(B, C, T).astype(np.float32))
    y = tf.constant(rng.randn(B, C, U).astype(np.float32))

    with tf.GradientTape() as tape:
        loss = SoftMSMLoss(c=1.0, gamma=0.1)(x, y)
    grad = tape.gradient(loss, x)
    assert tf.reduce_all(tf.math.is_finite(grad))
