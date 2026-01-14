# test_soft_dtw_equivalence.py
#
# Pytest file that checks equivalence between:
# - this module's PyTorch manual Soft-DTW implementation
# - aeon's reference implementations
#
# Adjust the import below to point at your implementation file/module.

from __future__ import annotations

import numpy as np
import pytest
import torch
from aeon.distances.elastic.soft._soft_dtw import (
    soft_dtw_alignment_matrix as aeon_alignment_matrix,
)
from aeon.distances.elastic.soft._soft_dtw import (
    soft_dtw_cost_matrix,
    soft_dtw_distance,
)
from aeon.distances.elastic.soft._soft_dtw import soft_dtw_grad_x as aeon_grad_x

# ---- CHANGE THIS IMPORT to your module path ----
# e.g. from soft_msm.torch.soft_dtw_manual import soft_dtw_cost, soft_dtw_alignment_matrix, soft_dtw_grad_x, pairwise_sq_dists
from soft_msm.torch._soft_dtw_autograd import (
    soft_dtw_alignment_matrix,
    soft_dtw_cost,
    soft_dtw_grad_x,
)


def _device_list() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


@pytest.mark.parametrize("device", _device_list())
@pytest.mark.parametrize(
    "B,C,T,U",
    [
        (1, 1, 5, 7),
        (2, 3, 10, 12),
        (4, 1, 6, 6),
    ],
)
@pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0])
def test_soft_dtw_equivalence_cost_alignment_gradx(
    device: str, B: int, C: int, T: int, U: int, gamma: float
):
    # Make random tests reproducible across devices
    torch.manual_seed(1234)
    np.random.seed(1234)

    dtype = torch.float32

    # Generate inputs
    x = torch.randn(B, C, T, device=device, dtype=dtype)
    y = torch.randn(B, C, U, device=device, dtype=dtype)

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    s_torch = soft_dtw_cost(x, y, gamma=gamma).detach().cpu().numpy()
    s_aeon = np.array(
        [soft_dtw_distance(x_np[i], y_np[i], gamma=gamma) for i in range(B)]
    )

    np.testing.assert_allclose(
        s_torch,
        s_aeon,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Cost mismatch on {device} (B={B},C={C},T={T},U={U},gamma={gamma})",
    )

    # -------------------------
    # 3) Alignment matrix E
    # -------------------------
    E_torch, s2_torch = soft_dtw_alignment_matrix(x, y, gamma=gamma)
    E_torch = E_torch.detach().cpu().numpy()

    E_aeon = np.stack(
        [aeon_alignment_matrix(x_np[i], y_np[i], gamma=gamma) for i in range(B)], axis=0
    )

    # Alignment can be a bit noisier (especially on GPU); still should match closely.
    np.testing.assert_allclose(
        E_torch,
        E_aeon,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Alignment matrix mismatch on {device} (B={B},C={C},T={T},U={U},gamma={gamma})",
    )

    # Optional: alignment-sum sanity checks (soft DTW expected alignment)
    # Sum of E is not necessarily exactly T+U-1 for soft DTW, but it should be positive and finite.
    assert np.isfinite(E_torch).all()
    assert (E_torch >= 0).all()

    # -------------------------
    # 4) Gradient wrt x
    # -------------------------
    dx_torch, s3_torch = soft_dtw_grad_x(x, y, gamma=gamma)
    dx_torch = dx_torch.detach().cpu().numpy()

    dx_aeon = np.stack(
        [aeon_grad_x(x_np[i], y_np[i], gamma=gamma) for i in range(B)], axis=0
    )

    # Gradients typically need slightly looser tolerances due to exp/log and ordering differences.
    np.testing.assert_allclose(
        dx_torch,
        dx_aeon,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Grad x mismatch on {device} (B={B},C={C},T={T},U={U},gamma={gamma})",
    )


def test_soft_dtw_equivalence_on_identical_series_gives_small_cost():
    # A simple property test: x==y should give cost near 0 (exactly 0 for squared euclidean local costs)
    device = "cpu"
    torch.manual_seed(0)

    B, C, T = 2, 3, 12
    gamma = 0.5
    x = torch.randn(B, C, T, device=device, dtype=torch.float32)
    y = x.clone()

    s = soft_dtw_cost(x, y, gamma=gamma).cpu().numpy()
    # soft-DTW of identical series should be very close to 0 (floating noise possible)
    assert np.max(np.abs(s)) < 1e-4
