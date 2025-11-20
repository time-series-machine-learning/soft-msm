import math
import pytest
import torch

from aeon.distances.elastic.soft import (
    soft_dtw_distance,
    soft_dtw_alignment_matrix as aeon_soft_dtw_alignment_matrix,
    soft_dtw_grad_x as aeon_soft_dtw_grad_x,
)

from soft_msm.torch import (
    SoftDTWLoss,
    soft_dtw_alignment_matrix,
    soft_dtw_grad_x,
)
from soft_msm.torch.tests._utils import check_arrays_close, \
    check_values_close

DEVICES = ["cpu"]

if torch.cuda.is_available():
    DEVICES.append("cuda")

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICES.append("mps")

GAMMAS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("C,T,U", [
    (1, 16, 16),
    (1, 20, 15),
    (3, 10, 12),
])
def test_soft_dtw_loss_equivalence(device, gamma, C, T, U):
    torch.manual_seed(0)
    x = torch.randn(1, C, T, device=device, requires_grad=True)
    y = torch.randn(1, C, U, device=device, requires_grad=True)

    # Torch implementation
    loss_fn = SoftDTWLoss(gamma=gamma, reduction="none")
    s_torch = loss_fn(x, y)[0].item()

    # Aeon (expects np arrays, shape (C, T) etc.)
    x_np = x.detach().cpu().numpy().squeeze(0)
    y_np = y.detach().cpu().numpy().squeeze(0)
    s_aeon = soft_dtw_distance(x_np, y_np, gamma=gamma)

    assert check_values_close(s_torch, s_aeon)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_soft_dtw_alignment_matrix_equivalence(device, gamma):
    torch.manual_seed(1)
    C, T, U = 2, 8, 6
    x = torch.randn(1, C, T, device=device, dtype=torch.float32)
    y = torch.randn(1, C, U, device=device, dtype=torch.float32)

    E_torch, s_torch = soft_dtw_alignment_matrix(x, y, gamma=gamma)
    E_torch = E_torch.squeeze(0).cpu().numpy()
    s_torch = s_torch.squeeze(0).item()

    x_np = x.cpu().numpy().squeeze(0)
    y_np = y.cpu().numpy().squeeze(0)
    E_aeon, s_aeon = aeon_soft_dtw_alignment_matrix(x_np, y_np, gamma=gamma)

    assert check_arrays_close(E_torch, E_aeon)
    assert check_values_close(s_torch, s_aeon)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_soft_dtw_grad_x_equivalence(device, gamma):
    torch.manual_seed(2)
    C, T, U = 3, 9, 11
    x = torch.randn(1, C, T, device=device, dtype=torch.float32)
    y = torch.randn(1, C, U, device=device, dtype=torch.float32)

    dx_torch, s_torch = soft_dtw_grad_x(x, y, gamma=gamma)
    dx_torch = dx_torch.squeeze(0).cpu().numpy()
    s_torch = s_torch.squeeze(0).item()

    x_np = x.cpu().numpy().squeeze(0)
    y_np = y.cpu().numpy().squeeze(0)
    dx_aeon, s_aeon = aeon_soft_dtw_grad_x(x_np, y_np, gamma=gamma)

    assert check_arrays_close(dx_torch, dx_aeon)
    assert check_values_close(s_torch, s_aeon)


@pytest.mark.parametrize("device", DEVICES)
def test_soft_dtw_autograd_smoke(device):
    torch.manual_seed(3)
    B, C, T, U = 4, 2, 12, 10
    x = torch.randn(B, C, T, device=device, requires_grad=True, dtype=torch.float32)
    y = torch.randn(B, C, U, device=device, requires_grad=True, dtype=torch.float32)
    loss = SoftDTWLoss(gamma=0.1)(x, y)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()