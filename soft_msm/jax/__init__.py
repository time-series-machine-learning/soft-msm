"""JAX implementation of soft distances.

Provides pure functions for computing Soft-DTW and Soft-MSM distances,
alignment matrices, and gradients. Compatible with jax.grad and jax.jit.
"""

from soft_msm.jax._soft_dtw_jax import (
    soft_dtw_alignment_matrix,
    soft_dtw_grad_x,
    soft_dtw_loss,
)
from soft_msm.jax._soft_msm_jax import (
    soft_msm_alignment_matrix,
    soft_msm_grad_x,
    soft_msm_loss,
)

__all__ = [
    "soft_dtw_loss",
    "soft_dtw_alignment_matrix",
    "soft_dtw_grad_x",
    "soft_msm_loss",
    "soft_msm_alignment_matrix",
    "soft_msm_grad_x",
]
