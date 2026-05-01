"""TensorFlow implementation of soft distances.

Provides classes and functions for computing Soft-DTW and Soft-MSM distances,
alignment matrices, and gradients. Compatible with tf.GradientTape.
"""

from soft_msm.tensorflow._soft_dtw_tf import (
    SoftDTWLoss,
    soft_dtw_alignment_matrix,
    soft_dtw_grad_x,
)
from soft_msm.tensorflow._soft_msm_tf import (
    SoftMSMLoss,
    soft_msm_alignment_matrix,
    soft_msm_grad_x,
)

__all__ = [
    "SoftDTWLoss",
    "soft_dtw_alignment_matrix",
    "soft_dtw_grad_x",
    "SoftMSMLoss",
    "soft_msm_alignment_matrix",
    "soft_msm_grad_x",
]
