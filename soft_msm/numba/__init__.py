"""Numba (CPU) implementation.

We don't explicitly define the soft distances in this repository instead we have
implemented them in the open source aeon package and are simply exposing them here
for convenience.
"""
__all__ = [
    "soft_dtw_alignment_matrix",
    "soft_dtw_alignment_path",
    "soft_dtw_cost_matrix",
    "soft_dtw_distance",
    "soft_dtw_pairwise_distance",
    "soft_msm_alignment_path",
    "soft_msm_distance",
    "soft_msm_cost_matrix",
    "soft_msm_pairwise_distance",
    "soft_msm_alignment_matrix",
    "soft_dtw_grad_x",
    "soft_dtw_divergence_distance",
    "soft_dtw_divergence_pairwise_distance",
    "soft_dtw_divergence_grad_x",
    "soft_msm_divergence_distance",
    "soft_msm_divergence_pairwise_distance",
    "soft_msm_divergence_grad_x",
]

from aeon.distances.elastic.soft import (
    soft_dtw_alignment_matrix,
    soft_dtw_alignment_path,
    soft_dtw_cost_matrix,
    soft_dtw_distance,
    soft_dtw_grad_x,
    soft_dtw_pairwise_distance,
    soft_dtw_divergence_distance,
    soft_dtw_divergence_grad_x,
    soft_dtw_divergence_pairwise_distance,
    soft_msm_alignment_matrix,
    soft_msm_alignment_path,
    soft_msm_cost_matrix,
    soft_msm_distance,
    soft_msm_pairwise_distance,
    soft_msm_divergence_distance,
    soft_msm_divergence_grad_x,
    soft_msm_divergence_pairwise_distance,
)
