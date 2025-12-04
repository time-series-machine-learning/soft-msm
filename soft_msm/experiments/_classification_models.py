from aeon.classification import BaseClassifier
from aeon.classification.distance_based import (
    KNeighborsTimeSeriesClassifier,
    NearestCentroid,
)

DEFAULT_MSM_PARAMS = {"c": 1.0}
DEFAULT_SOFT_MBA_PARAMS = {
    **DEFAULT_MSM_PARAMS,
    "distance": "soft_msm",
    "max_iters": 100,
    "tol": 1e-6,
    "init_barycenter": "mean",
}
DEFAULT_SOFT_DBA_PARAMS = {
    "distance": "soft_dtw",
    "max_iters": 100,
    "tol": 1e-6,
    "init_barycenter": "mean",
}


def soft_mba_hard_dist_nearest_centroid(
    gamma: float, random_state: int, n_jobs: int
) -> BaseClassifier:
    return NearestCentroid(
        distance="msm",
        average_method="soft",
        distance_params=DEFAULT_MSM_PARAMS,
        average_params={**DEFAULT_SOFT_MBA_PARAMS, "gamma": gamma},
        n_jobs=n_jobs,
    )


def soft_mba_nearest_centroid(
    gamma: float, random_state: int, n_jobs: int
) -> BaseClassifier:
    return NearestCentroid(
        distance="soft_msm",
        average_method="soft",
        distance_params={**DEFAULT_MSM_PARAMS, "gamma": gamma},
        average_params={**DEFAULT_SOFT_MBA_PARAMS, "gamma": gamma},
        n_jobs=n_jobs,
    )


def soft_dba_hard_dist_nearest_centroid(
    gamma: float, random_state: int, n_jobs: int
) -> BaseClassifier:
    return NearestCentroid(
        distance="dtw",
        average_method="soft",
        average_params={**DEFAULT_SOFT_DBA_PARAMS, "gamma": gamma},
        n_jobs=n_jobs,
    )


def soft_dba_nearest_centroid(
    gamma: float, random_state: int, n_jobs: int
) -> BaseClassifier:
    return NearestCentroid(
        distance="soft_dtw",
        average_method="soft",
        distance_params={"gamma": gamma},
        average_params={**DEFAULT_SOFT_DBA_PARAMS, "gamma": gamma},
        n_jobs=n_jobs,
    )


def knn_soft_dtw_classifier(
    gamma: float, random_state: int, n_jobs: int
) -> BaseClassifier:
    return KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        distance="soft_dtw",
        distance_params={"gamma": gamma},
        n_jobs=n_jobs,
    )


def knn_soft_msm_classifier(
    gamma: float, random_state: int, n_jobs: int
) -> BaseClassifier:
    return KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        distance="soft_msm",
        distance_params={**DEFAULT_MSM_PARAMS, "gamma": gamma},
        n_jobs=n_jobs,
    )


def knn_dtw_classifier(gamma: float, random_state: int, n_jobs: int) -> BaseClassifier:
    return KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        distance="dtw",
        n_jobs=n_jobs,
    )


def knn_msm_classifier(gamma: float, random_state: int, n_jobs: int) -> BaseClassifier:
    return KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        distance="msm",
        distance_params=DEFAULT_MSM_PARAMS,
        n_jobs=n_jobs,
    )


CLASSIFICATION_EXPERIMENT_MODELS = {
    "NearestCentroid-soft-MBA": soft_mba_nearest_centroid,
    "NearestCentroid-soft-DBA": soft_dba_nearest_centroid,
    "NearestCentroid-soft-MBA-hard-dist": soft_mba_hard_dist_nearest_centroid,
    "NearestCentroid-soft-DBA-hard-dist": soft_dba_hard_dist_nearest_centroid,
    "KNN-soft-MSM": knn_soft_msm_classifier,
    "KNN-soft-DTW": knn_soft_dtw_classifier,
    "KNN-DTW": knn_dtw_classifier,
    "KNN-MSM": knn_msm_classifier,
}
