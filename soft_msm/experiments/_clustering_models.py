from aeon.clustering import (
    KASBA,
    BaseClusterer,
    KSpectralCentroid,
    TimeSeriesKMeans,
    TimeSeriesKShape,
)

from soft_msm.experiments._classification_models import (
    DEFAULT_MSM_PARAMS,
    DEFAULT_SOFT_DBA_PARAMS,
    DEFAULT_SOFT_MBA_PARAMS,
)


def soft_mba_hard_dist_clusterer(
    n_clusters: int, gamma: float, random_state: int, n_jobs: int
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="msm",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=True,
        random_state=random_state,
        averaging_method="soft",
        distance_params=DEFAULT_MSM_PARAMS,
        average_params={**DEFAULT_SOFT_MBA_PARAMS, "gamma": gamma},
        n_jobs=n_jobs,
    )


def soft_mba_clusterer(
    n_clusters: int, gamma: float, random_state: int, n_jobs: int
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="soft_msm",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=True,
        random_state=random_state,
        averaging_method="soft",
        distance_params={**DEFAULT_MSM_PARAMS, "gamma": gamma},
        average_params={**DEFAULT_SOFT_MBA_PARAMS, "gamma": gamma},
        n_jobs=n_jobs,
    )


def soft_dba_hard_dist_clusterer(
    n_clusters: int, gamma: float, random_state: int, n_jobs: int
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="dtw",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=True,
        random_state=random_state,
        averaging_method="soft",
        average_params={**DEFAULT_SOFT_DBA_PARAMS, "gamma": gamma},
        n_jobs=n_jobs,
    )


def soft_dba_clusterer(
    n_clusters: int, gamma: float, random_state: int, n_jobs: int
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="soft_dtw",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=True,
        random_state=random_state,
        averaging_method="soft",
        distance_params={"gamma": gamma},
        average_params={**DEFAULT_SOFT_DBA_PARAMS, "gamma": gamma},
        n_jobs=n_jobs,
    )


def kasba_clusterer(n_clusters: int, random_state: int, *args) -> BaseClusterer:
    return KASBA(
        n_clusters=n_clusters,
        distance="msm",
        ba_subset_size=0.5,
        initial_step_size=0.05,
        max_iter=300,
        tol=1e-6,
        distance_params={"c": 1.0},
        decay_rate=0.1,
        verbose=False,
        random_state=random_state,
    )


def dba_clusterer(
    n_clusters: int, random_state: int, n_jobs: int, *args
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="dtw",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=True,
        random_state=random_state,
        averaging_method="ba",
        distance_params=None,
        average_params=None,
        n_jobs=n_jobs,
    )


def shape_dba_clusterer(
    n_clusters: int, random_state: int, n_jobs: int, *args
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="shape_dtw",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        averaging_method="ba",
        distance_params={"reach": 15},
        average_params={"reach": 15},
        n_jobs=n_jobs,
    )


def mba_clusterer(
    n_clusters: int, random_state: int, n_jobs: int, *args
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="msm",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=True,
        random_state=random_state,
        averaging_method="ba",
        distance_params={"c": 1.0},
        average_params={"c": 1.0},
        n_jobs=n_jobs,
    )


def euclid_clusterer(
    n_clusters: int, random_state: int, n_jobs: int, *args
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="euclidean",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        averaging_method="mean",
        distance_params=None,
        average_params=None,
        n_jobs=n_jobs,
    )


def msm_clusterer(
    n_clusters: int, random_state: int, n_jobs: int, *args
) -> BaseClusterer:
    return TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance="msm",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        averaging_method="mean",
        distance_params={"c": 1.0},
        average_params=None,  # This isn't used when mean selected
        n_jobs=n_jobs,
    )


def k_sc_clusterer(
    n_clusters: int, random_state: int, n_jobs: int, *args
) -> BaseClusterer:
    return KSpectralCentroid(
        n_clusters=n_clusters,
        max_shift=None,  # This means it will be calculated automatically to length m
        init="kmeans++",
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def kshape_clusterer(n_clusters: int, random_state: int, *args) -> BaseClusterer:
    return TimeSeriesKShape(
        n_clusters=n_clusters,
        centroid_init="kmeans++",
        max_iter=300,
        n_init=1,
        random_state=random_state,
        verbose=False,
        tol=1e-6,
    )


CLUSTERING_EXPERIMENT_MODELS = {
    "soft-MBA": soft_mba_clusterer,
    "soft-MBA-hard-dist": soft_mba_hard_dist_clusterer,
    "soft-DBA": soft_dba_clusterer,
    "soft-DBA-hard-dist": soft_dba_hard_dist_clusterer,
    "KASBA": kasba_clusterer,
    "DBA": dba_clusterer,
    "shape-DBA": shape_dba_clusterer,
    "MBA": mba_clusterer,
    "Euclid": euclid_clusterer,
    "MSM": msm_clusterer,
    "k-Shape": kshape_clusterer,
    "k-SC": k_sc_clusterer,
}
