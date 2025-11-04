import sys

import numpy as np
from aeon.clustering import TimeSeriesKMeans
from tsml_eval.experiments import (
    run_clustering_experiment as tsml_clustering_experiment,
)

from soft_msm.experiments._utils import (
    _parse_command_line_bool,
    check_experiment_results_exist,
    get_averaging_params,
    get_distance_default_params,
    load_and_validate_env,
    load_dataset_from_file,
    validate_distance_vs_averaging_method,
)


def run_clustering_experiment(
    dataset: str,
    distance: str,
    clusterer_str: str,
    dataset_path: str,
    results_path: str,
    averaging_method: str,
    combine_test_train: bool = False,
    resample_id: int = 0,
    n_jobs: int = -1,
):
    """Run clustering experiment.

    Parameters
    ----------
    dataset : str
        Dataset name.
    distance : str
        Distance string (assumed correct and final), e.g.:
        "msm", "dtw", "soft_msm", "soft_dtw",
        "soft_divergence_msm", "soft_divergence_dtw".
    clusterer_str : str
        Free-form label used only for naming/logging (not logic).
    dataset_path : str
        Path to the dataset.
    results_path : str
        Path to the results.
    averaging_method : str
        One of: "soft", "kasba", "petitjean_ba", "subgradient_ba".
    combine_test_train : bool, default=False
        Boolean indicating if data should be combined for test and train.
    resample_id : int, default=0
        Integer indicating the resample id.
    n_jobs : int default=-1
        Integer indicating the number of jobs to run in parallel.
    """
    model_name = f"{clusterer_str}-{distance}"
    print(
        f"==== Running clustering experiment for {model_name} on dataset {dataset} ===="
    )

    if check_experiment_results_exist(
        model_name=model_name,
        dataset=dataset,
        combine_test_train=combine_test_train,
        path_to_results=results_path,
        resample_id=resample_id,
    ):
        print(f"Results already exist for {model_name} on dataset {dataset}")
        print(
            f"==== Finished clustering experiment for {model_name} on dataset "
            f"{dataset} ===="
        )
        return

    X_train, y_train, X_test, y_test = load_dataset_from_file(
        dataset, dataset_path, normalize=True, combine_test_train=combine_test_train
    )
    n_clusters = np.unique(y_train).size

    # Validate the distance is compatible with the averaging method.
    validate_distance_vs_averaging_method(distance, averaging_method)

    # Get the default distance parameters.
    distance_params = get_distance_default_params(distance, X_train)

    # Get the default averaging parameters.
    average_params = get_averaging_params(averaging_method)

    # Combine the distance and averaging parameters.
    average_params = {"distance": distance, **average_params, **distance_params}

    clusterer = TimeSeriesKMeans(
        n_clusters=n_clusters,
        init="kmeans++",
        distance=distance,
        n_init=1,
        max_iter=300,
        tol=1e-6,
        verbose=True,
        random_state=resample_id,
        averaging_method=averaging_method,
        distance_params=distance_params,
        average_params=average_params,
        n_jobs=n_jobs,
    )

    tsml_clustering_experiment(
        X_train=X_train,
        y_train=y_train,
        clusterer=clusterer,
        results_path=results_path,
        X_test=X_test,
        y_test=y_test,
        n_clusters=n_clusters,
        clusterer_name=model_name,
        dataset_name=dataset,
        resample_id=resample_id,
        data_transforms=None,
        build_test_file=not combine_test_train,
        build_train_file=True,
        benchmark_time=True,
    )
    print(
        f"==== Finished clustering experiment for {model_name} on dataset "
        f"{dataset} ===="
    )


RUN_LOCALLY = True


if __name__ == "__main__":
    """NOTE: To run with command line arguments, set RUN_LOCALLY to False."""
    if RUN_LOCALLY:
        print("RUNNING WITH TEST CONFIG")
        dataset = "GunPoint"
        distance = "soft_msm"
        averaging_method = "soft"
        clusterer_str = "k-means"
        # Value can be "true" or "false"
        combine_test_train = _parse_command_line_bool("false")

        env = load_and_validate_env()
        dataset_path = env["DATASET_PATH"]
        result_path = env["RESULT_PATH"]

    else:
        if len(sys.argv) != 8:
            print(
                "Usage: python _clustering_experiment.py "
                "<dataset> <distance> <clusterer_name> <dataset_path> <result_path> "
                "<averaging_method> <combine_test_train>"
            )
            sys.exit(1)

        dataset = str(sys.argv[1])
        # one of: dtw | msm | soft_dtw | soft_msm | soft_divergence_dtw |
        # soft_divergence_msm
        distance = str(sys.argv[2])
        clusterer_str = str(sys.argv[3])
        dataset_path = str(sys.argv[4])
        result_path = str(sys.argv[5])
        # one of: soft | kasba | petitjean_ba | subgradient_ba
        averaging_method = str(sys.argv[6])
        combine_test_train = _parse_command_line_bool(sys.argv[7])

    run_clustering_experiment(
        dataset=dataset,
        distance=distance,
        clusterer_str=clusterer_str,
        dataset_path=dataset_path,
        results_path=result_path,
        averaging_method=averaging_method,
        combine_test_train=combine_test_train,
        n_jobs=-1,
    )
