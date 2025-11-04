import sys

from aeon.classification.distance_based import NearestCentroid
from tsml_eval.experiments import (
    run_classification_experiment as tsml_classification_experiment,
)

from soft_msm.experiments._utils import (
    check_experiment_results_exist,
    get_averaging_params,
    get_distance_default_params,
    load_and_validate_env,
    load_dataset_from_file,
    validate_distance_vs_averaging_method,
)


def run_classification_experiment(
    dataset: str,
    distance: str,
    dataset_path: str,
    results_path: str,
    averaging_method: str,
    resample_id: int = 0,
    n_jobs: int = -1,
):
    """
    Run NearestCentroid with explicit distance + averaging method.

    Parameters
    ----------
    dataset : str
        Dataset name.
    distance : str
        Distance string, e.g. "msm", "dtw", "soft_msm", "soft_dtw",
        "soft_divergence_msm", "soft_divergence_dtw".
    dataset_path : str
        Path to datasets.
    results_path : str
        Path to write results.
    averaging_method : str
        One of: "soft", "kasba", "petitjean_ba", "subgradient_ba".
    resample_id : int, default=0
        Resample seed passed to tsml runner.
    n_jobs : int, default=-1
        Parallelism for distance computations.
    """
    model_name = f"nearest-centroid-{averaging_method}-{distance}".replace("_", "-")
    print(
        f"==== Running classification experiment for {model_name} on dataset "
        f"{dataset} ===="
    )

    # classification uses split train/test -> combine_test_train = False
    if check_experiment_results_exist(
        model_name=model_name,
        dataset=dataset,
        combine_test_train=False,
        path_to_results=results_path,
        resample_id=resample_id,
    ):
        print(f"Results already exist for {model_name} on dataset {dataset}")
        print(
            f"==== Finished classification experiment for {model_name} on dataset "
            f"{dataset} ===="
        )
        return

    # Load split data
    X_train, y_train, X_test, y_test = load_dataset_from_file(
        dataset, dataset_path, normalize=True, combine_test_train=False
    )

    # Validate pairing
    validate_distance_vs_averaging_method(distance, averaging_method)

    # Params (mirrors clustering refactor)
    distance_params = get_distance_default_params(distance, X_train)
    average_params = {
        "distance": distance,
        **get_averaging_params(averaging_method),
        **distance_params,
    }

    classifier = NearestCentroid(
        distance=distance,
        average_method=averaging_method,
        distance_params=distance_params,
        average_params=average_params,
        verbose=False,
        n_jobs=n_jobs,
    )

    tsml_classification_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classifier=classifier,
        results_path=results_path,
        classifier_name=model_name,
        dataset_name=dataset,
        resample_id=resample_id,
        data_transforms=None,
        build_test_file=True,
        build_train_file=False,
        benchmark_time=False,
    )
    print(
        f"==== Finished classification experiment for {model_name} on dataset "
        f"{dataset} ===="
    )


RUN_LOCALLY = False

if __name__ == "__main__":
    """
    Usage (CLI):
      python _classification_experiment.py <dataset> <distance> <dataset_path>
      <result_path> <averaging_method>

    averaging_method ∈ {"soft", "kasba", "petitjean_ba", "subgradient_ba"}
    """
    if RUN_LOCALLY:
        print("RUNNING WITH TEST CONFIG")
        env = load_and_validate_env()
        dataset = "Gunpoint"
        distance = "soft_divergence_msm"
        averaging_method = "soft"
        dataset_path = env["DATASET_PATH"]
        result_path = env["RESULT_PATH"]
    else:
        if len(sys.argv) != 6:
            print(
                "Usage: python _classification_experiment.py "
                "<dataset> <distance> <dataset_path> <result_path> <averaging_method>"
            )
            sys.exit(1)

        dataset = str(sys.argv[1])
        distance = str(sys.argv[2])
        dataset_path = str(sys.argv[3])
        result_path = str(sys.argv[4])
        averaging_method = str(
            sys.argv[5]
        )  # "soft" | "kasba" | "petitjean_ba" | "subgradient_ba"

    run_classification_experiment(
        dataset=dataset,
        distance=distance,
        dataset_path=dataset_path,
        results_path=result_path,
        averaging_method=averaging_method,
        resample_id=0,
        n_jobs=-1,
    )
