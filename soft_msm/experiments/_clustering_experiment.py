import sys

import numpy as np
from tsml_eval.experiments import (
    run_clustering_experiment as tsml_clustering_experiment,
)

from soft_msm.experiments._clustering_models import CLUSTERING_EXPERIMENT_MODELS
from soft_msm.experiments._utils import (
    _parse_command_line_bool,
    check_experiment_results_exist,
    load_and_validate_env,
    load_dataset_from_file,
)


def run_threaded_clustering_experiment(
    dataset: str,
    clusterer_name: str,
    combine_test_train: bool,
    dataset_path: str,
    results_path: str,
    resample_id: int,
    gamma: float = None,
):
    if clusterer_name not in CLUSTERING_EXPERIMENT_MODELS:
        raise ValueError(f"Unknown classifier_name '{clusterer_name}'")

    if gamma is not None:
        model_output_name = f"{clusterer_name}-gamma-{gamma}"
    else:
        model_output_name = clusterer_name

    # Skip if results already exist
    if check_experiment_results_exist(
        model_name=model_output_name,
        dataset=dataset,
        path_to_results=results_path,
        resample_id=resample_id,
        check_only_test=not combine_test_train,
        combine_test_train=combine_test_train,
    ):
        return (
            f"[SKIP] {model_output_name} (resample {resample_id}): "
            f"results already exist."
        )

    X_train, y_train, X_test, y_test = load_dataset_from_file(
        dataset,
        dataset_path,
        normalize=True,
        combine_test_train=combine_test_train,
        resample_id=resample_id,
    )
    n_clusters = np.unique(y_train).size

    factory = CLUSTERING_EXPERIMENT_MODELS[clusterer_name]
    clusterer = factory(
        n_clusters=n_clusters,
        random_state=resample_id,
        n_jobs=-1,
        gamma=gamma,
    )

    tsml_clustering_experiment(
        X_train=X_train,
        y_train=y_train,
        clusterer=clusterer,
        results_path=results_path,
        X_test=X_test,
        y_test=y_test,
        n_clusters=n_clusters,
        clusterer_name=model_output_name,
        dataset_name=dataset,
        resample_id=resample_id,
        data_transforms=None,
        build_test_file=not combine_test_train,
        build_train_file=True,
        benchmark_time=True,
    )
    print(f"[DONE] {model_output_name} (resample {resample_id})")


RUN_LOCALLY = False

if __name__ == "__main__":
    """NOTE: To run with command line arguments, set RUN_LOCALLY to False."""
    if RUN_LOCALLY:
        print("RUNNING WITH TEST CONFIG")

        dataset = "GunPoint"
        classifier_name = "soft-DBA-hard-dist"

        env = load_and_validate_env()
        dataset_path = env["DATASET_PATH"]
        results_path = env["RESULT_PATH"]
        gamma = 0.001

        run_threaded_clustering_experiment(
            dataset=dataset,
            clusterer_name=classifier_name,
            combine_test_train=True,
            gamma=gamma,
            dataset_path=dataset_path,
            results_path=results_path,
            resample_id=0,
        )

    else:
        if len(sys.argv) < 6:
            print(
                "Usage: python _classification_experiment.py "
                "<dataset> <classifier_name> <gamma> <dataset_path> <result_path>"
            )
            sys.exit(1)

        dataset = str(sys.argv[1])
        classifier_name = str(sys.argv[2])
        combine_test_train = _parse_command_line_bool(sys.argv[3])
        dataset_path = str(sys.argv[4])
        results_path = str(sys.argv[5])
        if len(sys.argv) == 7:
            gamma = float(sys.argv[6])
        else:
            gamma = None

        run_threaded_clustering_experiment(
            dataset=dataset,
            clusterer_name=classifier_name,
            combine_test_train=combine_test_train,
            gamma=gamma,
            dataset_path=dataset_path,
            results_path=results_path,
            resample_id=0,
        )
