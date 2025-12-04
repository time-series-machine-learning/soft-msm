import sys

from tsml_eval.experiments import (
    run_classification_experiment as tsml_classification_experiment,
)

from soft_msm.experiments._classification_models import CLASSIFICATION_EXPERIMENT_MODELS
from soft_msm.experiments._utils import (
    check_experiment_results_exist,
    load_and_validate_env,
    load_dataset_from_file,
)


def run_threaded_classification_experiment(
    dataset: str,
    classifier_name: str,
    dataset_path: str,
    results_path: str,
    resample_id: int,
    gamma: float = None,
):
    if classifier_name not in CLASSIFICATION_EXPERIMENT_MODELS:
        raise ValueError(f"Unknown classifier_name '{classifier_name}'")

    if gamma is not None:
        model_output_name = f"{classifier_name}-gamma-{gamma}"
    else:
        model_output_name = classifier_name

    # Skip if results already exist
    if check_experiment_results_exist(
        model_name=model_output_name,
        dataset=dataset,
        path_to_results=results_path,
        resample_id=resample_id,
        check_only_test=True,
        combine_test_train=False,
    ):
        return (
            f"[SKIP] {model_output_name} (resample {resample_id}): "
            f"results already exist."
        )

    X_train, y_train, X_test, y_test = load_dataset_from_file(
        dataset,
        dataset_path,
        normalize=True,
        resample_id=resample_id,
    )

    factory = CLASSIFICATION_EXPERIMENT_MODELS[classifier_name]
    classifier = factory(
        gamma=gamma,
        random_state=resample_id,
        n_jobs=-1,
    )

    tsml_classification_experiment(
        X_train=X_train,
        y_train=y_train,
        classifier=classifier,
        results_path=results_path,
        X_test=X_test,
        y_test=y_test,
        classifier_name=model_output_name,
        dataset_name=dataset,
        resample_id=resample_id,
        data_transforms=None,
        build_test_file=True,
        build_train_file=False,
        benchmark_time=False,
    )
    print(f"[DONE] {model_output_name} (resample {resample_id})")


RUN_LOCALLY = False

if __name__ == "__main__":
    """NOTE: To run with command line arguments, set RUN_LOCALLY to False."""
    if RUN_LOCALLY:
        print("RUNNING WITH TEST CONFIG")

        dataset = "GunPoint"
        classifier_name = "NearestCentroid-soft-MBA"

        env = load_and_validate_env()
        dataset_path = env["DATASET_PATH"]
        results_path = env["RESULT_PATH"]
        gamma = 1.0

        run_threaded_classification_experiment(
            dataset=dataset,
            classifier_name=classifier_name,
            gamma=gamma,
            dataset_path=dataset_path,
            results_path=results_path,
            resample_id=0,
        )

    else:
        if len(sys.argv) < 5:
            print(
                "Usage: python _classification_experiment.py "
                "<dataset> <classifier_name> <gamma> <dataset_path> <result_path>"
            )
            sys.exit(1)

        dataset = str(sys.argv[1])
        classifier_name = str(sys.argv[2])
        dataset_path = str(sys.argv[3])
        results_path = str(sys.argv[4])
        if len(sys.argv) == 6:
            gamma = float(sys.argv[5])
        else:
            gamma = None

        run_threaded_classification_experiment(
            dataset=dataset,
            classifier_name=classifier_name,
            gamma=gamma,
            dataset_path=dataset_path,
            results_path=results_path,
            resample_id=0,
        )
