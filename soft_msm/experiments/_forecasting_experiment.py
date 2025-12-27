import sys

import matplotlib.pyplot as plt
import numpy as np

from soft_msm.experiments._forecasting_models import FORECASTING_EXPERIMENT_MODELS
from soft_msm.experiments._utils import (
    _parse_command_line_bool,
    check_experiment_results_exist,
    load_and_validate_env,
    load_dataset_from_file,
)


def run_threaded_forecasting_experiment(
    dataset: str,
    forecaster_name: str,
    combine_test_train: bool,
    dataset_path: str,
    results_path: str,
    resample_id: int,
    horizon: int,
    window: int,
    loss_gamma: float | None = None,
):
    if forecaster_name not in FORECASTING_EXPERIMENT_MODELS:
        raise ValueError(f"Unknown forecaster_name '{forecaster_name}'")

    if loss_gamma is not None:
        model_output_name = f"{forecaster_name}-gamma-{loss_gamma}"
    else:
        model_output_name = forecaster_name

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

    X_train = X_train.squeeze(axis=1)
    X_test = X_test.squeeze(axis=1)

    factory = FORECASTING_EXPERIMENT_MODELS[forecaster_name]

    forecaster = factory(
        horizon=horizon,
        window=window,
        axis=1,
        random_state=resample_id,
        device="mpu",
        # gamma=loss_gamma,
    )

    forecaster.fit(X_train)
    y_pred = forecaster.predict(X_test)

    ts_index = 50
    plt.figure(1, layout="constrained")
    plt.title("Multi-step ahead forecasting using MSE")
    plt.plot(X_test[ts_index].ravel())
    plt.plot(np.arange(150, 275), y_pred[ts_index], "r-")
    plt.show()

    # tsml_forecasting_experiment(
    #     y_train=y_train_series,
    #     forecaster=forecaster,
    #     results_path=results_path,
    #     y_test=y_test_series,
    #     forecaster_name=model_output_name,
    #     dataset_name=dataset,
    #     resample_id=resample_id,
    #     data_transforms=None,
    #     build_test_file=True,
    #     build_train_file=True,
    #     benchmark_time=True,
    #     # If tsml_eval supports these explicitly, pass them; otherwise remove:
    #     horizon=horizon,
    #     window=window,
    # )

    print(f"[DONE] {model_output_name} (resample {resample_id})")


RUN_LOCALLY = True

if __name__ == "__main__":
    """NOTE: To run with command line arguments, set RUN_LOCALLY to False."""
    if RUN_LOCALLY:
        print("RUNNING WITH TEST CONFIG")

        dataset = "GunPoint"
        forecaster_name = "MLP-Adam-MSE"

        env = load_and_validate_env()
        dataset_path = env["DATASET_PATH"]
        results_path = env["RESULT_PATH"]

        run_threaded_forecasting_experiment(
            dataset=dataset,
            forecaster_name=forecaster_name,
            combine_test_train=False,
            dataset_path=dataset_path,
            results_path=results_path,
            resample_id=0,
            horizon=3,
            window=24,
            loss_gamma=None,
        )

    else:
        if len(sys.argv) < 8:
            print(
                "Usage: python _forecasting_experiment.py "
                "<dataset> <forecaster_name> <combine_test_train> "
                "<horizon> <window> <dataset_path> <result_path> [gamma]"
            )
            sys.exit(1)

        dataset = str(sys.argv[1])
        forecaster_name = str(sys.argv[2])
        combine_test_train = _parse_command_line_bool(sys.argv[3])
        horizon = int(sys.argv[4])
        window = int(sys.argv[5])
        dataset_path = str(sys.argv[6])
        results_path = str(sys.argv[7])

        if len(sys.argv) == 9:
            loss_gamma = float(sys.argv[8])
        else:
            loss_gamma = None

        run_threaded_forecasting_experiment(
            dataset=dataset,
            forecaster_name=forecaster_name,
            combine_test_train=combine_test_train,
            dataset_path=dataset_path,
            results_path=results_path,
            resample_id=0,
            horizon=horizon,
            window=window,
            loss_gamma=loss_gamma,
        )
