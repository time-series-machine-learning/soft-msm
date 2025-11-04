import argparse

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


def _param_suffix(c_val, g_val):
    parts = []
    if c_val is not None:
        parts.append(f"c-{c_val}")
    if g_val is not None:
        parts.append(f"gamma-{g_val}")
    return ("-" + "-".join(str(p) for p in parts)) if parts else ""


def _grid_for_distance(distance: str, c_list, g_list):
    """Yield (c, gamma) combos appropriate for the distance."""
    is_soft = distance.startswith("soft_")
    is_msm = distance.endswith("msm")  # covers msm, soft_msm, soft_divergence_msm

    if is_soft and is_msm:
        # sweep both c and gamma
        for c in c_list:
            for g in g_list:
                yield (c, g)
        return

    if (not is_soft) and is_msm:
        # only c
        for c in c_list:
            yield (c, None)
        return

    if is_soft and (not is_msm):
        # only gamma (e.g., soft_dtw, soft_wdtw, etc.)
        for g in g_list:
            yield (None, g)
        return

    # everything else -> single run, no params
    yield (None, None)


def run_one(
    *,
    dataset: str,
    distance: str,
    dataset_path: str,
    results_path: str,
    averaging_method: str,
    resample_id: int,
    n_jobs: int,
    c_val,
    g_val,
):
    # model name includes averaging + distance + param suffix
    base_model = f"nearest-centroid-{averaging_method}-{distance}".replace("_", "-")
    model_name = f"{base_model}{_param_suffix(c_val, g_val)}"
    print(
        f"==== Running classification experiment for {model_name} on dataset "
        f"{dataset} ===="
    )

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

    # Base params
    distance_params = get_distance_default_params(distance, X_train)
    # Override c/gamma if provided in the sweep
    if c_val is not None:
        distance_params["c"] = float(c_val)
    if g_val is not None:
        distance_params["gamma"] = float(g_val)

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
        build_test_file=True,  # keep as in your version
        build_train_file=False,  # keep as in your version
        benchmark_time=False,  # keep as in your version
    )
    print(
        f"==== Finished classification experiment for {model_name} on dataset "
        f"{dataset} ===="
    )


RUN_LOCALLY = True

if __name__ == "__main__":
    """
    Usage:
      python _classification_experiment.py <dataset> <distance> <dataset_path>
      <result_path> <averaging_method>
      [--c_csv "0.1,0.5,1.0"] [--gamma_csv "1.0,0.1,0.01,0.001"] [--resample_id 0]
      [--n_jobs -1]

    averaging_method ∈ {"soft", "kasba", "petitjean_ba", "subgradient_ba"}

    Grid logic:
      - msm                         -> sweep c only
      - soft_msm / soft_divergence_msm -> sweep c × gamma
      - soft_* (non-msm)            -> sweep gamma only
      - others                      -> single run
    """
    if RUN_LOCALLY:
        print("RUNNING WITH TEST CONFIG")
        env = load_and_validate_env()
        dataset = "GunPoint"
        distance = "soft_msm"
        averaging_method = "soft"
        dataset_path = env["DATASET_PATH"]
        result_path = env["RESULT_PATH"]
        c_csv = "0.1,0.5,1.0,1.5,2.0"
        gamma_csv = "1.0,0.1,0.01,0.001"
        resample_id = 0
        n_jobs = -1
    else:
        p = argparse.ArgumentParser()
        p.add_argument("dataset", type=str)
        p.add_argument("distance", type=str)
        p.add_argument("dataset_path", type=str)
        p.add_argument("result_path", type=str)
        p.add_argument("averaging_method", type=str)
        p.add_argument("--c_csv", type=str, default="")
        p.add_argument("--gamma_csv", type=str, default="")
        p.add_argument("--resample_id", type=int, default=0)
        p.add_argument("--n_jobs", type=int, default=-1)
        args = p.parse_args()

        dataset = args.dataset
        distance = args.distance
        dataset_path = args.dataset_path
        result_path = args.result_path
        averaging_method = args.averaging_method
        c_csv = args.c_csv
        gamma_csv = args.gamma_csv
        resample_id = args.resample_id
        n_jobs = args.n_jobs

    # Parse grids (empty -> no sweep for that param)
    c_list = [s.strip() for s in c_csv.split(",") if s.strip()] if c_csv else []
    g_list = [s.strip() for s in gamma_csv.split(",") if s.strip()] if gamma_csv else []

    # Provide sensible defaults if the user wants a sweep but forgot to pass lists
    # (No harm if lists are empty; the grid logic will just do a single run.)
    if distance.endswith("msm") and not c_list:
        c_list = ["1.0"]  # default single value if none provided
    if distance.startswith("soft_") and not g_list:
        g_list = ["0.1"]  # default single value if none provided

    # Iterate over the grid appropriate for the distance
    for c_val, g_val in _grid_for_distance(distance, c_list, g_list):
        run_one(
            dataset=dataset,
            distance=distance,
            dataset_path=dataset_path,
            results_path=result_path,
            averaging_method=averaging_method,
            resample_id=resample_id,
            n_jobs=n_jobs,
            c_val=c_val,
            g_val=g_val,
        )
