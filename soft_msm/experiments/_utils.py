import os

import numpy as np
from aeon.datasets import load_from_ts_file
from aeon.transformations.collection import Normalizer
from dotenv import find_dotenv, load_dotenv
from tsml_eval.utils.resampling import stratified_resample_data


def check_experiment_results_exist(
    model_name: str,
    dataset: str,
    combine_test_train: bool,
    path_to_results: str,
    resample_id: int = 0,
    check_only_test: bool = False,
) -> bool:
    """
    Check if the results of the experiment already exist.

    Returns
    -------
    bool
        True if results already exist.
    """
    path_to_train = os.path.join(
        path_to_results,
        model_name,
        "Predictions",
        dataset,
        f"trainResample{resample_id}.csv",
    )
    path_to_test = os.path.join(
        path_to_results,
        model_name,
        "Predictions",
        dataset,
        f"testResample{resample_id}.csv",
    )

    if combine_test_train:
        return os.path.exists(path_to_train)
    else:
        if check_only_test:
            return os.path.exists(path_to_test)
        return os.path.exists(path_to_train) and os.path.exists(path_to_test)


def _normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize time series collection data."""
    scaler = Normalizer()
    return scaler.fit_transform(X)


def load_dataset_from_file(
    dataset_name: str,
    path_to_data: str,
    normalize: bool = True,
    combine_test_train: bool = False,
    resample_id: int | None = None,
):
    """Load dataset from file, optionally doing stratified resampling.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load. If it is one of the MONSTER datasets,
        the data is loaded from the MONSTER Hugging Face repo; otherwise,
        a UCR-style .ts train/test pair is loaded from `path_to_data`.
    path_to_data : str
        Path to the data (used for non-MONSTER datasets).
    normalize : bool, default=True
        Whether to normalize the data.
    combine_test_train : bool, default=False
        Whether to combine the test and train data.
    resample_id : int or None, default=None
        If > 0 and combine_test_train is False, perform a stratified resample of the
        original TRAIN/TEST pair using this as the random seed.
        If 0 or None, use the original train/test split.
    """
    path_to_train_data = os.path.join(
        path_to_data, f"{dataset_name}/{dataset_name}_TRAIN.ts"
    )
    path_to_test_data = os.path.join(
        path_to_data, f"{dataset_name}/{dataset_name}_TEST.ts"
    )

    X_train, y_train = load_from_ts_file(path_to_train_data)
    X_test, y_test = load_from_ts_file(path_to_test_data)

    if not combine_test_train and resample_id is not None and resample_id > 0:
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train,
            y_train,
            X_test,
            y_test,
            random_state=resample_id,
        )
    if combine_test_train:
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        if normalize:
            X = _normalize_data(X)
        return X, y, None, None
    else:
        if normalize:
            X_train = _normalize_data(X_train)
            X_test = _normalize_data(X_test)
        return X_train, y_train, X_test, y_test


def get_distance_default_params(dist_name: str, X: np.ndarray) -> dict:
    """Get default parameters for a distance function.

    Parameters
    ----------
    dist_name: str
        Distance name.
    X: np.ndarray, of shape (n_cases, n_channels, n_timepoints)
        Dataset values.

    Returns
    -------
    dict
        Distance parameters.
    """
    if "soft" in dist_name:
        if dist_name == "soft_dtw":
            return {"gamma": 0.01}
        if dist_name == "soft_msm":
            return {
                "c": 1.0,
                "gamma": 0.1,
            }
        if dist_name == "soft_divergence_dtw":
            return {"gamma": 0.001}
        if dist_name == "soft_divergence_msm":
            return {
                "c": 1.0,
                "gamma": 1.0,
            }
        else:
            return {}
    if dist_name == "lcss":
        return {"epsilon": 1.0}
    if dist_name == "erp":
        return {"g": X.std(axis=0).sum()}
    if dist_name == "msm":
        return {"c": 1.0}
    if dist_name == "edr":
        return {"epsilon": None}
    if dist_name == "twe":
        return {"nu": 0.001, "lmbda": 1.0}
    if dist_name == "adtw":
        return {"warp_penalty": 1.0}
    if dist_name == "shape_dtw":
        return {"descriptor": "identity", "reach": 15}
    if dist_name == "wdtw":
        return {"g": 0.05}
    return {}


def get_averaging_params(averaging_method: str) -> dict:
    """Return averaging parameters for the given averaging_method.

    Parameters
    ----------
    averaging_method: str
        Method name.

    Returns
    -------
    dict
        Averaging parameters.
    """
    base = {"max_iters": 100, "tol": 1e-3}

    if averaging_method == "soft":
        return {**base}

    if averaging_method == "kasba":
        return {
            **base,
            "ba_subset_size": 0.5,
            "initial_step_size": 0.05,
            "decay_rate": 0.1,
        }

    if averaging_method == "petitjean":
        return {**base}

    if averaging_method == "subgradient":
        return {
            **base,
            "initial_step_size": 0.05,
            "final_step_size": 0.005,
        }

    raise ValueError(
        f'Invalid averaging_method="{averaging_method}". '
        'Valid: "soft", "kasba", "petitjean_ba", "subgradient_ba".'
    )


def validate_distance_vs_averaging_method(distance: str, averaging_method: str) -> None:
    """Validate that the distance and averaging method are compatible.

    Parameters
    ----------
    distance: str
        Distance name.
    averaging_method: str
        Averaging method name.
    """
    is_soft_dist = distance.startswith("soft_")
    if averaging_method == "soft" and not is_soft_dist:
        raise ValueError(
            f'averaging_method="soft" requires a soft distance (got "{distance}"). '
            'Use one of: "soft_msm", "soft_dtw", "soft_divergence_msm", '
            '"soft_divergence_dtw".'
        )
    if averaging_method != "soft" and is_soft_dist:
        raise ValueError(
            f'averaging_method="{averaging_method}" does not expect a soft distance '
            f'(got "{distance}"). '
            "Pick a non-soft method or change the distance."
        )


def load_and_validate_env(required_vars=("DATASET_PATH", "RESULT_PATH")):
    """Load and validate required environment variables from a .env file.

    Parameters
    ----------
    required_vars : tuple[str]
        The names of environment variables that must exist.

    Raises
    ------
    FileNotFoundError
        If the .env file is not found.
    EnvironmentError
        If one or more required environment variables are missing.
    """
    env_path = find_dotenv()

    if not env_path:
        raise FileNotFoundError(
            "❌ .env file not found. Please create one in the project root "
            "with the required variables."
        )

    load_dotenv(env_path)

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise OSError(f"❌ Missing environment variables in .env: {', '.join(missing)}")

    return {var: os.getenv(var) for var in required_vars}


def _parse_command_line_bool(s: str) -> bool:
    """Parse a boolean from common CLI strings."""
    t = s.strip().lower()
    if t in {"1", "true", "t", "yes", "y"}:
        return True
    if t in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(
        f"Invalid boolean value: {s!r}. Use one of: true/false/1/0/yes/no."
    )
