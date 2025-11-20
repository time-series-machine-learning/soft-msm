from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from aeon.clustering.averaging import elastic_barycenter_average
from soft_msm.experiments._utils import (
    load_dataset_from_file,
    load_and_validate_env,
    get_averaging_params,
)


if __name__ == "__main__":
    # --- Load data ---
    env = load_and_validate_env()
    DATASET_PATH = Path(env["DATASET_PATH"])
    RESULT_PATH = Path(env["RESULT_PATH"]) / "average_results"
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    DATASET = "CricketX"

    X, y, _, _ = load_dataset_from_file(
        DATASET,
        DATASET_PATH,
        normalize=True,
        combine_test_train=False,
    )

    # X: (n_cases, 1, series_length)
    # y: (n_cases,)

    print("Unique labels in y:", np.unique(y))

    # --- Select only one class (change this if you want the other label) ---
    class_label = "12"  # adjust if needed after seeing the print above
    mask_class = y == class_label
    X_class = X[mask_class]  # (n_class_cases, 1, series_length)

    if X_class.shape[0] == 0:
        raise ValueError(
            f"No cases found for class {class_label}. "
            f"Available labels: {np.unique(y)}"
        )

    X_class_2d = X_class[:, 0, :]  # (n_class_cases, series_length)

    n_cases, series_length = X_class_2d.shape
    time_idx = np.arange(series_length)

    # --- Define methods + their parameter configs ---
    # Two different soft configurations (different gamma), plus petitjean + subgradient.
    labels = [
        "Soft-MSM (γ=0.001)",
        "Soft-DTW (γ=0.001)",
        "MBA",
        "SSG-MBA",
    ]

    methods = [
        "soft",
        "soft",
        "petitjean",
        "subgradient",
    ]

    kwargs_list = [
        {**get_averaging_params("soft"), "gamma": 0.001, "distance": "soft_msm"},
        {**get_averaging_params("soft"), "gamma": 0.001, "distance": "soft_dtw"},
        {**get_averaging_params("petitjean"), "distance": "msm"},
        {**get_averaging_params("subgradient"), "distance": "msm"}
    ]

    # --- Compute barycenters ---
    barycenters = {}

    for label, method, args in zip(labels, methods, kwargs_list):
        bary = elastic_barycenter_average(
            X_class_2d,
            method=method,
            random_state=1,
            **args,
        )

        bary = np.asarray(bary).reshape(-1)

        if bary.shape[0] != series_length:
            raise ValueError(
                f"Barycenter for {label} has length {bary.shape[0]}, "
                f"expected {series_length}."
            )

        barycenters[label] = bary

    # --- Plot: 4 subplots, one per method ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, label in zip(axes, labels):
        # Plot all series of the chosen class in black
        for i in range(n_cases):
            ax.plot(time_idx, X_class_2d[i], color="black", linewidth=0.5, alpha=0.5)

        # Plot barycenter in red, thicker
        ax.plot(
            time_idx,
            barycenters[label],
            color="red",
            linewidth=2.0,
        )

        ax.set_title(label)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(f"Class {class_label} averages on {DATASET}")
    plt.tight_layout()
    plt.show()