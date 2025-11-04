import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from aeon.clustering.averaging import elastic_barycenter_average
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.distances import pairwise_distance
from sklearn.utils import check_random_state

from soft_msm.experiments._utils import (
    _parse_command_line_bool,
    get_averaging_params,
    get_distance_default_params,
    load_and_validate_env,
    load_dataset_from_file,
    validate_distance_vs_averaging_method,
)


def classes_with_n_samples(y: np.ndarray, k: int = 10) -> list:
    """Return class labels with at least k samples."""
    (labels, counts) = np.unique(y, return_counts=True)
    return [lbl for lbl, c in zip(labels, counts) if c >= k]


def load_existing_df(root: Path, distance: str) -> pd.DataFrame:
    """Load existing CSVs for a distance into a single DataFrame."""
    files = list(root.glob(f"{distance}_*average*.csv"))
    if not files:
        return pd.DataFrame(
            columns=[
                "name",
                "dataset",
                "method",
                "soft_dist_loss",
                "original_dist_loss",
                "euclidean_loss",
                "time",
                "c",
                "gamma",
            ]
        )
    frames = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            for col in ("name", "dataset", "method"):
                if col in df.columns:
                    df[col] = df[col].astype(str)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] failed to read {f}: {e}")
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "name",
                "dataset",
                "method",
                "soft_dist_loss",
                "original_dist_loss",
                "euclidean_loss",
                "time",
                "c",
                "gamma",
            ]
        )
    )


def build_counts(existing: pd.DataFrame) -> Counter:
    """Count existing rows by (dataset, method, name, c, gamma) for resume logic."""
    if existing.empty:
        return Counter()
    # Handle missing param columns gracefully
    c_col = (
        existing["c"]
        if "c" in existing.columns
        else pd.Series([np.nan] * len(existing))
    )
    g_col = (
        existing["gamma"]
        if "gamma" in existing.columns
        else pd.Series([np.nan] * len(existing))
    )
    idx = list(
        zip(existing["dataset"], existing["method"], existing["name"], c_col, g_col)
    )
    return Counter(idx)


def append_buffer(root: Path, distance: str, rows: list[dict]) -> None:
    """Append buffered rows to canonical {distance}_average.csv."""
    if not rows:
        return
    out = root / f"{distance}_average.csv"
    df = pd.DataFrame(rows)
    header = not out.exists() or out.stat().st_size == 0
    df.to_csv(out, mode="a", header=header, index=False)
    rows.clear()


MSM_CS = [1.0]
GAMMAS = [1.0, 0.1, 0.01, 0.001]


def hypergrid(distance: str, base_dist_params: dict, averaging_method: str):
    """
    Yield tuples (name, dist_params, avg_params) for the current distance/method.

    - msm: sweep c over MSM_CS
    - soft_msm / soft_divergence_msm: sweep over c in MSM_CS AND gamma in GAMMAS
    - soft_* (non-msm): sweep gamma in GAMMAS
    - all other distances: single config

    NOTE: `name` is fixed as "{averaging_method}_{distance}". Params are separate
    columns.
    """
    avg_base = get_averaging_params(averaging_method)
    is_soft = distance.startswith("soft_")
    is_msm = distance.endswith("msm")  # covers "msm", "soft_msm", "soft_divergence_msm"

    base_name = f"{averaging_method}_{distance}"

    if is_soft and is_msm:
        for c in MSM_CS:
            for g in GAMMAS:
                dp = dict(base_dist_params)
                dp["c"] = c
                dp["gamma"] = g
                yield (base_name, dp, dict(avg_base))
        return

    if (not is_soft) and is_msm:
        for c in MSM_CS:
            dp = dict(base_dist_params)
            dp["c"] = c
            yield (base_name, dp, dict(avg_base))
        return

    if is_soft and (not is_msm):
        for g in GAMMAS:
            dp = dict(base_dist_params)
            dp["gamma"] = g
            yield (base_name, dp, dict(avg_base))
        return

    # default: single config
    yield (base_name, dict(base_dist_params), dict(avg_base))


def run_once(
    *,
    X: np.ndarray,
    y: np.ndarray,
    distance: str,
    averaging_method: str,
    name: str,
    dist_params: dict,
    avg_params: dict,
    dataset_name: str,
    rng_seed: int,
    k: int = 10,
) -> dict:
    """Pick a random class with ≥k samples, take k series, run EBA, compute losses."""
    rng = check_random_state(rng_seed)
    candidates = classes_with_n_samples(y, k)
    if not candidates:
        raise RuntimeError("No class with enough samples.")

    cls = rng.choice(candidates)
    idx = np.where(y == cls)[0]
    sel = rng.choice(idx, k, replace=False)
    S = X[sel]

    start = time.time()
    avg, loss = elastic_barycenter_average(
        S,
        distance=distance,
        method=averaging_method,
        random_state=rng,
        return_cost=True,
        n_jobs=-1,
        **avg_params,
        **dist_params,
    )
    dt = time.time() - start

    # soft loss vs original base-distance loss
    soft_loss = float(loss) if distance.startswith("soft_") else 0.0
    if distance.startswith("soft_"):
        base = distance[len("soft_") :]
        if base.startswith("divergence_"):
            base = base[len("divergence_") :]
        base_params = {k: v for k, v in dist_params.items() if k != "gamma"}
        orig = float(
            np.sum(pairwise_distance(S, avg, method=base, n_jobs=-1, **base_params))
        )
    else:
        orig = float(loss)

    eucl = float(np.sum(pairwise_distance(S, avg, method="euclidean")))
    return {
        "name": name,
        "dataset": dataset_name,
        "method": averaging_method,
        "soft_dist_loss": abs(soft_loss),
        "original_dist_loss": orig,
        "euclidean_loss": eucl,
        "time": f"{dt:.5f}",
        "c": dist_params.get("c"),
        "gamma": dist_params.get("gamma"),
    }


# -------------------- main --------------------

RUN_LOCALLY = True

if __name__ == "__main__":
    """
    CLI (when RUN_LOCALLY=False):
      python average_experiment.py <distances_csv> <averaging_method> <repeats:int>
        <combine_test_train:bool>
    """
    env = load_and_validate_env()
    DATASET_PATH = Path(env["DATASET_PATH"])
    RESULT_PATH = Path(env["RESULT_PATH"]) / "average_results"
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    if RUN_LOCALLY:
        print("RUNNING WITH TEST CONFIG")
        distances_csv = "msm"  # e.g. "soft_msm,soft_dtw"
        averaging_method = (
            "subgradient"  # e.g. "soft" | "kasba" | "petitjean" | "subgradient"
        )
        REPEATS = 10
        combine_test_train = True
    else:
        if len(sys.argv) != 5:
            print(
                "Usage: python average_experiment.py <distances_csv> "
                "<averaging_method> <repeats:int> <combine_test_train:bool>"
            )
            sys.exit(1)
        distances_csv = str(sys.argv[1])
        averaging_method = str(sys.argv[2])
        REPEATS = int(sys.argv[3])
        combine_test_train = _parse_command_line_bool(sys.argv[4])

    distances = [d.strip() for d in distances_csv.split(",") if d.strip()]
    datasets = list(univariate_equal_length)

    total_run = total_skip = 0

    for dist in distances:
        print(f"\n=== Distance: {dist} | averaging_method: {averaging_method} ===")
        try:
            validate_distance_vs_averaging_method(dist, averaging_method)
        except ValueError as e:
            print(f"[SKIP] {dist}: {e}")
            continue

        existing = load_existing_df(RESULT_PATH, dist)
        have = build_counts(existing)
        buffer: list[dict] = []

        for di, ds in enumerate(datasets, 1):
            print(f"\nDataset {ds} ({di}/{len(datasets)})")
            X, y, _, _ = load_dataset_from_file(
                str(ds),
                str(DATASET_PATH),
                normalize=True,
                combine_test_train=combine_test_train,
            )
            if not classes_with_n_samples(y, 10):
                print("  -> SKIP DATASET: no class with ≥10 samples")
                continue

            base_dp = get_distance_default_params(dist, X)

            for name, dp, ap in hypergrid(dist, base_dp, averaging_method):
                key = (
                    str(ds),
                    averaging_method,
                    name,
                    dp.get("c"),
                    dp.get("gamma"),
                )  # <-- include params
                for rep in range(have.get(key, 0), REPEATS):
                    c_str = f", c={dp['c']}" if "c" in dp else ""
                    g_str = f", gamma={dp['gamma']}" if "gamma" in dp else ""
                    print(
                        f"  -> RUN : {dist} | {ds} | {averaging_method}{c_str}{g_str} "
                        f"(rep {rep+1}/{REPEATS})"
                    )
                    row = run_once(
                        X=X,
                        y=y,
                        distance=dist,
                        averaging_method=averaging_method,
                        name=name,
                        dist_params=dp,
                        avg_params=ap,
                        dataset_name=str(ds),
                        rng_seed=rep,
                        k=10,
                    )
                    buffer.append(row)
                    total_run += 1
                    have[key] = rep + 1
            # Flush per dataset to be safe
            append_buffer(RESULT_PATH, dist, buffer)

        # Final flush per distance
        append_buffer(RESULT_PATH, dist, buffer)
        print(
            f"\nFinished distance '{dist}'. New runs: {total_run}, skipped: "
            f"{total_skip}"
        )

    print(f"\nALL DONE. New runs: {total_run}, skipped: {total_skip}")
