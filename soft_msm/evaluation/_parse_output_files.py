"""
Load inertia results from a precomputed CSV and print LaTeX tables.

Input (required):
  - final_inertia_by_model_and_gamma.csv
    Produced by the earlier parsing script. Format:
      dataset,<model_1>,<model_2>,...

This script:
- Loads the CSV into memory
- Computes "Better (%)" for:
    soft-MBA-hard-dist-gamma-{g} vs MBA
    soft-DBA-hard-dist-gamma-{g} vs DBA
  using only datasets where BOTH entries are present (non-empty).
- Prints the LaTeX table.

Notes
-----
- "Better (%)" = percentage of datasets where soft inertia < baseline inertia
- Percentages cannot be negative with this definition.
"""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "results" / "clustering" / "inertia_values.csv"

GAMMAS = ["1.0", "0.1", "0.01", "0.001"]  # order in table


def _parse_float(cell: str) -> float | None:
    s = cell.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_inertia_matrix(
    csv_path: Path,
) -> tuple[list[str], list[str], dict[tuple[str, str], float | None]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0].strip().lower() != "dataset":
            raise ValueError("CSV must have first column header 'dataset'.")

        models = [h.strip() for h in header[1:] if h.strip()]
        datasets: list[str] = []
        inertias: dict[tuple[str, str], float | None] = {}

        for row in reader:
            if not row:
                continue
            ds = row[0].strip()
            if not ds:
                continue

            datasets.append(ds)

            # Ensure row length matches header; missing trailing cells treated as empty
            values = row[1:] + [""] * max(0, len(models) - (len(row) - 1))
            for model, cell in zip(models, values):
                inertias[(model, ds)] = _parse_float(cell)

    return datasets, models, inertias


def compare_soft_vs_base(
    inertias: dict[tuple[str, str], float | None],
    datasets: list[str],
    soft_base: str,
    base_model: str,
    gammas: list[str],
) -> dict[str, float]:
    results: dict[str, float] = {}

    for g in gammas:
        soft_model = f"{soft_base}-gamma-{g}"

        better = 0
        compared = 0

        for ds in datasets:
            soft_val = inertias.get((soft_model, ds))
            base_val = inertias.get((base_model, ds))

            if soft_val is None or base_val is None:
                continue

            compared += 1
            if soft_val < base_val:
                better += 1

        results[g] = float("nan") if compared == 0 else 100.0 * better / compared

        print(
            f"{soft_model} vs {base_model}: "
            f"{better}/{compared} better = {results[g]:.2f}%"
        )

    return results


def latex_percent(x: float) -> str:
    if x != x:  # NaN
        return "--"
    return f"{x:.1f}\\,\\%"


def print_latex_table(
    mba_results: dict[str, float],
    dba_results: dict[str, float],
    gammas: list[str],
) -> None:
    print("\nLaTeX table:\n")
    print(r"\begin{table}[t]")
    print(
        r"\caption{Percentage of datasets on which the soft variants achieve a lower "
        r"inertia than the standard variants.}"
    )
    print(r"\label{tab:soft_hard_dist_vs_baselines_by_gamma}")
    print(r"\centering")
    print(r"\begin{tabular}{c|c|c}")
    print(r"\toprule")
    print(r"$\gamma$ & \textbf{MBA} & \textbf{DBA} \\")
    print(r" & Better (\%) & Better (\%) \\")
    print(r"\midrule")

    for g in gammas:
        print(
            f"{g} & {latex_percent(mba_results[g])} & "
            f"{latex_percent(dba_results[g])} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    datasets, models, inertias = load_inertia_matrix(INPUT_CSV)

    for required in ["MBA", "DBA"]:
        if required not in models:
            raise ValueError(
                f"Required model column '{required}' not found in CSV header."
            )

    mba_results = compare_soft_vs_base(
        inertias=inertias,
        datasets=datasets,
        soft_base="soft-MBA-hard-dist",
        base_model="MBA",
        gammas=GAMMAS,
    )

    dba_results = compare_soft_vs_base(
        inertias=inertias,
        datasets=datasets,
        soft_base="soft-DBA-hard-dist",
        base_model="DBA",
        gammas=GAMMAS,
    )

    print_latex_table(mba_results, dba_results, GAMMAS)
