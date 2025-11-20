#!/usr/bin/env python3
"""
Script to generate LaTeX tables comparing soft methods to baseline methods.
"""

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_average_results_data(csv_path):
    """Load CSV data and compute average losses per dataset/method/gamma."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            method = row["method"]
            gamma = row["gamma"] if row["gamma"] else None
            loss = float(row["original_dist_loss"])
            data[dataset][method][gamma].append(loss)

    # Compute averages
    averages = defaultdict(lambda: defaultdict(dict))
    for dataset in data:
        for method in data[dataset]:
            for gamma in data[dataset][method]:
                losses = data[dataset][method][gamma]
                averages[dataset][method][gamma] = sum(losses) / len(losses)

    return averages


def calculate_percentages(
    data, soft_method="soft", baseline1="petitjean", baseline2="subgradient"
):
    """Calculate percentage of datasets where soft method is better for each gamma."""
    gammas = ["1.0", "0.1", "0.01", "0.001"]
    results = {gamma: {"baseline1": 0, "baseline2": 0, "total": 0} for gamma in gammas}

    for dataset in data:
        # Get baseline losses (no gamma)
        baseline1_loss = data[dataset].get(baseline1, {}).get(None)
        baseline2_loss = data[dataset].get(baseline2, {}).get(None)

        if baseline1_loss is None or baseline2_loss is None:
            continue

        for gamma in gammas:
            soft_loss = data[dataset].get(soft_method, {}).get(gamma)
            if soft_loss is None:
                continue

            results[gamma]["total"] += 1
            if soft_loss < baseline1_loss:
                results[gamma]["baseline1"] += 1
            if soft_loss < baseline2_loss:
                results[gamma]["baseline2"] += 1

    # Convert to percentages
    percentages = {}
    for gamma in gammas:
        total = results[gamma]["total"]
        if total > 0:
            percentages[gamma] = {
                "baseline1": (results[gamma]["baseline1"] / total) * 100,
                "baseline2": (results[gamma]["baseline2"] / total) * 100,
            }
        else:
            percentages[gamma] = {"baseline1": 0, "baseline2": 0}

    return percentages


def get_percentage_dataframe(method: str) -> pd.DataFrame:
    """
    Calculate percentage of datasets where soft method is better for each gamma.

    Parameters
    ----------
    method : str
        Either "msm" or "dtw"

    Returns
    -------
    pd.DataFrame
        DataFrame with gamma values as index and baseline percentages as columns.
        Columns are named after the baseline methods (e.g., "DBA", "SSG-DBA").
    """
    if method.lower() not in ["msm", "dtw"]:
        raise ValueError(f"method must be 'msm' or 'dtw', got '{method}'")

    base_dir = Path("full_results/average_results")
    csv_file = f"{method.lower()}_final_average.csv"
    csv_path = base_dir / csv_file

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    # Load data
    data = load_average_results_data(csv_path)

    # Determine baseline names based on method
    if method.lower() == "dtw":
        baseline1_name = "DBA"
        baseline2_name = "SSG-DBA"
    else:  # msm
        baseline1_name = "MBA"
        baseline2_name = "SSG-MBA"

    # Calculate percentages
    percentages = calculate_percentages(
        data, soft_method="soft", baseline1="petitjean", baseline2="subgradient"
    )

    # Convert to DataFrame
    gammas = ["1.0", "0.1", "0.01", "0.001"]
    rows = []
    for gamma in gammas:
        rows.append(
            {
                baseline1_name: percentages[gamma]["baseline1"],
                baseline2_name: percentages[gamma]["baseline2"],
            }
        )

    df = pd.DataFrame(rows, index=gammas)
    return df


def generate_latex_from_dataframe(df: pd.DataFrame, method: str) -> str:
    """
    Generate LaTeX table code from a percentage DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with gamma values as index and baseline percentages as columns.
        Should be the output of get_percentage_dataframe().
    method : str
        Either "msm" or "dtw" (used to determine caption and label)

    Returns
    -------
    str
        LaTeX table code as a string.
    """
    if method.lower() not in ["msm", "dtw"]:
        raise ValueError(f"method must be 'msm' or 'dtw', got '{method}'")

    if df.empty or len(df.columns) != 2:
        raise ValueError("DataFrame must have exactly 2 columns")

    baseline1_name = df.columns[0]
    baseline2_name = df.columns[1]

    # Determine caption and label based on method
    if method.lower() == "dtw":
        caption = (
            "Percentage of datasets on which the Soft-DTW Barycentre Average "
            "achieves a lower DTW loss compared to Dynamic Barycentre Averaging "
            "(DBA) and Stochastic Subgradient Dynamic Barycentre Averaging (SSG-DBA)."
        )
        label = "tab:soft_dtw_vs_baselines_by_gamma"
    else:  # msm
        caption = (
            "Percentage of datasets on which the Soft-MSM Barycentre Average "
            "achieves a lower MSM loss compared to Moving Barycentre Averaging "
            "(MBA) and Stochastic Subgradient Moving Barycentre Averaging (SSG-MBA)."
        )
        label = "tab:soft_msm_vs_baselines_by_gamma"

    gamma_display = {"1.0": "1", "0.1": "0.1", "0.01": "0.01", "0.001": "0.001"}
    gammas = ["1.0", "0.1", "0.01", "0.001"]

    lines = [
        "\\begin{table}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\centering",
        "\\makebox[\\textwidth][c]{%",
        "\\begin{tabular}{l|c|c}",
        "\\toprule",
        "\\multirow{2}{*}{$\\gamma$} \\\\",
        f"& \\textbf{{{baseline1_name}}} & \\textbf{{{baseline2_name}}} \\\\",
        "& Better (\\%) & Better (\\%) \\\\",
        "\\midrule",
    ]

    for gamma in gammas:
        if gamma not in df.index:
            continue
        p1 = df.loc[gamma, baseline1_name]
        p2 = df.loc[gamma, baseline2_name]
        lines.append(f"{gamma_display[gamma]}   & {p1:.1f}\\,\\% & {p2:.1f}\\,\\% \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])

    return "\n".join(lines)


def main():
    """Example usage of the functions."""
    # Process DTW data
    print("Processing DTW data...")
    dtw_df = get_percentage_dataframe("dtw")
    dtw_table = generate_latex_from_dataframe(dtw_df, "dtw")

    # Process MSM data
    print("Processing MSM data...")
    msm_df = get_percentage_dataframe("msm")
    msm_table = generate_latex_from_dataframe(msm_df, "msm")

    # Write output files
    output_dir = Path("full_results")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "dtw_table.tex", "w") as f:
        f.write(dtw_table)

    with open(output_dir / "msm_table.tex", "w") as f:
        f.write(msm_table)

    print(f"\nGenerated tables:")
    print(f"  - {output_dir / 'dtw_table.tex'}")
    print(f"  - {output_dir / 'msm_table.tex'}")

    # Also print to console
    print("\n" + "=" * 80)
    print("DTW TABLE:")
    print("=" * 80)
    print(dtw_table)
    print("\n" + "=" * 80)
    print("MSM TABLE:")
    print("=" * 80)
    print(msm_table)


if __name__ == "__main__":
    main()
