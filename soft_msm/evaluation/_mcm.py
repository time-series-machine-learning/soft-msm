import os

import matplotlib.pyplot as plt
import pandas as pd
from aeon.visualisation import create_multi_comparison_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "clustering",
    "clacc_mean.csv",
)

if __name__ == "__main__":
    # ---- read real data ----
    df = pd.read_csv(CSV_PATH, index_col=0)

    # The first row "Estimators:" is a label, so we rename the index to remove it
    df.index.name = None

    # ---- create the MCM diagram ----
    fig = create_multi_comparison_matrix(
        results=df,
        save_path=os.path.join(PROJECT_ROOT, "results", "clustering", "mcm_diagram"),
        formats=("pdf",),
        statistic_name="CLAcc",
        higher_stat_better=True,
        pvalue_correction=None,
        plot_1v1_comparisons=False,
        font_size=16,
        fig_size=(20, 8),
        pvalue_threshold=0.1,
        pvalue_test_params={"zero_method": "wilcox", "alternative": "greater"},
        show_symmetry=False,
    )

    plt.show()
    save_path = os.path.join(PROJECT_ROOT, "results", "clustering", "mcm_diagram")
