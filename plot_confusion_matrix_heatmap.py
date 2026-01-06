import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sort_key(x):
    """Sort similarity bins like 0.0–0.1, 0.1–0.2, ..., Invalid last."""
    try:
        return float(x.split("–")[0])
    except:
        return 999


def main():
    # Load the confusion-matrix-style CSV
    df = pd.read_csv("confusion_matrix.csv")

    # Sort bins numerically
    df = df.sort_values(by="similarity_bin", key=lambda col: col.map(sort_key))

    # Convert to heatmap-friendly format (1 row, many columns)
    heatmap_df = df.set_index("similarity_bin").T

    # Plot heatmap
    plt.figure(figsize=(14, 3), dpi=300)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar=True
    )

    plt.title("Tanimoto Similarity Distribution Heatmap", fontsize=16, fontweight="bold")
    plt.xlabel("Similarity Bin")
    plt.ylabel("")

    plt.tight_layout()

    # Save at 1200 dpi
    plt.savefig("confusion_matrix_heatmap_1200dpi.png", dpi=1200, bbox_inches="tight")

    print("Saved confusion_matrix_heatmap_1200dpi.png")


if __name__ == "__main__":
    main()
