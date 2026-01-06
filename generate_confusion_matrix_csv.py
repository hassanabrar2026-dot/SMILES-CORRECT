import pandas as pd
import numpy as np


def bin_similarity(x):
    """Convert Tanimoto similarity into bins like 0.0–0.1, 0.1–0.2, etc."""
    if pd.isna(x):
        return "Invalid"
    lower = np.floor(x * 10) / 10
    upper = lower + 0.1
    return f"{lower:.1f}–{upper:.1f}"


def main():
    print("Loading distribution_data.csv ...")

    # Load the CSV created earlier
    dist = pd.read_csv("distribution_data.csv")

    # Create similarity bins
    dist["similarity_bin"] = dist["tanimoto"].apply(bin_similarity)

    # Count how many predictions fall into each bin
    confusion_like = (
        dist["similarity_bin"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    confusion_like.columns = ["similarity_bin", "count"]

    # Save to CSV
    confusion_like.to_csv("confusion_matrix.csv", index=False)

    print("Saved confusion_matrix.csv")


if __name__ == "__main__":
    main()
