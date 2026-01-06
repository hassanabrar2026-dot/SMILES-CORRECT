import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the confusion-matrix-style CSV
    df = pd.read_csv("confusion_matrix.csv")

    # Sort bins in correct order (0.0–0.1, 0.1–0.2, ...)
    def sort_key(x):
        try:
            return float(x.split("–")[0])
        except:
            return 999  # "Invalid" goes last

    df = df.sort_values(by="similarity_bin", key=lambda col: col.map(sort_key))

    # Plot
    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_theme(style="whitegrid", font_scale=1.4)

    sns.barplot(
        data=df,
        x="similarity_bin",
        y="count",
        color="#4C72B0"
    )

    plt.xlabel("Tanimoto Similarity Bin", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.title("Confusion-Matrix Style Similarity Distribution", fontsize=18, fontweight="bold")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save at 1200 dpi
    plt.savefig("confusion_matrix_1200dpi.png", dpi=1200, bbox_inches="tight")

    print("Saved confusion_matrix_1200dpi.png")

if __name__ == "__main__":
    main()
