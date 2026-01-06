import pandas as pd
import numpy as np

# Load previously generated CSVs
scatter = pd.read_csv("scatter_data.csv")
dist = pd.read_csv("distribution_data.csv")

# -----------------------------
# CONFUSION-MATRIX STYLE BINNING
# -----------------------------
def bin_similarity(x):
    if pd.isna(x):
        return "Invalid"
    return f"{np.floor(x*10)/10:.1f}–{np.floor(x*10)/10 + 0.1:.1f}"

dist["similarity_bin"] = dist["tanimoto"].apply(bin_similarity)
confusion_like = dist["similarity_bin"].value_counts().sort_index().reset_index()
confusion_like.columns = ["similarity_bin", "count"]

# -----------------------------
# ERROR TYPE SUMMARY
# -----------------------------
def classify_error(d):
    if d == 0:
        return "Perfect match (0 edits)"
    elif d <= 5:
        return "Small error (1–5 edits)"
    elif d <= 15:
        return "Medium error (6–15 edits)"
    else:
        return "Large error (>15 edits)"

scatter["error_type"] = scatter["levenshtein"].apply(classify_error)
error_summary = scatter["error_type"].value_counts().reset_index()
error_summary.columns = ["error_type", "count"]

# -----------------------------
# BEST & WORST PREDICTIONS
# -----------------------------
valid_dist = dist.dropna(subset=["tanimoto"])

best20 = valid_dist.sort_values("tanimoto", ascending=False).head(20)
worst20 = valid_dist.sort_values("tanimoto", ascending=True).head(20)

# -----------------------------
# LENGTH STATISTICS
# -----------------------------
length_stats = dist["pred_length"].describe().to_frame().reset_index()
length_stats.columns = ["statistic", "value"]

# -----------------------------
# VALIDITY SUMMARY
# -----------------------------
validity_summary = dist["valid"].value_counts().reset_index()
validity_summary.columns = ["valid", "count"]

# -----------------------------
# SAVE EVERYTHING TO EXCEL
# -----------------------------
with pd.ExcelWriter("analysis_results.xlsx") as writer:
    confusion_like.to_excel(writer, sheet_name="Similarity Bins", index=False)
    error_summary.to_excel(writer, sheet_name="Error Types", index=False)
    best20.to_excel(writer, sheet_name="Best Predictions", index=False)
    worst20.to_excel(writer, sheet_name="Worst Predictions", index=False)
    length_stats.to_excel(writer, sheet_name="Length Stats", index=False)
    validity_summary.to_excel(writer, sheet_name="Validity Summary", index=False)

print("Saved analysis_results.xlsx")
