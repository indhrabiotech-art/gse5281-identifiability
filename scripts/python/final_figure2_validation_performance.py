import os
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "06_Manuscript/Figures"
os.makedirs(OUTDIR, exist_ok=True)

INPUT = (
    "04_ML/Final_Model/Validation/"
    "FINAL_MODEL_PERFORMANCE.csv"
)

df = pd.read_csv(INPUT)

# Keep the final three-gene model
df = df[
    df["Signature"].astype(str).str.contains(
        "ABCA6.*CRLF1.*TNFRSF11B",
        regex=True,
        na=False
    )
].copy()

# If signature filtering does not match, use the 3-gene rows
if len(df) == 0:
    df = pd.read_csv(INPUT)
    df = df[df["N_genes"] == 3].copy()

# ------------------------------------------------------------
# Standardize cohort labels
# ------------------------------------------------------------

def cohort_label(x):
    x = str(x)

    if "Training" in x:
        return "Discovery\nGSE48350"
    elif "HeldOut" in x:
        return "Held-out\nGSE48350"
    elif "5281" in x:
        return "External\nGSE5281"
    return x

df["Cohort"] = df["Dataset"].apply(cohort_label)

df = df.drop_duplicates("Cohort")

# ------------------------------------------------------------
# Print data used
# ------------------------------------------------------------

print("=" * 80)
print("FIGURE 2 — FINAL VALIDATION PERFORMANCE")
print("=" * 80)

print(
    df[
        ["Cohort", "ROC_AUC", "PR_AUC"]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# Plot ROC-AUC
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

x = range(len(df))

ax.plot(
    list(x),
    df["ROC_AUC"],
    marker="o",
    linewidth=2,
    label="ROC-AUC"
)

ax.plot(
    list(x),
    df["PR_AUC"],
    marker="s",
    linewidth=2,
    label="PR-AUC"
)

ax.set_xticks(list(x))
ax.set_xticklabels(
    df["Cohort"],
    fontsize=10
)

ax.set_ylim(0, 1.0)

ax.set_ylabel(
    "Performance",
    fontsize=11
)

ax.set_title(
    "Three-Gene Signature Performance Across Cohorts",
    fontsize=13,
    fontweight="bold"
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend(
    frameon=False
)

# Add numerical values

for i, (_, row) in enumerate(df.iterrows()):

    ax.text(
        i,
        row["ROC_AUC"] + 0.035,
        f"{row['ROC_AUC']:.3f}",
        ha="center",
        fontsize=9
    )

    ax.text(
        i,
        row["PR_AUC"] - 0.065,
        f"{row['PR_AUC']:.3f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

png = (
    f"{OUTDIR}/"
    "Figure_2_ThreeGene_Validation_Performance.png"
)

pdf = (
    f"{OUTDIR}/"
    "Figure_2_ThreeGene_Validation_Performance.pdf"
)

plt.savefig(
    png,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    pdf,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(png)
print(pdf)

print("\nFIGURE 2 COMPLETE")

