import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

OUT = "07_Figures/Final_3gene"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------
# Load training data
# ------------------------------------------------------------

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

# ------------------------------------------------------------
# Extract 3-gene expression
# ------------------------------------------------------------

X = train[GENES].copy()

# Standardize genes for visualization
Xz = pd.DataFrame(
    StandardScaler().fit_transform(X),
    index=X.index,
    columns=GENES
)

# ------------------------------------------------------------
# Order samples by diagnosis
# ------------------------------------------------------------

order = (
    train_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .sort_values()
    .index
)

Xz = Xz.iloc[order]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 5)
)

im = ax.imshow(
    Xz.T,
    aspect="auto",
    interpolation="nearest"
)

ax.set_yticks(range(len(GENES)))
ax.set_yticklabels(GENES)

ax.set_xlabel("GSE48350 samples")
ax.set_ylabel("Gene")

ax.set_title(
    "GSE48350 — 3-Gene Alzheimer’s Disease Signature"
)

plt.colorbar(
    im,
    ax=ax,
    label="Standardized expression (Z-score)"
)

plt.tight_layout()

outfile = os.path.join(
    OUT,
    "Figure_Heatmap_3gene_GSE48350.png"
)

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 70)
print("3-GENE EXPRESSION HEATMAP GENERATED")
print("=" * 70)
print()
print("Output:")
print(outfile)
print()
print("=" * 70)
