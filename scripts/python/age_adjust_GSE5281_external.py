import numpy as np
import pandas as pd

print("=" * 60)
print("GSE5281 — TRAINING-DERIVED AGE ADJUSTMENT")
print("=" * 60)

# ------------------------------------------------------------
# Load GSE48350 training-derived age coefficients
# ------------------------------------------------------------

age_stats = pd.read_csv(
    "04_ML/GSE48350_RMA_training_age_adjustment_stats.csv"
)

age_stats = age_stats.set_index("gene")

# ------------------------------------------------------------
# Load GSE5281 gene-level expression
# ------------------------------------------------------------

external = pd.read_csv(
    "03_Preprocessing/GSE5281_RMA_genelevel.csv",
    index_col=0
)

# ------------------------------------------------------------
# Load GSE5281 metadata
# ------------------------------------------------------------

meta = pd.read_csv(
    "02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv",
)

# ------------------------------------------------------------
# Extract GSM IDs
# ------------------------------------------------------------

gsm_external = [
    x.split(".")[0]
    for x in external.columns
]

external.columns = gsm_external

# ------------------------------------------------------------
# Align metadata
# ------------------------------------------------------------

meta = meta.set_index("GSM")

missing_meta = set(external.columns) - set(meta.index)

if missing_meta:
    raise ValueError(
        "Missing metadata for: " +
        ", ".join(sorted(missing_meta))
    )

meta = meta.loc[external.columns]

# ------------------------------------------------------------
# Check age
# ------------------------------------------------------------

external_age = pd.to_numeric(
    meta["Age_used_years"]
)

if external_age.isna().any():
    raise ValueError("Missing external ages detected.")

# ------------------------------------------------------------
# Check required genes
# ------------------------------------------------------------

genes = age_stats.index.tolist()

missing_genes = [
    g for g in genes
    if g not in external.index
]

if missing_genes:
    raise ValueError(
        "Genes missing from GSE5281: " +
        ", ".join(missing_genes[:20])
    )

# ------------------------------------------------------------
# Training mean age
# ------------------------------------------------------------

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

train_age = pd.to_numeric(
    train_meta["Age"]
)

train_mean_age = train_age.mean()

print("GSE48350 training mean age:",
      round(train_mean_age, 4))

print("GSE5281 samples:",
      external.shape[1])

print("Genes available:",
      len(genes))

# ------------------------------------------------------------
# Age adjustment
#
# IMPORTANT:
# beta values come ONLY from GSE48350 training data.
#
# external_adjusted =
# expression - beta * (external_age - training_mean_age)
# ------------------------------------------------------------

external_adj = external.copy()

for gene in genes:

    beta = age_stats.loc[
        gene,
        "age_coefficient"
    ]

    external_adj.loc[gene, :] = (
        external.loc[gene, :]
        - beta *
        (external_age.values - train_mean_age)
    )

# ------------------------------------------------------------
# Save adjusted external matrix
# ------------------------------------------------------------

output = (
    "04_ML/"
    "GSE5281_RMA_genelevel_age_adjusted.csv"
)

external_adj.to_csv(output)

# ------------------------------------------------------------
# QC
# ------------------------------------------------------------

print("\n============================================")
print("EXTERNAL AGE-ADJUSTMENT QC")
print("============================================")

print("Adjusted matrix:",
      external_adj.shape)

print("Missing values:",
      int(external_adj.isna().sum().sum()))

print("External diagnosis:")
print(meta["diagnosis"].value_counts())

print("\nSaved:")
print(output)

print("\n============================================")
print("GSE5281 AGE ADJUSTMENT COMPLETE")
print("============================================")
