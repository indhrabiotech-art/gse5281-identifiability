import pandas as pd
import numpy as np

print("=" * 60)
print("GSE5281 — EXTERNAL VALIDATION MATRIX")
print("=" * 60)

# ------------------------------------------------------------
# Candidate genes from LASSO stability analysis
# ------------------------------------------------------------

genes = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B",
    "PLA2G7",
    "SORBS1",
    "PMS2P2",
    "TAB2"
]

# ------------------------------------------------------------
# Load GSE5281 gene-level matrix
# ------------------------------------------------------------

external = pd.read_csv(
    "03_Preprocessing/GSE5281_RMA_genelevel.csv",
    index_col=0
)

print("Original GSE5281 matrix:", external.shape)

# ------------------------------------------------------------
# Check candidate genes
# ------------------------------------------------------------

missing_genes = [
    gene for gene in genes
    if gene not in external.index
]

if missing_genes:
    raise ValueError(
        "Genes missing from GSE5281: "
        + ", ".join(missing_genes)
    )

# ------------------------------------------------------------
# Select genes and transpose
# ------------------------------------------------------------

validation = external.loc[genes].T

# Extract GSM IDs
validation.index = validation.index.str.extract(
    r"^(GSM[0-9]+)",
    expand=False
)

# ------------------------------------------------------------
# Load GSE5281 metadata
# ------------------------------------------------------------

meta = pd.read_csv(
    "02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv"
)

# ------------------------------------------------------------
# Verify sample matching
# ------------------------------------------------------------

missing_metadata = sorted(
    set(validation.index) - set(meta["GSM"])
)

missing_expression = sorted(
    set(meta["GSM"]) - set(validation.index)
)

print("\nValidation samples:", len(validation))
print("Metadata samples:", len(meta))

print("\nMissing metadata samples:")
print(missing_metadata)

print("\nMissing expression samples:")
print(missing_expression)

if missing_metadata or missing_expression:
    raise ValueError("GSE5281 sample alignment failed.")

# ------------------------------------------------------------
# Align metadata to expression
# ------------------------------------------------------------

meta = meta.set_index("GSM")
meta = meta.loc[validation.index]

# ------------------------------------------------------------
# Final QC
# ------------------------------------------------------------

print("\nCandidate genes:")
print(validation.columns.tolist())

print("\nValidation matrix:")
print(validation.shape)

print("\nDiagnosis:")
print(meta["diagnosis"].value_counts())

print("\nMissing values:")
print(validation.isna().sum().sum())

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

import os
os.makedirs("04_ML/External_Validation", exist_ok=True)

validation.to_csv(
    "04_ML/External_Validation/GSE5281_LASSO_candidate_matrix.csv"
)

meta.to_csv(
    "04_ML/External_Validation/GSE5281_validation_metadata.csv"
)

print("\nSaved:")
print("04_ML/External_Validation/GSE5281_LASSO_candidate_matrix.csv")
print("04_ML/External_Validation/GSE5281_validation_metadata.csv")

print("\n" + "=" * 60)
print("GSE5281 EXTERNAL MATRIX COMPLETE")
print("=" * 60)
