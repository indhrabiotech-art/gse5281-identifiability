import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# Files
# ------------------------------------------------------------
EXPR_FILE = "06_Validation/GSE5281_hippocampus_15gene_validation.csv"
META_FILE = "02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv"
BETA_FILE = "06_Validation/GSE48350_age_coefficients.csv"
MEAN_AGE_FILE = "06_Validation/GSE48350_train_mean_age.txt"

OUTPUT = "06_Validation/GSE5281_hippocampus_15gene_ageadjusted.csv"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
expr = pd.read_csv(EXPR_FILE, index_col=0)
meta = pd.read_csv(META_FILE)
beta = pd.read_csv(BETA_FILE)

with open(MEAN_AGE_FILE) as f:
    train_mean_age = float(f.read().strip())

print("GSE5281 expression:", expr.shape)
print("GSE5281 metadata:", meta.shape)
print("GSE48350 beta coefficients:", beta.shape)
print("GSE48350 training mean age:", train_mean_age)

# ------------------------------------------------------------
# Verify sample alignment
# ------------------------------------------------------------
if list(expr.columns) != list(meta["GSM"]):
    raise RuntimeError(
        "GSE5281 expression columns and metadata GSM order do not match."
    )

# ------------------------------------------------------------
# Signature genes
# ------------------------------------------------------------
genes = [
    "SLC25A46",
    "FAM170A",
    "CD5",
    "LINC02987",
    "RAE1",
    "HSPA12A",
    "ANKIB1",
    "BTK",
    "ZNF621",
    "SPDEF",
    "KCNJ5",
    "TRIM49",
    "PART1",
    "MEFV",
    "P3H3"
]

# Check all genes
missing_expr = [g for g in genes if g not in expr.index]
missing_beta = [g for g in genes if g not in beta["gene"].values]

if missing_expr:
    raise RuntimeError(f"Missing genes in GSE5281 expression: {missing_expr}")

if missing_beta:
    raise RuntimeError(f"Missing genes in GSE48350 beta coefficients: {missing_beta}")

# ------------------------------------------------------------
# Extract beta values in EXACT signature order
# ------------------------------------------------------------
beta_lookup = beta.set_index("gene")["beta_age"]

beta_signature = beta_lookup.loc[genes]

# ------------------------------------------------------------
# Extract GSE5281 ages
# ------------------------------------------------------------
ages = meta.set_index("GSM").loc[expr.columns, "Age_used_years"]

if ages.isna().any():
    raise RuntimeError("Missing GSE5281 age values.")

# ------------------------------------------------------------
# Apply FROZEN GSE48350 age correction
#
# X_adj = X - beta_age * (Age_validation - mean_age_training)
# ------------------------------------------------------------
age_centered = ages.values - train_mean_age

adjusted = (
    expr.loc[genes].astype(float)
    - np.outer(beta_signature.values, age_centered)
)

adjusted = pd.DataFrame(
    adjusted,
    index=genes,
    columns=expr.columns
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
os.makedirs("06_Validation", exist_ok=True)

adjusted.to_csv(OUTPUT)

print("\n==============================================")
print("GSE5281 FROZEN AGE-ADJUSTMENT COMPLETE")
print("==============================================")

print("Genes:", adjusted.shape[0])
print("Samples:", adjusted.shape[1])

print("\nAge range:")
print("Minimum:", ages.min())
print("Maximum:", ages.max())

print("\nDiagnosis counts:")
print(meta["diagnosis"].value_counts())

print("\nBeta coefficients used:")
print(beta_signature.to_string())

print("\nOutput:")
print(OUTPUT)
