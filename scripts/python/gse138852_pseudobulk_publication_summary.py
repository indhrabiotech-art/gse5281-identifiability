import os
import pandas as pd
import numpy as np

BASE = os.path.expanduser(
    "~/project_ml/11_SingleCell/GSE138852"
)

RESULTS = os.path.join(BASE, "results")

INPUT = os.path.join(
    RESULTS,
    "GSE138852_ABCA6_CRLF1_pseudobulk_statistics.csv"
)

OUTPUT = os.path.join(
    RESULTS,
    "GSE138852_ABCA6_CRLF1_publication_effects.csv"
)

df = pd.read_csv(INPUT)

# Remove mathematically unstable fold-change estimates
# where the control mean is zero.
df["Fold_change_interpretable"] = np.where(
    df["Control_mean"] > 0,
    df["AD_mean"] / df["Control_mean"],
    np.nan
)

df["log2_FC_interpretable"] = np.where(
    df["Control_mean"] > 0,
    np.log2(
        df["AD_mean"] / df["Control_mean"]
    ),
    np.nan
)

# Effect-size interpretation
def interpret_d(d):

    if pd.isna(d):
        return "Not estimable"

    a = abs(d)

    if a < 0.2:
        return "Negligible"

    if a < 0.5:
        return "Small"

    if a < 0.8:
        return "Moderate"

    return "Large"


df["Effect_size_interpretation"] = (
    df["Cohens_d"]
    .apply(interpret_d)
)

# Flag statistically interpretable results
df["Statistical_status"] = np.where(
    df["Exact_permutation_p"] < 0.05,
    "Nominally significant",
    "Not significant"
)

# Strongest biologically interpretable effects
df["Absolute_effect_size"] = (
    df["Cohens_d"].abs()
)

df = df.sort_values(
    "Absolute_effect_size",
    ascending=False
)

# Save
df.to_csv(
    OUTPUT,
    index=False
)

print("=" * 80)
print("GSE138852 — PUBLICATION-SAFE EFFECT SIZE SUMMARY")
print("=" * 80)

cols = [
    "Gene",
    "Cell_type",
    "AD_mean",
    "Control_mean",
    "Fold_change_interpretable",
    "log2_FC_interpretable",
    "Cohens_d",
    "Effect_size_interpretation",
    "Exact_permutation_p",
    "FDR_BH",
    "Statistical_status"
]

print(
    df[cols].to_string(index=False)
)

print("\nSaved:")
print(OUTPUT)

print("\n" + "=" * 80)
print("PUBLICATION EFFECT SUMMARY COMPLETE")
print("=" * 80)
