import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr


OUTDIR = "04_ML/Final_Model/Validation"
MANUSCRIPT_OUT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT_OUT, exist_ok=True)


GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]


# ============================================================
# LOAD MODEL COEFFICIENTS
# ============================================================

coef_file = (
    "04_ML/Final_Characterization/"
    "3gene_final_coefficients.csv"
)

coef = pd.read_csv(coef_file)

coef = coef[
    ["gene", "coefficient", "abs_coefficient"]
].copy()

coef.columns = [
    "Gene",
    "Model_Coefficient",
    "Absolute_Model_Coefficient"
]

coef["Coefficient_Rank"] = (
    coef["Absolute_Model_Coefficient"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# ============================================================
# LOAD CROSS-DATASET EFFECT SIZES
# ============================================================

effect_file = (
    "04_ML/Final_Model/Validation/"
    "three_gene_cross_dataset_concordance.csv"
)

effects = pd.read_csv(effect_file)

effects = effects[
    effects["Gene"].isin(GENES)
].copy()


# ============================================================
# CREATE EFFECT-SIZE MATRIX
# ============================================================

effect_matrix = (
    effects
    .pivot(
        index="Gene",
        columns="Dataset",
        values="Cohens_d"
    )
    .reset_index()
)

effect_matrix.columns.name = None


# Rename datasets

effect_matrix = effect_matrix.rename(
    columns={
        "GSE48350_Training":
            "Training_Cohens_d",

        "GSE5281_External":
            "External_Hippocampus_Cohens_d",

        "GSE63060_Blood":
            "Blood_Cohens_d"
    }
)


# ============================================================
# MERGE
# ============================================================

result = coef.merge(
    effect_matrix,
    on="Gene",
    how="left"
)


# ============================================================
# RANK EFFECT SIZES
# ============================================================

for col, rank_col in [
    (
        "Training_Cohens_d",
        "Training_Effect_Rank"
    ),
    (
        "External_Hippocampus_Cohens_d",
        "External_Effect_Rank"
    ),
    (
        "Blood_Cohens_d",
        "Blood_Effect_Rank"
    )
]:

    result[rank_col] = (
        result[col]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )


# ============================================================
# SIGN / DIRECTION
# ============================================================

result["Training_Direction"] = np.where(
    result["Training_Cohens_d"] > 0,
    "UP_AD",
    "DOWN_AD"
)

result["External_Direction"] = np.where(
    result["External_Hippocampus_Cohens_d"] > 0,
    "UP_AD",
    "DOWN_AD"
)

result["Blood_Direction"] = np.where(
    result["Blood_Cohens_d"] > 0,
    "UP_AD",
    "DOWN_AD"
)


# ============================================================
# TRANSPORTABILITY SCORE
# ============================================================

# Direction preserved from training -> external hippocampus
result["Brain_Direction_Preserved"] = (
    np.sign(result["Training_Cohens_d"])
    ==
    np.sign(result["External_Hippocampus_Cohens_d"])
)

# Direction preserved from training -> blood
result["Blood_Direction_Preserved"] = (
    np.sign(result["Training_Cohens_d"])
    ==
    np.sign(result["Blood_Cohens_d"])
)

# Overall direction preservation
result["CrossTissue_Direction_Preserved"] = (
    result["Brain_Direction_Preserved"]
    &
    result["Blood_Direction_Preserved"]
)


# Magnitude retention relative to training

result["External_Magnitude_Retention"] = (
    np.abs(
        result["External_Hippocampus_Cohens_d"]
    )
    /
    np.abs(
        result["Training_Cohens_d"]
    )
)

result["Blood_Magnitude_Retention"] = (
    np.abs(
        result["Blood_Cohens_d"]
    )
    /
    np.abs(
        result["Training_Cohens_d"]
    )
)


# ============================================================
# RANK CONCORDANCE
# ============================================================

print("=" * 80)
print("THREE-GENE MODEL–BIOLOGY CONCORDANCE")
print("=" * 80)

print("\nGene-level results:")
print(
    result.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


def correlation(x, y, method):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if method == "spearman":
        r, p = spearmanr(x, y)
    else:
        r, p = pearsonr(x, y)

    return r, p


# ============================================================
# MODEL COEFFICIENT vs EFFECT SIZE
# ============================================================

comparisons = [
    (
        "Training",
        "Model_Coefficient",
        "Training_Cohens_d"
    ),
    (
        "External_Hippocampus",
        "Model_Coefficient",
        "External_Hippocampus_Cohens_d"
    ),
    (
        "Blood",
        "Model_Coefficient",
        "Blood_Cohens_d"
    )
]

corr_rows = []


for name, xcol, ycol in comparisons:

    pearson_r, pearson_p = correlation(
        result[xcol],
        result[ycol],
        "pearson"
    )

    spearman_r, spearman_p = correlation(
        result[xcol],
        result[ycol],
        "spearman"
    )

    corr_rows.append({
        "Comparison": name,
        "Pearson_r": pearson_r,
        "Pearson_p": pearson_p,
        "Spearman_rho": spearman_r,
        "Spearman_p": spearman_p,
        "N_genes": len(result)
    })


corr_df = pd.DataFrame(corr_rows)


print("\n" + "=" * 80)
print("MODEL COEFFICIENT vs EFFECT-SIZE CORRELATION")
print("=" * 80)

print(
    corr_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# RANK CONCORDANCE
# ============================================================

rank_comparisons = [
    (
        "Training",
        "Coefficient_Rank",
        "Training_Effect_Rank"
    ),
    (
        "External_Hippocampus",
        "Coefficient_Rank",
        "External_Effect_Rank"
    ),
    (
        "Blood",
        "Coefficient_Rank",
        "Blood_Effect_Rank"
    )
]

rank_rows = []


for name, xcol, ycol in rank_comparisons:

    rho, p = spearmanr(
        result[xcol],
        result[ycol]
    )

    rank_rows.append({
        "Comparison": name,
        "Spearman_Rho": rho,
        "P_value": p
    })


rank_df = pd.DataFrame(rank_rows)


print("\n" + "=" * 80)
print("MODEL vs BIOLOGICAL RANK CONCORDANCE")
print("=" * 80)

print(
    rank_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# TRANSPORTABILITY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("GENE TRANSPORTABILITY")
print("=" * 80)

for _, row in result.iterrows():

    print(
        f"\n{row['Gene']}"
    )

    print(
        f"  Brain direction preserved: "
        f"{row['Brain_Direction_Preserved']}"
    )

    print(
        f"  Blood direction preserved: "
        f"{row['Blood_Direction_Preserved']}"
    )

    print(
        f"  External magnitude retention: "
        f"{row['External_Magnitude_Retention']:.3f}"
    )

    print(
        f"  Blood magnitude retention: "
        f"{row['Blood_Magnitude_Retention']:.3f}"
    )


# ============================================================
# SAVE
# ============================================================

gene_out = (
    f"{OUTDIR}/"
    "three_gene_model_biology_concordance.csv"
)

corr_out = (
    f"{OUTDIR}/"
    "three_gene_model_effect_correlations.csv"
)

rank_out = (
    f"{OUTDIR}/"
    "three_gene_model_rank_concordance.csv"
)

result.to_csv(
    gene_out,
    index=False
)

corr_df.to_csv(
    corr_out,
    index=False
)

rank_df.to_csv(
    rank_out,
    index=False
)


# Manuscript copies

result.to_csv(
    f"{MANUSCRIPT_OUT}/"
    "Table_three_gene_model_biology_concordance.csv",
    index=False
)

corr_df.to_csv(
    f"{MANUSCRIPT_OUT}/"
    "Table_three_gene_model_effect_correlations.csv",
    index=False
)

rank_df.to_csv(
    f"{MANUSCRIPT_OUT}/"
    "Table_three_gene_model_rank_concordance.csv",
    index=False
)


print("\n" + "=" * 80)
print("MODEL–BIOLOGY CONCORDANCE COMPLETE")
print("=" * 80)

print("\nSaved:")
print(gene_out)
print(corr_out)
print(rank_out)

