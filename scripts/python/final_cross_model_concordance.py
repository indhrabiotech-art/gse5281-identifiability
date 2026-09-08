import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau


OUTDIR = "04_ML/Final_Model/Validation"
MANUSCRIPT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

print("=" * 80)
print("FINAL CROSS-MODEL THREE-GENE CONCORDANCE")
print("=" * 80)


# ============================================================
# 1. LOGISTIC MODEL
# ============================================================

logistic = pd.read_csv(
    "04_ML/Final_Characterization/3gene_final_coefficients.csv"
)

logistic = logistic[
    ["gene", "coefficient", "abs_coefficient"]
].copy()

logistic.columns = [
    "Gene",
    "Logistic_Coefficient",
    "Logistic_Abs"
]


# ============================================================
# 2. RANDOM FOREST
# ============================================================

rf = pd.read_csv(
    "04_ML/Final_Model/Validation/"
    "FINAL_THREE_GENE_RANDOM_FOREST_IMPORTANCE.csv"
)

rf = rf[
    ["Gene", "RF_Gini_Importance"]
].copy()


# ============================================================
# 3. XGBOOST
# ============================================================

xgb = pd.read_csv(
    "04_ML/Final_Model/Validation/"
    "FINAL_THREE_GENE_XGBOOST_IMPORTANCE.csv"
)

xgb = xgb[
    ["Gene", "XGB_Gain_Importance_Normalized"]
].copy()


# ============================================================
# 4. TRAINING EFFECT SIZE
# ============================================================

effects = pd.read_csv(
    "04_ML/Final_Model/Validation/"
    "three_gene_cross_dataset_concordance.csv"
)

training = effects[
    effects["Dataset"] == "GSE48350_Training"
][
    ["Gene", "Cohens_d"]
].copy()

training.columns = [
    "Gene",
    "Training_Cohens_d"
]


# ============================================================
# 5. EXTERNAL HIPPOCAMPUS EFFECT
# ============================================================

external = effects[
    effects["Dataset"] == "GSE5281_External"
][
    ["Gene", "Cohens_d"]
].copy()

external.columns = [
    "Gene",
    "External_Cohens_d"
]


# ============================================================
# 6. BLOOD EFFECT
# ============================================================

blood = effects[
    effects["Dataset"] == "GSE63060_Blood"
][
    ["Gene", "Cohens_d"]
].copy()

blood.columns = [
    "Gene",
    "Blood_Cohens_d"
]


# ============================================================
# 7. LASSO BOOTSTRAP STABILITY
# ============================================================

lasso = pd.read_csv(
    "04_ML/LASSO/"
    "GSE48350_three_gene_bootstrap_stability.csv"
)

lasso = lasso[
    ["gene", "selection_frequency"]
].copy()

lasso.columns = [
    "Gene",
    "LASSO_Selection_Frequency"
]


# ============================================================
# MERGE
# ============================================================

df = pd.DataFrame({"Gene": GENES})

for table in [
    logistic,
    rf,
    xgb,
    training,
    external,
    blood,
    lasso
]:
    df = df.merge(
        table,
        on="Gene",
        how="left"
    )


# ============================================================
# RANKS
# ============================================================

df["Logistic_Rank"] = (
    df["Logistic_Abs"]
    .rank(
        ascending=False,
        method="min"
    )
)

df["RF_Rank"] = (
    df["RF_Gini_Importance"]
    .rank(
        ascending=False,
        method="min"
    )
)

df["XGB_Rank"] = (
    df["XGB_Gain_Importance_Normalized"]
    .rank(
        ascending=False,
        method="min"
    )
)

df["Training_Effect_Rank"] = (
    df["Training_Cohens_d"]
    .abs()
    .rank(
        ascending=False,
        method="min"
    )
)

df["External_Effect_Rank"] = (
    df["External_Cohens_d"]
    .abs()
    .rank(
        ascending=False,
        method="min"
    )
)

df["Blood_Effect_Rank"] = (
    df["Blood_Cohens_d"]
    .abs()
    .rank(
        ascending=False,
        method="min"
    )
)

df["LASSO_Stability_Rank"] = (
    df["LASSO_Selection_Frequency"]
    .rank(
        ascending=False,
        method="min"
    )
)


# ============================================================
# MODEL RANK CONSENSUS
# ============================================================

rank_columns = [
    "Logistic_Rank",
    "RF_Rank",
    "XGB_Rank",
    "Training_Effect_Rank"
]

df["Core_Rank_Mean"] = df[
    rank_columns
].mean(axis=1)

df["Core_Rank_SD"] = df[
    rank_columns
].std(axis=1)

df["Core_Rank"] = (
    df["Core_Rank_Mean"]
    .rank(
        ascending=True,
        method="min"
    )
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 80)
print("GENE-LEVEL CROSS-MODEL CONCORDANCE")
print("=" * 80)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# CORRELATIONS
# ============================================================

comparisons = [
    (
        "Logistic_vs_RF",
        "Logistic_Abs",
        "RF_Gini_Importance"
    ),
    (
        "Logistic_vs_XGBoost",
        "Logistic_Abs",
        "XGB_Gain_Importance_Normalized"
    ),
    (
        "RF_vs_XGBoost",
        "RF_Gini_Importance",
        "XGB_Gain_Importance_Normalized"
    ),
    (
        "Training_Effect_vs_Logistic",
        "Training_Cohens_d",
        "Logistic_Abs"
    ),
    (
        "Training_Effect_vs_RF",
        "Training_Cohens_d",
        "RF_Gini_Importance"
    ),
    (
        "Training_Effect_vs_XGBoost",
        "Training_Cohens_d",
        "XGB_Gain_Importance_Normalized"
    )
]

corr_rows = []

for name, xcol, ycol in comparisons:

    x = df[xcol].values
    y = df[ycol].values

    rho, p = spearmanr(x, y)
    tau, kp = kendalltau(x, y)

    corr_rows.append({
        "Comparison": name,
        "Spearman_Rho": rho,
        "Spearman_P": p,
        "Kendall_Tau": tau,
        "Kendall_P": kp,
        "N_genes": len(df)
    })


correlations = pd.DataFrame(corr_rows)


print("\n" + "=" * 80)
print("CROSS-MODEL RANK CORRELATIONS")
print("=" * 80)

print(
    correlations.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

out1 = (
    OUTDIR +
    "/FINAL_CROSS_MODEL_CONCORDANCE.csv"
)

out2 = (
    OUTDIR +
    "/FINAL_CROSS_MODEL_CORRELATIONS.csv"
)

out3 = (
    MANUSCRIPT +
    "/Table_final_cross_model_concordance.csv"
)

df.to_csv(
    out1,
    index=False
)

correlations.to_csv(
    out2,
    index=False
)

df.to_csv(
    out3,
    index=False
)


print("\n" + "=" * 80)
print("CROSS-MODEL CONCORDANCE COMPLETE")
print("=" * 80)

print("\nSaved:")
print(out1)
print(out2)
print(out3)

