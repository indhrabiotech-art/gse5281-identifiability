import os
import numpy as np
import pandas as pd


# ============================================================
# FINAL EVIDENCE INTEGRATION
# ============================================================

OUTDIR = "04_ML/Final_Model/Validation"
MANUSCRIPT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT, exist_ok=True)


print("=" * 80)
print("FINAL THREE-GENE EVIDENCE INTEGRATION")
print("=" * 80)


# ============================================================
# HELPER
# ============================================================

def read_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"WARNING: missing file -> {path}")
    return None


# ============================================================
# 1. MODEL PERFORMANCE / ABLATION
# ============================================================

# Canonical locked ablation validation
locked_ablation = read_if_exists(
    "04_ML/Final_Model/Validation/Locked_Ablation/"
    "locked_three_gene_ablation_results.csv"
)

locked_bootstrap = read_if_exists(
    "04_ML/Final_Model/Validation/Locked_Ablation/"
    "locked_ablation_bootstrap_results.csv"
)

paired_delta = read_if_exists(
    "04_ML/Final_Model/Validation/Locked_Ablation/"
    "paired_ablation_delta_auc.csv"
)


# ============================================================
# 2. CROSS-DATASET EFFECTS
# ============================================================

concordance = read_if_exists(
    "04_ML/Final_Model/Validation/"
    "three_gene_cross_dataset_concordance.csv"
)

direction_summary = read_if_exists(
    "04_ML/Final_Model/Validation/"
    "three_gene_directional_concordance_summary.csv"
)

meta_analysis = read_if_exists(
    "04_ML/Final_Model/Validation/"
    "three_gene_effect_meta_analysis.csv"
)


# ============================================================
# 3. MODEL-BIOLOGY CONCORDANCE
# ============================================================

biology = read_if_exists(
    "04_ML/Final_Model/Validation/"
    "three_gene_model_biology_concordance.csv"
)


# ============================================================
# 4. BLOOD COHORT ANALYSES
# ============================================================

blood_incremental = read_if_exists(
    "09_CrossTissue/results/"
    "GSE63060_three_gene_incremental_models.csv"
)

blood_lrt = read_if_exists(
    "09_CrossTissue/results/"
    "GSE63060_three_gene_incremental_LRT.csv"
)

abca6_adjusted = read_if_exists(
    "09_CrossTissue/results/"
    "GSE63060_ABCA6_demographic_adjustment.csv"
)

mci_ad = read_if_exists(
    "09_CrossTissue/results/"
    "GSE63060_MCI_vs_AD_gene_decomposition.csv"
)

blood_composite_pairwise = read_if_exists(
    "09_CrossTissue/results/"
    "GSE63060_three_gene_composite_pairwise.csv"
)


# ============================================================
# 5. LASSO STABILITY
# ============================================================

lasso_bootstrap = read_if_exists(
    "04_ML/LASSO/"
    "GSE48350_lasso_bootstrap_selection_stability.csv"
)

lasso_three_gene = read_if_exists(
    "04_ML/LASSO/"
    "GSE48350_three_gene_bootstrap_stability.csv"
)


# ============================================================
# MASTER PERFORMANCE TABLE
# ============================================================

performance_rows = []

if locked_ablation is not None:

    full = locked_ablation[
        locked_ablation["Gene_set"]
        == "ABCA6+CRLF1+TNFRSF11B"
    ].copy()

    for _, r in full.iterrows():

        performance_rows.append({
            "Dataset": r["Gene_set"],
            "N_genes": 3,
            "Signature": r["Gene_set"],
            "Training_AUC": r["Training_AUC"],
            "Test_AUC": r["Test_AUC"],
            "External_AUC": r["External_AUC"],
            "Training_AP": r["Training_AP"],
            "Test_AP": r["Test_AP"],
            "External_AP": r["External_AP"]
        })

performance = pd.DataFrame(performance_rows)

performance.to_csv(
    OUTDIR +
    "/FINAL_MODEL_PERFORMANCE.csv",
    index=False
)

performance.to_csv(
    MANUSCRIPT +
    "/Table_final_model_performance.csv",
    index=False
)


# ============================================================
# FINAL ROBUSTNESS TABLE
# ============================================================

robustness = pd.DataFrame()

if locked_ablation is not None:

    robustness = locked_ablation.copy()

    robustness = robustness.rename(columns={
        "Gene_set": "Signature",
        "Training_AUC": "Training_ROC_AUC",
        "Test_AUC": "Test_ROC_AUC",
        "External_AUC": "External_ROC_AUC",
        "Training_AP": "Training_PR_AUC",
        "Test_AP": "Test_PR_AUC",
        "External_AP": "External_PR_AUC"
    })

    # Add paired ΔAUC evidence where available
    if paired_delta is not None:
        paired_delta.to_csv(
            OUTDIR +
            "/FINAL_PAIRED_ABLATION_DELTA_AUC.csv",
            index=False
        )

        paired_delta.to_csv(
            MANUSCRIPT +
            "/Table_final_paired_ablation_delta_auc.csv",
            index=False
        )

robustness.to_csv(
    OUTDIR +
    "/FINAL_ROBUSTNESS_TABLE.csv",
    index=False
)

robustness.to_csv(
    MANUSCRIPT +
    "/Table_final_robustness.csv",
    index=False
)


# ============================================================
# BIOLOGICAL EVIDENCE TABLE
# ============================================================

bio_rows = []


if concordance is not None:

    for _, r in concordance.iterrows():

        bio_rows.append({
            "Dataset": r["Dataset"],
            "Gene": r["Gene"],
            "N_AD": r["N_AD"],
            "N_Control": r["N_Control"],
            "Cohens_d": r["Cohens_d"],
            "P_value": r["P_value"],
            "FDR": r["FDR"],
            "Direction": r["Direction"],
            "Reference_Direction":
                r["Reference_Direction"],
            "Direction_Concordant":
                r["Direction_Concordant"]
        })


biological = pd.DataFrame(
    bio_rows
)


biological.to_csv(
    OUTDIR +
    "/FINAL_BIOLOGICAL_EVIDENCE_TABLE.csv",
    index=False
)

biological.to_csv(
    MANUSCRIPT +
    "/Table_final_biological_evidence.csv",
    index=False
)


# ============================================================
# META-ANALYSIS SUMMARY
# ============================================================

if meta_analysis is not None:

    meta_analysis.to_csv(
        OUTDIR +
        "/FINAL_META_ANALYSIS_TABLE.csv",
        index=False
    )

    meta_analysis.to_csv(
        MANUSCRIPT +
        "/Table_final_meta_analysis.csv",
        index=False
    )


# ============================================================
# LASSO STABILITY SUMMARY
# ============================================================

if lasso_bootstrap is not None:

    target_genes = [
        "ABCA6",
        "CRLF1",
        "TNFRSF11B"
    ]

    stability = lasso_bootstrap[
        lasso_bootstrap["gene"].isin(
            target_genes
        )
    ].copy()

    stability = stability[
        [
            "gene",
            "selection_count",
            "selection_frequency",
            "positive_frequency",
            "negative_frequency",
            "sign_consistency",
            "mean_coefficient",
            "sd_coefficient"
        ]
    ]

    stability.to_csv(
        OUTDIR +
        "/FINAL_LASSO_BOOTSTRAP_STABILITY.csv",
        index=False
    )

    stability.to_csv(
        MANUSCRIPT +
        "/Table_final_lasso_bootstrap_stability.csv",
        index=False
    )


# ============================================================
# BLOOD EVIDENCE SUMMARY
# ============================================================

blood_rows = []


# ---- Incremental model ----

if blood_incremental is not None:

    for _, r in blood_incremental.iterrows():

        blood_rows.append({
            "Evidence_Type":
                "Incremental_Model",
            "Model":
                r["Model"],
            "AUC":
                r["AUC"],
            "AIC":
                r["AIC"],
            "McFadden_Pseudo_R2":
                r["McFadden_Pseudo_R2"],
            "Delta_AUC_vs_AgeSex":
                r["Delta_AUC_vs_AgeSex"]
        })


blood_summary = pd.DataFrame(
    blood_rows
)


blood_summary.to_csv(
    OUTDIR +
    "/FINAL_BLOOD_MODEL_EVIDENCE.csv",
    index=False
)

blood_summary.to_csv(
    MANUSCRIPT +
    "/Table_final_blood_model_evidence.csv",
    index=False
)


# ============================================================
# CLAIM AUDIT
# ============================================================

claims = [
    {
        "Claim":
            "Three-gene signature shows strong discrimination "
            "in the discovery cohort.",
        "Evidence":
            "GSE48350 training ROC-AUC approximately 0.855.",
        "Verdict":
            "SUPPORTED"
    },

    {
        "Claim":
            "Three-gene signature generalizes to an "
            "independent hippocampal cohort.",
        "Evidence":
            "GSE5281 external ROC-AUC 0.662.",
        "Verdict":
            "SUPPORTED_WITH_ATTENUATION"
    },

    {
        "Claim":
            "Three genes are highly stable individual "
            "LASSO-selected features.",
        "Evidence":
            "Bootstrap selection frequencies: "
            "TNFRSF11B 15.15%, ABCA6 9.15%, CRLF1 5.65%.",
        "Verdict":
            "NOT_SUPPORTED"
    },

    {
        "Claim":
            "ABCA6 preserves direction across brain and blood.",
        "Evidence":
            "Positive effect in training, external hippocampus "
            "and blood.",
        "Verdict":
            "SUPPORTED"
    },

    {
        "Claim":
            "CRLF1 preserves direction across evaluated cohorts.",
        "Evidence":
            "Positive effects in training, hippocampus and blood.",
        "Verdict":
            "SUPPORTED"
    },

    {
        "Claim":
            "TNFRSF11B is directionally conserved across brain "
            "and blood.",
        "Evidence":
            "Positive in training and hippocampus; negative in blood.",
        "Verdict":
            "NOT_SUPPORTED"
    },

    {
        "Claim":
            "ABCA6 adds information beyond age and sex in "
            "GSE63060.",
        "Evidence":
            "Age+Sex AUC 0.631 versus Age+Sex+ABCA6 AUC 0.682; "
            "likelihood-ratio p approximately 0.00042.",
        "Verdict":
            "SUPPORTED_AS_EXPLORATORY"
    },

    {
        "Claim":
            "The full three-gene model is clearly superior to "
            "the strongest two-gene alternative.",
        "Evidence":
            "External AUC 0.662 for the three-gene model versus "
            "0.731 for ABCA6+TNFRSF11B; paired bootstrap "
            "Delta AUC = -0.069 with 95% CI -0.159 to 0.278 "
            "and P(Delta AUC > 0) = 0.736.",
        "Verdict":
            "NOT_SUPPORTED"
    },

    {
        "Claim":
            "The signature is clinically validated.",
        "Evidence":
            "No prospective clinical validation cohort.",
        "Verdict":
            "NOT_SUPPORTED"
    },

    {
        "Claim":
            "The signature demonstrates cross-tissue "
            "biological transportability.",
        "Evidence":
            "ABCA6 and CRLF1 retain direction; TNFRSF11B "
            "shows tissue-dependent reversal.",
        "Verdict":
            "PARTIALLY_SUPPORTED"
    },

    {
        "Claim":
            "Three genes are individually necessary in every cohort.",
        "Evidence":
            "Ablation shows cohort-dependent performance.",
        "Verdict":
            "NOT_SUPPORTED"
    }
]


claim_audit = pd.DataFrame(
    claims
)


claim_audit.to_csv(
    OUTDIR +
    "/FINAL_CLAIM_AUDIT.csv",
    index=False
)

claim_audit.to_csv(
    MANUSCRIPT +
    "/Table_final_claim_audit.csv",
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("FINAL EVIDENCE INTEGRATION COMPLETE")
print("=" * 80)

print()

print("MASTER PERFORMANCE")
print(performance.to_string(index=False))

print()
print("LASSO BOOTSTRAP STABILITY")

if lasso_bootstrap is not None:
    print(
        stability.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

print()
print("CLAIM AUDIT")
print(
    claim_audit[
        ["Claim", "Verdict"]
    ].to_string(index=False)
)

print()
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

files = [
    "FINAL_MODEL_PERFORMANCE.csv",
    "FINAL_ROBUSTNESS_TABLE.csv",
    "FINAL_BIOLOGICAL_EVIDENCE_TABLE.csv",
    "FINAL_META_ANALYSIS_TABLE.csv",
    "FINAL_LASSO_BOOTSTRAP_STABILITY.csv",
    "FINAL_BLOOD_MODEL_EVIDENCE.csv",
    "FINAL_CLAIM_AUDIT.csv"
]

for f in files:
    print(
        os.path.join(OUTDIR, f)
    )

print()
print("Manuscript copies written to:")
print(MANUSCRIPT)
print("=" * 80)
