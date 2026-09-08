import os
import pandas as pd

OUTDIR = "04_ML/Final_Model/Validation"
MANUSCRIPT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

# ------------------------------------------------------------
# LOAD EXISTING EVIDENCE
# ------------------------------------------------------------

cross_model = pd.read_csv(
    "04_ML/Final_Model/Validation/"
    "FINAL_CROSS_MODEL_CONCORDANCE.csv"
)

biology = pd.read_csv(
    "04_ML/Final_Model/Validation/"
    "three_gene_model_biology_concordance.csv"
)

string = pd.read_csv(
    "10_STRING/results/"
    "FINAL_THREE_GENE_MECHANISTIC_MODULES.csv"
)

# ------------------------------------------------------------
# SELECT CORE MODEL EVIDENCE
# ------------------------------------------------------------

model = cross_model[
    [
        "Gene",
        "Logistic_Coefficient",
        "RF_Gini_Importance",
        "XGB_Gain_Importance_Normalized",
        "LASSO_Selection_Frequency",
        "Core_Rank"
    ]
].copy()

# ------------------------------------------------------------
# SELECT BIOLOGICAL EVIDENCE
# ------------------------------------------------------------

bio = biology[
    [
        "Gene",
        "Training_Cohens_d",
        "External_Hippocampus_Cohens_d",
        "Blood_Cohens_d",
        "Brain_Direction_Preserved",
        "Blood_Direction_Preserved",
        "CrossTissue_Direction_Preserved"
    ]
].copy()

# ------------------------------------------------------------
# STRING MODULE SUMMARY
# ------------------------------------------------------------

string_summary = (
    string
    .groupby("Seed_Gene")
    .agg(
        STRING_Modules=(
            "Mechanistic_Module",
            lambda x: "; ".join(sorted(set(x)))
        ),
        STRING_Best_FDR=(
            "Best_FDR",
            "min"
        ),
        STRING_Evidence_Count=(
            "Evidence_Count",
            "sum"
        )
    )
    .reset_index()
    .rename(columns={"Seed_Gene": "Gene"})
)

# ------------------------------------------------------------
# MERGE
# ------------------------------------------------------------

final = (
    model
    .merge(bio, on="Gene", how="left")
    .merge(string_summary, on="Gene", how="left")
)

# ------------------------------------------------------------
# BIOLOGICAL INTERPRETATION
# ------------------------------------------------------------

interpretation = {
    "ABCA6":
        "Lipid transport/homeostasis axis; strong brain effect and preserved direction in blood.",

    "CRLF1":
        "Cytokine/IL-6-type signaling axis with preserved direction but weaker external effect.",

    "TNFRSF11B":
        "TNFR/RANKL-associated inflammatory axis; strong model importance but blood direction reversal."
}

final["Integrated_Interpretation"] = (
    final["Gene"].map(interpretation)
)

# ------------------------------------------------------------
# PRINT
# ------------------------------------------------------------

print("=" * 90)
print("FINAL GENE-LEVEL EVIDENCE–MECHANISM MATRIX")
print("=" * 90)

print(
    final.to_string(index=False)
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

final.to_csv(
    f"{OUTDIR}/FINAL_GENE_EVIDENCE_MECHANISM_MATRIX.csv",
    index=False
)

final.to_csv(
    f"{MANUSCRIPT}/Table_final_gene_evidence_mechanism_matrix.csv",
    index=False
)

print("\n" + "=" * 90)
print("FINAL EVIDENCE–MECHANISM MATRIX COMPLETE")
print("=" * 90)

print("\nSaved:")
print(
    f"{OUTDIR}/FINAL_GENE_EVIDENCE_MECHANISM_MATRIX.csv"
)

print(
    f"{MANUSCRIPT}/Table_final_gene_evidence_mechanism_matrix.csv"
)

