import pandas as pd
from pathlib import Path

ROOT = Path.home() / "project_ml"
OUT = ROOT / "04_ML" / "Reviewer_Robustness_V3"
OUT.mkdir(parents=True, exist_ok=True)

rows = []

# ============================================================
# 1. CANONICAL MODEL PERFORMANCE
# ============================================================

perf = pd.read_csv(
    ROOT / "04_ML/Final_Characterization/3gene_model_performance_summary.csv"
)

for _, r in perf.iterrows():
    rows.append({
        "Evidence": "Canonical model performance",
        "Dataset": r["dataset"],
        "Metric": "ROC_AUC",
        "Value": r["ROC_AUC"],
        "CI_low": None,
        "CI_high": None,
        "P_value": None,
        "Interpretation": "Canonical discrimination"
    })

    rows.append({
        "Evidence": "Canonical model performance",
        "Dataset": r["dataset"],
        "Metric": "Average_Precision",
        "Value": r["Average_Precision"],
        "CI_low": None,
        "CI_high": None,
        "P_value": None,
        "Interpretation": "Canonical precision-recall performance"
    })

# ============================================================
# 2. EXTERNAL RAW BOOTSTRAP
# ============================================================

boot = pd.read_csv(
    ROOT / "04_ML/Reviewer_Robustness_V3/GSE5281_bootstrap_metrics.csv"
).iloc[0]

rows.append({
    "Evidence": "External raw bootstrap",
    "Dataset": "GSE5281_external",
    "Metric": "ROC_AUC",
    "Value": boot["AUC"],
    "CI_low": boot["AUC_CI_lower"],
    "CI_high": boot["AUC_CI_upper"],
    "P_value": None,
    "Interpretation": "Bootstrap uncertainty of raw external discrimination"
})

# ============================================================
# 3. PERMUTATION TEST
# ============================================================

perm = pd.read_csv(
    ROOT / "04_ML/External_Validation/GSE5281_3gene_permutation_test.csv"
).iloc[0]

rows.append({
    "Evidence": "External permutation test",
    "Dataset": "GSE5281_external",
    "Metric": "Permutation_p",
    "Value": perm["observed_auc"],
    "CI_low": None,
    "CI_high": None,
    "P_value": perm["empirical_p_value"],
    "Interpretation": "Empirical significance against label permutation"
})

# ============================================================
# 4. RAW VS HARMONIZED
# ============================================================

robust = pd.read_csv(
    ROOT / "04_ML/External_Validation/GSE5281_3gene_raw_vs_harmonized_robustness.csv"
)

r = robust[
    robust["analysis"] == "3-gene_model"
].iloc[0]

rows.append({
    "Evidence": "External harmonization sensitivity",
    "Dataset": "GSE5281_external",
    "Metric": "Raw_AUC",
    "Value": r["raw_AUC"],
    "CI_low": None,
    "CI_high": None,
    "P_value": None,
    "Interpretation": "Raw external result"
})

rows.append({
    "Evidence": "External harmonization sensitivity",
    "Dataset": "GSE5281_external",
    "Metric": "Harmonized_AUC",
    "Value": r["harmonized_AUC"],
    "CI_low": None,
    "CI_high": None,
    "P_value": None,
    "Interpretation": "Sensitivity analysis after harmonization"
})

# ============================================================
# 5. LOCKED PAIRED ABLATION
# ============================================================

paired = pd.read_csv(
    ROOT / "04_ML/Final_Model/Validation/Locked_Ablation/paired_ablation_delta_auc.csv"
)

for _, r in paired.iterrows():
    rows.append({
        "Evidence": "Paired ablation",
        "Dataset": r["Dataset"],
        "Metric": "Delta_AUC",
        "Value": r["Observed_Delta_AUC"],
        "CI_low": r["Delta_AUC_CI_lower"],
        "CI_high": r["Delta_AUC_CI_upper"],
        "P_value": None,
        "Interpretation": r["Comparison"]
    })

# ============================================================
# 6. LASSO STABILITY
# ============================================================

stability = pd.read_csv(
    ROOT / "04_ML/LASSO/GSE48350_three_gene_bootstrap_stability.csv"
)

for _, r in stability.iterrows():
    rows.append({
        "Evidence": "LASSO bootstrap stability",
        "Dataset": "GSE48350_training",
        "Metric": "Selection_frequency",
        "Value": r["selection_frequency"],
        "CI_low": None,
        "CI_high": None,
        "P_value": None,
        "Interpretation":
            f'{r["gene"]}; positive_frequency={r["positive_frequency"]}; '
            f'sign_consistency={r["sign_consistency"]}'
    })

# ============================================================
# SAVE
# ============================================================

matrix = pd.DataFrame(rows)

outfile = OUT / "FINAL_REVIEWER_EVIDENCE_MATRIX.csv"

matrix.to_csv(
    outfile,
    index=False
)

print("=" * 80)
print("FINAL REVIEWER EVIDENCE MATRIX")
print("=" * 80)

print(matrix.to_string(index=False))

print()
print("Saved:")
print(outfile)
