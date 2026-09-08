import os
import pandas as pd

print("=" * 80)
print("FINAL NUMERICAL CONSISTENCY AUDIT")
print("=" * 80)

# ------------------------------------------------------------
# Canonical files
# ------------------------------------------------------------

perf = pd.read_csv(
    "04_ML/Final_Characterization/3gene_model_performance_summary.csv"
)

boot = pd.read_csv(
    "04_ML/Reviewer_Robustness_V3/GSE5281_bootstrap_metrics.csv"
)

perm = pd.read_csv(
    "04_ML/External_Validation/GSE5281_3gene_permutation_test.csv"
)

robust = pd.read_csv(
    "04_ML/External_Validation/GSE5281_3gene_raw_vs_harmonized_robustness.csv"
)

# Canonical sample-level prediction files
train_pred = pd.read_csv(
    "04_ML/Final_Model/GSE48350_training_predictions.csv"
)

test_pred = pd.read_csv(
    "04_ML/Final_Model/GSE48350_test_predictions.csv"
)

external_pred = pd.read_csv(
    "04_ML/Final_Model/GSE5281_external_predictions.csv"
)

# ------------------------------------------------------------
# Extract canonical values
# ------------------------------------------------------------

train = perf[perf["dataset"] == "GSE48350_train"].iloc[0]
test = perf[perf["dataset"] == "GSE48350_test"].iloc[0]
external = robust[
    robust["analysis"] == "3-gene_model"
].iloc[0]

observed_auc = float(boot["AUC"].iloc[0])
ci_low = float(boot["AUC_CI_lower"].iloc[0])
ci_high = float(boot["AUC_CI_upper"].iloc[0])
p_value = float(perm["empirical_p_value"].iloc[0])

raw_auc = float(
    robust.loc[
        robust["analysis"] == "3-gene_model",
        "raw_AUC"
    ].iloc[0]
)

harm_auc = float(
    robust.loc[
        robust["analysis"] == "3-gene_model",
        "harmonized_AUC"
    ].iloc[0]
)

# ------------------------------------------------------------
# Expected manuscript values
# ------------------------------------------------------------

expected = {
    "Training AUC": (float(train["ROC_AUC"]), 0.854902),
    "Internal test AUC": (float(test["ROC_AUC"]), 0.694444),
    "External AUC": (float(external["raw_AUC"]), 0.661538),
    "Bootstrap AUC": (observed_auc, 0.661538),
    "Bootstrap CI lower": (ci_low, 0.416667),
    "Bootstrap CI upper": (ci_high, 0.876923),
    "Harmonized permutation p": (p_value, 0.144686),
    "Raw external AUC": (raw_auc, 0.661538),
    "Harmonized external AUC": (harm_auc, 0.684615),
}

print("\nCANONICAL VALUES")
print("-" * 80)

all_pass = True

for name, (actual, expected_value) in expected.items():

    passed = abs(actual - expected_value) < 1e-5

    status = "PASS" if passed else "FAIL"

    print(
        f"{status:7s} "
        f"{name:30s} "
        f"actual={actual:.6f} "
        f"expected={expected_value:.6f}"
    )

    if not passed:
        all_pass = False

# ------------------------------------------------------------
# Sample sizes
# ------------------------------------------------------------

print("\nSAMPLE SIZE CHECK")
print("-" * 80)

sample_checks = {
    "Training n": (len(train_pred), 49),
    "Internal test n": (len(test_pred), 13),
    "External n": (len(external_pred), 23),
    "External AD": (
        int((external_pred["Diagnosis"] == "AD").sum()),
        10
    ),
    "External Control": (
        int((external_pred["Diagnosis"] == "Control").sum()),
        13
    ),
}

for name, (actual, expected_value) in sample_checks.items():

    passed = actual == expected_value

    print(
        f"{'PASS' if passed else 'FAIL':7s} "
        f"{name:25s} "
        f"actual={actual} expected={expected_value}"
    )

    if not passed:
        all_pass = False

# ------------------------------------------------------------
# Manuscript files
# ------------------------------------------------------------

print("\nMANUSCRIPT PACKAGE CHECK")
print("-" * 80)

required = [
    "06_Manuscript/Abstract_3gene_signature.txt",
    "06_Manuscript/Methods/Methods_3gene_signature.txt",
    "06_Manuscript/Results/Results_3gene_signature.txt",
    "06_Manuscript/Results/Figure_Legends_3gene_signature.txt",
    "06_Manuscript/Discussion/Discussion_3gene_signature.txt",

    "06_Manuscript/Figure_ROC_3gene.png",
    "06_Manuscript/Figure_PR_3gene.png",
    "06_Manuscript/Figure_Heatmap_3gene_GSE48350.png",

    "06_Manuscript/Tables/Table_3gene_model_performance.csv",
    "06_Manuscript/Tables/Table_3gene_gene_statistics.csv",
    "06_Manuscript/Tables/Table_3gene_coefficients.csv",
    "06_Manuscript/Tables/Table_external_bootstrap.csv",
    "06_Manuscript/Tables/Table_external_permutation.csv",
    "06_Manuscript/Tables/Table_harmonization_audit.csv",
    "06_Manuscript/Tables/Table_raw_vs_harmonized.csv",
    "06_Manuscript/Tables/Table_biological_evidence.csv",
]

for path in required:

    exists = os.path.exists(path)

    print(
        f"{'PASS' if exists else 'FAIL':7s} {path}"
    )

    if not exists:
        all_pass = False

# ------------------------------------------------------------
# Final status
# ------------------------------------------------------------

print("\n" + "=" * 80)

if all_pass:
    print("FINAL AUDIT STATUS: PASS")
    print()
    print("The canonical numerical results and manuscript package")
    print("are internally consistent.")
else:
    print("FINAL AUDIT STATUS: REVIEW REQUIRED")
    print()
    print("One or more values/files do not match the canonical outputs.")

print("=" * 80)
