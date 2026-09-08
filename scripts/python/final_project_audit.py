import os
import pandas as pd
import numpy as np

print("=" * 75)
print("FINAL ML PROJECT AUDIT — 3-GENE ALZHEIMER'S SIGNATURE")
print("=" * 75)

# ------------------------------------------------------------
# Expected key files
# ------------------------------------------------------------

files = [
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
    "04_ML/GSE48350_RMA_training_age_adjustment_stats.csv",

    "04_ML/Final_Model/GSE48350_final_7gene_coefficients.csv",
    "04_ML/Final_Model/GSE48350_final_7gene_training_predictions.csv",

    "04_ML/External_Validation/GSE5281_7gene_predictions.csv",
    "04_ML/External_Validation/GSE5281_7gene_validation_metrics.csv",
    "04_ML/External_Validation/GSE5281_7gene_direction_diagnostic.csv",

    "04_ML/GSE5281_RMA_genelevel_harmonized.csv",

    "04_ML/External_Validation/GSE5281_3gene_bootstrap_auc.csv",
    "04_ML/External_Validation/GSE5281_3gene_permutation_test.csv",

    "04_ML/Final_Characterization",
    "07_Figures/Final_3gene/Figure_ROC_3gene.png",
    "07_Figures/Final_3gene/Figure_PR_3gene.png",
    "07_Figures/Final_3gene/Figure_Heatmap_3gene_GSE48350.png"
]

print("\nFILE QC")
print("-" * 75)

missing = []

for f in files:
    exists = os.path.exists(f)

    status = "OK" if exists else "MISSING"

    print(f"{status:8s} {f}")

    if not exists:
        missing.append(f)

# ------------------------------------------------------------
# 3-gene characterization
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("FINAL 3-GENE PERFORMANCE")
print("=" * 75)

char_dir = "04_ML/Final_Characterization"

csv_files = []

if os.path.isdir(char_dir):
    csv_files = [
        f for f in os.listdir(char_dir)
        if f.endswith(".csv")
    ]

for f in sorted(csv_files):

    path = os.path.join(char_dir, f)

    try:
        df = pd.read_csv(path)

        print(f"\n{f}")
        print("Shape:", df.shape)

    except Exception as e:
        print("Could not read:", f)
        print(e)

# ------------------------------------------------------------
# Candidate genes
# ------------------------------------------------------------

genes = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

print("\n" + "=" * 75)
print("FINAL CANDIDATE GENES")
print("=" * 75)

for g in genes:
    print("✓", g)

# ------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------

bootstrap_file = (
    "04_ML/External_Validation/"
    "GSE5281_3gene_bootstrap_auc.csv"
)

if os.path.exists(bootstrap_file):

    df = pd.read_csv(bootstrap_file)

    print("\n" + "=" * 75)
    print("BOOTSTRAP CHECK")
    print("=" * 75)

    print(df.to_string(index=False))

# ------------------------------------------------------------
# Permutation
# ------------------------------------------------------------

perm_file = (
    "04_ML/External_Validation/"
    "GSE5281_3gene_permutation_test.csv"
)

if os.path.exists(perm_file):

    df = pd.read_csv(perm_file)

    print("\n" + "=" * 75)
    print("PERMUTATION CHECK")
    print("=" * 75)

    print(df.to_string(index=False))

# ------------------------------------------------------------
# Final interpretation
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("FINAL PROJECT STATUS")
print("=" * 75)

if missing:
    print("\nWARNING:")
    print("Some expected files are missing.")

    for f in missing:
        print("  -", f)

else:
    print("\n✓ All major analysis outputs are present.")

print("""
Current interpretation:

1. A 3-gene signature was derived:
   ABCA6, CRLF1, TNFRSF11B

2. The 3-gene model performs better externally than
   the original 7-gene model.

3. External validation is encouraging but the external
   sample size is small (n=23).

4. Bootstrap uncertainty is substantial.

5. Permutation testing does not establish statistically
   significant discrimination at the conventional 0.05 level.

Therefore the signature should currently be described as:

"an externally evaluated candidate gene signature"

rather than as a clinically validated diagnostic biomarker.

Next stage:
biological interpretation + manuscript-ready tables/results.
""")

print("=" * 75)
print("FINAL AUDIT COMPLETE")
print("=" * 75)
