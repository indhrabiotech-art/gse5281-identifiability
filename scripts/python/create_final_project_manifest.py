import os
from pathlib import Path

ROOT = Path(".")

print("=" * 80)
print("FINAL PROJECT MANIFEST")
print("=" * 80)

directories = [
    "01_Raw_Data",
    "02_Metadata",
    "03_Preprocessing",
    "04_ML",
    "05_Biology",
    "06_Manuscript",
    "07_Figures",
    "Python_scripts",
    "R_scripts"
]

print("\nDIRECTORY STATUS")
print("-" * 80)

for d in directories:
    p = ROOT / d
    if p.exists():
        files = sum(1 for x in p.rglob("*") if x.is_file())
        print(f"OK       {d:25s} {files} files")
    else:
        print(f"MISSING  {d}")

print("\nKEY FINAL OUTPUTS")
print("-" * 80)

files = [
    "04_ML/Final_Characterization/3gene_model_performance_summary.csv",
    "04_ML/Final_Characterization/3gene_gene_wise_statistics.csv",
    "04_ML/Final_Characterization/3gene_final_coefficients.csv",

    "04_ML/External_Validation/GSE5281_3gene_bootstrap_auc.csv",
    "04_ML/External_Validation/GSE5281_3gene_permutation_test.csv",
    "04_ML/External_Validation/GSE5281_harmonization_method_audit.csv",
    "04_ML/External_Validation/GSE5281_3gene_raw_vs_harmonized_robustness.csv",

    "05_Biology/Final_Summary/final_3gene_biological_evidence_table.csv",
    "05_Biology/Final_Summary/manuscript_3gene_biology_table.csv",

    "06_Manuscript/Abstract_3gene_signature.txt",
    "06_Manuscript/Methods/Methods_3gene_signature.txt",
    "06_Manuscript/Results/Results_3gene_signature.txt",
    "06_Manuscript/Results/Figure_Legends_3gene_signature.txt",
    "06_Manuscript/Discussion/Discussion_3gene_signature.txt",

    "06_Manuscript/Figure_ROC_3gene.png",
    "06_Manuscript/Figure_PR_3gene.png",
    "06_Manuscript/Figure_Heatmap_3gene_GSE48350.png"
]

for f in files:
    status = "OK" if Path(f).exists() else "MISSING"
    print(f"{status:8s} {f}")

print("\n" + "=" * 80)
print("MANIFEST COMPLETE")
print("=" * 80)
