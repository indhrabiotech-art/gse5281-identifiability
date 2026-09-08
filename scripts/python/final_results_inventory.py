import os

ROOTS = [
    "03_Preprocessing",
    "04_ML",
    "05_Biology",
    "07_Figures"
]

print("=" * 80)
print("FINAL PROJECT RESULTS INVENTORY")
print("=" * 80)

for root in ROOTS:

    print("\n" + "=" * 80)
    print(root)
    print("=" * 80)

    if not os.path.exists(root):
        print("MISSING DIRECTORY")
        continue

    for dirpath, dirnames, filenames in os.walk(root):

        # Skip unnecessary cache directories
        dirnames[:] = [
            d for d in dirnames
            if d not in ["__pycache__", ".Rproj.user"]
        ]

        for filename in sorted(filenames):

            path = os.path.join(dirpath, filename)

            try:
                size = os.path.getsize(path)
            except OSError:
                size = -1

            print(f"{path}\t{size:,} bytes")

print("\n" + "=" * 80)
print("KEY FINAL OUTPUTS")
print("=" * 80)

key_outputs = [
    "04_ML/External_Validation/GSE5281_3gene_bootstrap_auc.csv",
    "04_ML/External_Validation/GSE5281_3gene_permutation_test.csv",
    "04_ML/External_Validation/GSE5281_harmonization_method_audit.csv",
    "04_ML/External_Validation/GSE5281_3gene_raw_vs_harmonized_robustness.csv",

    "05_Biology/Enrichment/GO_Biological_Process_3gene.csv",
    "05_Biology/Enrichment/GO_Molecular_Function_3gene.csv",
    "05_Biology/Enrichment/KEGG_3gene.csv",
    "05_Biology/Enrichment/Reactome_3gene.csv",

    "05_Biology/PPI_FirstShell/STRING_first_shell_raw.csv",
    "05_Biology/PPI_FirstShell/signature_gene_network_degree.csv",

    "05_Biology/Final_Summary/final_3gene_biological_evidence_table.csv",
    "05_Biology/Final_Summary/manuscript_3gene_biology_table.csv",
    "05_Biology/Final_Summary/final_3gene_biology_summary.txt",

    "07_Figures/Final_3gene/Figure_ROC_3gene.png",
    "07_Figures/Final_3gene/Figure_PR_3gene.png",
    "07_Figures/Final_3gene/Figure_Heatmap_3gene_GSE48350.png"
]

print()

for path in key_outputs:

    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"OK       {path}\t{size:,} bytes")
    else:
        print(f"MISSING  {path}")

print("\n" + "=" * 80)
print("INVENTORY COMPLETE")
print("=" * 80)
