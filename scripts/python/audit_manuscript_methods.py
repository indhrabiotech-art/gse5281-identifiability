import os
import re

METHODS = "06_Manuscript/Methods/Methods_3gene_signature.txt"

print("=" * 75)
print("MANUSCRIPT METHODS — PIPELINE CONSISTENCY AUDIT")
print("=" * 75)

with open(METHODS, "r") as f:
    text = f.read()

checks = {
    "GSE48350": "GSE48350" in text,
    "GSE5281": "GSE5281" in text,
    "RMA": "RMA" in text,
    "probe-to-gene annotation": "Probe-to-Gene Annotation" in text,
    "21,367 genes": "21,367" in text,
    "49 training samples": "49 samples" in text,
    "13 internal test samples": "13 samples" in text,
    "23 external samples": "23 samples" in text,
    "LASSO": "LASSO" in text,
    "ABCA6": "ABCA6" in text,
    "CRLF1": "CRLF1" in text,
    "TNFRSF11B": "TNFRSF11B" in text,
    "bootstrap": "10,000 bootstrap" in text,
    "permutation": "10,000 permutations" in text,
    "GO": "Gene Ontology" in text,
    "KEGG": "KEGG" in text,
    "Reactome": "Reactome" in text,
    "STRING": "STRING" in text,
}

print("\nMETHODS CONTENT CHECK")
print("-" * 75)

failed = []

for name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"{status:<8} {name}")

    if not passed:
        failed.append(name)

print("\n" + "=" * 75)

if failed:
    print("AUDIT STATUS: REVIEW REQUIRED")
    print("\nMissing/uncertain items:")
    for item in failed:
        print(" -", item)
else:
    print("AUDIT STATUS: BASIC CONSISTENCY CHECK PASSED")

print("=" * 75)

# ------------------------------------------------------------
# Check actual project scripts involved in the workflow
# ------------------------------------------------------------

print("\nKEY PIPELINE SCRIPTS")
print("-" * 75)

script_patterns = [
    "preprocess_raw_CEL.R",
    "annotate_RMA_genelevel.R",
    "qc_filter_split.R",
    "training_variance_filter.R",
    "training_top25_variance_filter.R",
    "age_adjust_training_only.R",
    "lasso_feature_selection.py",
    "final_3gene_characterization.py",
    "validate_3gene_external.py",
    "bootstrap_external_auc.py",
    "permutation_external_auc.py",
    "audit_harmonization_method.py",
    "robustness_raw_vs_harmonized_3gene.py",
    "enrichment_3gene.py",
    "string_first_shell_3gene.py",
]

for script in script_patterns:
    found = False

    for root in ["R_scripts", "Python_scripts"]:
        path = os.path.join(root, script)
        if os.path.exists(path):
            found = True
            break

    print(f"{'FOUND' if found else 'MISSING':<8} {script}")

print("\n" + "=" * 75)
print("METHODS AUDIT COMPLETE")
print("=" * 75)
