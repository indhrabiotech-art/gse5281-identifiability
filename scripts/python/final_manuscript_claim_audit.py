from pathlib import Path
import pandas as pd
import re

ROOT = Path("06_Manuscript")

print("=" * 85)
print("FINAL MANUSCRIPT-WIDE SCIENTIFIC CLAIM AUDIT")
print("=" * 85)

# ============================================================
# 1. LOAD CANONICAL NUMERICAL RESULTS
# ============================================================

perf = pd.read_csv(
    "04_ML/Final_Characterization/3gene_model_performance_summary.csv"
)

boot = pd.read_csv(
    "04_ML/External_Validation/GSE5281_3gene_bootstrap_auc.csv"
)

perm = pd.read_csv(
    "04_ML/External_Validation/GSE5281_3gene_permutation_test.csv"
)

robust = pd.read_csv(
    "04_ML/External_Validation/GSE5281_3gene_raw_vs_harmonized_robustness.csv"
)

genes = pd.read_csv(
    "04_ML/Final_Characterization/3gene_gene_wise_statistics.csv"
)

# ============================================================
# 2. CANONICAL VALUES
# ============================================================

train_auc = float(
    perf.loc[
        perf["dataset"] == "GSE48350_train",
        "ROC_AUC"
    ].iloc[0]
)

test_auc = float(
    perf.loc[
        perf["dataset"] == "GSE48350_test",
        "ROC_AUC"
    ].iloc[0]
)

external_auc = float(
    perf.loc[
        perf["dataset"] == "GSE5281_external",
        "ROC_AUC"
    ].iloc[0]
)

external_ap = float(
    perf.loc[
        perf["dataset"] == "GSE5281_external",
        "Average_Precision"
    ].iloc[0]
)

bootstrap_low = float(boot["ci_lower_95"].iloc[0])
bootstrap_high = float(boot["ci_upper_95"].iloc[0])

perm_p = float(
    perm["empirical_p_value"].iloc[0]
)

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

auc_difference = float(
    robust.loc[
        robust["analysis"] == "3-gene_model",
        "AUC_difference"
    ].iloc[0]
)

print("\nCANONICAL VALUES")
print("-" * 85)
print(f"Training AUC       : {train_auc:.6f}")
print(f"Internal test AUC  : {test_auc:.6f}")
print(f"External AUC       : {external_auc:.6f}")
print(f"External AP        : {external_ap:.6f}")
print(f"Bootstrap 95% CI   : {bootstrap_low:.6f} – {bootstrap_high:.6f}")
print(f"Permutation p      : {perm_p:.6f}")
print(f"Raw external AUC   : {raw_auc:.6f}")
print(f"Harmonized AUC     : {harm_auc:.6f}")
print(f"AUC difference     : {auc_difference:.6f}")

# ============================================================
# 3. LOAD MANUSCRIPT
# ============================================================

manuscript_files = [
    ROOT / "Abstract_3gene_signature.txt",
    ROOT / "Methods" / "Methods_3gene_signature.txt",
    ROOT / "Results" / "Results_3gene_signature.txt",
    ROOT / "Discussion" / "Discussion_3gene_signature.txt",
    ROOT / "Results" / "Figure_Legends_3gene_signature.txt",
]

texts = {}

print("\nMANUSCRIPT FILES")
print("-" * 85)

for path in manuscript_files:
    if path.exists():
        texts[path.name] = path.read_text()
        print(f"FOUND    {path}")
    else:
        print(f"MISSING  {path}")

combined = "\n".join(texts.values())

# ============================================================
# 4. GENE CHECK
# ============================================================

print("\nGENE SIGNATURE CHECK")
print("-" * 85)

for gene in ["ABCA6", "CRLF1", "TNFRSF11B"]:
    if gene in combined:
        print(f"PASS     {gene}")
    else:
        print(f"FAIL     {gene}")

# ============================================================
# 5. NUMERICAL CLAIM CHECK
# ============================================================

print("\nNUMERICAL CLAIM CHECK")
print("-" * 85)

claims = {
    "Training AUC 0.855":
        ("0.855" in combined or "0.8549" in combined),

    "Internal test AUC 0.694":
        ("0.694" in combined),

    "External AUC 0.685":
        ("0.685" in combined or "0.6846" in combined),

    "External AP 0.635":
        ("0.635" in combined),

    "Bootstrap CI 0.442–0.886":
        (
            ("0.442" in combined or "0.4417" in combined)
            and
            ("0.886" in combined or "0.8864" in combined)
        ),

    "Permutation p 0.145":
        ("0.145" in combined or "0.1447" in combined),

    "Raw AUC 0.662":
        ("0.662" in combined or "0.6615" in combined),

    "Harmonized AUC 0.685":
        ("0.685" in combined or "0.6846" in combined),

    "AUC difference 0.023":
        ("0.023" in combined),

    "External n=23":
        ("23 samples" in combined),

    "External AD n=10":
        ("10 AD" in combined),

    "External control n=13":
        ("13 control" in combined),
}

for name, passed in claims.items():
    print(
        f"{'PASS' if passed else 'FAIL':8s} {name}"
    )

# ============================================================
# 6. STATISTICAL CAUTION CHECK
# ============================================================

print("\nSTATISTICAL INTERPRETATION CHECK")
print("-" * 85)

statistical_terms = {
    "small external cohort":
        ("small" in combined.lower() and "external" in combined.lower()),

    "broad bootstrap uncertainty":
        ("bootstrap" in combined.lower() and "uncertainty" in combined.lower()),

    "permutation non-significance":
        (
            "0.145" in combined
            and
            (
                "not statistically" in combined.lower()
                or
                "non-significant" in combined.lower()
                or
                "not establish" in combined.lower()
            )
        ),

    "candidate signature wording":
        (
            "candidate signature" in combined.lower()
            or
            "candidate gene signature" in combined.lower()
        ),

    "clinical caution":
        (
            "clinically validated" in combined.lower()
            and
            "rather than" in combined.lower()
        ),
}

for name, passed in statistical_terms.items():
    print(
        f"{'PASS' if passed else 'REVIEW':8s} {name}"
    )

# ============================================================
# 7. OVERCLAIM CHECK
# ============================================================

print("\nOVERCLAIMING CHECK")
print("-" * 85)

dangerous_terms = [
    "mechanistic",
    "causal",
    "clinically validated",
    "clinical diagnostic",
    "definitive biomarker",
    "therapeutic target",
]

for term in dangerous_terms:
    count = combined.lower().count(term.lower())

    if count:
        print(
            f"REVIEW   '{term}' appears {count} time(s)"
        )
    else:
        print(
            f"PASS     '{term}' not present"
        )

# ============================================================
# 8. METHODS-SPECIFIC CHECK
# ============================================================

print("\nMETHODS CHECK")
print("-" * 85)

methods = texts.get(
    "Methods_3gene_signature.txt",
    ""
)

methods_checks = {
    "RMA":
        "RMA" in methods,

    "GSE48350":
        "GSE48350" in methods,

    "GSE5281":
        "GSE5281" in methods,

    "LASSO":
        "LASSO" in methods,

    "10,000 bootstrap":
        "10,000 bootstrap" in methods,

    "10,000 permutations":
        "10,000 permutations" in methods,

    "empirical quantile mapping":
        "empirical" in methods.lower()
        and
        "quantile" in methods.lower(),

    "external labels not used":
        (
            "not used for feature selection" in methods.lower()
            or
            "not used during feature selection" in methods.lower()
            or
            "not used to train" in methods.lower()
            or
            "no external diagnostic labels were used to calculate the expression transformation" in methods.lower()
        ),

    "GO":
        "Gene Ontology" in methods,

    "KEGG":
        "KEGG" in methods,

    "Reactome":
        "Reactome" in methods,

    "STRING":
        "STRING" in methods,
}

for name, passed in methods_checks.items():
    print(
        f"{'PASS' if passed else 'FAIL':8s} {name}"
    )

# ============================================================
# 9. TERMINAL CONTAMINATION CHECK
# ============================================================

print("\nTERMINAL / SCRIPT CONTAMINATION CHECK")
print("-" * 85)

contamination_patterns = [
    r"bharathwaj@",
    r"\(\.venv_ml\)",
    r"command not found",
    r"Traceback \(most recent call last\)",
    r"Permission denied",
    r"^python ",
    r"^source ",
]

found_contamination = False

for name, text in texts.items():
    for pattern in contamination_patterns:
        if re.search(pattern, text, flags=re.MULTILINE):
            print(
                f"REVIEW   {name}: {pattern}"
            )
            found_contamination = True

if not found_contamination:
    print("PASS     No terminal contamination detected")

# ============================================================
# 10. FINAL STATUS
# ============================================================

print("\n" + "=" * 85)
print("FINAL MANUSCRIPT CLAIM AUDIT COMPLETE")
print("=" * 85)

print("""
Interpretation:

PASS    = internally supported and present
REVIEW  = requires human inspection
FAIL    = missing or inconsistent

Do NOT modify manuscript text based on this audit automatically.
Review every REVIEW/FAIL item first.
""")

