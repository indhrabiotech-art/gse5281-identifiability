from pathlib import Path
import subprocess
import sys
import re

ROOT = Path.home() / "project_ml"
MANUSCRIPT = ROOT / "06_Manuscript"
TABLES = MANUSCRIPT / "Tables"
ARCHIVE = TABLES / "ARCHIVED_SUPERSEDED"

EXTERNAL_AUC = "0.661538"
HARMONIZED_EXTERNAL_AUC = "0.684615"
TWO_GENE_EXTERNAL = "0.730769"
DELTA_AUC = "0.069231"
DELTA_CI_LOW = "-0.158730"
DELTA_CI_HIGH = "0.277778"
DELTA_PROB = "0.7357"

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

# ============================================================
# 1. CLAIM AUDIT ALREADY EXISTS
# ============================================================

section("1. VERIFYING CLAIM AUDIT")

claim_audit = TABLES / "Table_final_claim_audit.csv"

if not claim_audit.exists():
    raise SystemExit("FAIL: Table_final_claim_audit.csv missing")

print("PASS:", claim_audit)

# ============================================================
# 2. RUN FINAL EVIDENCE INTEGRATION
# ============================================================

section("2. FINAL EVIDENCE INTEGRATION")

integration = ROOT / "Python_scripts" / "final_evidence_integration.py"

if integration.exists():
    result = subprocess.run(
        [sys.executable, str(integration)],
        cwd=ROOT,
        text=True
    )

    if result.returncode != 0:
        raise SystemExit(
            "FAIL: final_evidence_integration.py returned an error."
        )

    print("PASS: evidence integration completed")
else:
    print("WARNING: final_evidence_integration.py not found")

# ============================================================
# 3. NUMERICAL CONSISTENCY AUDIT
# ============================================================

section("3. NUMERICAL CONSISTENCY AUDIT")

audit = ROOT / "Python_scripts" / "final_numerical_consistency_audit.py"

if audit.exists():

    result = subprocess.run(
        [sys.executable, str(audit)],
        cwd=ROOT,
        text=True
    )

    if result.returncode != 0:
        raise SystemExit(
            "FAIL: final_numerical_consistency_audit.py returned an error."
        )

    print("PASS: numerical consistency audit completed")

else:
    print("WARNING: final_numerical_consistency_audit.py not found")

# ============================================================
# 4. VERIFY ACTIVE TABLES
# ============================================================

section("4. VERIFYING ACTIVE FINAL TABLES")

required_tables = [
    "Table_final_model_performance.csv",
    "Table_final_robustness.csv",
    "Table_final_paired_ablation_delta_auc.csv",
    "Table_final_claim_audit.csv",
    "Table_final_lasso_bootstrap_stability.csv",
    "Table_final_meta_analysis.csv",
    "Table_final_biological_evidence.csv",
    "Table_final_results_evidence_matrix.csv",
    "Table_final_cross_model_concordance.csv",
]

missing = []

for name in required_tables:
    path = TABLES / name

    if path.exists():
        print("PASS:", name)
    else:
        print("FAIL:", name)
        missing.append(name)

if missing:
    raise SystemExit(
        "FAIL: missing required tables: " +
        ", ".join(missing)
    )

# ============================================================
# 5. VERIFY STALE TABLES ARE NOT ACTIVE
# ============================================================

section("5. VERIFYING SUPERSEDED TABLES")

stale = [
    "Table_final_three_gene_ablation.csv",
    "Table_three_gene_ablation_bootstrap.csv",
]

for name in stale:

    active = TABLES / name
    archived = ARCHIVE / name

    if active.exists():
        raise SystemExit(
            f"FAIL: stale table is still active: {name}"
        )

    if archived.exists():
        print("PASS: archived:", name)
    else:
        print("WARNING: archive missing:", name)

# ============================================================
# 6. VERIFY CANONICAL THREE-GENE PERFORMANCE
# ============================================================

section("6. VERIFYING CANONICAL THREE-GENE PERFORMANCE")

model = TABLES / "Table_final_model_performance.csv"
robustness = TABLES / "Table_final_robustness.csv"
paired = TABLES / "Table_final_paired_ablation_delta_auc.csv"

model_text = model.read_text(errors="replace")
robust_text = robustness.read_text(errors="replace")
paired_text = paired.read_text(errors="replace")

if EXTERNAL_AUC in model_text:
    print("PASS: final model external AUC =", EXTERNAL_AUC)
else:
    raise SystemExit(
        "FAIL: canonical external AUC missing from final model table"
    )

if EXTERNAL_AUC in robust_text:
    print("PASS: robustness table contains external AUC =", EXTERNAL_AUC)
else:
    raise SystemExit(
        "FAIL: canonical external AUC missing from robustness table"
    )

if TWO_GENE_EXTERNAL in paired_text:
    print("PASS: strongest two-gene external AUC =", TWO_GENE_EXTERNAL)
else:
    raise SystemExit(
        "FAIL: two-gene comparator missing from paired ablation table"
    )

# The CSV stores the full floating-point value
# (e.g. 0.0692307692307693), while DELTA_AUC is
# the manuscript-rounded representation (0.069231).
#
# Therefore compare numerically rather than by exact string matching.

import csv

delta_found = False

try:
    with paired.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        if row.get("Dataset") == "GSE5281_External":
            observed = float(row["Observed_Delta_AUC"])

            if abs(observed - float(DELTA_AUC)) < 1e-5:
                delta_found = True
                print(
                    "PASS: observed paired delta = "
                    f"{observed:.6f}"
                )
                break

except Exception as e:
    print("WARNING: numerical paired-delta check failed:", e)

if not delta_found:
    raise SystemExit(
        "FAIL: paired delta missing or inconsistent"
    )

# ============================================================
# 7. VERIFY PAIRED BOOTSTRAP INTERPRETATION
# ============================================================

section("7. VERIFYING EXTERNAL SUPERIORITY INTERPRETATION")

claim_text = claim_audit.read_text(errors="replace")

required_claims = [
    "The full three-gene model is clearly superior to the strongest two-gene alternative.",
    "NOT_SUPPORTED",
]

for item in required_claims:

    if item in claim_text:
        print("PASS:", item)
    else:
        print("WARNING: expected claim-audit item not found:", item)

# ============================================================
# 8. FINAL TEXT SCAN
# ============================================================

section("8. FINAL MANUSCRIPT PROSE AUDIT")

prose_files = [
    MANUSCRIPT / "Abstract_3gene_signature.txt",
    MANUSCRIPT / "Methods" / "Methods_3gene_signature.txt",
    MANUSCRIPT / "Results" / "Results_3gene_signature.txt",
    MANUSCRIPT / "Results" / "Figure_Legends_3gene_signature.txt",
    MANUSCRIPT / "Discussion" / "Discussion_3gene_signature.txt",
]

old_patterns = [
    r"0\.669230769",
    r"0\.66923",
    r"\b0\.669\b",
    r"GSE5281.{0,100}0\.669",
    r"external.{0,100}0\.669",
]

overclaim_patterns = [
    r"clinically validated",
    r"clinical biomarker",
    r"diagnostic biomarker",
    r"diagnostic validation",
    r"three-gene model was superior",
    r"three-gene signature was superior",
    r"three-gene model outperformed",
    r"three-gene signature outperformed",
    r"all three genes were necessary",
    r"all three genes are necessary",
]

old_found = []
overclaim_found = []

for path in prose_files:

    if not path.exists():
        continue

    text = path.read_text(errors="replace")

    for pattern in old_patterns:

        if re.search(pattern, text, flags=re.I):
            old_found.append(
                f"{path.relative_to(ROOT)} :: {pattern}"
            )

    for pattern in overclaim_patterns:

        if re.search(pattern, text, flags=re.I):
            overclaim_found.append(
                f"{path.relative_to(ROOT)} :: {pattern}"
            )

if old_found:

    print()
    print("FAIL — STALE EXTERNAL AUC REFERENCES:")
    for x in old_found:
        print(" ", x)

else:
    print(
        "PASS: manuscript prose contains no stale 0.669 external-AUC claim"
    )

if overclaim_found:

    print()
    print("FAIL — POTENTIAL OVERCLAIMS:")

    for x in overclaim_found:
        print(" ", x)

else:
    print(
        "PASS: manuscript prose contains no predefined major overclaims"
    )

# ============================================================
# 9. RAW VS HARMONIZED CHECK
# ============================================================

section("9. RAW VS HARMONIZED EXTERNAL RESULT CHECK")

raw_harmonized = TABLES / "Table_raw_vs_harmonized.csv"

if raw_harmonized.exists():

    text = raw_harmonized.read_text(errors="replace")

    if "3-gene_model,0.6615384615384615,0.6846153846153846" in text:
        print(
            "PASS: raw/harmonized three-gene values are correctly distinguished"
        )
    else:
        print(
            "WARNING: raw/harmonized exact row not found; inspect table manually"
        )

else:
    print("WARNING: Table_raw_vs_harmonized.csv missing")

# ============================================================
# 10. FINAL MANUSCRIPT INVENTORY
# ============================================================

section("10. FINAL ACTIVE MANUSCRIPT INVENTORY")

active_files = []

for path in MANUSCRIPT.rglob("*"):

    if not path.is_file():
        continue

    if "ARCHIVED_SUPERSEDED" in path.parts:
        continue

    active_files.append(path.relative_to(ROOT))

for path in sorted(active_files):
    print(path)

# ============================================================
# 11. FINAL STATUS
# ============================================================

section("FINAL STATUS")

if old_found:
    print("STATUS: FAIL — stale numerical references remain")

elif overclaim_found:
    print("STATUS: FAIL — potential overclaims remain")

else:
    print("STATUS: PASS — manuscript prose audit clean")

print()
print("LOCKED EXTERNAL THREE-GENE AUC:", EXTERNAL_AUC)
print("HARMONIZED EXTERNAL AUC:", HARMONIZED_EXTERNAL_AUC)
print("STRONGEST TWO-GENE EXTERNAL AUC:", TWO_GENE_EXTERNAL)
print("OBSERVED EXTERNAL DELTA:", DELTA_AUC)
print("PAIRED BOOTSTRAP CI:", DELTA_CI_LOW, "to", DELTA_CI_HIGH)
print("P(DELTA > 0):", DELTA_PROB)

print()
print("Scientific interpretation:")
print(
    "The three-gene signature demonstrates independent external "
    "discrimination, but external performance is attenuated and "
    "superiority over the strongest two-gene alternative is not established."
)

print()
print("=" * 80)
print("FINAL CONTINUATION COMPLETE")
print("=" * 80)
