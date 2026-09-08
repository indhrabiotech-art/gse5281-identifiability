from pathlib import Path
import shutil
import re
import subprocess
import sys
from datetime import datetime

from pathlib import Path as _P
import shutil as _sh
def _safe_copy(src, dst):
    src, dst = _P(src), _P(dst)
    if not src.exists(): return False
    if dst.exists() and src.resolve() == dst.resolve(): return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    _sh.copy2(src, dst); return True


ROOT = Path.home() / "project_ml"
MANUSCRIPT = ROOT / "06_Manuscript"
TABLES = MANUSCRIPT / "Tables"
ARCHIVE = TABLES / "ARCHIVED_SUPERSEDED"

# ============================================================
# CANONICAL LOCKED VALUES
# ============================================================

TRAIN_AUC = "0.854902"
TEST_AUC = "0.694444"
EXTERNAL_AUC = "0.661538"
HARMONIZED_EXTERNAL_AUC = "0.684615"

TRAIN_AP = "0.8281"
TEST_AP = "0.69375"
EXTERNAL_AP = "0.6030"

BOOTSTRAP_CI_LOW = "0.416667"
BOOTSTRAP_CI_HIGH = "0.876923"

PERM_P = "0.144686"

N_TRAIN = "49"
N_TEST = "13"
N_EXTERNAL = "23"
N_EXTERNAL_AD = "10"
N_EXTERNAL_CTRL = "13"

TWO_GENE_EXTERNAL = "0.730769"
DELTA_AUC = "0.069231"
DELTA_CI_LOW = "-0.158730"
DELTA_CI_HIGH = "0.277778"
DELTA_PROB = "0.7357"

GENES = "ABCA6+CRLF1+TNFRSF11B"

# ============================================================
# LOGGING
# ============================================================

def log(msg):
    print(msg)

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

# ============================================================
# BACKUP
# ============================================================

section("1. CREATING MANUSCRIPT BACKUP")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_root = ROOT / "04_ML" / "Reviewer_Robustness_V3" / f"MANUSCRIPT_BACKUP_{stamp}"

if backup_root.exists():
    shutil.rmtree(backup_root)

shutil.copytree(
    MANUSCRIPT,
    backup_root,
    ignore=shutil.ignore_patterns("ARCHIVED_SUPERSEDED")
)

log(f"Backup created: {backup_root}")

# ============================================================
# ARCHIVE STALE ABLATION TABLES
# ============================================================

section("2. ARCHIVING SUPERSEDED ABLATION TABLES")

ARCHIVE.mkdir(parents=True, exist_ok=True)

stale_files = [
    TABLES / "Table_final_three_gene_ablation.csv",
    TABLES / "Table_three_gene_ablation_bootstrap.csv",
]

for f in stale_files:
    if f.exists():
        destination = ARCHIVE / f.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(f), str(destination))
        log(f"ARCHIVED: {f.name}")
    else:
        log(f"OK: {f.name} already absent")

# ============================================================
# TEXT REPLACEMENT ENGINE
# ============================================================

def replace_in_file(path, replacements):
    if not path.exists():
        return

    text = path.read_text(errors="replace")
    original = text

    for old, new in replacements:
        text = text.replace(old, new)

    if text != original:
        path.write_text(text)
        log(f"UPDATED: {path.relative_to(ROOT)}")

# ============================================================
# ABSTRACT
# ============================================================

section("3. LOCKING ABSTRACT")

abstract = MANUSCRIPT / "Abstract_3gene_signature.txt"

replace_in_file(
    abstract,
    [
        # External AUC
        ("0.669 in the independent GSE5281 cohort",
         "0.662 in the independent GSE5281 cohort"),

        ("0.669 in the independent GSE5281",
         "0.662 in the independent GSE5281"),

        ("0.669 in GSE5281",
         "0.662 in GSE5281"),

        # Canonical numerical values
        ("0.6692307692307692", EXTERNAL_AUC),
        ("0.669230769", EXTERNAL_AUC),

        # Explicitly discourage clinical language
        ("clinically validated biomarker",
         "candidate molecular signature"),

        ("clinical biomarker",
         "candidate molecular signature"),

        ("diagnostic biomarker",
         "candidate molecular signature"),

        # Strong generalization language
        ("demonstrated clinical utility",
         "showed exploratory external discrimination"),

        ("validated clinically",
         "evaluated in an independent research cohort"),
    ]
)

# ============================================================
# RESULTS
# ============================================================

section("4. LOCKING RESULTS")

results = MANUSCRIPT / "Results" / "Results_3gene_signature.txt"

replace_in_file(
    results,
    [
        ("0.6692307692307692", EXTERNAL_AUC),
        ("0.669230769", EXTERNAL_AUC),
        ("0.66923", EXTERNAL_AUC),

        ("0.669 in the raw external",
         "0.662 in the raw external"),

        ("0.669 in the external",
         "0.662 in the external"),

        ("external ROC-AUC of 0.669",
         "external ROC-AUC of 0.662"),

        ("external ROC-AUC of approximately 0.669",
         "external ROC-AUC of approximately 0.662"),

        ("ROC-AUC of 0.669",
         "ROC-AUC of 0.662"),

        # Canonical model values
        ("training ROC-AUC of 0.855",
         "training ROC-AUC of 0.855"),

        # Avoid implying significant superiority
        ("The three-gene model outperformed the two-gene model",
         "The three-gene model did not demonstrate superior external discrimination over the strongest two-gene alternative"),

        ("three-gene model outperformed",
         "three-gene model did not demonstrate superior performance over"),

        ("three-gene signature outperformed",
         "three-gene signature did not demonstrate superior performance over"),

        ("clearly superior to the two-gene model",
         "not demonstrably superior to the strongest two-gene alternative"),

        ("clearly superior",
         "not demonstrably superior"),
    ]
)

# ============================================================
# DISCUSSION
# ============================================================

section("5. LOCKING DISCUSSION")

discussion = MANUSCRIPT / "Discussion" / "Discussion_3gene_signature.txt"

replace_in_file(
    discussion,
    [
        ("0.6692307692307692", EXTERNAL_AUC),
        ("0.669230769", EXTERNAL_AUC),
        ("0.66923", EXTERNAL_AUC),

        ("0.669 in the raw GSE5281",
         "0.662 in the raw GSE5281"),

        ("ROC-AUC of approximately 0.669",
         "ROC-AUC of approximately 0.662"),

        ("external ROC-AUC of 0.669",
         "external ROC-AUC of 0.662"),

        # Overclaim prevention
        ("clinically validated",
         "clinically unvalidated"),

        ("clinical validation",
         "external research-cohort evaluation"),

        ("diagnostic biomarker",
         "candidate molecular signature"),

        ("diagnostic signature",
         "candidate molecular signature"),

        ("clinical biomarker",
         "candidate molecular signature"),

        # Three-gene superiority
        ("The three-gene model was superior to the two-gene model",
         "The three-gene model was not demonstrably superior to the strongest two-gene alternative"),

        ("three-gene model was superior",
         "three-gene model was not demonstrably superior"),

        ("three-gene signature was superior",
         "three-gene signature was not demonstrably superior"),

        ("three-gene model outperformed the two-gene model",
         "the three-gene model did not outperform the strongest two-gene alternative on external ROC-AUC"),

        ("three-gene signature outperformed the two-gene model",
         "the three-gene signature did not outperform the strongest two-gene alternative on external ROC-AUC"),

        ("outperformed the two-gene",
         "did not outperform the strongest two-gene"),

        # Necessity overclaim
        ("all three genes were necessary",
         "the three genes were not individually necessary across cohorts"),

        ("each gene was necessary",
         "individual gene necessity was not established"),

        ("all three genes are necessary",
         "individual necessity of all three genes is not established"),

        # Stability overclaim
        ("highly stable LASSO features",
         "low-frequency LASSO-selected features"),

        ("highly stable individual features",
         "not individually stable features"),

        # Transportability
        ("complete cross-tissue transportability",
         "partial cross-tissue transportability"),

        ("fully transportable across tissues",
         "only partially transportable across tissues"),
    ]
)

# ============================================================
# METHODS
# ============================================================

section("6. LOCKING METHODS TERMINOLOGY")

methods = MANUSCRIPT / "Methods" / "Methods_3gene_signature.txt"

replace_in_file(
    methods,
    [
        ("clinically validated",
         "externally evaluated"),

        ("clinical validation cohort",
         "independent research cohort"),

        ("diagnostic validation",
         "external validation"),

        ("clinical biomarker",
         "candidate molecular signature"),
    ]
)

# ============================================================
# FIGURE LEGENDS
# ============================================================

section("7. LOCKING FIGURE LEGENDS")

legends = MANUSCRIPT / "Results" / "Figure_Legends_3gene_signature.txt"

replace_in_file(
    legends,
    [
        ("0.6692307692307692", EXTERNAL_AUC),
        ("0.669230769", EXTERNAL_AUC),
        ("0.66923", EXTERNAL_AUC),
        ("external AUC of 0.669",
         "external AUC of 0.662"),
        ("ROC-AUC of 0.669",
         "ROC-AUC of 0.662"),
        ("clinically validated",
         "externally evaluated"),
        ("diagnostic biomarker",
         "candidate molecular signature"),
    ]
)

# ============================================================
# GENERATE A CANONICAL MANUSCRIPT FACT SHEET
# ============================================================

section("8. WRITING CANONICAL MANUSCRIPT FACT SHEET")

fact_sheet = MANUSCRIPT / "CANONICAL_RESULTS_LOCKED.txt"

fact_sheet.write_text(
f"""CANONICAL RESULTS — LOCKED
============================

Signature:
{GENES}

Discovery cohort:
GSE48350

Training:
n={N_TRAIN}
ROC-AUC={TRAIN_AUC}
PR-AUC={TRAIN_AP}

Held-out internal test:
n={N_TEST}
ROC-AUC={TEST_AUC}
PR-AUC={TEST_AP}

Independent external hippocampal cohort:
GSE5281
n={N_EXTERNAL}
AD={N_EXTERNAL_AD}
Control={N_EXTERNAL_CTRL}

Raw external:
ROC-AUC={EXTERNAL_AUC}
PR-AUC={EXTERNAL_AP}

External bootstrap:
ROC-AUC={EXTERNAL_AUC}
95% CI={BOOTSTRAP_CI_LOW}–{BOOTSTRAP_CI_HIGH}

Harmonized external:
ROC-AUC={HARMONIZED_EXTERNAL_AUC}

Harmonized permutation:
p={PERM_P}

Strongest external two-gene alternative:
ABCA6+TNFRSF11B
ROC-AUC={TWO_GENE_EXTERNAL}

Three-gene external:
ROC-AUC={EXTERNAL_AUC}

External observed difference:
two-gene minus three-gene = {DELTA_AUC}

Paired bootstrap:
95% CI={DELTA_CI_LOW} to {DELTA_CI_HIGH}
P(delta > 0)={DELTA_PROB}

Interpretation:
- External discrimination is moderate and attenuated relative to discovery.
- The three-gene model is NOT established as superior to the strongest two-gene alternative.
- The confidence interval for the paired external AUC difference spans zero.
- Individual LASSO selection frequencies are low.
- TNFRSF11B shows tissue-dependent directional behavior.
- ABCA6 and CRLF1 show more consistent direction across evaluated datasets.
- No prospective clinical validation was performed.
- The signature should be described as a candidate/exploratory molecular signature,
  not as a clinically validated diagnostic biomarker.
- Raw and harmonized GSE5281 results must not be conflated.
"""
)

# ============================================================
# CLAIM AUDIT — REBUILD CANONICAL VERSION
# ============================================================

section("9. REBUILDING CLAIM AUDIT")

claim_audit = TABLES / "Table_final_claim_audit.csv"

claim_audit.write_text(
"""Claim,Evidence,Verdict
Three-gene signature shows strong discrimination in the discovery cohort.,GSE48350 training ROC-AUC 0.855.,SUPPORTED
Three-gene signature generalizes to an independent hippocampal cohort.,GSE5281 raw external ROC-AUC 0.662 with attenuation relative to discovery.,SUPPORTED_WITH_ATTENUATION
Three genes are highly stable individual LASSO-selected features.,"Bootstrap selection frequencies: TNFRSF11B 15.15%, ABCA6 9.15%, CRLF1 5.65%.",NOT_SUPPORTED
ABCA6 preserves direction across brain and blood.,"Positive effect in training, external hippocampus and blood.",SUPPORTED
CRLF1 preserves direction across evaluated cohorts.,"Positive effects in training, hippocampus and blood.",SUPPORTED
TNFRSF11B is directionally conserved across brain and blood.,Positive in training and hippocampus; negative in blood.,NOT_SUPPORTED
ABCA6 adds information beyond age and sex in GSE63060.,"Age+Sex AUC 0.631 versus Age+Sex+ABCA6 AUC 0.682; likelihood-ratio p approximately 0.00042.",SUPPORTED_AS_EXPLORATORY
The full three-gene model is clearly superior to the strongest two-gene alternative.,"External AUC 0.662 for the three-gene model versus 0.731 for ABCA6+TNFRSF11B; paired bootstrap difference does not establish superiority.",NOT_SUPPORTED
The signature is clinically validated.,No prospective clinical validation cohort.,NOT_SUPPORTED
The signature demonstrates cross-tissue biological transportability.,"ABCA6 and CRLF1 retain direction; TNFRSF11B shows tissue-dependent reversal.",PARTIALLY_SUPPORTED
Three genes are individually necessary in every cohort.,Ablation shows cohort-dependent performance.,NOT_SUPPORTED
"""
)

# ============================================================
# MANUSCRIPT COPY
# ============================================================

_safe_copy(
    claim_audit,
    TABLES / "Table_final_claim_audit.csv"
)

# ============================================================
# REBUILD FINAL EVIDENCE INTEGRATION
# ============================================================

section("10. RUNNING FINAL EVIDENCE INTEGRATION")

integration = ROOT / "Python_scripts" / "final_evidence_integration.py"

if integration.exists():
    result = subprocess.run(
        [sys.executable, str(integration)],
        cwd=ROOT,
        text=True
    )

    if result.returncode != 0:
        raise SystemExit(
            "ERROR: final_evidence_integration.py failed."
        )
else:
    log("WARNING: final_evidence_integration.py not found")

# ============================================================
# COPY CLAIM AUDIT AFTER INTEGRATION
# ============================================================

if claim_audit.exists():
    manuscript_claim = TABLES / "Table_final_claim_audit.csv"
    _safe_copy(claim_audit, manuscript_claim)

# ============================================================
# FINAL NUMERICAL AUDIT
# ============================================================

section("11. RUNNING FINAL NUMERICAL CONSISTENCY AUDIT")

audit = ROOT / "Python_scripts" / "final_numerical_consistency_audit.py"

if audit.exists():
    result = subprocess.run(
        [sys.executable, str(audit)],
        cwd=ROOT,
        text=True
    )

    if result.returncode != 0:
        raise SystemExit(
            "ERROR: numerical consistency audit failed."
        )
else:
    log("WARNING: numerical consistency audit script not found")

# ============================================================
# FINAL TEXT SCAN
# ============================================================

section("12. FINAL MANUSCRIPT TEXT SCAN")

text_files = [
    MANUSCRIPT / "Abstract_3gene_signature.txt",
    MANUSCRIPT / "Methods" / "Methods_3gene_signature.txt",
    MANUSCRIPT / "Results" / "Results_3gene_signature.txt",
    MANUSCRIPT / "Results" / "Figure_Legends_3gene_signature.txt",
    MANUSCRIPT / "Discussion" / "Discussion_3gene_signature.txt",
]

old_patterns = [
    r"0\.669230769",
    r"0\.66923",
    r"0\.669\b",
    r"external.*0\.669",
    r"GSE5281.*0\.669",
]

dangerous_patterns = [
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

found_old = []
found_dangerous = []

for f in text_files:
    if not f.exists():
        continue

    text = f.read_text(errors="replace")

    for pat in old_patterns:
        if re.search(pat, text, flags=re.I):
            found_old.append(
                f"{f.relative_to(ROOT)} :: {pat}"
            )

    for pat in dangerous_patterns:
        if re.search(pat, text, flags=re.I):
            found_dangerous.append(
                f"{f.relative_to(ROOT)} :: {pat}"
            )

if found_old:
    print()
    print("WARNING — OLD EXTERNAL AUC REFERENCES FOUND:")
    for x in found_old:
        print("  ", x)

if found_dangerous:
    print()
    print("WARNING — POTENTIAL OVERCLAIMS FOUND:")
    for x in found_dangerous:
        print("  ", x)

if not found_old:
    print("PASS: no stale 0.669 external AUC in manuscript prose")

if not found_dangerous:
    print("PASS: no major predefined clinical/superiority overclaims")

# ============================================================
# VERIFY ACTIVE TABLES
# ============================================================

section("13. VERIFYING ACTIVE FINAL TABLES")

required = [
    TABLES / "Table_final_model_performance.csv",
    TABLES / "Table_final_robustness.csv",
    TABLES / "Table_final_paired_ablation_delta_auc.csv",
    TABLES / "Table_final_claim_audit.csv",
    TABLES / "Table_final_lasso_bootstrap_stability.csv",
    TABLES / "Table_final_meta_analysis.csv",
    TABLES / "Table_final_biological_evidence.csv",
]

missing = []

for f in required:
    if f.exists():
        print("PASS", f.name)
    else:
        print("FAIL", f.name)
        missing.append(f)

if missing:
    raise SystemExit(
        "ERROR: one or more required final tables are missing."
    )

# ============================================================
# VERIFY CANONICAL THREE-GENE EXTERNAL AUC
# ============================================================

section("14. VERIFYING CANONICAL THREE-GENE AUC")

model_table = TABLES / "Table_final_model_performance.csv"

if model_table.exists():
    text = model_table.read_text()

    if EXTERNAL_AUC in text:
        print(
            f"PASS: canonical three-gene external AUC {EXTERNAL_AUC}"
        )
    else:
        raise SystemExit(
            "FAIL: canonical external AUC not found in final model table."
        )

# ============================================================
# VERIFY STALE TABLES ARE ARCHIVED
# ============================================================

section("15. VERIFYING STALE TABLE ARCHIVE")

for name in [
    "Table_final_three_gene_ablation.csv",
    "Table_three_gene_ablation_bootstrap.csv",
]:
    active = TABLES / name
    archived = ARCHIVE / name

    if active.exists():
        raise SystemExit(
            f"FAIL: stale table still active: {name}"
        )

    if archived.exists():
        print(f"PASS: archived {name}")
    else:
        print(f"WARNING: archive copy missing: {name}")

# ============================================================
# FINAL SUMMARY
# ============================================================

section("FINAL MANUSCRIPT LOCKDOWN STATUS")

print("Canonical signature:")
print(" ", GENES)

print()
print("Canonical performance:")
print(f"  Training AUC       = {TRAIN_AUC}")
print(f"  Internal test AUC  = {TEST_AUC}")
print(f"  External AUC       = {EXTERNAL_AUC}")
print(f"  Harmonized AUC     = {HARMONIZED_EXTERNAL_AUC}")

print()
print("External comparison:")
print(f"  Strongest 2-gene   = {TWO_GENE_EXTERNAL}")
print(f"  Three-gene         = {EXTERNAL_AUC}")
print(f"  Delta              = {DELTA_AUC}")
print(f"  95% CI             = {DELTA_CI_LOW} to {DELTA_CI_HIGH}")
print(f"  P(delta > 0)       = {DELTA_PROB}")

print()
print("Interpretive lock:")
print("  - External validation = SUPPORTED_WITH_ATTENUATION")
print("  - Three-gene superiority = NOT_SUPPORTED")
print("  - Clinical validation = NOT_SUPPORTED")
print("  - Individual LASSO stability = NOT_SUPPORTED")
print("  - Cross-tissue transportability = PARTIALLY_SUPPORTED")
print("  - TNFRSF11B directional conservation = NOT_SUPPORTED")

print()
print("Backup:")
print(f"  {backup_root}")

print()
print("=" * 80)
print("MANUSCRIPT LOCKDOWN COMPLETE")
print("=" * 80)
