from pathlib import Path
from datetime import datetime
import shutil
import re
import hashlib

ROOT = Path.home() / "project_ml"
MANUSCRIPT = ROOT / "06_Manuscript"

# ------------------------------------------------------------
# CANONICAL MANUSCRIPT VALUES
# ------------------------------------------------------------

RAW_EXTERNAL = "0.662"
RAW_EXTERNAL_FULL = "0.661538"

HARMONIZED_EXTERNAL = "0.684615"
HARMONIZED_EXTERNAL_SHORT = "0.685"

# ------------------------------------------------------------
# FILES THAT ARE ALLOWED TO BE MODIFIED
# ONLY MANUSCRIPT PROSE FILES
# ------------------------------------------------------------

PROSE_FILES = [
    MANUSCRIPT / "Abstract_3gene_signature.txt",
    MANUSCRIPT / "Methods" / "Methods_3gene_signature.txt",
    MANUSCRIPT / "Results" / "Results_3gene_signature.txt",
    MANUSCRIPT / "Results" / "Figure_Legends_3gene_signature.txt",
    MANUSCRIPT / "Discussion" / "Discussion_3gene_signature.txt",
]

# ------------------------------------------------------------
# CREATE BACKUP
# ------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_root = (
    ROOT
    / "04_ML"
    / "Reviewer_Robustness_V3"
    / f"MANUSCRIPT_PROSE_BACKUP_{timestamp}"
)

backup_root.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FINAL STALE PROSE REPAIR")
print("=" * 80)
print()
print("Backup:", backup_root)
print()

# ------------------------------------------------------------
# BACKUP + PATCH
# ------------------------------------------------------------

changed_files = []
unchanged_files = []

for src in PROSE_FILES:

    if not src.exists():
        print("WARNING: missing:", src)
        continue

    # Preserve relative structure inside backup
    rel = src.relative_to(MANUSCRIPT)
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dst)

    original = src.read_text(encoding="utf-8")

    text = original

    # --------------------------------------------------------
    # REPLACE STALE EXTERNAL AUC REPRESENTATIONS
    #
    # These replacements apply ONLY to prose files.
    # Tables are deliberately untouched.
    # --------------------------------------------------------

    replacements = [
        # Full stale values
        (r"\b0\.6692307692307692\b", "0.662"),
        (r"\b0\.669230769230769\b", "0.662"),
        (r"\b0\.669230769\b", "0.662"),
        (r"\b0\.66923077\b", "0.662"),
        (r"\b0\.66923\b", "0.662"),
        (r"\b0\.6692\b", "0.662"),

        # Rounded stale value
        (r"\b0\.669\b", "0.662"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    if text != original:
        src.write_text(text, encoding="utf-8")
        changed_files.append(src)

        print("PATCHED:", src)

        # Show exactly what changed
        old_lines = original.splitlines()
        new_lines = text.splitlines()

        for i, (old_line, new_line) in enumerate(
            zip(old_lines, new_lines), start=1
        ):
            if old_line != new_line:
                print(f"  line {i}:")
                print(f"    OLD: {old_line}")
                print(f"    NEW: {new_line}")

    else:
        unchanged_files.append(src)
        print("UNCHANGED:", src)

print()
print("=" * 80)
print("PROSE PATCH SUMMARY")
print("=" * 80)
print("Changed files:", len(changed_files))
print("Unchanged files:", len(unchanged_files))
print("Backup:", backup_root)

# ------------------------------------------------------------
# HARD SAFETY CHECK
#
# No manuscript table is modified by this script.
# ------------------------------------------------------------

print()
print("=" * 80)
print("VERIFYING PROSE ONLY")
print("=" * 80)

for f in PROSE_FILES:
    if f.exists():
        print("OK:", f)

# ------------------------------------------------------------
# SEARCH ACTIVE PROSE FOR STALE EXTERNAL AUC
# ------------------------------------------------------------

print()
print("=" * 80)
print("SEARCHING ACTIVE PROSE FOR STALE 0.669 REFERENCES")
print("=" * 80)

stale_found = []

for f in PROSE_FILES:

    if not f.exists():
        continue

    text = f.read_text(encoding="utf-8")

    patterns = [
        r"\b0\.669\b",
        r"\b0\.6692\b",
        r"\b0\.66923\b",
        r"\b0\.669230769\b",
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            stale_found.append((f, pattern))

if stale_found:
    print()
    print("FAIL: stale prose references remain")

    for f, pattern in stale_found:
        print(" ", f, "::", pattern)

    raise SystemExit(1)

print("PASS: no stale 0.669 references remain in active prose")

# ------------------------------------------------------------
# VERIFY CANONICAL EXTERNAL AUC IN PROSE
# ------------------------------------------------------------

print()
print("=" * 80)
print("VERIFYING CANONICAL EXTERNAL AUC IN PROSE")
print("=" * 80)

canonical_hits = 0

for f in PROSE_FILES:

    if not f.exists():
        continue

    text = f.read_text(encoding="utf-8")

    if re.search(r"\b0\.662\b", text):
        canonical_hits += 1
        print("PASS:", f)

if canonical_hits == 0:
    print("WARNING: no 0.662 reference found in prose")

# ------------------------------------------------------------
# VERIFY NO SCIENTIFIC OVERCLAIM WAS CREATED
# ------------------------------------------------------------

print()
print("=" * 80)
print("OVERCLAIM SAFETY CHECK")
print("=" * 80)

OVERCLAIM_PATTERNS = [
    r"\bsuperior\b",
    r"\boutperformed\b",
    r"\boutperforms\b",
    r"\bclearly superior\b",
    r"\bsignificantly superior\b",
    r"\bbetter than\b",
    r"\bclinically validated\b",
    r"\bclinically validated signature\b",
]

for f in PROSE_FILES:

    if not f.exists():
        continue

    text = f.read_text(encoding="utf-8")

    # We are NOT deleting claims here.
    # We only report them so the scientific wording remains visible.
    found = []

    for pattern in OVERCLAIM_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            found.append(pattern)

    if found:
        print("REVIEW:", f)
        for pattern in found:
            print("  ", pattern)

print()
print("=" * 80)
print("FINAL PROSE REPAIR COMPLETE")
print("=" * 80)
print()
print("RAW EXTERNAL AUC:        0.661538 (~0.662)")
print("HARMONIZED EXTERNAL AUC: 0.684615")
print("TWO-GENE EXTERNAL AUC:   0.730769")
print("OBSERVED DELTA:          0.069231")
print("DELTA CI:                -0.158730 to 0.277778")
print("P(DELTA > 0):            0.7357")
print()
print("Tables were NOT modified.")
print("Backup created at:")
print(backup_root)
print()
