#!/usr/bin/env python3

from pathlib import Path
import gzip
import re
import pandas as pd
import numpy as np
import sys
import subprocess
from collections import defaultdict

ROOT = Path.home() / "project_ml"

print("=" * 90)
print("GSE5281 BATCH / DIAGNOSIS STRUCTURE VERIFICATION")
print("=" * 90)
print()

# ============================================================
# 1. SEARCH THE PROJECT FOR EXISTING GSE5281 METADATA
# ============================================================

print("=" * 90)
print("1. SEARCHING PROJECT FOR EXISTING GSE5281 METADATA")
print("=" * 90)

patterns = [
    "*GSE5281*",
    "*gse5281*",
    "*.soft",
    "*.soft.gz",
    "*series_matrix*",
    "*SOFT*",
]

found = set()

for pattern in patterns:
    for p in ROOT.rglob(pattern):
        if "ARCHIVED_SUPERSEDED" not in str(p):
            found.add(p)

for p in sorted(found):
    print(p)

print()
print(f"Files found: {len(found)}")
print()

# ============================================================
# 2. SPECIFIC SAMPLE GROUPS
# ============================================================

HIP_CONTROL = [f"GSM{i}" for i in range(119628, 119641)]
HIP_AD = [f"GSM{i}" for i in range(238799, 238809)]

print("=" * 90)
print("2. HIPPOCAMPAL SAMPLE GROUPS")
print("=" * 90)

print("Controls:")
print(", ".join(HIP_CONTROL))

print()
print("AD:")
print(", ".join(HIP_AD))

print()

# ============================================================
# 3. READ LOCAL METADATA FILES
# ============================================================

metadata_candidates = []

for p in found:
    name = p.name.lower()

    if (
        "gse5281" in name
        and (
            "metadata" in name
            or "characteristic" in name
            or "sample" in name
            or "soft" in name
            or "matrix" in name
        )
    ):
        metadata_candidates.append(p)

print("=" * 90)
print("3. CANDIDATE METADATA FILES")
print("=" * 90)

if metadata_candidates:
    for p in metadata_candidates:
        print(p)
else:
    print("NO LOCAL SOFT/METADATA FILE FOUND.")

print()

# ============================================================
# 4. PARSE SOFT FILE IF AVAILABLE
# ============================================================

soft_files = [
    p for p in metadata_candidates
    if p.suffix.lower() == ".soft"
    or p.name.lower().endswith(".soft.gz")
]

records = {}

def parse_soft(path):

    print("=" * 90)
    print(f"4. PARSING SOFT FILE: {path}")
    print("=" * 90)

    opener = gzip.open if str(path).endswith(".gz") else open

    current = None
    current_block = None
    data = {}

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:

        for raw in fh:

            line = raw.rstrip("\n")

            # ------------------------------------------------
            # Sample block
            # ------------------------------------------------

            m = re.match(r"^\^SAMPLE\s*=\s*(GSM\d+)", line)

            if m:

                current = m.group(1)
                current_block = {}
                data[current] = current_block
                continue

            if current is None:
                continue

            # ------------------------------------------------
            # Sample-level metadata
            # ------------------------------------------------

            if line.startswith("!Sample_"):

                parts = line.split("\t", 1)

                if len(parts) == 2:

                    key = parts[0].strip()
                    value = parts[1].strip()

                    current_block.setdefault(key, []).append(value)

    return data


if soft_files:

    for sf in soft_files:

        try:
            parsed = parse_soft(sf)

            for gsm in HIP_CONTROL + HIP_AD:

                if gsm in parsed:
                    records[gsm] = parsed[gsm]

        except Exception as e:

            print(f"ERROR parsing {sf}: {e}")

else:

    print("No local SOFT file available.")
    print()

# ============================================================
# 5. DISPLAY ALL METADATA FOR HIPPOCAMPAL SAMPLES
# ============================================================

print("=" * 90)
print("5. HIPPOCAMPAL SAMPLE METADATA")
print("=" * 90)

if not records:

    print("No locally parsed SOFT records.")
    print()
    print("This does NOT invalidate the GEO finding.")
    print("It means the raw SOFT file needs to be downloaded locally.")
    print()

else:

    for gsm in HIP_CONTROL + HIP_AD:

        print()
        print("-" * 90)
        print(gsm)
        print("-" * 90)

        rec = records.get(gsm, {})

        for key, values in rec.items():

            print(f"{key}:")
            for value in values:
                print(f"    {value}")

# ============================================================
# 6. EXTRACT IMPORTANT BATCH-RELATED FIELDS
# ============================================================

IMPORTANT_KEYS = [
    "submission_date",
    "last_update_date",
    "title",
    "source_name_ch1",
    "characteristics_ch1",
    "molecule_ch1",
    "extract_protocol_ch1",
    "label_ch1",
    "label_protocol_ch1",
    "hyb_protocol",
    "scan_protocol",
    "description",
    "data_processing",
    "platform_id",
]

print()
print("=" * 90)
print("6. BATCH-RELEVANT METADATA SUMMARY")
print("=" * 90)

rows = []

for gsm in HIP_CONTROL + HIP_AD:

    rec = records.get(gsm, {})

    row = {
        "GSM": gsm,
        "Group": (
            "Control"
            if gsm in HIP_CONTROL
            else "AD"
        )
    }

    for key in IMPORTANT_KEYS:

        values = rec.get(f"!Sample_{key}", [])

        if not values:
            values = rec.get(key, [])

        row[key] = " | ".join(values)

    rows.append(row)

if rows:

    df = pd.DataFrame(rows)

    print(df.to_string(index=False))

    out = ROOT / "04_ML" / "External_Validation" / \
        "GSE5281_hippocampus_batch_metadata_audit.csv"

    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out, index=False)

    print()
    print("Saved:")
    print(out)

else:

    print("No records available.")

# ============================================================
# 7. CHECK WHETHER SUBMISSION DATE IS PERFECTLY NESTED
# ============================================================

print()
print("=" * 90)
print("7. SUBMISSION-DATE NESTING TEST")
print("=" * 90)

submission_dates = {}

for gsm in HIP_CONTROL + HIP_AD:

    rec = records.get(gsm, {})

    vals = (
        rec.get("!Sample_submission_date", [])
        or rec.get("submission_date", [])
    )

    submission_dates[gsm] = vals[0] if vals else ""

if any(submission_dates.values()):

    for gsm, date in submission_dates.items():

        group = (
            "Control"
            if gsm in HIP_CONTROL
            else "AD"
        )

        print(
            f"{gsm:12s} {group:10s} {date}"
        )

    control_dates = {
        submission_dates[g]
        for g in HIP_CONTROL
        if submission_dates[g]
    }

    ad_dates = {
        submission_dates[g]
        for g in HIP_AD
        if submission_dates[g]
    }

    print()
    print("Control submission dates:")
    print(control_dates)

    print()
    print("AD submission dates:")
    print(ad_dates)

    if (
        len(control_dates) == 1
        and len(ad_dates) == 1
        and control_dates != ad_dates
    ):

        print()
        print(
            "RESULT: DIAGNOSIS IS PERFECTLY NESTED "
            "WITHIN SUBMISSION DATE."
        )

    else:

        print()
        print(
            "RESULT: Submission date is NOT perfectly nested "
            "according to the available local metadata."
        )

else:

    print(
        "Submission dates were not available in the local SOFT file."
    )

# ============================================================
# 8. CHECK SCAN PROTOCOL
# ============================================================

print()
print("=" * 90)
print("8. SCAN PROTOCOL TEST")
print("=" * 90)

scan_values = defaultdict(set)

for gsm in HIP_CONTROL + HIP_AD:

    rec = records.get(gsm, {})

    vals = (
        rec.get("!Sample_scan_protocol", [])
        or rec.get("scan_protocol", [])
    )

    group = (
        "Control"
        if gsm in HIP_CONTROL
        else "AD"
    )

    for value in vals:
        scan_values[group].add(value)

print("CONTROL SCAN PROTOCOLS:")
for x in scan_values["Control"]:
    print("  ", x)

print()
print("AD SCAN PROTOCOLS:")
for x in scan_values["AD"]:
    print("  ", x)

# ============================================================
# 9. CHECK HYBRIDIZATION PROTOCOL
# ============================================================

print()
print("=" * 90)
print("9. HYBRIDIZATION PROTOCOL TEST")
print("=" * 90)

hyb_values = defaultdict(set)

for gsm in HIP_CONTROL + HIP_AD:

    rec = records.get(gsm, {})

    vals = (
        rec.get("!Sample_hyb_protocol", [])
        or rec.get("hyb_protocol", [])
    )

    group = (
        "Control"
        if gsm in HIP_CONTROL
        else "AD"
    )

    for value in vals:
        hyb_values[group].add(value)

print("CONTROL HYBRIDIZATION PROTOCOLS:")
for x in hyb_values["Control"]:
    print("  ", x)

print()
print("AD HYBRIDIZATION PROTOCOLS:")
for x in hyb_values["AD"]:
    print("  ", x)

# ============================================================
# 10. CHECK DATA PROCESSING
# ============================================================

print()
print("=" * 90)
print("10. DATA PROCESSING TEST")
print("=" * 90)

processing_values = defaultdict(set)

for gsm in HIP_CONTROL + HIP_AD:

    rec = records.get(gsm, {})

    vals = (
        rec.get("!Sample_data_processing", [])
        or rec.get("data_processing", [])
    )

    group = (
        "Control"
        if gsm in HIP_CONTROL
        else "AD"
    )

    for value in vals:
        processing_values[group].add(value)

print("CONTROL DATA PROCESSING:")
for x in processing_values["Control"]:
    print("  ", x)

print()
print("AD DATA PROCESSING:")
for x in processing_values["AD"]:
    print("  ", x)

# ============================================================
# 11. CHECK EXISTING PCA RESULT
# ============================================================

print()
print("=" * 90)
print("11. EXISTING PROJECT PCA / BATCH EVIDENCE")
print("=" * 90)

pca_candidates = []

for p in ROOT.rglob("*"):
    if "ARCHIVED_SUPERSEDED" in str(p):
        continue

    if p.is_file():

        name = p.name.lower()

        if (
            "pca" in name
            or "batch" in name
            or "harmonization" in name
        ):
            pca_candidates.append(p)

for p in sorted(pca_candidates):
    print(p)

# ============================================================
# 12. CHECK CURRENT MANUSCRIPT LANGUAGE
# ============================================================

print()
print("=" * 90)
print("12. CURRENT MANUSCRIPT BATCH / VALIDATION LANGUAGE")
print("=" * 90)

text_files = [
    ROOT / "06_Manuscript" / "Abstract_3gene_signature.txt",
    ROOT / "06_Manuscript" / "Results" / "Results_3gene_signature.txt",
    ROOT / "06_Manuscript" / "Discussion" / "Discussion_3gene_signature.txt",
    ROOT / "06_Manuscript" / "Methods" / "Methods_3gene_signature.txt",
]

keywords = [
    "independent",
    "external",
    "validation",
    "batch",
    "harmon",
    "GSE5281",
    "0.662",
    "0.6615",
]

for path in text_files:

    if not path.exists():
        continue

    print()
    print("-" * 90)
    print(path)
    print("-" * 90)

    text = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):

        if any(
            key.lower() in line.lower()
            for key in keywords
        ):

            print(
                f"{i:04d}: {line}"
            )

# ============================================================
# 13. FINAL INTERPRETATION
# ============================================================

print()
print("=" * 90)
print("13. FINAL INTERPRETATION")
print("=" * 90)

print(
"""
IMPORTANT:

This audit distinguishes three different claims:

A. SUBMISSION-DATE NESTING
   Can diagnosis be completely separated by GEO submission date?

B. EXPERIMENTAL BATCH NESTING
   Does GEO explicitly identify a laboratory, scan,
   hybridization, processing, or other experimental batch
   that is perfectly confounded with diagnosis?

C. BIOLOGICAL VALIDATION
   Does the external cohort provide an independent estimate
   of disease-associated signal?

A is not automatically B.

B is not automatically proof that all disease signal is artifact.

However, if A/B prevent independent estimation of disease
and batch effects, then the external AUC should NOT be described
as clean independent biological validation.

The appropriate terminology is:

    "cross-dataset transfer test"

rather than:

    "independent external validation"

unless the metadata demonstrate genuine independence.
"""
)

print()
print("=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)
