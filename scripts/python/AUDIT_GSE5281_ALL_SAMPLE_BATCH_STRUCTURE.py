#!/usr/bin/env python3

from pathlib import Path
import gzip
import re
import pandas as pd
from collections import defaultdict

ROOT = Path.home() / "project_ml"

SOFT = (
    ROOT
    / "01_GEO_Data/GSE5281/"
      "GSE5281_family.soft.gz"
)

OUT = (
    ROOT
    / "04_ML/External_Validation/"
)

OUT.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("GSE5281 — ALL-SAMPLE SUBMISSION / COHORT STRUCTURE AUDIT")
print("=" * 90)

records = []

current = None
current_record = None

def clean_value(x):

    x = x.strip()

    if x.startswith('"') and x.endswith('"'):
        x = x[1:-1]

    return x


with gzip.open(
    SOFT,
    "rt",
    encoding="utf-8",
    errors="replace"
) as f:

    for raw in f:

        line = raw.rstrip("\n")

        m = re.match(
            r"^\^SAMPLE\s*=\s*(GSM\d+)",
            line
        )

        if m:

            if current_record is not None:
                records.append(
                    current_record
                )

            current = m.group(1)

            current_record = {
                "GSM": current
            }

            continue

        if current_record is None:
            continue

        if not line.startswith("!Sample_"):
            continue

        if "=" not in line:
            continue

        key, value = line.split(
            "=",
            1
        )

        key = key.strip()
        value = clean_value(value)

        current_record[key] = value


if current_record is not None:
    records.append(
        current_record
    )


print(
    "\nTotal GEO sample records:",
    len(records)
)


df = pd.DataFrame(
    records
)


print(
    "\nAvailable metadata columns:"
)

for c in df.columns:

    print(
        " ",
        c
    )


# ============================================================
# SAVE COMPLETE METADATA
# ============================================================

complete_out = (
    OUT
    / "GSE5281_ALL_SAMPLE_SOFT_METADATA.csv"
)

df.to_csv(
    complete_out,
    index=False
)

print(
    "\nSaved:",
    complete_out
)


# ============================================================
# SUBMISSION DATE
# ============================================================

submission_col = (
    "!Sample_submission_date"
)

if submission_col not in df.columns:

    print(
        "\nERROR:"
    )

    print(
        "No !Sample_submission_date field found."
    )

else:

    print(
        "\n" + "=" * 90
    )

    print(
        "SUBMISSION DATE DISTRIBUTION"
    )

    print(
        "=" * 90
    )

    dates = (
        df[submission_col]
        .fillna("MISSING")
        .value_counts()
    )

    print(
        dates.to_string()
    )

    dates.to_csv(
        OUT
        / "GSE5281_ALL_SAMPLE_submission_date_counts.csv"
    )


# ============================================================
# CHARACTERISTICS
# ============================================================

characteristic_cols = [
    c
    for c in df.columns
    if "characteristics" in c.lower()
]


print(
    "\n" + "=" * 90
)

print(
    "CHARACTERISTICS FIELDS"
)

print(
    "=" * 90
)

for c in characteristic_cols:

    print(
        "\nCOLUMN:",
        c
    )

    values = (
        df[c]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .head(50)
    )

    print(
        values.to_string()
    )


# ============================================================
# PLATFORM
# ============================================================

platform_cols = [
    c
    for c in df.columns
    if "platform" in c.lower()
]


print(
    "\n" + "=" * 90
)

print(
    "PLATFORM STRUCTURE"
)

print(
    "=" * 90
)

for c in platform_cols:

    print(
        "\nCOLUMN:",
        c
    )

    print(
        df[c]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .to_string()
    )


# ============================================================
# ALL SAMPLE SUMMARY
# ============================================================

summary_cols = [
    "GSM",
    "!Sample_submission_date"
]

summary_cols = [
    c
    for c in summary_cols
    if c in df.columns
]


summary = df[
    summary_cols
].copy()


summary.to_csv(
    OUT
    / "GSE5281_ALL_SAMPLE_submission_structure.csv",
    index=False
)


print(
    "\n" + "=" * 90
)

print(
    "FINAL SUMMARY"
)

print(
    "=" * 90
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 90
)

print(
    "AUDIT COMPLETE"
)

print(
    "=" * 90
)

