#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path.home() / "project_ml"

META = (
    ROOT /
    "04_ML/External_Validation/"
    "GSE5281_ALL_SAMPLE_SOFT_METADATA.csv"
)

OUT = (
    ROOT /
    "04_ML/External_Validation/"
    "GSE5281_submission_cohort_composition.csv"
)

print("=" * 90)
print("GSE5281 SUBMISSION-COHORT COMPOSITION AUDIT")
print("=" * 90)

meta = pd.read_csv(META)

print()
print("Total samples:", len(meta))

# ------------------------------------------------------------
# Basic submission-date distribution
# ------------------------------------------------------------

date_col = "!Sample_submission_date"

print()
print("=" * 90)
print("SUBMISSION DATE DISTRIBUTION")
print("=" * 90)

print(
    meta[date_col]
    .value_counts(dropna=False)
    .to_string()
)

# ------------------------------------------------------------
# Examine biological/sample descriptors
# ------------------------------------------------------------

columns_to_check = [
    "!Sample_title",
    "!Sample_source_name_ch1",
    "!Sample_description",
    "!Sample_characteristics_ch1",
    "!Sample_platform_id"
]

for col in columns_to_check:

    if col not in meta.columns:
        continue

    print()
    print("=" * 90)
    print(col)
    print("=" * 90)

    result = (
        meta
        .groupby(
            [date_col, col],
            dropna=False
        )
        .size()
        .reset_index(name="n")
    )

    print(
        result.to_string(index=False)
    )

# ------------------------------------------------------------
# Save compact summary
# ------------------------------------------------------------

rows = []

for date, subset in meta.groupby(
    date_col,
    dropna=False
):

    titles = []

    if "!Sample_title" in subset.columns:
        titles = (
            subset["!Sample_title"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    sources = []

    if "!Sample_source_name_ch1" in subset.columns:
        sources = (
            subset["!Sample_source_name_ch1"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    rows.append({
        "submission_date": date,
        "n_samples": len(subset),
        "titles": " | ".join(titles),
        "sources": " | ".join(sources)
    })

out = pd.DataFrame(rows)

out.to_csv(
    OUT,
    index=False
)

print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(OUT)

print()
print("=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)
