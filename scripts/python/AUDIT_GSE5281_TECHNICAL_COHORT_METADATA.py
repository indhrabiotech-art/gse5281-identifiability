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
    "GSE5281_technical_cohort_metadata_audit.csv"
)

meta = pd.read_csv(META)

DATE = "!Sample_submission_date"

print("=" * 100)
print("GSE5281 TECHNICAL / COHORT METADATA AUDIT")
print("=" * 100)

print()
print("Total samples:", len(meta))

technical_columns = [
    "!Sample_contact_name",
    "!Sample_contact_email",
    "!Sample_contact_institute",
    "!Sample_contact_address",
    "!Sample_contact_city",
    "!Sample_contact_country",
    "!Sample_extract_protocol_ch1",
    "!Sample_label_protocol_ch1",
    "!Sample_hyb_protocol",
    "!Sample_scan_protocol",
    "!Sample_label_ch1",
    "!Sample_data_processing",
    "!Sample_platform_id",
    "!Sample_supplementary_file",
    "!Sample_relation",
    "!Sample_series_id"
]

summary_rows = []

for col in technical_columns:

    if col not in meta.columns:
        continue

    print()
    print("=" * 100)
    print(col)
    print("=" * 100)

    result = (
        meta
        .groupby([DATE, col], dropna=False)
        .size()
        .reset_index(name="n")
    )

    print(result.to_string(index=False))

    for _, row in result.iterrows():

        summary_rows.append({
            "submission_date": row[DATE],
            "metadata_field": col,
            "metadata_value": row[col],
            "n": row["n"]
        })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT,
    index=False
)

print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(OUT)

print()
print("=" * 100)
print("AUDIT COMPLETE")
print("=" * 100)
