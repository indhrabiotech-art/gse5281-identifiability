#!/usr/bin/env python3

import gzip
import re
import pandas as pd
from pathlib import Path

SOFT = Path(
    "01_GEO_Data/GSE5281/GSE5281_family.soft.gz"
)

OUT = Path(
    "04_ML/External_Validation/"
    "GSE5281_hippocampus_SOFT_complete_metadata.csv"
)

TARGETS = (
    {f"GSM{i}" for i in range(119628,119641)}
    |
    {f"GSM{i}" for i in range(238799,238809)}
)

records = {}
current = None

with gzip.open(
    SOFT,
    "rt",
    encoding="utf-8",
    errors="replace"
) as f:

    for raw in f:

        line = raw.rstrip("\n")

        m = re.match(
            r'^\^SAMPLE\s*=\s*(GSM\d+)',
            line
        )

        if m:

            current = m.group(1)

            if current in TARGETS:
                records[current] = {
                    "GSM": current
                }

            continue

        if current not in records:
            continue

        if line.startswith("!Sample_"):

            m = re.match(
                r'^!Sample_([^ ]+)\s*=\s*(.*)$',
                line
            )

            if not m:
                continue

            key = m.group(1)
            value = m.group(2).strip()

            if key in records[current]:

                records[current][key] += " | " + value

            else:

                records[current][key] = value


# ------------------------------------------------------------
# Build dataframe
# ------------------------------------------------------------

df = pd.DataFrame.from_dict(
    records,
    orient="index"
)

df.index.name = "row"

df["Group"] = "UNKNOWN"

df.loc[
    df["GSM"].isin(
        [f"GSM{i}" for i in range(119628,119641)]
    ),
    "Group"
] = "Control"

df.loc[
    df["GSM"].isin(
        [f"GSM{i}" for i in range(238799,238809)]
    ),
    "Group"
] = "AD"


# ------------------------------------------------------------
# Print important fields
# ------------------------------------------------------------

important = [
    "GSM",
    "Group",
    "title",
    "submission_date",
    "last_update_date",
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
    "platform_id"
]

existing = [
    x for x in important
    if x in df.columns
]

print("=" * 100)
print("GSE5281 HIPPOCAMPUS — COMPLETE SOFT METADATA")
print("=" * 100)

print()

print(
    df[
        existing
    ].to_string(index=False)
)

print()
print("=" * 100)
print("UNIQUE VALUES BY GROUP")
print("=" * 100)

for col in existing:

    if col in ("GSM","Group"):
        continue

    print()
    print("-" * 100)
    print(col)

    for group in ["Control","AD"]:

        vals = (
            df.loc[
                df["Group"] == group,
                col
            ]
            .fillna("")
            .astype(str)
            .unique()
        )

        print()
        print(group)

        for v in vals:
            print("  ",repr(v))


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df[
    existing
].to_csv(
    OUT,
    index=False
)

print()
print("=" * 100)
print("SAVED")
print(OUT)
print("=" * 100)

