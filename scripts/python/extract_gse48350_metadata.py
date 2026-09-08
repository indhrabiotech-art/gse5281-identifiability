#!/usr/bin/env python3

import gzip
import re
import pandas as pd
from pathlib import Path


# ============================================================
# GSE48350 METADATA EXTRACTION
# Alzheimer's Disease Gene Expression Dataset
# ============================================================

# -----------------------------
# 1. File paths
# -----------------------------

INPUT_FILE = Path("01_GEO_Data/GSE48350_series_matrix.txt.gz")
OUTPUT_DIR = Path("02_Metadata")
OUTPUT_FILE = OUTPUT_DIR / "GSE48350_metadata.csv"


# -----------------------------
# 2. Check input file
# -----------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nERROR: Input file not found:\n{INPUT_FILE}\n"
        "Check that GSE48350_series_matrix.txt.gz is inside 01_GEO_Data/"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 3. Read GEO metadata
# -----------------------------

metadata = {}

print("\nReading GSE48350 metadata...")

with gzip.open(INPUT_FILE, "rt", encoding="utf-8") as f:

    for line in f:

        # GEO sample metadata lines
        if line.startswith("!Sample_"):

            parts = line.rstrip("\n").split("\t")

            key = parts[0]

            values = [
                value.strip('"')
                for value in parts[1:]
            ]

            metadata[key] = values


# -----------------------------
# 4. Check required fields
# -----------------------------

required_fields = [
    "!Sample_geo_accession",
    "!Sample_title"
]

for field in required_fields:

    if field not in metadata:

        raise ValueError(
            f"\nERROR: Required metadata field missing: {field}"
        )


# -----------------------------
# 5. Number of samples
# -----------------------------

n_samples = len(
    metadata["!Sample_geo_accession"]
)

print(f"Number of samples found: {n_samples}")


# -----------------------------
# 6. Create basic dataframe
# -----------------------------

df = pd.DataFrame({

    "GSM": metadata["!Sample_geo_accession"],

    "Title": metadata["!Sample_title"]

})


# ============================================================
# 7. Extract brain region
# ============================================================

def extract_brain_region(title):

    title_lower = title.lower()

    if "entorhinal cortex" in title_lower:
        return "entorhinal_cortex"

    elif "entorhinalcortex" in title_lower:
        return "entorhinal_cortex"

    elif "hippocampus" in title_lower:
        return "hippocampus"

    elif "post-central gyrus" in title_lower:
        return "post-central_gyrus"

    elif "postcentralgyrus" in title_lower:
        return "post-central_gyrus"

    elif "superior frontal gyrus" in title_lower:
        return "superior_frontal_gyrus"

    elif "superiorfrontalgyrus" in title_lower:
        return "superior_frontal_gyrus"

    else:
        return None


df["Brain_region"] = df["Title"].apply(
    extract_brain_region
)


# ============================================================
# 8. Extract sex
# ============================================================

df["Sex"] = df["Title"].str.extract(
    r"_(female|male)_",
    flags=re.IGNORECASE,
    expand=False
)

df["Sex"] = df["Sex"].str.lower()


# ============================================================
# 9. Extract age
# ============================================================

def extract_age(title):

    # Control:
    # ..._45yrs_indiv12

    match_control = re.search(
        r"_(\d+)yrs_",
        title
    )

    if match_control:
        return int(match_control.group(1))

    # AD:
    # ..._76_AD_33

    match_ad = re.search(
        r"_(\d+)_AD_",
        title
    )

    if match_ad:
        return int(match_ad.group(1))

    return None


df["Age"] = df["Title"].apply(
    extract_age
)
# ============================================================
# 10. Extract individual ID
# ============================================================
#
# Control example:
# Hippocampus_male_45yrs_indiv12
#
# AD example:
# hippocampus_male_76_AD_33
#
# ============================================================

def extract_individual(title):

    # Control format: ..._indiv12
    match_control = re.search(
        r"_indiv([\w-]+)$",
        title
    )

    if match_control:
        return match_control.group(1)

    # AD format: ..._AD_33
    match_ad = re.search(
        r"_AD_([\w-]+)$",
        title
    )

    if match_ad:
        return match_ad.group(1)

    return None


df["Individual"] = df["Title"].apply(
    extract_individual
)
# ============================================================
# 11. Determine diagnosis
# ============================================================
#
# AD samples contain "_AD_" in their title.
#
# Example:
# hippocampus_female_60_AD_20
#
# Control samples do not contain "_AD_".
#
# Example:
# Hippocampus_male_45yrs_indiv12
#
# ============================================================

df["Diagnosis"] = (

    df["Title"]

    .str.contains(
        r"_AD_",
        regex=True,
        na=False
    )

    .map({
        True: "AD",
        False: "Control"
    })

)


# ============================================================
# 12. Reorder columns
# ============================================================

df = df[
    [
        "GSM",
        "Title",
        "Individual",
        "Brain_region",
        "Sex",
        "Age",
        "Diagnosis"
    ]
]


# ============================================================
# 13. Save metadata
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 14. QC SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("GSE48350 METADATA QC SUMMARY")
print("=" * 60)


# Total samples

print(
    f"\nTotal samples: {len(df)}"
)


# Diagnosis

print("\nDiagnosis distribution:")

print(
    df["Diagnosis"].value_counts(
        dropna=False
    )
)


# Brain regions

print("\nBrain-region distribution:")

print(
    df["Brain_region"].value_counts(
        dropna=False
    )
)


# Diagnosis × brain region

print("\nDiagnosis by brain region:")

print(
    pd.crosstab(
        df["Brain_region"],
        df["Diagnosis"]
    )
)


# Sex

print("\nSex distribution:")

print(
    df["Sex"].value_counts(
        dropna=False
    )
)


# Age

print("\nAge summary:")

print(
    df["Age"].describe()
)


# Unique individuals

print(
    "\nUnique individuals:",
    df["Individual"].nunique()
)


# Samples per individual

print("\nSamples per individual:")

print(
    df.groupby("Individual")
      .size()
      .value_counts()
      .sort_index()
)


# Missing values

print("\nMissing values:")

print(
    df.isna().sum()
)


# ============================================================
# 15. Validation checks
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)


# Check expected sample count

if len(df) == 253:

    print("✓ Sample count = 253")

else:

    print(
        f"⚠ WARNING: Expected 253 samples, found {len(df)}"
    )


# Check diagnosis

ad_count = (
    df["Diagnosis"] == "AD"
).sum()

control_count = (
    df["Diagnosis"] == "Control"
).sum()


print(
    f"✓ AD samples: {ad_count}"
)

print(
    f"✓ Control samples: {control_count}"
)


# Check expected AD count

if ad_count == 80:

    print("✓ AD count matches expected 80")

else:

    print(
        f"⚠ WARNING: Expected 80 AD samples, found {ad_count}"
    )


# Check expected control count

if control_count == 173:

    print("✓ Control count matches expected 173")

else:

    print(
        f"⚠ WARNING: Expected 173 control samples, found {control_count}"
    )


# Check brain regions

expected_regions = {
    "entorhinal_cortex",
    "hippocampus",
    "post-central_gyrus",
    "superior_frontal_gyrus"
}

observed_regions = set(
    df["Brain_region"].dropna().unique()
)

if observed_regions == expected_regions:

    print("✓ Four expected brain regions detected")

else:

    print(
        "⚠ WARNING: Brain-region check needs inspection"
    )

    print(
        "Observed:",
        observed_regions
    )


# ============================================================
# 16. Display first samples
# ============================================================

print("\nFirst 15 samples:")

print(
    df.head(15).to_string(
        index=False
    )
)


# ============================================================
# 17. Final output
# ============================================================

print("\n" + "=" * 60)

print(
    f"Metadata saved to:\n{OUTPUT_FILE}"
)

print("=" * 60)

print("\nMetadata extraction completed successfully.\n")
