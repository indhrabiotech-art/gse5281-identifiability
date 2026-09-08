"""
STEP 1-4 of the master log's "IMMEDIATE NEXT STEPS" section.
"""

import pandas as pd

IN_PATH  = "02_Metadata/GSE48350_metadata.csv"
OUT_PATH = "02_Metadata/GSE48350_metadata_grouped.csv"

df = pd.read_csv(IN_PATH)

print("Loaded metadata:", df.shape[0], "rows")
print(df.head())

# STEP 1: Construct a robust grouping identifier.
# The raw "Individual" field alone is not globally unique (Section 18).
df["subject_group_id"] = (
    df["Diagnosis"].astype(str) + "_" +
    df["Sex"].astype(str) + "_" +
    df["Age"].astype(str) + "_" +
    df["Individual"].astype(str)
)

# STEP 2: Verify no group contains both AD and Control.
mixed_diagnosis = df.groupby("subject_group_id")["Diagnosis"].nunique()
bad_groups = mixed_diagnosis[mixed_diagnosis > 1]

if len(bad_groups) > 0:
    print("\nERROR: the following subject_group_id values contain "
          "BOTH AD and Control samples:")
    print(bad_groups)
else:
    print("\nOK: every subject_group_id maps to a single diagnosis "
          "(no AD/Control mixing).")

regions_per_group = df.groupby("subject_group_id")["Brain_region"].nunique()
over_max = regions_per_group[regions_per_group > 4]
if len(over_max) > 0:
    print("\nWARNING: groups with more than 4 distinct brain regions:")
    print(over_max)

# STEP 3: Calculate unique subjects per diagnosis and per region.
unique_subjects = df.drop_duplicates("subject_group_id")

print("\n--- Unique subject counts ---")
print("Total unique subjects:", unique_subjects.shape[0])
print(unique_subjects["Diagnosis"].value_counts())

print("\n--- Samples per subject_group_id (distribution) ---")
samples_per_subject = df.groupby("subject_group_id").size()
print(samples_per_subject.value_counts().sort_index())

print("\n--- Unique subjects per brain region x diagnosis ---")
region_diag_subjects = (
    df.drop_duplicates(["subject_group_id", "Brain_region"])
      .groupby(["Brain_region", "Diagnosis"])
      .size()
      .unstack(fill_value=0)
)
print(region_diag_subjects)

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved grouped metadata to {OUT_PATH}")
