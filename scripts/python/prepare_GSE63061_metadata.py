import gzip
import csv
import pandas as pd

FILE = "09_CrossTissue/GSE63061/GSE63061_normalized.txt.gz"
OUT = "09_CrossTissue/results/GSE63061_sample_metadata_clean.csv"

rows = {}

with gzip.open(FILE, "rt") as fh:

    reader = csv.reader(fh, delimiter="\t")

    for row in reader:

        if not row:
            continue

        key = row[0]

        if key == "!Sample_geo_accession":
            rows["GSM"] = [x.strip('"') for x in row[1:]]

        elif key == "!Sample_title":
            rows["Title"] = [x.strip('"') for x in row[1:]]

        elif key == "!Sample_characteristics_ch1":
            value = [x.strip('"') for x in row[1:]]

            if value and value[0].startswith("status:"):
                rows.setdefault("Status_raw", value)

            elif value and value[0].startswith("ethnicity:"):
                rows["Ethnicity"] = value

            elif value and value[0].startswith("age:"):
                rows["Age"] = value

            elif value and value[0].startswith("gender:"):
                rows["Gender"] = value


meta = pd.DataFrame(rows)

print("=" * 70)
print("GSE63061 METADATA")
print("=" * 70)

print("\nTotal samples:", len(meta))

print("\nRAW STATUS DISTRIBUTION")
print(meta["Status_raw"].value_counts(dropna=False))

# ------------------------------------------------------------
# STRICT PRIMARY VALIDATION GROUPS
# ------------------------------------------------------------

def classify_status(x):

    x = str(x).strip().lower()

    if x == "status: ctl":
        return "Control"

    if x == "status: mci":
        return "MCI"

    if x == "status: ad":
        return "AD"

    return pd.NA


meta["Group"] = meta["Status_raw"].map(classify_status)

print("\n" + "=" * 70)
print("PRIMARY VALIDATION GROUPS")
print("=" * 70)

print(meta["Group"].value_counts(dropna=False))

print("\nExcluded ambiguous/other samples:")

excluded = meta.loc[
    meta["Group"].isna(),
    "Status_raw"
].value_counts()

print(excluded)

meta.to_csv(OUT, index=False)

print("\nSaved:")
print(OUT)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
