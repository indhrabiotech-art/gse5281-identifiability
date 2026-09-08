import gzip
import csv
import re
import pandas as pd

INPUT = "01_GEO_Data/GSE5281/GSE5281_series_matrix.txt.gz"
OUTPUT = "02_Metadata/GSE5281_hippocampus_metadata_age.csv"

target_gsms = [
    "GSM119628","GSM119629","GSM119630","GSM119631",
    "GSM119632","GSM119633","GSM119634","GSM119635",
    "GSM119636","GSM119637","GSM119638","GSM119639",
    "GSM119640",
    "GSM238799","GSM238800","GSM238801","GSM238802",
    "GSM238803","GSM238804","GSM238805","GSM238806",
    "GSM238807","GSM238808"
]

with gzip.open(INPUT, "rt", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

def parse_geo_line(prefix):
    for line in lines:
        if line.startswith(prefix):
            return next(csv.reader([line], delimiter="\t"))
    raise RuntimeError(f"Could not find {prefix}")

gsm_row = parse_geo_line("!Sample_geo_accession")
title_row = parse_geo_line("!Sample_title")
source_row = parse_geo_line("!Sample_source_name_ch1")

age_row = None

for line in lines:
    if line.startswith("!Sample_characteristics_ch1"):
        fields = next(csv.reader([line], delimiter="\t"))
        if any("Age:" in x for x in fields):
            age_row = fields
            break

if age_row is None:
    raise RuntimeError("Age metadata not found")

# Build GSM -> metadata mapping
records = []

for i in range(1, len(gsm_row)):
    gsm = gsm_row[i].strip('"')

    if gsm not in target_gsms:
        continue

    title = title_row[i].strip('"')
    source = source_row[i].strip('"')
    age_text = age_row[i].strip('"')

    match = re.search(r'Age:\s*([0-9]+(?:\.[0-9]+)?)\s*(years?|days?)',
                      age_text, re.I)

    age_years = None

    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()

        if "day" in unit:
            age_years = value / 365.25
        else:
            age_years = value

    if "control" in title.lower():
        diagnosis = "Control"
    elif "affected" in title.lower():
        diagnosis = "AD"
    else:
        diagnosis = "Unknown"

    records.append({
        "GSM": gsm,
        "title": title,
        "source": source,
        "age_text": age_text,
        "Age_years": age_years,
        "diagnosis": diagnosis
    })

df = pd.DataFrame(records)

# Preserve the intended cohort order
df["order"] = df["GSM"].map({gsm: i for i, gsm in enumerate(target_gsms)})
df = df.sort_values("order").drop(columns="order")

print("\n=== GSE5281 HIPPOCAMPUS AGE METADATA ===")
print(df.to_string(index=False))

print("\n=== COUNTS ===")
print(df["diagnosis"].value_counts())

print("\n=== AGE SUMMARY ===")
print(df.groupby("diagnosis")["Age_years"].describe())

print("\nMissing ages:", df["Age_years"].isna().sum())

if df["GSM"].duplicated().any():
    raise RuntimeError("Duplicate GSM detected")

if len(df) != 23:
    raise RuntimeError(f"Expected 23 samples, found {len(df)}")

if df["Age_years"].isna().any():
    raise RuntimeError("One or more validation samples have missing age")

df.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)
