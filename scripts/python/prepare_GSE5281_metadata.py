import gzip
import pandas as pd

file = "01_GEO_Data/GSE5281/GSE5281_series_matrix.txt.gz"

sample_accessions = []
sample_titles = []
sample_sources = []

with gzip.open(file, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("!Sample_geo_accession"):
            sample_accessions = line.rstrip("\n").split("\t")[1:]
        elif line.startswith("!Sample_title"):
            sample_titles = line.rstrip("\n").split("\t")[1:]
        elif line.startswith("!Sample_source_name"):
            sample_sources = line.rstrip("\n").split("\t")[1:]

metadata = pd.DataFrame({
    "GSM": [x.strip('"') for x in sample_accessions],
    "title": [x.strip('"') for x in sample_titles],
    "source": [x.strip('"') for x in sample_sources]
})

# Keep only hippocampus validation samples
hip = metadata[
    metadata["title"].str.match(r"^HIP (control|affected)|^HIP_affected", case=False, na=False)
].copy()

hip["diagnosis"] = hip["title"].apply(
    lambda x: "Control" if "control" in x.lower() else "AD"
)

print("\n=== GSE5281 HIPPOCAMPUS COHORT ===")
print(hip.to_string(index=False))

print("\n=== GROUP COUNTS ===")
print(hip["diagnosis"].value_counts())

hip.to_csv(
    "02_Metadata/GSE5281_hippocampus_metadata.csv",
    index=False
)

print("\nSaved:")
print("02_Metadata/GSE5281_hippocampus_metadata.csv")
