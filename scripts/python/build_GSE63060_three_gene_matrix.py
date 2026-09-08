import pandas as pd
import numpy as np
import os

INPUT = "09_CrossTissue/GSE63060/GSE63060_normalized.txt.gz"
META = "09_CrossTissue/results/GSE63060_sample_metadata_raw.csv"

OUTDIR = "09_CrossTissue/results"
os.makedirs(OUTDIR, exist_ok=True)

# Frozen probe mapping according to the project rule:
# multiple probes -> highest mean RMA expression
PROBE_MAP = {
    "ABCA6": "ILMN_1795507",
    "CRLF1": "ILMN_1681515",
    "TNFRSF11B": "ILMN_1676663"
}

print("=" * 70)
print("GSE63060 THREE-GENE MATRIX CONSTRUCTION")
print("=" * 70)

# ------------------------------------------------------------
# Read metadata
# ------------------------------------------------------------

meta = pd.read_csv(META)

print("\nMetadata shape:", meta.shape)
print("Metadata columns:")
print(meta.columns.tolist())

# ------------------------------------------------------------
# Read normalized matrix
# ------------------------------------------------------------

print("\nReading expression matrix...")

# Find matrix boundaries manually because GEO series matrix
# contains metadata before the expression table.

import gzip
import csv

rows = []

with gzip.open(INPUT, "rt") as fh:

    reader = csv.reader(fh, delimiter="\t")

    inside = False
    header = None

    for row in reader:

        if not row:
            continue

        if row[0] == "!series_matrix_table_begin":
            inside = True
            continue

        if row[0] == "!series_matrix_table_end":
            break

        if not inside:
            continue

        if header is None:
            header = row
            continue

        rows.append(row)

print("Expression rows:", len(rows))
print("Expression columns:", len(header))

# ------------------------------------------------------------
# Convert to dataframe
# ------------------------------------------------------------

expr = pd.DataFrame(
    rows,
    columns=header
)

expr = expr.set_index("ID_REF")

# Remove GEO quotation marks if present
expr.index = (
    expr.index
    .astype(str)
    .str.strip('"')
)

expr.columns = (
    expr.columns
    .astype(str)
    .str.strip('"')
)

expr = expr.apply(
    pd.to_numeric,
    errors="coerce"
)

print("\nMatrix shape:", expr.shape)

# ------------------------------------------------------------
# Extract frozen probes
# ------------------------------------------------------------

missing = [
    probe
    for probe in PROBE_MAP.values()
    if probe not in expr.index
]

if missing:
    raise ValueError(
        f"Required probes missing: {missing}"
    )

gene_matrix = pd.DataFrame(
    {
        gene: expr.loc[probe]
        for gene, probe in PROBE_MAP.items()
    }
)

# ------------------------------------------------------------
# Check orientation
# ------------------------------------------------------------

gene_matrix.index.name = "GSM"

print("\nThree-gene matrix:")
print(gene_matrix.head())

print("\nShape:", gene_matrix.shape)

# ------------------------------------------------------------
# Missing-value check
# ------------------------------------------------------------

print("\nMissing values:")
print(gene_matrix.isna().sum())

if gene_matrix.isna().any().any():
    raise ValueError(
        "Missing values detected in three-gene matrix."
    )

# ------------------------------------------------------------
# Expression summary
# ------------------------------------------------------------

print("\nExpression summary:")
print(
    gene_matrix.describe().T[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

outfile = (
    f"{OUTDIR}/GSE63060_three_gene_genelevel_matrix.csv"
)

gene_matrix.to_csv(outfile)

# ------------------------------------------------------------
# Save probe mapping
# ------------------------------------------------------------

mapping = pd.DataFrame(
    [
        {
            "Gene": gene,
            "Probe": probe,
            "Selection_rule":
                "Highest mean RMA expression across samples"
        }
        for gene, probe in PROBE_MAP.items()
    ]
)

mapping_file = (
    f"{OUTDIR}/GSE63060_final_three_gene_probe_mapping.csv"
)

mapping.to_csv(
    mapping_file,
    index=False
)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print("\nSaved:")
print(outfile)
print(mapping_file)

