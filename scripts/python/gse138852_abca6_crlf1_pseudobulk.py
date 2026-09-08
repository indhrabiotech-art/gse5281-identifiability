import os
import gzip
import pandas as pd
import numpy as np

BASE = os.path.expanduser(
    "~/project_ml/11_SingleCell/GSE138852"
)

RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

COUNTS = os.path.join(
    BASE,
    "GSE138852_counts.csv.gz"
)

META = os.path.join(
    BASE,
    "GSE138852_covariates.csv.gz"
)

OUTFILE = os.path.join(
    RESULTS,
    "GSE138852_ABCA6_CRLF1_pseudobulk.csv"
)

GENES = [
    "ABCA6",
    "CRLF1"
]

print("=" * 80)
print("GSE138852 — ABCA6 / CRLF1 SAMPLE-LEVEL PSEUDOBULK")
print("=" * 80)

# ============================================================
# METADATA
# ============================================================

meta = pd.read_csv(META)

meta = meta.rename(columns={
    meta.columns[0]: "cell_id",
    "oupSample.batchCond": "condition",
    "oupSample.cellType": "cell_type",
    "oupSample.cellType_batchCond": "cell_type_condition",
    "oupSample.subclustID": "subcluster",
    "oupSample.subclustCond": "subcluster_condition"
})

meta["cell_id"] = meta["cell_id"].astype(str)

# Extract sample group from barcode
meta["sample_group"] = (
    meta["cell_id"]
    .str.rsplit("_", n=2)
    .str[-2:].str.join("_")
)

print("\nSample groups:")
print(
    meta.groupby(
        ["sample_group", "condition"]
    ).size()
)

# ============================================================
# READ ONLY THE TWO GENES
# ============================================================

found = {}

with gzip.open(COUNTS, "rt") as f:

    header = next(f).rstrip("\n").split(",")

    header = [
        x.strip('"')
        for x in header
    ]

    cell_ids = header[1:]

    for line in f:

        parts = line.rstrip("\n").split(",")

        gene = parts[0].strip('"')

        if gene in GENES:

            found[gene] = pd.to_numeric(
                parts[1:],
                errors="coerce"
            )

print("\nGene availability:")

for gene in GENES:

    print(
        f"{gene}: "
        + ("FOUND" if gene in found else "NOT FOUND")
    )

# ============================================================
# EXPRESSION MATRIX
# ============================================================

expr = pd.DataFrame(
    found,
    index=cell_ids
)

expr.index.name = "cell_id"

# ============================================================
# ALIGN
# ============================================================

common = meta["cell_id"].isin(expr.index)

meta = meta.loc[common].copy()

meta = meta.set_index("cell_id")

expr = expr.loc[meta.index]

data = pd.concat(
    [
        meta[
            [
                "condition",
                "cell_type",
                "sample_group"
            ]
        ],
        expr
    ],
    axis=1
)

print("\nAligned cells:", len(data))

# ============================================================
# PSEUDOBULK
# ============================================================

records = []

for (
    sample_group,
    condition,
    cell_type
), subset in data.groupby(
    [
        "sample_group",
        "condition",
        "cell_type"
    ],
    observed=True
):

    row = {
        "Sample_group": sample_group,
        "Condition": condition,
        "Cell_type": cell_type,
        "N_cells": len(subset)
    }

    for gene in GENES:

        values = subset[gene].dropna()

        row[f"{gene}_total_count"] = values.sum()

        row[f"{gene}_mean_expression"] = values.mean()

        row[f"{gene}_median_expression"] = values.median()

        row[f"{gene}_expressing_cells"] = (
            values > 0
        ).sum()

        row[f"{gene}_detection_fraction"] = (
            values > 0
        ).mean()

    records.append(row)

pseudobulk = pd.DataFrame(records)

# ============================================================
# SAVE
# ============================================================

pseudobulk = pseudobulk.sort_values(
    [
        "Cell_type",
        "Condition",
        "Sample_group"
    ]
)

pseudobulk.to_csv(
    OUTFILE,
    index=False
)

print("\n" + "=" * 80)
print("PSEUDOBULK SUMMARY")
print("=" * 80)

print(
    pseudobulk.to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTFILE)

print("\n" + "=" * 80)
print("PSEUDOBULK ANALYSIS COMPLETE")
print("=" * 80)
