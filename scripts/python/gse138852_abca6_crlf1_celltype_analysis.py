import os
import pandas as pd
import numpy as np

BASE = os.path.expanduser(
    "~/project_ml/11_SingleCell/GSE138852"
)

RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

SUMMARY = os.path.join(
    RESULTS,
    "GSE138852_ABCA6_CRLF1_celltype_summary.csv"
)

meta = pd.read_csv(
    os.path.join(BASE, "GSE138852_covariates.csv.gz")
)

meta = meta.rename(columns={
    meta.columns[0]: "cell_id",
    "oupSample.batchCond": "condition",
    "oupSample.cellType": "cell_type",
    "oupSample.cellType_batchCond": "cell_type_condition",
    "oupSample.subclustID": "subcluster",
    "oupSample.subclustCond": "subcluster_condition"
})

meta["cell_id"] = meta["cell_id"].astype(str)

# ------------------------------------------------------------
# Read only ABCA6 and CRLF1
# ------------------------------------------------------------

genes = ["ABCA6", "CRLF1"]

counts = {}

import gzip

with gzip.open(
    os.path.join(BASE, "GSE138852_counts.csv.gz"),
    "rt"
) as f:

    header = next(f).rstrip("\n").split(",")

    header = [x.strip('"') for x in header]

    cell_ids = header[1:]

    for line in f:

        parts = line.rstrip("\n").split(",")

        gene = parts[0].strip('"')

        if gene in genes:

            counts[gene] = pd.to_numeric(
                parts[1:],
                errors="coerce"
            )

expr = pd.DataFrame(
    counts,
    index=cell_ids
)

expr.index.name = "cell_id"

# ------------------------------------------------------------
# Align
# ------------------------------------------------------------

meta = meta[
    meta["cell_id"].isin(expr.index)
].copy()

meta = meta.set_index("cell_id")

expr = expr.loc[meta.index]

data = pd.concat(
    [
        meta[["condition", "cell_type"]],
        expr
    ],
    axis=1
)

# ------------------------------------------------------------
# Cell-type summary
# ------------------------------------------------------------

rows = []

for gene in genes:

    for cell_type in sorted(
        data["cell_type"].dropna().unique()
    ):

        for condition in ["AD", "ct"]:

            subset = data[
                (data["cell_type"] == cell_type) &
                (data["condition"] == condition)
            ][gene].dropna()

            if len(subset) == 0:
                continue

            rows.append({
                "Gene": gene,
                "Cell_type": cell_type,
                "Condition": condition,
                "N_cells": len(subset),
                "Mean_expression": subset.mean(),
                "Median_expression": subset.median(),
                "Expressing_cells": int(
                    (subset > 0).sum()
                ),
                "Detection_fraction": (
                    subset > 0
                ).mean()
            })

summary = pd.DataFrame(rows)

summary.to_csv(
    SUMMARY,
    index=False
)

print("=" * 80)
print("GSE138852 — ABCA6 / CRLF1 CELL-TYPE ANALYSIS")
print("=" * 80)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSaved:")
print(SUMMARY)

print("\n" + "=" * 80)
print("CELL-TYPE ANALYSIS COMPLETE")
print("=" * 80)
