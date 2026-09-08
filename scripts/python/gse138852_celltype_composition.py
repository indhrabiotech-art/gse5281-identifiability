import os
import pandas as pd
import numpy as np

BASE = os.path.expanduser(
    "~/project_ml/11_SingleCell/GSE138852"
)

RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

INPUT = os.path.join(
    BASE,
    "GSE138852_covariates.csv.gz"
)

OUTPUT = os.path.join(
    RESULTS,
    "GSE138852_celltype_composition.csv"
)

meta = pd.read_csv(INPUT)

meta = meta.rename(columns={
    meta.columns[0]: "cell_id",
    "oupSample.batchCond": "condition",
    "oupSample.cellType": "cell_type",
    "oupSample.cellType_batchCond": "cell_type_condition",
    "oupSample.subclustID": "subcluster",
    "oupSample.subclustCond": "subcluster_condition"
})

meta["cell_id"] = meta["cell_id"].astype(str)

# Extract sample group
meta["sample_group"] = (
    meta["cell_id"]
    .str.rsplit("_", n=2)
    .str[-2:]
    .str.join("_")
)

# ------------------------------------------------------------
# Cell counts
# ------------------------------------------------------------

counts = (
    meta
    .groupby(
        [
            "sample_group",
            "condition",
            "cell_type"
        ]
    )
    .size()
    .reset_index(
        name="N_cells"
    )
)

# ------------------------------------------------------------
# Total cells per sample
# ------------------------------------------------------------

totals = (
    counts
    .groupby(
        "sample_group"
    )["N_cells"]
    .sum()
    .reset_index(
        name="Total_cells"
    )
)

counts = counts.merge(
    totals,
    on="sample_group",
    how="left"
)

counts["Cell_fraction"] = (
    counts["N_cells"]
    /
    counts["Total_cells"]
)

counts["Cell_percentage"] = (
    counts["Cell_fraction"]
    * 100
)

# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

counts = counts.sort_values(
    [
        "cell_type",
        "condition",
        "sample_group"
    ]
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

counts.to_csv(
    OUTPUT,
    index=False
)

print("=" * 80)
print("GSE138852 — CELL-TYPE COMPOSITION ANALYSIS")
print("=" * 80)

print("\nSample totals:")
print(
    totals.to_string(
        index=False
    )
)

print("\nCell-type composition:")
print(
    counts.to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT)

print("\n" + "=" * 80)
print("CELL-TYPE COMPOSITION COMPLETE")
print("=" * 80)
