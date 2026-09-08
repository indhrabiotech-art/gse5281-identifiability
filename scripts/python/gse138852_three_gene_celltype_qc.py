import gzip
import os
import pandas as pd
import numpy as np

BASE = "~/project_ml/11_SingleCell/GSE138852"
BASE = os.path.expanduser(BASE)

COUNTS = os.path.join(BASE, "GSE138852_counts.csv.gz")
META = os.path.join(BASE, "GSE138852_covariates.csv.gz")

OUTDIR = os.path.join(BASE, "results")
os.makedirs(OUTDIR, exist_ok=True)

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

print("=" * 80)
print("GSE138852 — THREE-GENE SINGLE-CELL EXPRESSION QC")
print("=" * 80)

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

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

print("\nMetadata:")
print("Cells:", len(meta))
print("Conditions:")
print(meta["condition"].value_counts())

print("\nCell types:")
print(meta["cell_type"].value_counts())

# ------------------------------------------------------------
# Read only required genes from count matrix
# ------------------------------------------------------------

found = {}
header = None

with gzip.open(COUNTS, "rt") as f:

    header = next(f).rstrip("\n").split(",")

    # remove quotation marks
    header = [x.strip('"') for x in header]

    cell_ids = header[1:]

    target_set = set(GENES)

    for line in f:

        parts = line.rstrip("\n").split(",")

        gene = parts[0].strip('"')

        if gene in target_set:
            values = parts[1:]

            found[gene] = pd.to_numeric(
                values,
                errors="coerce"
            )

print("\nGene availability:")

for gene in GENES:
    if gene in found:
        print(f"{gene}: FOUND")
    else:
        print(f"{gene}: NOT FOUND")

# ------------------------------------------------------------
# Build expression dataframe
# ------------------------------------------------------------

expr = pd.DataFrame(index=cell_ids)

for gene in GENES:

    if gene in found:

        expr[gene] = found[gene]

    else:

        expr[gene] = np.nan

expr.index.name = "cell_id"

# ------------------------------------------------------------
# Align metadata and expression
# ------------------------------------------------------------

common = meta["cell_id"].isin(expr.index)

meta = meta.loc[common].copy()

meta = meta.set_index("cell_id")

expr = expr.loc[meta.index]

print("\nAligned cells:", len(expr))

# ------------------------------------------------------------
# Detection summary
# ------------------------------------------------------------

summary = []

for gene in GENES:

    x = expr[gene]

    summary.append({
        "Gene": gene,
        "Present_in_matrix": bool(x.notna().any()),
        "Cells_with_nonzero_expression":
            int((x.fillna(0) > 0).sum()),
        "Detection_fraction":
            float((x.fillna(0) > 0).mean()),
        "Mean_expression":
            float(x.mean()) if x.notna().any() else np.nan,
        "Median_expression":
            float(x.median()) if x.notna().any() else np.nan
    })

summary = pd.DataFrame(summary)

print("\n" + "=" * 80)
print("GENE DETECTION SUMMARY")
print("=" * 80)

print(summary.to_string(index=False))

# ------------------------------------------------------------
# Cell-type summaries
# ------------------------------------------------------------

long = expr.copy()

long["cell_type"] = meta["cell_type"]
long["condition"] = meta["condition"]

records = []

for gene in GENES:

    if gene not in long:
        continue

    grouped = long.groupby(
        ["cell_type", "condition"],
        observed=True
    )[gene]

    for (cell_type, condition), values in grouped:

        values = values.dropna()

        records.append({
            "Gene": gene,
            "Cell_type": cell_type,
            "Condition": condition,
            "N_cells": len(values),
            "Mean_expression": values.mean(),
            "Median_expression": values.median(),
            "Expressing_cells":
                int((values > 0).sum()),
            "Detection_fraction":
                float((values > 0).mean())
        })

celltype_summary = pd.DataFrame(records)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

summary_file = os.path.join(
    OUTDIR,
    "GSE138852_three_gene_detection_summary.csv"
)

celltype_file = os.path.join(
    OUTDIR,
    "GSE138852_three_gene_celltype_expression.csv"
)

summary.to_csv(summary_file, index=False)

celltype_summary.to_csv(
    celltype_file,
    index=False
)

print("\nSaved:")
print(summary_file)
print(celltype_file)

print("\n" + "=" * 80)
print("SINGLE-CELL THREE-GENE QC COMPLETE")
print("=" * 80)
