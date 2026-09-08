import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

BASE = os.path.expanduser(
    "~/project_ml/11_SingleCell/GSE138852"
)

RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

COMPOSITION = os.path.join(
    RESULTS,
    "GSE138852_celltype_composition.csv"
)

PSEUDOBULK = os.path.join(
    RESULTS,
    "GSE138852_ABCA6_CRLF1_pseudobulk.csv"
)

OUTPUT_TABLE = os.path.join(
    RESULTS,
    "GSE138852_composition_ABCA6_CRLF1_correlation.csv"
)

LIBRARY_TABLE = os.path.join(
    RESULTS,
    "GSE138852_composition_expression_library_table.csv"
)

# ============================================================
# LOAD COMPOSITION
# ============================================================

comp = pd.read_csv(COMPOSITION)

# Handle capitalization from our previous output
comp.columns = [
    c.strip()
    for c in comp.columns
]

oligo = comp[
    comp["cell_type"].str.lower() == "oligo"
].copy()

oligo = oligo[
    [
        "sample_group",
        "condition",
        "Cell_percentage"
    ]
].rename(
    columns={
        "Cell_percentage":
        "Oligodendrocyte_percentage"
    }
)

# ============================================================
# LOAD PSEUDOBULK
# ============================================================

pb = pd.read_csv(PSEUDOBULK)

pb.columns = [
    c.strip()
    for c in pb.columns
]

print("=" * 80)
print("GSE138852 — CELL COMPOSITION / GENE EXPRESSION CORRELATION")
print("=" * 80)

print("\nPseudobulk columns:")
print(pb.columns.tolist())

# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

# Convert sample-group naming consistently
if "Sample_group" in pb.columns:
    pb = pb.rename(
        columns={
            "Sample_group":
            "sample_group"
        }
    )

if "Condition" in pb.columns:
    pb = pb.rename(
        columns={
            "Condition":
            "condition"
        }
    )

if "Cell_type" in pb.columns:
    pb = pb.rename(
        columns={
            "Cell_type":
            "cell_type"
        }
    )

# ============================================================
# OLIGODENDROCYTE PSEUDOBULK
# ============================================================

pb_oligo = pb[
    pb["cell_type"].str.lower() == "oligo"
].copy()

# Expected columns:
# ABCA6_mean_expression
# CRLF1_mean_expression

required = [
    "sample_group",
    "condition",
    "ABCA6_mean_expression",
    "CRLF1_mean_expression"
]

missing = [
    c for c in required
    if c not in pb_oligo.columns
]

if missing:

    raise ValueError(
        "Missing expected pseudobulk columns: "
        + ", ".join(missing)
    )

# Keep one row per library
pb_oligo = pb_oligo[
    [
        "sample_group",
        "condition",
        "ABCA6_mean_expression",
        "CRLF1_mean_expression"
    ]
].copy()

# ============================================================
# MERGE
# ============================================================

df = oligo.merge(
    pb_oligo,
    on=[
        "sample_group",
        "condition"
    ],
    how="inner"
)

# ============================================================
# CHECK
# ============================================================

print("\nMerged library-level dataset:")
print(
    df.to_string(index=False)
)

print("\nNumber of libraries:", len(df))

# ============================================================
# CORRELATION
# ============================================================

results = []

for gene in [
    "ABCA6",
    "CRLF1"
]:

    expression_column = (
        gene +
        "_mean_expression"
    )

    valid = df[
        [
            "Oligodendrocyte_percentage",
            expression_column
        ]
    ].dropna()

    if len(valid) < 3:

        print(
            "\nInsufficient libraries for:",
            gene
        )

        continue

    rho, spearman_p = spearmanr(
        valid[
            "Oligodendrocyte_percentage"
        ],
        valid[
            expression_column
        ]
    )

    r, pearson_p = pearsonr(
        valid[
            "Oligodendrocyte_percentage"
        ],
        valid[
            expression_column
        ]
    )

    results.append({

        "Gene": gene,

        "N_libraries": len(valid),

        "Spearman_rho": rho,

        "Spearman_p": spearman_p,

        "Pearson_r": r,

        "Pearson_p": pearson_p
    })

results = pd.DataFrame(results)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    LIBRARY_TABLE,
    index=False
)

results.to_csv(
    OUTPUT_TABLE,
    index=False
)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 80)
print("CORRELATION RESULTS")
print("=" * 80)

print(
    results.to_string(
        index=False
    )
)

print("\nSaved:")
print(LIBRARY_TABLE)
print(OUTPUT_TABLE)

print("\n" + "=" * 80)
print("COMPOSITION / EXPRESSION CORRELATION COMPLETE")
print("=" * 80)
