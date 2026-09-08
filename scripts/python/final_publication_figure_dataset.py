import os
import numpy as np
import pandas as pd

OUTDIR = "04_ML/Final_Model/Validation"
MANUSCRIPT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]


def add_scores(df, means, sds):

    out = df.copy()

    for gene in GENES:

        out[gene + "_z"] = (
            out[gene] - means[gene]
        ) / sds[gene]

    out["Three_gene_signature"] = (
        out["ABCA6_z"] +
        out["CRLF1_z"] +
        out["TNFRSF11B_z"]
    ) / 3

    return out


# ============================================================
# GSE48350 TRAINING
# ============================================================

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

means = train[GENES].mean()
sds = train[GENES].std(ddof=1)

train = add_scores(
    train,
    means,
    sds
)

train["Diagnosis"] = (
    train_meta["Diagnosis"]
    .astype(str)
    .str.strip()
)

train["Dataset"] = "GSE48350"
train["Cohort"] = "Training"


# ============================================================
# GSE48350 HELD-OUT TEST
# ============================================================

test = pd.read_csv(
    "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
    index_col=0
)

test_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_test_meta.csv"
)

test = add_scores(
    test,
    means,
    sds
)

test["Diagnosis"] = (
    test_meta["Diagnosis"]
    .astype(str)
    .str.strip()
)

test["Dataset"] = "GSE48350"
test["Cohort"] = "HeldOut_Test"


# ============================================================
# GSE5281 EXTERNAL
# ============================================================

external = pd.read_csv(
    "04_ML/Final_Characterization/"
    "GSE5281_3gene_predictions_final.csv"
)

external = external.set_index("GSM")

external = add_scores(
    external,
    means,
    sds
)

external["Dataset"] = "GSE5281"
external["Cohort"] = "External_Hippocampus"

external = external.rename(
    columns={
        "Diagnosis": "Diagnosis"
    }
)


# ============================================================
# GSE63060 BLOOD
# ============================================================

blood = pd.read_csv(
    "09_CrossTissue/results/"
    "GSE63060_three_gene_genelevel_matrix.csv",
    index_col=0
)

blood_meta = pd.read_csv(
    "09_CrossTissue/results/"
    "GSE63060_sample_metadata_raw.csv"
)

blood_meta["GSM"] = (
    blood_meta["GSM"]
    .astype(str)
)

blood.index = blood.index.astype(str)

blood_meta = blood_meta.set_index("GSM")

common = blood.index.intersection(
    blood_meta.index
)

blood = blood.loc[common]
blood_meta = blood_meta.loc[common]

blood["Diagnosis"] = (
    blood_meta["Characteristic_1"]
    .astype(str)
    .str.replace(
        "status:",
        "",
        regex=False
    )
    .str.strip()
)

blood = add_scores(
    blood,
    means,
    sds
)

blood["Dataset"] = "GSE63060"
blood["Cohort"] = "Blood"


# ============================================================
# COMMON FORMAT
# ============================================================

columns = [
    "Dataset",
    "Cohort",
    "Diagnosis",
    "ABCA6",
    "CRLF1",
    "TNFRSF11B",
    "ABCA6_z",
    "CRLF1_z",
    "TNFRSF11B_z",
    "Three_gene_signature"
]

datasets = []

for x in [
    train,
    test,
    external,
    blood
]:

    available = [
        c for c in columns
        if c in x.columns
    ]

    datasets.append(
        x[available].copy()
    )


final = pd.concat(
    datasets,
    axis=0,
    ignore_index=True
)

final.insert(
    0,
    "Sample_ID",
    np.arange(
        1,
        len(final) + 1
    )
)


# ============================================================
# SAVE
# ============================================================

out = (
    OUTDIR +
    "/FINAL_PUBLICATION_FIGURE_DATASET.csv"
)

table = (
    MANUSCRIPT +
    "/Table_final_publication_figure_dataset.csv"
)

final.to_csv(
    out,
    index=False
)

final.to_csv(
    table,
    index=False
)


# ============================================================
# QC
# ============================================================

print("=" * 80)
print("FINAL PUBLICATION FIGURE DATASET")
print("=" * 80)

print()
print("Shape:", final.shape)

print()
print("Cohort counts:")
print(
    final.groupby(
        ["Dataset", "Cohort", "Diagnosis"],
        dropna=False
    ).size()
)

print()
print("Genes:")
print(GENES)

print()
print("Signature summary:")
print(
    final.groupby(
        ["Dataset", "Cohort"]
    )["Three_gene_signature"]
    .agg(
        N="count",
        Mean="mean",
        SD="std",
        Median="median"
    )
)

print()
print("Saved:")
print(out)
print(table)

print()
print("=" * 80)
print("PUBLICATION FIGURE DATASET COMPLETE")
print("=" * 80)
