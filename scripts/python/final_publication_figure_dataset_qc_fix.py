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


def standardize(df, means, sds):

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
# TRAINING
# ============================================================

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

# Explicit GSM alignment
train_meta["index"] = train_meta["index"].astype(str)
train.index = train.index.astype(str)

train_meta = train_meta.set_index("index")

common = train.index.intersection(train_meta.index)

train = train.loc[common].copy()
train_meta = train_meta.loc[common].copy()

train["Diagnosis"] = train_meta["Diagnosis"].astype(str).str.strip()

train["Dataset"] = "GSE48350"
train["Cohort"] = "Training"


# ============================================================
# LOCKED TRAINING STANDARDIZATION
# ============================================================

means = train[GENES].mean()
sds = train[GENES].std(ddof=1)

train = standardize(
    train,
    means,
    sds
)


# ============================================================
# HELD-OUT TEST
# ============================================================

test = pd.read_csv(
    "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
    index_col=0
)

test_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_test_meta.csv"
)

test.index = test.index.astype(str)

# Inspect likely GSM identifier column
if "index" in test_meta.columns:
    test_meta["index"] = test_meta["index"].astype(str)
    test_meta = test_meta.set_index("index")
elif "GSM" in test_meta.columns:
    test_meta["GSM"] = test_meta["GSM"].astype(str)
    test_meta = test_meta.set_index("GSM")
else:
    raise ValueError(
        "No GSM/index identifier found in held-out metadata."
    )

common = test.index.intersection(test_meta.index)

test = test.loc[common].copy()
test_meta = test_meta.loc[common].copy()

test["Diagnosis"] = (
    test_meta["Diagnosis"]
    .astype(str)
    .str.strip()
)

test["Dataset"] = "GSE48350"
test["Cohort"] = "HeldOut_Test"

test = standardize(
    test,
    means,
    sds
)


# ============================================================
# GSE5281
# ============================================================

external = pd.read_csv(
    "04_ML/Final_Characterization/"
    "GSE5281_3gene_predictions_final.csv"
)

external["GSM"] = external["GSM"].astype(str)
external = external.set_index("GSM")

external["Diagnosis"] = (
    external["Diagnosis"]
    .astype(str)
    .str.strip()
)

external["Dataset"] = "GSE5281"
external["Cohort"] = "External_Hippocampus"

external = standardize(
    external,
    means,
    sds
)


# ============================================================
# GSE63060
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

blood.index = blood.index.astype(str)
blood_meta["GSM"] = blood_meta["GSM"].astype(str)

blood_meta = blood_meta.set_index("GSM")

common = blood.index.intersection(
    blood_meta.index
)

blood = blood.loc[common].copy()
blood_meta = blood_meta.loc[common].copy()

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

blood["Dataset"] = "GSE63060"
blood["Cohort"] = "Blood"

blood = standardize(
    blood,
    means,
    sds
)


# ============================================================
# COMMON DATASET
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

frames = []

for frame in [
    train,
    test,
    external,
    blood
]:

    frames.append(
        frame[columns].copy()
    )


final = pd.concat(
    frames,
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
# QC
# ============================================================

print("=" * 80)
print("FINAL PUBLICATION FIGURE DATASET — QC FIX")
print("=" * 80)

print("\nShape:")
print(final.shape)

print("\nDiagnosis counts:")
print(
    final.groupby(
        ["Dataset", "Cohort", "Diagnosis"],
        dropna=False
    ).size()
)

print("\nMissing diagnosis:")
print(
    final["Diagnosis"].isna().sum()
)

print("\nMissing gene values:")
print(
    final[GENES].isna().sum()
)

print("\nSignature summary:")
print(
    final.groupby(
        ["Dataset", "Cohort", "Diagnosis"],
        dropna=False
    )["Three_gene_signature"]
    .agg(
        N="count",
        Mean="mean",
        SD="std",
        Median="median"
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

print("\nSaved:")
print(out)
print(table)

print("\n" + "=" * 80)
print("PUBLICATION FIGURE DATASET QC COMPLETE")
print("=" * 80)
