import numpy as np
import pandas as pd

print("=" * 70)
print("GSE48350 vs GSE5281 — CROSS-DATASET DISTRIBUTION CHECK")
print("=" * 70)

genes = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B",
    "PLA2G7",
    "SORBS1",
    "PMS2P2",
    "TAB2"
]

# ------------------------------------------------------------
# Load GSE48350 training data
# ------------------------------------------------------------

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

# ------------------------------------------------------------
# Load GSE5281 age-adjusted data
# ------------------------------------------------------------

external = pd.read_csv(
    "04_ML/GSE5281_RMA_genelevel_age_adjusted.csv",
    index_col=0
).T

external.index = (
    external.index
    .str.replace(".CEL.gz", "", regex=False)
)

external_meta = pd.read_csv(
    "02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv"
)

external_meta["GSM"] = (
    external_meta["GSM"]
    .astype(str)
    .str.strip()
)

external_meta = (
    external_meta
    .set_index("GSM")
    .loc[external.index]
    .reset_index()
)

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

train_ad = (train_meta["Diagnosis"] == "AD").to_numpy()
train_control = (train_meta["Diagnosis"] == "Control").to_numpy()

ext_ad = (external_meta["diagnosis"] == "AD").to_numpy()
ext_control = (external_meta["diagnosis"] == "Control").to_numpy()

# ------------------------------------------------------------
# Distribution comparison
# ------------------------------------------------------------

results = []

for gene in genes:

    t = train[gene]
    e = external[gene]

    results.append({
        "gene": gene,

        "train_mean": t.mean(),
        "train_sd": t.std(),
        "train_median": t.median(),
        "train_min": t.min(),
        "train_max": t.max(),

        "external_mean": e.mean(),
        "external_sd": e.std(),
        "external_median": e.median(),
        "external_min": e.min(),
        "external_max": e.max(),

        "train_AD_mean":
            train.loc[train_ad, gene].mean(),

        "train_Control_mean":
            train.loc[train_control, gene].mean(),

        "external_AD_mean":
            external.loc[ext_ad, gene].mean(),

        "external_Control_mean":
            external.loc[ext_control, gene].mean()
    })

results = pd.DataFrame(results)

print("\n" + "=" * 70)
print("RAW DISTRIBUTION")
print("=" * 70)

print(
    results.to_string(index=False)
)

# ------------------------------------------------------------
# Dataset-level standardized comparison
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STANDARDIZED MEANS")
print("=" * 70)

for gene in genes:

    train_z = (
        (train[gene] - train[gene].mean())
        / train[gene].std()
    )

    external_z = (
        (external[gene] - external[gene].mean())
        / external[gene].std()
    )

    print(
        f"{gene:12s} "
        f"GSE48350 mean Z = {train_z.mean(): .4f} | "
        f"GSE5281 mean Z = {external_z.mean(): .4f}"
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results.to_csv(
    "04_ML/External_Validation/"
    "GSE5281_cross_dataset_distribution.csv",
    index=False
)

print("\nSaved:")
print(
    "04_ML/External_Validation/"
    "GSE5281_cross_dataset_distribution.csv"
)

print("\n" + "=" * 70)
print("DISTRIBUTION CHECK COMPLETE")
print("=" * 70)
