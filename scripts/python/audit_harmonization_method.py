import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

TRAIN_FILE = "03_Preprocessing/GSE48350_RMA_genelevel.csv"
EXT_RAW = "03_Preprocessing/GSE5281_RMA_genelevel.csv"
EXT_HARM = "04_ML/GSE5281_RMA_genelevel_harmonized.csv"
META = "02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv"

train = pd.read_csv(TRAIN_FILE, index_col=0)
ext_raw = pd.read_csv(EXT_RAW, index_col=0)
ext_harm = pd.read_csv(EXT_HARM, index_col=0)

meta = pd.read_csv(META)
meta["GSM"] = meta["GSM"].astype(str).str.strip()

# Convert genes x samples -> samples x genes
ext_raw = ext_raw.T
ext_harm = ext_harm.T

ext_raw.index = ext_raw.index.str.replace(
    ".CEL.gz", "", regex=False
)

ext_harm.index = ext_harm.index.str.replace(
    ".CEL.gz", "", regex=False
)

meta = (
    meta.set_index("GSM")
    .loc[ext_raw.index]
)

y = meta["diagnosis"].map({
    "Control": 0,
    "AD": 1
}).values

print("=" * 75)
print("HARMONIZATION METHODOLOGICAL AUDIT")
print("=" * 75)

print("\nExternal samples:", len(y))
print("AD:", int(y.sum()))
print("Control:", int((y == 0).sum()))

results = []

for gene in GENES:

    train_x = train.loc[gene].dropna()
    raw_x = ext_raw[gene].astype(float)
    harm_x = ext_harm[gene].astype(float)

    train_mean = train_x.mean()
    train_sd = train_x.std()

    raw_auc = roc_auc_score(y, raw_x)
    harm_auc = roc_auc_score(y, harm_x)

    raw_mean = raw_x.mean()
    harm_mean = harm_x.mean()

    raw_sd = raw_x.std()
    harm_sd = harm_x.std()

    results.append({
        "gene": gene,
        "training_mean": train_mean,
        "training_sd": train_sd,
        "external_raw_mean": raw_mean,
        "external_harmonized_mean": harm_mean,
        "external_raw_sd": raw_sd,
        "external_harmonized_sd": harm_sd,
        "raw_external_AUC": raw_auc,
        "harmonized_external_AUC": harm_auc,
        "AUC_change": harm_auc - raw_auc
    })

results = pd.DataFrame(results)

print("\n" + "=" * 75)
print("GENE-WISE HARMONIZATION EFFECT")
print("=" * 75)

print(results.to_string(index=False))

# ------------------------------------------------------------
# Check whether harmonization preserves within-dataset ranking
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("RANK CORRELATION: RAW vs HARMONIZED")
print("=" * 75)

for gene in GENES:

    raw = ext_raw[gene].astype(float)
    harm = ext_harm[gene].astype(float)

    spearman = raw.rank().corr(
        harm.rank(),
        method="pearson"
    )

    print(f"{gene:12s} Spearman rank correlation = {spearman:.6f}")

# ------------------------------------------------------------
# AD/control direction
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("AD vs CONTROL DIRECTION")
print("=" * 75)

for gene in GENES:

    raw_ad = ext_raw.loc[y == 1, gene].mean()
    raw_ctrl = ext_raw.loc[y == 0, gene].mean()

    harm_ad = ext_harm.loc[y == 1, gene].mean()
    harm_ctrl = ext_harm.loc[y == 0, gene].mean()

    raw_delta = raw_ad - raw_ctrl
    harm_delta = harm_ad - harm_ctrl

    print(
        f"{gene:12s}"
        f" raw Δ={raw_delta:+.6f}"
        f" | harmonized Δ={harm_delta:+.6f}"
    )

# ------------------------------------------------------------
# Save audit
# ------------------------------------------------------------

out = (
    "04_ML/External_Validation/"
    "GSE5281_harmonization_method_audit.csv"
)

results.to_csv(out, index=False)

print("\nSaved:")
print(out)

print("\n" + "=" * 75)
print("HARMONIZATION AUDIT COMPLETE")
print("=" * 75)
