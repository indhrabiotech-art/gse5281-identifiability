import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

TRAIN = "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv"
TRAIN_META = "04_ML/GSE48350_RMA_train_meta.csv"

EXT_RAW = "03_Preprocessing/GSE5281_RMA_genelevel.csv"
EXT_HARM = "04_ML/GSE5281_RMA_genelevel_harmonized.csv"
EXT_META = "02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv"

# ------------------------------------------------------------
# Load training
# ------------------------------------------------------------

train = pd.read_csv(TRAIN, index_col=0)
train_meta = pd.read_csv(TRAIN_META)

X_train = train[GENES].values

y_train = (
    train_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .values
)

# ------------------------------------------------------------
# Load external raw
# ------------------------------------------------------------

raw = pd.read_csv(
    EXT_RAW,
    index_col=0
).T

raw.index = raw.index.str.replace(
    ".CEL.gz", "", regex=False
)

# ------------------------------------------------------------
# Load external harmonized
# ------------------------------------------------------------

harm = pd.read_csv(
    EXT_HARM,
    index_col=0
).T

harm.index = harm.index.str.replace(
    ".CEL.gz", "", regex=False
)

# ------------------------------------------------------------
# External metadata
# ------------------------------------------------------------

meta = pd.read_csv(EXT_META)

meta["GSM"] = (
    meta["GSM"]
    .astype(str)
    .str.strip()
)

meta = (
    meta
    .set_index("GSM")
    .loc[raw.index]
)

y_ext = (
    meta["diagnosis"]
    .map({"Control": 0, "AD": 1})
    .values
)

# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=0.3,
    max_iter=10000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

# ------------------------------------------------------------
# RAW external
# ------------------------------------------------------------

X_raw = raw[GENES].values

X_raw_scaled = scaler.transform(
    X_raw
)

p_raw = model.predict_proba(
    X_raw_scaled
)[:, 1]

auc_raw = roc_auc_score(
    y_ext,
    p_raw
)

# ------------------------------------------------------------
# HARMONIZED external
# ------------------------------------------------------------

X_harm = harm[GENES].values

X_harm_scaled = scaler.transform(
    X_harm
)

p_harm = model.predict_proba(
    X_harm_scaled
)[:, 1]

auc_harm = roc_auc_score(
    y_ext,
    p_harm
)

# ------------------------------------------------------------
# Individual gene AUCs
# ------------------------------------------------------------

print("=" * 75)
print("3-GENE ROBUSTNESS — RAW vs HARMONIZED EXTERNAL DATA")
print("=" * 75)

print("\nModel-level AUC:")
print(f"Raw external AUC:        {auc_raw:.4f}")
print(f"Harmonized external AUC: {auc_harm:.4f}")
print(f"AUC difference:          {auc_harm - auc_raw:+.4f}")

print("\nIndividual gene AUCs:")

for gene in GENES:

    raw_auc = roc_auc_score(
        y_ext,
        raw[gene].values
    )

    harm_auc = roc_auc_score(
        y_ext,
        harm[gene].values
    )

    print(
        f"{gene:12s}"
        f" raw={raw_auc:.4f}"
        f" harmonized={harm_auc:.4f}"
        f" change={harm_auc - raw_auc:+.4f}"
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

result = pd.DataFrame({
    "analysis": [
        "3-gene_model",
        "ABCA6",
        "CRLF1",
        "TNFRSF11B"
    ],
    "raw_AUC": [
        auc_raw,
        roc_auc_score(y_ext, raw["ABCA6"]),
        roc_auc_score(y_ext, raw["CRLF1"]),
        roc_auc_score(y_ext, raw["TNFRSF11B"])
    ],
    "harmonized_AUC": [
        auc_harm,
        roc_auc_score(y_ext, harm["ABCA6"]),
        roc_auc_score(y_ext, harm["CRLF1"]),
        roc_auc_score(y_ext, harm["TNFRSF11B"])
    ]
})

result["AUC_difference"] = (
    result["harmonized_AUC"] -
    result["raw_AUC"]
)

out = (
    "04_ML/External_Validation/"
    "GSE5281_3gene_raw_vs_harmonized_robustness.csv"
)

result.to_csv(
    out,
    index=False
)

print("\nSaved:")
print(out)

print("\n" + "=" * 75)
print("ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 75)
