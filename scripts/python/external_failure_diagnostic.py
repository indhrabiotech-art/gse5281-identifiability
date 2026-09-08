import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

print("=" * 70)
print("GSE48350 → GSE5281 EXTERNAL VALIDATION FAILURE DIAGNOSTIC")
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
# Load training data
# ------------------------------------------------------------

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

# ------------------------------------------------------------
# Load external age-adjusted data
# ------------------------------------------------------------

external = pd.read_csv(
    "04_ML/GSE5281_RMA_genelevel_age_adjusted.csv",
    index_col=0
)

external = external.T

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

y_train = (
    train_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
)

y_external = (
    external_meta["diagnosis"]
    .map({"Control": 0, "AD": 1})
)

print("\nTraining:")
print("AD =", int(y_train.sum()))
print("Control =", int((y_train == 0).sum()))

print("\nExternal:")
print("AD =", int(y_external.sum()))
print("Control =", int((y_external == 0).sum()))

# ------------------------------------------------------------
# Gene-wise direction analysis
# ------------------------------------------------------------

results = []

for gene in genes:

    train_ad = train.loc[
        y_train.values == 1,
        gene
    ]

    train_control = train.loc[
        y_train.values == 0,
        gene
    ]

    ext_ad = external.loc[
        y_external.values == 1,
        gene
    ]

    ext_control = external.loc[
        y_external.values == 0,
        gene
    ]

    train_delta = (
        train_ad.mean() -
        train_control.mean()
    )

    ext_delta = (
        ext_ad.mean() -
        ext_control.mean()
    )

    train_auc = roc_auc_score(
        y_train,
        train[gene]
    )

    ext_auc = roc_auc_score(
        y_external,
        external[gene]
    )

    results.append({
        "gene": gene,
        "train_AD_mean": train_ad.mean(),
        "train_Control_mean": train_control.mean(),
        "train_delta_AD_minus_Control": train_delta,
        "train_direction": (
            "UP" if train_delta > 0 else "DOWN"
        ),
        "train_AUC": train_auc,
        "external_AD_mean": ext_ad.mean(),
        "external_Control_mean": ext_control.mean(),
        "external_delta_AD_minus_Control": ext_delta,
        "external_direction": (
            "UP" if ext_delta > 0 else "DOWN"
        ),
        "external_AUC": ext_auc,
        "direction_same": (
            np.sign(train_delta) ==
            np.sign(ext_delta)
        )
    })

results = pd.DataFrame(results)

print("\n" + "=" * 70)
print("GENE-WISE DIRECTION COMPARISON")
print("=" * 70)

print(
    results[
        [
            "gene",
            "train_delta_AD_minus_Control",
            "train_direction",
            "train_AUC",
            "external_delta_AD_minus_Control",
            "external_direction",
            "external_AUC",
            "direction_same"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# Prediction distribution
# ------------------------------------------------------------

pred = pd.read_csv(
    "04_ML/External_Validation/GSE5281_7gene_predictions.csv"
)

prob_col = [
    c for c in pred.columns
    if "probability" in c.lower()
]

if prob_col:

    p = pred[prob_col[0]]

    print("\n" + "=" * 70)
    print("EXTERNAL PREDICTION DISTRIBUTION")
    print("=" * 70)

    print("Minimum probability:", p.min())
    print("Maximum probability:", p.max())
    print("Mean probability:", p.mean())
    print("Median probability:", p.median())

    print("\nProbability by diagnosis:")

    pred["Diagnosis"] = external_meta["diagnosis"].values

    print(
        pred.groupby("Diagnosis")[prob_col[0]]
        .describe()
    )

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

results.to_csv(
    "04_ML/External_Validation/"
    "GSE5281_7gene_direction_diagnostic.csv",
    index=False
)

print("\nSaved:")
print(
    "04_ML/External_Validation/"
    "GSE5281_7gene_direction_diagnostic.csv"
)

print("\n" + "=" * 70)
print("EXTERNAL FAILURE DIAGNOSTIC COMPLETE")
print("=" * 70)
