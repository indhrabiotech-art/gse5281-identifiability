import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

OUTDIR = "04_ML/Final_Characterization"
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 70)
print("FINAL 3-GENE SIGNATURE CHARACTERIZATION")
print("=" * 70)

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
# Load internal test data
# ------------------------------------------------------------

test = pd.read_csv(
    "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
    index_col=0
)

test_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_test_meta.csv"
)

# ------------------------------------------------------------
# Load external harmonized data
# ------------------------------------------------------------

external = pd.read_csv(
    "04_ML/GSE5281_RMA_genelevel_harmonized.csv",
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

y_train = (
    train_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .values
)

y_test = (
    test_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .values
)

y_external = (
    external_meta["diagnosis"]
    .map({"Control": 0, "AD": 1})
    .values
)

# ------------------------------------------------------------
# Gene-wise statistics
# ------------------------------------------------------------

rows = []

for gene in GENES:

    train_ad = train.loc[y_train == 1, gene]
    train_control = train.loc[y_train == 0, gene]

    test_ad = test.loc[y_test == 1, gene]
    test_control = test.loc[y_test == 0, gene]

    ext_ad = external.loc[y_external == 1, gene]
    ext_control = external.loc[y_external == 0, gene]

    train_delta = train_ad.mean() - train_control.mean()
    test_delta = test_ad.mean() - test_control.mean()
    ext_delta = ext_ad.mean() - ext_control.mean()

    train_auc = roc_auc_score(y_train, train[gene])
    test_auc = roc_auc_score(y_test, test[gene])
    ext_auc = roc_auc_score(y_external, external[gene])

    rows.append({
        "gene": gene,

        "train_AD_mean": train_ad.mean(),
        "train_Control_mean": train_control.mean(),
        "train_delta": train_delta,
        "train_AUC": train_auc,

        "test_AD_mean": test_ad.mean(),
        "test_Control_mean": test_control.mean(),
        "test_delta": test_delta,
        "test_AUC": test_auc,

        "external_AD_mean": ext_ad.mean(),
        "external_Control_mean": ext_control.mean(),
        "external_delta": ext_delta,
        "external_AUC": ext_auc,

        "train_external_direction_same":
            np.sign(train_delta) == np.sign(ext_delta)
    })

gene_stats = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("GENE-WISE PERFORMANCE")
print("=" * 70)

print(
    gene_stats[
        [
            "gene",
            "train_delta",
            "train_AUC",
            "test_delta",
            "test_AUC",
            "external_delta",
            "external_AUC",
            "train_external_direction_same"
        ]
    ].to_string(index=False)
)

gene_stats.to_csv(
    f"{OUTDIR}/3gene_gene_wise_statistics.csv",
    index=False
)

# ------------------------------------------------------------
# Fit locked 3-gene model
# ------------------------------------------------------------

X_train = train[GENES].values
X_test = test[GENES].values
X_external = external[GENES].values

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_external_scaled = scaler.transform(X_external)

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
# Predictions
# ------------------------------------------------------------

datasets = {
    "GSE48350_train": (X_train_scaled, y_train),
    "GSE48350_test": (X_test_scaled, y_test),
    "GSE5281_external": (X_external_scaled, y_external)
}

summary = []

for name, (X, y) in datasets.items():

    probability = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, probability)

    ap = average_precision_score(
        y,
        probability
    )

    fpr, tpr, thresholds = roc_curve(
        y,
        probability
    )

    precision, recall, pr_thresholds = precision_recall_curve(
        y,
        probability
    )

    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds
    }).to_csv(
        f"{OUTDIR}/{name}_ROC.csv",
        index=False
    )

    pd.DataFrame({
        "precision": precision,
        "recall": recall
    }).to_csv(
        f"{OUTDIR}/{name}_PR.csv",
        index=False
    )

    summary.append({
        "dataset": name,
        "n": len(y),
        "AD": int(y.sum()),
        "Control": int((y == 0).sum()),
        "ROC_AUC": auc,
        "Average_Precision": ap
    })

summary = pd.DataFrame(summary)

print("\n" + "=" * 70)
print("COMBINED 3-GENE MODEL")
print("=" * 70)

print(summary.to_string(index=False))

summary.to_csv(
    f"{OUTDIR}/3gene_model_performance_summary.csv",
    index=False
)

# ------------------------------------------------------------
# Coefficients
# ------------------------------------------------------------

coef_df = pd.DataFrame({
    "gene": GENES,
    "coefficient": model.coef_[0]
})

coef_df["abs_coefficient"] = (
    coef_df["coefficient"].abs()
)

coef_df = coef_df.sort_values(
    "abs_coefficient",
    ascending=False
)

coef_df.to_csv(
    f"{OUTDIR}/3gene_final_coefficients.csv",
    index=False
)

print("\nFinal coefficients:")
print(coef_df.to_string(index=False))

# ------------------------------------------------------------
# External prediction table
# ------------------------------------------------------------

external_probability = model.predict_proba(
    X_external_scaled
)[:, 1]

external_output = pd.DataFrame({
    "GSM": external.index,
    "Diagnosis": external_meta["diagnosis"].values,
    "ABCA6": external["ABCA6"].values,
    "CRLF1": external["CRLF1"].values,
    "TNFRSF11B": external["TNFRSF11B"].values,
    "Predicted_AD_probability": external_probability
})

external_output.to_csv(
    f"{OUTDIR}/GSE5281_3gene_predictions_final.csv",
    index=False
)

print("\nSaved final characterization files to:")
print(OUTDIR)

print("\n" + "=" * 70)
print("FINAL 3-GENE CHARACTERIZATION COMPLETE")
print("=" * 70)
