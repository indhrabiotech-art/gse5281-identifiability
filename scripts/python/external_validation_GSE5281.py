import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


print("=" * 60)
print("GSE5281 — INDEPENDENT EXTERNAL VALIDATION")
print("=" * 60)


# ============================================================
# 1. Candidate genes
# ============================================================

genes = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B",
    "PLA2G7",
    "SORBS1",
    "PMS2P2",
    "TAB2"
]


# ============================================================
# 2. Load GSE48350 training data
# ============================================================

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)


# ============================================================
# 3. Load GSE5281 external data
# ============================================================

external = pd.read_csv(
    "04_ML/GSE5281_RMA_genelevel_age_adjusted.csv",
    index_col=0
)

external_meta = pd.read_csv(
    "02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv"
)


print("\nTraining matrix:")
print(train.shape)

print("\nExternal matrix:")
print(external.shape)

print("\nExternal metadata:")
print(external_meta.shape)


# ============================================================
# 4. Convert external matrix
#
# GSE5281 currently:
# genes x samples
#
# We need:
# samples x genes
# ============================================================

external = external.T


# Extract clean GSM IDs

external.index = (
    external.index
    .str.replace(".CEL.gz", "", regex=False)
)


# ============================================================
# 5. Clean metadata GSM IDs
# ============================================================

external_meta["GSM"] = (
    external_meta["GSM"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 6. Verify external sample alignment
# ============================================================

missing_meta = set(external.index) - set(external_meta["GSM"])
missing_expr = set(external_meta["GSM"]) - set(external.index)

if missing_meta:
    raise ValueError(
        "External expression samples missing metadata: "
        + ", ".join(sorted(missing_meta))
    )

if missing_expr:
    raise ValueError(
        "External metadata samples missing expression: "
        + ", ".join(sorted(missing_expr))
    )


# Reorder metadata to expression order

external_meta = (
    external_meta
    .set_index("GSM")
    .loc[external.index]
    .reset_index()
)


print("\nExternal sample alignment: VERIFIED")

print("\nExternal diagnosis:")
print(external_meta["diagnosis"].value_counts())


# ============================================================
# 7. Verify candidate genes
# ============================================================

missing_train = [
    g for g in genes
    if g not in train.columns
]

missing_external = [
    g for g in genes
    if g not in external.columns
]

if missing_train:
    raise ValueError(
        "Genes missing from training data: "
        + ", ".join(missing_train)
    )

if missing_external:
    raise ValueError(
        "Genes missing from external data: "
        + ", ".join(missing_external)
    )


print("\nAll 7 genes present in both datasets.")


# ============================================================
# 8. Training expression
# ============================================================

X_train = train[genes].values

y_train = (
    train_meta["Diagnosis"]
    .map({
        "Control": 0,
        "AD": 1
    })
    .values
)


# ============================================================
# 9. External expression
# ============================================================

X_external = external[genes].values

y_external = (
    external_meta["diagnosis"]
    .map({
        "Control": 0,
        "AD": 1
    })
    .values
)


# ============================================================
# 10. Fit scaler ONLY on GSE48350
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_external_scaled = scaler.transform(
    X_external
)


# ============================================================
# 11. Fit locked LASSO model
# ============================================================

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


# ============================================================
# 12. External prediction
# ============================================================

external_probability = model.predict_proba(
    X_external_scaled
)[:, 1]

external_prediction = (
    external_probability >= 0.5
).astype(int)


# ============================================================
# 13. External validation metrics
# ============================================================

auc = roc_auc_score(
    y_external,
    external_probability
)

accuracy = accuracy_score(
    y_external,
    external_prediction
)

cm = confusion_matrix(
    y_external,
    external_prediction
)


tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan

specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan


print("\n" + "=" * 60)
print("GSE5281 EXTERNAL VALIDATION RESULTS")
print("=" * 60)

print(f"\nROC-AUC       : {auc:.4f}")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Sensitivity   : {sensitivity:.4f}")
print(f"Specificity   : {specificity:.4f}")

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(
    classification_report(
        y_external,
        external_prediction,
        target_names=["Control", "AD"],
        zero_division=0
    )
)


# ============================================================
# 14. Save predictions
# ============================================================

os.makedirs(
    "04_ML/External_Validation",
    exist_ok=True
)


prediction_table = external_meta.copy()

prediction_table["observed_class"] = np.where(
    y_external == 1,
    "AD",
    "Control"
)

prediction_table["predicted_probability_AD"] = (
    external_probability
)

prediction_table["predicted_class"] = np.where(
    external_prediction == 1,
    "AD",
    "Control"
)


prediction_table.to_csv(
    "04_ML/External_Validation/"
    "GSE5281_7gene_predictions.csv",
    index=False
)


# ============================================================
# 15. Save metrics
# ============================================================

metrics = pd.DataFrame({
    "metric": [
        "ROC_AUC",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "TN",
        "FP",
        "FN",
        "TP"
    ],
    "value": [
        auc,
        accuracy,
        sensitivity,
        specificity,
        tn,
        fp,
        fn,
        tp
    ]
})


metrics.to_csv(
    "04_ML/External_Validation/"
    "GSE5281_7gene_validation_metrics.csv",
    index=False
)


# ============================================================
# 16. Save model coefficients
# ============================================================

coef_table = pd.DataFrame({
    "gene": genes,
    "coefficient": model.coef_[0]
})


coef_table.to_csv(
    "04_ML/External_Validation/"
    "GSE5281_locked_model_coefficients.csv",
    index=False
)


print("\nSaved:")
print(
    "04_ML/External_Validation/"
    "GSE5281_7gene_predictions.csv"
)

print(
    "04_ML/External_Validation/"
    "GSE5281_7gene_validation_metrics.csv"
)

print(
    "04_ML/External_Validation/"
    "GSE5281_locked_model_coefficients.csv"
)


print("\n" + "=" * 60)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 60)
