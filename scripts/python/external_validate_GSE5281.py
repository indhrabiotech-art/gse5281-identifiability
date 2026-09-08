import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

GENES = [
    "SLC25A46",
    "FAM170A",
    "CD5",
    "LINC02987",
    "RAE1",
    "HSPA12A",
    "ANKIB1",
    "BTK",
    "ZNF621",
    "SPDEF",
    "KCNJ5",
    "TRIM49",
    "PART1",
    "MEFV",
    "P3H3"
]

TRAIN_EXPR = (
    "04_ML/GSE48350_hippocampus_15gene_signature.csv"
)

TRAIN_META = (
    "04_ML/GSE48350_hippocampus_train_meta.csv"
)

VALID_EXPR = (
    "06_Validation/GSE5281_hippocampus_15gene_ageadjusted.csv"
)

VALID_META = (
    "02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv"
)

OUT_DIR = "06_Validation"
RESULT_DIR = "08_Results"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD GSE48350 DISCOVERY DATA
# ============================================================

train_expr = pd.read_csv(
    TRAIN_EXPR,
    index_col=0
)

train_meta = pd.read_csv(TRAIN_META)

print("=" * 70)
print("FROZEN 15-GENE MODEL — GSE48350 → GSE5281")
print("=" * 70)

print("\nGSE48350 signature matrix:", train_expr.shape)

# Keep only the frozen 15 genes and correct order
train_expr = train_expr.loc[GENES]

# samples × genes
X_all = train_expr.T

# Restrict to the ORIGINAL GSE48350 TRAINING samples only
common_train = [
    gsm for gsm in train_meta["GSM"]
    if gsm in X_all.index
]

X_train = X_all.loc[common_train]

meta_train = train_meta.set_index("GSM").loc[common_train]

y_train = (
    meta_train["Diagnosis"] == "AD"
).astype(int).values

print("Training samples:", X_train.shape[0])
print("Training genes:", X_train.shape[1])

print(
    "Training classes:",
    dict(zip(
        ["Control", "AD"],
        [
            int((y_train == 0).sum()),
            int((y_train == 1).sum())
        ]
    ))
)


# ============================================================
# LOAD GSE5281 EXTERNAL VALIDATION DATA
# ============================================================

valid_expr = pd.read_csv(
    VALID_EXPR,
    index_col=0
)

valid_meta = pd.read_csv(VALID_META)

print("\nGSE5281 validation matrix:", valid_expr.shape)

# Check that every frozen gene exists
missing_train = [
    g for g in GENES
    if g not in train_expr.index
]

missing_valid = [
    g for g in GENES
    if g not in valid_expr.index
]

if missing_train:
    raise ValueError(
        f"Missing genes in GSE48350: {missing_train}"
    )

if missing_valid:
    raise ValueError(
        f"Missing genes in GSE5281: {missing_valid}"
    )

valid_expr = valid_expr.loc[GENES]

# samples × genes
X_valid = valid_expr.T

valid_meta = valid_meta.set_index("GSM")

common_valid = X_valid.index.intersection(valid_meta.index)

X_valid = X_valid.loc[common_valid]
meta_valid = valid_meta.loc[common_valid]

y_valid = (
    meta_valid["diagnosis"] == "AD"
).astype(int).values

print("Validation samples:", X_valid.shape[0])
print("Validation genes:", X_valid.shape[1])

print(
    "Validation classes:",
    dict(zip(
        ["Control", "AD"],
        [
            int((y_valid == 0).sum()),
            int((y_valid == 1).sum())
        ]
    ))
)


# ============================================================
# STANDARDIZATION
#
# CRITICAL:
# FIT ONLY ON GSE48350 TRAINING DATA.
#
# DO NOT fit scaler on GSE5281.
# ============================================================

print("\n" + "=" * 70)
print("FITTING FROZEN TRAINING SCALER")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_valid_scaled = scaler.transform(X_valid)

print("Scaler fitted on GSE48350 training samples only.")
print("GSE5281 transformed using frozen GSE48350 parameters.")


# ============================================================
# RANDOM FOREST
#
# Same model configuration used by baseline_ml.py
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE
)

print("\n" + "=" * 70)
print("TRAINING FINAL FROZEN RANDOM FOREST")
print("=" * 70)

model.fit(
    X_train_scaled,
    y_train
)

print("Model trained on GSE48350 TRAINING SET only.")


# ============================================================
# EXTERNAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("GSE5281 EXTERNAL VALIDATION")
print("=" * 70)

probabilities = model.predict_proba(
    X_valid_scaled
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

auc = roc_auc_score(
    y_valid,
    probabilities
)

accuracy = accuracy_score(
    y_valid,
    predictions
)

precision = precision_score(
    y_valid,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_valid,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_valid,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_valid,
    predictions
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

print(f"\nROC-AUC:     {auc:.4f}")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score:    {f1:.4f}")

print("\nConfusion matrix:")
print(cm)


# ============================================================
# SAMPLE-LEVEL PREDICTIONS
# ============================================================

prediction_df = meta_valid.copy()

prediction_df["true_label"] = y_valid
prediction_df["predicted_label"] = predictions
prediction_df["predicted_probability_AD"] = probabilities

prediction_df["true_diagnosis"] = np.where(
    y_valid == 1,
    "AD",
    "Control"
)

prediction_df["predicted_diagnosis"] = np.where(
    predictions == 1,
    "AD",
    "Control"
)

prediction_df.to_csv(
    f"{OUT_DIR}/GSE5281_15gene_predictions.csv"
)


# ============================================================
# METRICS TABLE
# ============================================================

metrics_df = pd.DataFrame({
    "dataset": ["GSE5281"],
    "model": ["Random Forest"],
    "features": [15],
    "training_dataset": ["GSE48350"],
    "validation_dataset": ["GSE5281"],
    "ROC_AUC": [auc],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Sensitivity": [recall],
    "Specificity": [specificity],
    "F1": [f1],
    "TN": [tn],
    "FP": [fp],
    "FN": [fn],
    "TP": [tp]
})

metrics_file = (
    f"{RESULT_DIR}/GSE5281_15gene_external_validation_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# ROC DATA
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_valid,
    probabilities
)

roc_df = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr,
    "threshold": thresholds
})

roc_file = (
    f"{RESULT_DIR}/GSE5281_15gene_external_validation_ROC.csv"
)

roc_df.to_csv(
    roc_file,
    index=False
)


# ============================================================
# SAVE FROZEN SCALER PARAMETERS
# ============================================================

scaler_df = pd.DataFrame({
    "gene": GENES,
    "training_mean": scaler.mean_,
    "training_sd": scaler.scale_
})

scaler_file = (
    f"{OUT_DIR}/GSE48350_15gene_frozen_scaler_parameters.csv"
)

scaler_df.to_csv(
    scaler_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 70)

print("\nTraining:")
print("  Dataset: GSE48350")
print("  Samples:", len(X_train))
print("  Genes: 15")

print("\nValidation:")
print("  Dataset: GSE5281")
print("  Samples:", len(X_valid))
print("  Genes: 15")

print("\nOutputs:")
print(" ", f"{OUT_DIR}/GSE5281_15gene_predictions.csv")
print(" ", metrics_file)
print(" ", roc_file)
print(" ", scaler_file)

print("\nIMPORTANT:")
print("  GSE5281 was NOT used for model fitting.")
print("  GSE5281 was NOT used to fit the scaler.")
print("  The model was trained only on GSE48350 training samples.")

print("\nDONE.")
