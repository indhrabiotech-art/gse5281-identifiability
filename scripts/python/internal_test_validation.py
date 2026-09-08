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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


print("=" * 60)
print("GSE48350 — INTERNAL TEST VALIDATION")
print("=" * 60)


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
# Load AGE-ADJUSTED training data
# ------------------------------------------------------------

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)


# ------------------------------------------------------------
# Load AGE-ADJUSTED internal test data
# ------------------------------------------------------------

test = pd.read_csv(
    "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
    index_col=0
)

test_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_test_meta.csv"
)


# ------------------------------------------------------------
# Alignment checks
# ------------------------------------------------------------

if list(train.index) != list(train_meta["index"]):
    raise ValueError("Training alignment failed.")

if list(test.index) != list(test_meta["index"]):
    raise ValueError("Test alignment failed.")


if list(train.columns) != list(test.columns):
    raise ValueError("Training/test gene alignment failed.")


print("\nTraining:", train.shape)
print("Test:", test.shape)


# ------------------------------------------------------------
# Select 7 genes
# ------------------------------------------------------------

X_train = train[genes].values
X_test = test[genes].values

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


print("\nTraining diagnosis:")
print(train_meta["Diagnosis"].value_counts())

print("\nTest diagnosis:")
print(test_meta["Diagnosis"].value_counts())


# ------------------------------------------------------------
# Fit scaler ONLY on training data
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# Fit locked LASSO model
# ------------------------------------------------------------

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
# Predict untouched test set
# ------------------------------------------------------------

prob = model.predict_proba(X_test_scaled)[:, 1]

pred = (prob >= 0.5).astype(int)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

auc = roc_auc_score(
    y_test,
    prob
)

accuracy = accuracy_score(
    y_test,
    pred
)

cm = confusion_matrix(
    y_test,
    pred
)

tn, fp, fn, tp = cm.ravel()

sensitivity = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else np.nan
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else np.nan
)


print("\n" + "=" * 60)
print("INTERNAL TEST RESULTS")
print("=" * 60)

print(f"\nROC-AUC     : {auc:.4f}")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"Specificity : {specificity:.4f}")

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        pred,
        target_names=["Control", "AD"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# Individual predictions
# ------------------------------------------------------------

results = test_meta.copy()

results["predicted_probability_AD"] = prob

results["predicted_class"] = np.where(
    pred == 1,
    "AD",
    "Control"
)

print("\nIndividual test predictions:")
print(
    results[
        [
            "index",
            "Diagnosis",
            "predicted_probability_AD",
            "predicted_class"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

os.makedirs(
    "04_ML/Internal_Test",
    exist_ok=True
)

results.to_csv(
    "04_ML/Internal_Test/"
    "GSE48350_7gene_internal_test_predictions.csv",
    index=False
)

pd.DataFrame({
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
}).to_csv(
    "04_ML/Internal_Test/"
    "GSE48350_7gene_internal_test_metrics.csv",
    index=False
)


print("\nSaved internal validation results.")

print("=" * 60)
print("INTERNAL TEST VALIDATION COMPLETE")
print("=" * 60)
