import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
# FILES
# ============================================================

EXPR_FILE = "04_ML/GSE48350_hippocampus_15gene_signature.csv"
META_FILE = "02_Metadata/GSE48350_metadata_grouped.csv"

OUT_METRICS = "08_Results/GSE48350_15gene_model_metrics.csv"
OUT_ROC = "08_Results/GSE48350_15gene_model_ROC.csv"

# ============================================================
# LOAD EXPRESSION
# ============================================================

expr = pd.read_csv(EXPR_FILE, index_col=0)

print("Expression matrix:")
print(expr.shape)

# genes × samples → samples × genes
X = expr.T

# ============================================================
# LOAD METADATA
# ============================================================

meta = pd.read_csv(META_FILE)

print("\nMetadata columns:")
print(meta.columns.tolist())

# Find GSM column
gsm_col = None
for col in meta.columns:
    if col.lower() in ["gsm", "geo_accession", "sample"]:
        gsm_col = col
        break

if gsm_col is None:
    raise ValueError("Could not identify GSM column in metadata.")

# Find diagnosis column
diag_col = None
for col in meta.columns:
    if col.lower() in ["diagnosis", "group", "status", "condition"]:
        diag_col = col
        break

if diag_col is None:
    raise ValueError("Could not identify diagnosis column in metadata.")

meta = meta.set_index(gsm_col)

# ============================================================
# ALIGN SAMPLES
# ============================================================

common = X.index.intersection(meta.index)

print("\nCommon samples:", len(common))

X = X.loc[common]
y_raw = meta.loc[common, diag_col]

print("\nDiagnosis values:")
print(y_raw.value_counts())

# Convert diagnosis to binary
y = y_raw.map({
    "Control": 0,
    "AD": 1,
    "control": 0,
    "ad": 1
})

if y.isna().any():
    print("\nUnrecognized diagnosis values:")
    print(y_raw[y.isna()].unique())
    raise ValueError("Diagnosis mapping failed.")

# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
    max_features="sqrt"
)

# ============================================================
# 5-FOLD CROSS-VALIDATED PREDICTIONS
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

probabilities = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

predictions = (probabilities >= 0.5).astype(int)

# ============================================================
# METRICS
# ============================================================

auc = roc_auc_score(y, probabilities)
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, zero_division=0)
recall = recall_score(y, predictions, zero_division=0)
f1 = f1_score(y, predictions, zero_division=0)

cm = confusion_matrix(y, predictions)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) else 0

print("\n" + "=" * 60)
print("15-GENE SIGNATURE — 5-FOLD CV")
print("=" * 60)

print(f"ROC-AUC:     {auc:.4f}")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-score:    {f1:.4f}")
print(f"Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")

print("\nConfusion matrix:")
print(cm)

# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "model": ["Random Forest"],
    "features": [15],
    "cv_folds": [5],
    "ROC_AUC": [auc],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall_Sensitivity": [recall],
    "Specificity": [specificity],
    "F1": [f1],
    "TN": [tn],
    "FP": [fp],
    "FN": [fn],
    "TP": [tp]
})

metrics.to_csv(OUT_METRICS, index=False)

# ============================================================
# ROC DATA
# ============================================================

fpr, tpr, thresholds = roc_curve(y, probabilities)

roc_df = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr,
    "threshold": thresholds
})

roc_df.to_csv(OUT_ROC, index=False)

# ============================================================
# FIT FINAL MODEL ON ALL 23 SAMPLES
# ============================================================

model.fit(X, y)

print("\nFinal model fitted on all samples.")

print("\nSaved:")
print(OUT_METRICS)
print(OUT_ROC)

print("\nDONE.")

