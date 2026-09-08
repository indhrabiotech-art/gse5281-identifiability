import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("GSE48350 → GSE5281 COHORT-WISE Z-SCORE VALIDATION")
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
# LOAD TRAINING DATA
# ------------------------------------------------------------

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

# ------------------------------------------------------------
# LOAD AGE-ADJUSTED EXTERNAL DATA
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
# LABELS
# ------------------------------------------------------------

y_train = (
    train_meta["Diagnosis"]
    .map({
        "Control": 0,
        "AD": 1
    })
    .values
)

y_external = (
    external_meta["diagnosis"]
    .map({
        "Control": 0,
        "AD": 1
    })
    .values
)

# ------------------------------------------------------------
# CHECK GENES
# ------------------------------------------------------------

missing_train = [g for g in genes if g not in train.columns]
missing_external = [g for g in genes if g not in external.columns]

if missing_train:
    raise ValueError(
        "Missing training genes: " +
        ", ".join(missing_train)
    )

if missing_external:
    raise ValueError(
        "Missing external genes: " +
        ", ".join(missing_external)
    )

print("\nTraining:", train.shape)
print("External:", external.shape)
print("Candidate genes:", len(genes))

# ------------------------------------------------------------
# EXTRACT 7 GENES
# ------------------------------------------------------------

X_train_raw = train[genes].copy()
X_external_raw = external[genes].copy()

# ------------------------------------------------------------
# COHORT-WISE Z-SCORE
#
# IMPORTANT:
# No diagnosis information is used here.
#
# Each dataset is standardized independently.
# ------------------------------------------------------------

train_scaler = StandardScaler()

X_train = train_scaler.fit_transform(
    X_train_raw
)

external_scaler = StandardScaler()

X_external = external_scaler.fit_transform(
    X_external_raw
)

# ------------------------------------------------------------
# CHECK DISTRIBUTIONS
# ------------------------------------------------------------

print("\nTraining standardized means:")
print(np.mean(X_train, axis=0))

print("\nExternal standardized means:")
print(np.mean(X_external, axis=0))

print("\nTraining standardized SD:")
print(np.std(X_train, axis=0))

print("\nExternal standardized SD:")
print(np.std(X_external, axis=0))

# ------------------------------------------------------------
# LOCKED LASSO MODEL
# ------------------------------------------------------------

model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=0.3,
    max_iter=10000,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ------------------------------------------------------------
# EXTERNAL PREDICTION
# ------------------------------------------------------------

probability = model.predict_proba(
    X_external
)[:, 1]

prediction = (
    probability >= 0.5
).astype(int)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

auc = roc_auc_score(
    y_external,
    probability
)

accuracy = accuracy_score(
    y_external,
    prediction
)

cm = confusion_matrix(
    y_external,
    prediction
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

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COHORT-WISE Z-SCORE EXTERNAL VALIDATION")
print("=" * 70)

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
        prediction,
        target_names=[
            "Control",
            "AD"
        ],
        zero_division=0
    )
)

# ------------------------------------------------------------
# PROBABILITY DISTRIBUTION
# ------------------------------------------------------------

print("\nProbability distribution:")
print("Minimum :", probability.min())
print("Maximum :", probability.max())
print("Mean    :", probability.mean())
print("Median  :", np.median(probability))

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

out = pd.DataFrame({
    "GSM": external.index,
    "Diagnosis": external_meta["diagnosis"].values,
    "AD_probability": probability,
    "Predicted_class": np.where(
        prediction == 1,
        "AD",
        "Control"
    )
})

out.to_csv(
    "04_ML/External_Validation/"
    "GSE5281_7gene_cohort_zscore_predictions.csv",
    index=False
)

metrics = pd.DataFrame([{
    "method": "cohort_wise_zscore",
    "AUC": auc,
    "accuracy": accuracy,
    "sensitivity": sensitivity,
    "specificity": specificity
}])

metrics.to_csv(
    "04_ML/External_Validation/"
    "GSE5281_7gene_cohort_zscore_metrics.csv",
    index=False
)

print("\nSaved:")
print(
    "04_ML/External_Validation/"
    "GSE5281_7gene_cohort_zscore_predictions.csv"
)

print(
    "04_ML/External_Validation/"
    "GSE5281_7gene_cohort_zscore_metrics.csv"
)

print("\n" + "=" * 70)
print("COHORT-WISE Z-SCORE VALIDATION COMPLETE")
print("=" * 70)
