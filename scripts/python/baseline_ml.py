"""
Day 5: Baseline ML (Logistic Regression + Random Forest)

Reads the FROZEN train/test split produced by qc_filter_split.R.
Uses train_meta.csv / test_meta.csv (GSM lists) to subset the full
filtered matrix - avoids needing to re-export .rds files from R.

IMPORTANT: the test set is touched exactly ONCE, at the very end,
for final evaluation only. All model selection / tuning happens via
cross-validation on the training set alone (Rule 1/Rule 2).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)

RANDOM_STATE = 42  # matches the seed used for the R train/test split

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
full_matrix = pd.read_csv("04_ML/GSE48350_hippocampus_ML_matrix_AGEADJUSTED_v2.csv", index_col=0)
train_meta  = pd.read_csv("04_ML/GSE48350_hippocampus_train_meta.csv")
test_meta   = pd.read_csv("04_ML/GSE48350_hippocampus_test_meta.csv")

print("Full matrix shape (genes x samples):", full_matrix.shape)
print("Train samples:", len(train_meta), " Test samples:", len(test_meta))

# Transpose so rows = samples, columns = genes (sklearn convention)
full_matrix_T = full_matrix.T

X_train = full_matrix_T.loc[train_meta["GSM"]]
X_test  = full_matrix_T.loc[test_meta["GSM"]]

y_train = (train_meta["Diagnosis"] == "AD").astype(int).values
y_test  = (test_meta["Diagnosis"] == "AD").astype(int).values

print("\nTrain class balance - AD:", y_train.sum(), " Control:", len(y_train) - y_train.sum())
print("Test class balance  - AD:", y_test.sum(), " Control:", len(y_test) - y_test.sum())

# ------------------------------------------------------------
# Standardize features (fit on TRAIN ONLY, applied to both)
# ------------------------------------------------------------
# WHY fit-on-train-only: fitting the scaler on the full dataset would
# leak test-set distribution information into training - same leakage
# principle as feature selection (Rule 1), applied to preprocessing.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ------------------------------------------------------------
# Cross-validation on TRAINING SET ONLY
# ------------------------------------------------------------
# WHY 5-fold and not more: with only 46 training samples (14 AD/32
# Control), each fold already has ~9 samples - higher fold counts
# would leave very few AD cases per validation fold, making the
# per-fold AUC unstable. 5-fold is a reasonable, defensible choice
# given this sample size.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n" + "=" * 60)
print("CROSS-VALIDATION ON TRAINING SET (46 samples, 5-fold)")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        penalty="l2", C=0.1, max_iter=5000, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=2,
        random_state=RANDOM_STATE
    ),
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    cv_results[name] = scores
    print(f"\n{name}:")
    print(f"  ROC-AUC per fold: {np.round(scores, 3)}")
    print(f"  Mean ROC-AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")

# ------------------------------------------------------------
# Fit final models on FULL training set
# ------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# FINAL EVALUATION ON LOCKED TEST SET (touched once, here only)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL EVALUATION ON LOCKED TEST SET (16 samples)")
print("=" * 60)
print("NOTE: test set is small (5 AD / 11 Control) - treat these")
print("numbers as a confirmatory check, not the primary evidence.")
print("The cross-validation results above are the more stable signal.\n")

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n--- {name} ---")
    print("ROC-AUC:  ", round(roc_auc_score(y_test, y_prob), 3))
    print("Accuracy: ", round(accuracy_score(y_test, y_pred), 3))
    print("F1:       ", round(f1_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
    print("Recall:   ", round(recall_score(y_test, y_pred, zero_division=0), 3))
    print("\nConfusion matrix (rows=true, cols=pred; order=[Control, AD]):")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["Control", "AD"], zero_division=0))

print("\nDone. Do NOT re-run this script with different hyperparameters")
print("to chase a better test-set number - that reintroduces the exact")
print("leakage the frozen split was built to prevent. Iterate on the")
print("cross-validation results instead; touch the test set only once")
print("more, at the very end, for the final reported number.")
