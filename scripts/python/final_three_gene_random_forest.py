import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

OUTDIR = "04_ML/Final_Model/Validation"
os.makedirs(OUTDIR, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

test = pd.read_csv(
    "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
    index_col=0
)

test_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_test_meta.csv"
)

external = pd.read_csv(
    "04_ML/Final_Characterization/"
    "GSE5281_3gene_predictions_final.csv"
)


# ============================================================
# PREPARE LABELS
# ============================================================

y_train = (
    train_meta["Diagnosis"]
    .astype(str)
    .str.strip()
    .eq("AD")
    .astype(int)
    .values
)

y_test = (
    test_meta["Diagnosis"]
    .astype(str)
    .str.strip()
    .eq("AD")
    .astype(int)
    .values
)

y_external = (
    external["Diagnosis"]
    .astype(str)
    .str.strip()
    .eq("AD")
    .astype(int)
    .values
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X_train = train[GENES].copy()
X_test = test[GENES].copy()
X_external = external[GENES].copy()

assert list(X_train.columns) == GENES
assert list(X_test.columns) == GENES
assert list(X_external.columns) == GENES


# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

pred_train = rf.predict_proba(X_train)[:, 1]
pred_test = rf.predict_proba(X_test)[:, 1]
pred_external = rf.predict_proba(X_external)[:, 1]


# ============================================================
# METRICS
# ============================================================

def evaluate(name, y, pred):

    roc = roc_auc_score(y, pred)

    pr = average_precision_score(y, pred)

    return {
        "Dataset": name,
        "N": len(y),
        "AD": int(y.sum()),
        "Control": int((y == 0).sum()),
        "ROC_AUC": roc,
        "PR_AUC": pr
    }


results = pd.DataFrame([
    evaluate(
        "GSE48350_Training",
        y_train,
        pred_train
    ),

    evaluate(
        "GSE48350_HeldOut_Test",
        y_test,
        pred_test
    ),

    evaluate(
        "GSE5281_External",
        y_external,
        pred_external
    )
])


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Gene": GENES,
    "RF_Gini_Importance": rf.feature_importances_
})

importance = importance.sort_values(
    "RF_Gini_Importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 75)
print("FINAL THREE-GENE RANDOM FOREST")
print("=" * 75)

print("\nGenes:")
for gene in GENES:
    print(" ", gene)

print("\nRandom Forest parameters:")
print("n_estimators:", rf.n_estimators)
print("max_depth:", rf.max_depth)
print("min_samples_leaf:", rf.min_samples_leaf)
print("max_features:", rf.max_features)
print("class_weight:", rf.class_weight)

print("\n" + "=" * 75)
print("MODEL PERFORMANCE")
print("=" * 75)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 75)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 75)

print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


results.to_csv(
    f"{OUTDIR}/FINAL_THREE_GENE_RANDOM_FOREST_PERFORMANCE.csv",
    index=False
)

importance.to_csv(
    f"{OUTDIR}/FINAL_THREE_GENE_RANDOM_FOREST_IMPORTANCE.csv",
    index=False
)

predictions = pd.DataFrame({
    "Dataset": (
        ["GSE48350_Training"] * len(y_train)
        +
        ["GSE48350_HeldOut_Test"] * len(y_test)
        +
        ["GSE5281_External"] * len(y_external)
    ),

    "Observed_AD": np.concatenate([
        y_train,
        y_test,
        y_external
    ]),

    "RF_predicted_probability_AD": np.concatenate([
        pred_train,
        pred_test,
        pred_external
    ])
})

predictions.to_csv(
    f"{OUTDIR}/FINAL_THREE_GENE_RANDOM_FOREST_PREDICTIONS.csv",
    index=False
)

print("\nSaved:")
print(
    f"{OUTDIR}/FINAL_THREE_GENE_RANDOM_FOREST_PERFORMANCE.csv"
)
print(
    f"{OUTDIR}/FINAL_THREE_GENE_RANDOM_FOREST_IMPORTANCE.csv"
)
print(
    f"{OUTDIR}/FINAL_THREE_GENE_RANDOM_FOREST_PREDICTIONS.csv"
)

print("\n" + "=" * 75)
print("RANDOM FOREST ANALYSIS COMPLETE")
print("=" * 75)
