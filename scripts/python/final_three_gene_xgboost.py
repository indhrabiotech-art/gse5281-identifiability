import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
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
# LABELS
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
# FEATURES — LOCKED THREE GENES ONLY
# ============================================================

X_train = train[GENES].copy()
X_test = test[GENES].copy()
X_external = external[GENES].copy()

assert list(X_train.columns) == GENES
assert list(X_test.columns) == GENES
assert list(X_external.columns) == GENES


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=1.0,
    reg_alpha=0.5,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    device="cpu",
    random_state=RANDOM_STATE,
    n_jobs=1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

pred_train = model.predict_proba(X_train)[:, 1]
pred_test = model.predict_proba(X_test)[:, 1]
pred_external = model.predict_proba(X_external)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, y, pred):

    return {
        "Dataset": name,
        "N": len(y),
        "AD": int(y.sum()),
        "Control": int((y == 0).sum()),
        "ROC_AUC": roc_auc_score(y, pred),
        "PR_AUC": average_precision_score(y, pred)
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
# XGBOOST FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Gene": GENES,
    "XGB_Gain_Importance": [
        model.get_booster()
        .get_score(importance_type="gain")
        .get(gene, 0.0)
        for gene in GENES
    ]
})

importance["XGB_Gain_Importance_Normalized"] = (
    importance["XGB_Gain_Importance"]
    /
    importance["XGB_Gain_Importance"].sum()
)

importance = importance.sort_values(
    "XGB_Gain_Importance_Normalized",
    ascending=False
).reset_index(drop=True)


# ============================================================
# PRINT
# ============================================================

print("=" * 75)
print("FINAL THREE-GENE XGBOOST")
print("=" * 75)

print("\nGenes:")
for gene in GENES:
    print(" ", gene)

print("\nXGBoost parameters:")
print("n_estimators:", model.n_estimators)
print("max_depth:", model.max_depth)
print("learning_rate:", model.learning_rate)
print("min_child_weight:", model.min_child_weight)
print("subsample:", model.subsample)
print("reg_alpha:", model.reg_alpha)
print("reg_lambda:", model.reg_lambda)
print("device: CPU")

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
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 75)

print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    f"{OUTDIR}/FINAL_THREE_GENE_XGBOOST_PERFORMANCE.csv",
    index=False
)

importance.to_csv(
    f"{OUTDIR}/FINAL_THREE_GENE_XGBOOST_IMPORTANCE.csv",
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

    "XGB_predicted_probability_AD": np.concatenate([
        pred_train,
        pred_test,
        pred_external
    ])
})

predictions.to_csv(
    f"{OUTDIR}/FINAL_THREE_GENE_XGBOOST_PREDICTIONS.csv",
    index=False
)


print("\nSaved:")
print(
    f"{OUTDIR}/FINAL_THREE_GENE_XGBOOST_PERFORMANCE.csv"
)
print(
    f"{OUTDIR}/FINAL_THREE_GENE_XGBOOST_IMPORTANCE.csv"
)
print(
    f"{OUTDIR}/FINAL_THREE_GENE_XGBOOST_PREDICTIONS.csv"
)

print("\n" + "=" * 75)
print("XGBOOST ANALYSIS COMPLETE")
print("=" * 75)
