import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("FINAL 3-GENE LASSO MODEL")
print("GSE48350 → GSE48350 TEST → GSE5281")
print("=" * 60)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

C_VALUE = 0.3

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
# Load held-out GSE48350 test
# ------------------------------------------------------------

test = pd.read_csv(
    "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
    index_col=0
)

test_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_test_meta.csv"
)

# ------------------------------------------------------------
# Load GSE5281 external validation
# ------------------------------------------------------------

external = pd.read_csv(
    "04_ML/External_Validation/GSE5281_LASSO_candidate_matrix.csv",
    index_col=0
)

external_meta = pd.read_csv(
    "04_ML/External_Validation/GSE5281_validation_metadata.csv",
    index_col=0
)

# ------------------------------------------------------------
# Check genes
# ------------------------------------------------------------

for gene in GENES:

    if gene not in train.columns:
        raise ValueError(f"{gene} missing from training data.")

    if gene not in test.columns:
        raise ValueError(f"{gene} missing from GSE48350 test.")

    if gene not in external.columns:
        raise ValueError(f"{gene} missing from GSE5281.")

# ------------------------------------------------------------
# Extract features
# ------------------------------------------------------------

X_train = train[GENES].values
X_test = test[GENES].values
X_external = external[GENES].values

# ------------------------------------------------------------
# Encode diagnosis
# AD = 1
# Control = 0
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
# Verify alignment
# ------------------------------------------------------------

if list(train.index) != list(train_meta["index"]):
    raise ValueError("Training alignment failed.")

if list(test.index) != list(test_meta["index"]):
    raise ValueError("Test alignment failed.")

if list(external.index) != list(external_meta.index):
    raise ValueError("External validation alignment failed.")

# ------------------------------------------------------------
# Final model
# ------------------------------------------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "lasso",
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C_VALUE,
            max_iter=5000,
            random_state=42
        )
    )
])

print("\nTraining final model...")
print("Genes:", GENES)
print("C:", C_VALUE)
print("Training samples:", len(y_train))

model.fit(X_train, y_train)

# ------------------------------------------------------------
# Model coefficients
# ------------------------------------------------------------

lasso = model.named_steps["lasso"]

coefficients = pd.DataFrame({
    "gene": GENES,
    "coefficient": lasso.coef_[0]
})

print("\nFinal model coefficients:")
print(coefficients.to_string(index=False))

print("\nIntercept:")
print(lasso.intercept_[0])

# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------

def evaluate_dataset(name, X, y):

    probability = model.predict_proba(X)[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    auc = roc_auc_score(y, probability)
    accuracy = accuracy_score(y, prediction)

    cm = confusion_matrix(
        y,
        prediction,
        labels=[0, 1]
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
    print(name)
    print("=" * 60)

    print("Samples:", len(y))
    print("AD:", int(sum(y == 1)))
    print("Control:", int(sum(y == 0)))

    print(f"AUC:         {auc:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    return {
        "dataset": name,
        "n": len(y),
        "AD": int(sum(y == 1)),
        "Control": int(sum(y == 0)),
        "AUC": auc,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }, probability, prediction

# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

results = []

r_train, p_train, pred_train = evaluate_dataset(
    "GSE48350 TRAINING",
    X_train,
    y_train
)

r_test, p_test, pred_test = evaluate_dataset(
    "GSE48350 HELD-OUT TEST",
    X_test,
    y_test
)

r_external, p_external, pred_external = evaluate_dataset(
    "GSE5281 EXTERNAL VALIDATION",
    X_external,
    y_external
)

results.extend([
    r_train,
    r_test,
    r_external
])

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

os.makedirs(
    "04_ML/Final_Model",
    exist_ok=True
)

coefficients.to_csv(
    "04_ML/Final_Model/GSE48350_final_3gene_coefficients.csv",
    index=False
)

pd.DataFrame(results).to_csv(
    "04_ML/Final_Model/GSE48350_final_3gene_validation_results.csv",
    index=False
)

# ------------------------------------------------------------
# Save prediction scores
# ------------------------------------------------------------

pd.DataFrame({
    "GSM": train_meta["index"],
    "Diagnosis": train_meta["Diagnosis"],
    "Predicted_probability_AD": p_train,
    "Predicted_class": pred_train
}).to_csv(
    "04_ML/Final_Model/GSE48350_training_predictions.csv",
    index=False
)

pd.DataFrame({
    "GSM": test_meta["index"],
    "Diagnosis": test_meta["Diagnosis"],
    "Predicted_probability_AD": p_test,
    "Predicted_class": pred_test
}).to_csv(
    "04_ML/Final_Model/GSE48350_test_predictions.csv",
    index=False
)

pd.DataFrame({
    "GSM": external_meta.index,
    "Diagnosis": external_meta["diagnosis"],
    "Predicted_probability_AD": p_external,
    "Predicted_class": pred_external
}).to_csv(
    "04_ML/Final_Model/GSE5281_external_predictions.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPLETE")
print("=" * 60)

print("\nSaved:")
print("04_ML/Final_Model/GSE48350_final_3gene_coefficients.csv")
print("04_ML/Final_Model/GSE48350_final_3gene_validation_results.csv")
print("04_ML/Final_Model/GSE48350_training_predictions.csv")
print("04_ML/Final_Model/GSE48350_test_predictions.csv")
print("04_ML/Final_Model/GSE5281_external_predictions.csv")

