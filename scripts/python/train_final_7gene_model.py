import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


print("=" * 60)
print("GSE48350 — FINAL 7-GENE CLASSIFIER")
print("=" * 60)


# ------------------------------------------------------------
# Candidate genes from LASSO stability selection
# ------------------------------------------------------------

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
# Load training expression
# ------------------------------------------------------------

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)


# ------------------------------------------------------------
# Alignment
# ------------------------------------------------------------

if list(train.index) != list(meta["index"]):
    raise ValueError("Sample alignment failed.")


# ------------------------------------------------------------
# Check genes
# ------------------------------------------------------------

missing = [g for g in genes if g not in train.columns]

if missing:
    raise ValueError(
        "Candidate genes missing: " +
        ", ".join(missing)
    )


X = train[genes].copy()

y = (
    meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .values
)


print("\nTraining samples:", X.shape[0])
print("Candidate genes:", X.shape[1])

print("\nDiagnosis:")
print(meta["Diagnosis"].value_counts())


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
            C=0.3,
            max_iter=10000,
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# Fit ONLY on GSE48350 training cohort
# ------------------------------------------------------------

model.fit(X, y)


# ------------------------------------------------------------
# Extract coefficients
# ------------------------------------------------------------

coef = model.named_steps["lasso"].coef_[0]

coef_table = pd.DataFrame({
    "gene": genes,
    "coefficient": coef,
    "abs_coefficient": np.abs(coef)
})

coef_table = coef_table.sort_values(
    "abs_coefficient",
    ascending=False
)


print("\n============================================================")
print("FINAL 7-GENE MODEL COEFFICIENTS")
print("============================================================")

print(
    coef_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\nNon-zero coefficients:")

nonzero = coef_table[
    coef_table["coefficient"] != 0
]

print(nonzero.to_string(index=False))


# ------------------------------------------------------------
# Save model coefficients
# ------------------------------------------------------------

os.makedirs("04_ML/Final_Model", exist_ok=True)

coef_table.to_csv(
    "04_ML/Final_Model/GSE48350_final_7gene_coefficients.csv",
    index=False
)


# ------------------------------------------------------------
# Save training predictions
# ------------------------------------------------------------

prob = model.predict_proba(X)[:, 1]
pred = model.predict(X)

pred_table = meta.copy()

pred_table["predicted_probability_AD"] = prob
pred_table["predicted_class"] = np.where(
    pred == 1,
    "AD",
    "Control"
)

pred_table.to_csv(
    "04_ML/Final_Model/GSE48350_final_7gene_training_predictions.csv",
    index=False
)


print("\nSaved:")
print(
    "04_ML/Final_Model/"
    "GSE48350_final_7gene_coefficients.csv"
)

print(
    "04_ML/Final_Model/"
    "GSE48350_final_7gene_training_predictions.csv"
)


print("\n============================================================")
print("FINAL 7-GENE MODEL COMPLETE")
print("============================================================")
