import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path.home() / "project_ml"

FINAL = ROOT / "04_ML/Final_Model"

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

print("=" * 90)
print("CANONICAL FINAL 3-GENE MODEL AUDIT")
print("=" * 90)

# ------------------------------------------------------------
# Canonical coefficients
# ------------------------------------------------------------

coef = pd.read_csv(
    FINAL / "GSE48350_final_3gene_coefficients.csv"
)

print("\n===== COEFFICIENTS =====")
print(coef.to_string(index=False))

assert coef["gene"].tolist() == GENES

# ------------------------------------------------------------
# Canonical validation
# ------------------------------------------------------------

validation = pd.read_csv(
    FINAL / "GSE48350_final_3gene_validation_results.csv"
)

print("\n===== CANONICAL PERFORMANCE =====")
print(validation.to_string(index=False))

# ------------------------------------------------------------
# Canonical predictions
# ------------------------------------------------------------

files = {
    "GSE48350_training":
        FINAL / "GSE48350_training_predictions.csv",

    "GSE48350_test":
        FINAL / "GSE48350_test_predictions.csv",

    "GSE5281_external":
        FINAL / "GSE5281_external_predictions.csv"
}

for name, path in files.items():

    df = pd.read_csv(path)

    print("\n" + "-" * 90)
    print(name)

    print("N:", len(df))
    print("Columns:", df.columns.tolist())

    print(
        "Probability range:",
        df["Predicted_probability_AD"].min(),
        "to",
        df["Predicted_probability_AD"].max()
    )

    print(
        "Predicted AD:",
        int((df["Predicted_class"] == 1).sum())
    )

    print(
        "Predicted Control:",
        int((df["Predicted_class"] == 0).sum())
    )

    print(
        "Observed AD:",
        int((df["Diagnosis"] == "AD").sum())
    )

    print(
        "Observed Control:",
        int((df["Diagnosis"] == "Control").sum())
    )

# ------------------------------------------------------------
# Expected canonical values
# ------------------------------------------------------------

expected = {
    "GSE48350 TRAINING": 0.8549019607843137,
    "GSE48350 HELD-OUT TEST": 0.6944444444444444,
    "GSE5281 EXTERNAL VALIDATION": 0.6615384615384615
}

print("\n" + "=" * 90)
print("CANONICAL AUC CHECK")
print("=" * 90)

for _, row in validation.iterrows():

    dataset = row["dataset"]
    observed = row["AUC"]
    target = expected[dataset]

    print(
        f"{dataset}: "
        f"{observed:.10f} "
        f"expected={target:.10f} "
        f"match={np.isclose(observed, target)}"
    )

    assert np.isclose(
        observed,
        target
    )

# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

manifest = pd.DataFrame([
    {
        "model": "Final 3-gene LASSO logistic regression",
        "training_dataset": "GSE48350",
        "genes": "ABCA6;CRLF1;TNFRSF11B",
        "C": 0.3,
        "penalty": "L1",
        "solver": "liblinear",
        "scaling": "StandardScaler fitted on training data",
        "random_state": 42,
        "decision_threshold": 0.5,
        "training_AUC": 0.8549019608,
        "heldout_AUC": 0.6944444444,
        "external_AUC": 0.6615384615,
        "status": "CANONICAL_LOCKED_MODEL"
    }
])

out = FINAL / "Validation" / "FINAL_CANONICAL_MODEL_MANIFEST.csv"

manifest.to_csv(
    out,
    index=False
)

print("\nSaved:")
print(out)

print("\n" + "=" * 90)
print("CANONICAL MODEL AUDIT COMPLETE")
print("=" * 90)
