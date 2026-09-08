import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path.home() / "project_ml"

EXPR = ROOT / "04_ML/GSE48350_RMA_ML_matrix.csv"
TRAIN_META = ROOT / "04_ML/GSE48350_RMA_train_meta.csv"
TEST_META = ROOT / "04_ML/GSE48350_RMA_test_meta.csv"

TRAIN_PRED = ROOT / "04_ML/Final_Model/GSE48350_training_predictions.csv"
TEST_PRED = ROOT / "04_ML/Final_Model/GSE48350_test_predictions.csv"

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

COEF = {
    "ABCA6": 0.47542366314463647,
    "CRLF1": 0.37970767240788367,
    "TNFRSF11B": 0.43940704735699226,
}

print("=" * 100)
print("FINAL 3-GENE MODEL REPRODUCIBILITY CHECK")
print("=" * 100)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

expr = pd.read_csv(EXPR)
expr = expr.set_index(expr.columns[0])

train_meta = pd.read_csv(TRAIN_META)
test_meta = pd.read_csv(TEST_META)

train_pred = pd.read_csv(TRAIN_PRED)
test_pred = pd.read_csv(TEST_PRED)

# ------------------------------------------------------------
# ALIGN
# ------------------------------------------------------------

def prepare(meta):
    ids = meta["index"].astype(str)
    common = [x for x in ids if x in expr.index]

    out = meta.set_index("index").loc[common].copy()
    X = expr.loc[common, GENES].copy()

    return X, out

X_train, m_train = prepare(train_meta)
X_test, m_test = prepare(test_meta)

print("\nSamples:")
print("Training:", len(X_train))
print("Held-out:", len(X_test))

# ------------------------------------------------------------
# CHECK GENE AVAILABILITY
# ------------------------------------------------------------

print("\nGenes:")
for g in GENES:
    print(
        f"{g:12s}",
        "present" if g in expr.columns else "MISSING"
    )

# ------------------------------------------------------------
# RAW EXPRESSION SCORE
# ------------------------------------------------------------

coef_vector = np.array([COEF[g] for g in GENES])

raw_train = X_train.values @ coef_vector
raw_test = X_test.values @ coef_vector

# ------------------------------------------------------------
# CORRELATION WITH STORED PREDICTIONS
# ------------------------------------------------------------

def compare(name, raw_score, stored):
    p = stored["Predicted_probability_AD"].values

    corr = np.corrcoef(raw_score, p)[0, 1]

    print("\n" + "-" * 90)
    print(name)
    print("-" * 90)

    print("Stored probability range:",
          float(p.min()),
          "to",
          float(p.max()))

    print("Raw linear-score range:",
          float(raw_score.min()),
          "to",
          float(raw_score.max()))

    print("Correlation:",
          round(corr, 8))

    print("\nFirst 10 comparison rows:")

    comparison = pd.DataFrame({
        "stored_probability": p[:10],
        "reconstructed_score": raw_score[:10],
    })

    print(comparison.to_string(index=False))

compare(
    "TRAINING",
    raw_train,
    train_pred
)

compare(
    "HELD-OUT",
    raw_test,
    test_pred
)

# ------------------------------------------------------------
# AUC OF STORED PREDICTIONS
# ------------------------------------------------------------

def metrics(name, pred):
    y = (pred["Diagnosis"] == "AD").astype(int)
    p = pred["Predicted_probability_AD"]

    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)

    print("\n" + "-" * 90)
    print(name)
    print("-" * 90)
    print("n =", len(y))
    print("AD =", int(y.sum()))
    print("Control =", int((y == 0).sum()))
    print("ROC-AUC =", round(auc, 6))
    print("PR-AUC =", round(ap, 6))

metrics(
    "STORED TRAINING PREDICTIONS",
    train_pred
)

metrics(
    "STORED HELD-OUT PREDICTIONS",
    test_pred
)

# ------------------------------------------------------------
# SAVE RECONSTRUCTED SCORES
# ------------------------------------------------------------

out_train = m_train.copy()
out_train["three_gene_linear_score"] = raw_train
out_train["ABCA6"] = X_train["ABCA6"].values
out_train["CRLF1"] = X_train["CRLF1"].values
out_train["TNFRSF11B"] = X_train["TNFRSF11B"].values

out_test = m_test.copy()
out_test["three_gene_linear_score"] = raw_test
out_test["ABCA6"] = X_test["ABCA6"].values
out_test["CRLF1"] = X_test["CRLF1"].values
out_test["TNFRSF11B"] = X_test["TNFRSF11B"].values

out_train.to_csv(
    ROOT / "04_ML/Reviewer_Robustness_V3/"
    "FINAL_3GENE_RECONSTRUCTED_TRAINING_SCORES.csv"
)

out_test.to_csv(
    ROOT / "04_ML/Reviewer_Robustness_V3/"
    "FINAL_3GENE_RECONSTRUCTED_HELDOUT_SCORES.csv"
)

print("\nSaved reconstructed scores.")

print("\n" + "=" * 100)
print("CHECK COMPLETE")
print("=" * 100)
