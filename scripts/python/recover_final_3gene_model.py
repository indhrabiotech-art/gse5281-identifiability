import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path.home() / "project_ml"

EXPR = ROOT / "04_ML/GSE48350_RMA_ML_matrix.csv"
TRAIN_META = ROOT / "04_ML/GSE48350_RMA_train_meta.csv"
TEST_META = ROOT / "04_ML/GSE48350_RMA_test_meta.csv"

TRAIN_PRED = ROOT / "04_ML/Final_Model/GSE48350_training_predictions.csv"
TEST_PRED = ROOT / "04_ML/Final_Model/GSE48350_test_predictions.csv"

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

COEF = np.array([
    0.47542366314463647,
    0.37970767240788367,
    0.43940704735699226
])

print("=" * 100)
print("RECOVERING EXACT FINAL 3-GENE MODEL TRANSFORMATION")
print("=" * 100)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

expr = pd.read_csv(EXPR).set_index("Unnamed: 0")

train_meta = pd.read_csv(TRAIN_META)
test_meta = pd.read_csv(TEST_META)

train_pred = pd.read_csv(TRAIN_PRED)
test_pred = pd.read_csv(TEST_PRED)

# ------------------------------------------------------------
# ALIGN
# ------------------------------------------------------------

def get_X(meta):
    ids = meta["index"].astype(str).tolist()
    ids = [x for x in ids if x in expr.index]

    return expr.loc[ids, GENES].astype(float)

X_train = get_X(train_meta)
X_test = get_X(test_meta)

p_train = train_pred["Predicted_probability_AD"].values
p_test = test_pred["Predicted_probability_AD"].values

y_train = (train_pred["Diagnosis"] == "AD").astype(int).values
y_test = (test_pred["Diagnosis"] == "AD").astype(int).values

print("\nX shapes:")
print("Training:", X_train.shape)
print("Held-out:", X_test.shape)

# ------------------------------------------------------------
# METHOD 1: RAW EXPRESSION × COEFFICIENT
# ------------------------------------------------------------

raw_train = X_train.values @ COEF
raw_test = X_test.values @ COEF

# ------------------------------------------------------------
# METHOD 2: TRAINING STANDARDIZATION
# ------------------------------------------------------------

mu = X_train.mean(axis=0).values
sd = X_train.std(axis=0, ddof=0).values

Z_train = (X_train.values - mu) / sd
Z_test = (X_test.values - mu) / sd

std_train = Z_train @ COEF
std_test = Z_test @ COEF

# ------------------------------------------------------------
# FIT INTERCEPT TO STORED PROBABILITIES
# ------------------------------------------------------------

def best_intercept(score, p):
    logit_p = np.log(p / (1 - p))
    return np.mean(logit_p - score)

b_raw = best_intercept(raw_train, p_train)
b_std = best_intercept(std_train, p_train)

pred_raw = expit(raw_train + b_raw)
pred_std = expit(std_train + b_std)

# ------------------------------------------------------------
# COMPARE
# ------------------------------------------------------------

def report(name, observed, predicted, y):

    corr = np.corrcoef(observed, predicted)[0, 1]
    rmse = np.sqrt(np.mean((observed - predicted) ** 2))
    auc = roc_auc_score(y, predicted)
    ap = average_precision_score(y, predicted)

    print("\n" + "-" * 90)
    print(name)
    print("-" * 90)

    print("Correlation:", round(corr, 8))
    print("RMSE:", round(rmse, 8))
    print("AUC:", round(auc, 8))
    print("AP:", round(ap, 8))

    print("\nFirst 5:")
    print(
        pd.DataFrame({
            "stored": observed[:5],
            "reconstructed": predicted[:5]
        }).to_string(index=False)
    )

report(
    "RAW EXPRESSION + FITTED INTERCEPT — TRAIN",
    p_train,
    pred_raw,
    y_train
)

report(
    "STANDARDIZED EXPRESSION + FITTED INTERCEPT — TRAIN",
    p_train,
    pred_std,
    y_train
)

# ------------------------------------------------------------
# TEST USING SAME TRAINING STANDARDIZATION
# ------------------------------------------------------------

pred_std_test = expit(std_test + b_std)

report(
    "STANDARDIZED EXPRESSION + TRAINING INTERCEPT — HELD-OUT",
    p_test,
    pred_std_test,
    y_test
)

# ------------------------------------------------------------
# RECOVER INTERCEPT USING LOGISTIC REGRESSION
# ------------------------------------------------------------

lr = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=10000
)

lr.fit(Z_train, y_train)

print("\n" + "=" * 100)
print("INDEPENDENT LOGISTIC REGRESSION FIT")
print("=" * 100)

print("Intercept:")
print(lr.intercept_[0])

print("\nCoefficients:")
for g, c in zip(GENES, lr.coef_[0]):
    print(f"{g:12s} {c:.12f}")

print("\nStored coefficients:")
for g, c in zip(GENES, COEF):
    print(f"{g:12s} {c:.12f}")

# ------------------------------------------------------------
# SAVE RECOVERY INFORMATION
# ------------------------------------------------------------

summary = pd.DataFrame([{
    "raw_intercept": b_raw,
    "standardized_intercept": b_std,
    "train_standardized_correlation": np.corrcoef(
        p_train, pred_std
    )[0,1],
    "test_standardized_correlation": np.corrcoef(
        p_test, pred_std_test
    )[0,1],
    "train_auc_reconstructed": roc_auc_score(
        y_train, pred_std
    ),
    "test_auc_reconstructed": roc_auc_score(
        y_test, pred_std_test
    )
}])

out = ROOT / "04_ML/Reviewer_Robustness_V3/final_3gene_model_recovery.csv"

summary.to_csv(out, index=False)

print("\nSaved:")
print(out)

print("\nDONE")
