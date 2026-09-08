import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

OUTDIR = "04_ML/Final_Model/Validation"
MANUSCRIPT_OUT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT_OUT, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

train_y = (
    train_meta["Diagnosis"]
    .astype(str)
    .str.strip()
    .eq("AD")
    .astype(int)
    .to_numpy()
)

test_y = (
    test_meta["Diagnosis"]
    .astype(str)
    .str.strip()
    .eq("AD")
    .astype(int)
    .to_numpy()
)

external_y = (
    external["Diagnosis"]
    .astype(str)
    .str.strip()
    .eq("AD")
    .astype(int)
    .to_numpy()
)

# ------------------------------------------------------------
# LOCKED EQUAL-WEIGHT THREE-GENE SCORE
# ------------------------------------------------------------

def make_score(df):
    z = pd.DataFrame(index=df.index)

    for gene in GENES:
        z[gene] = (
            df[gene] - train[gene].mean()
        ) / train[gene].std(ddof=1)

    return z[GENES].mean(axis=1).to_numpy()


train_score = make_score(train)
test_score = make_score(test)

external_score = external[
    GENES
].copy()

for gene in GENES:
    external_score[gene] = (
        external_score[gene] - train[gene].mean()
    ) / train[gene].std(ddof=1)

external_score = external_score.mean(axis=1).to_numpy()

# ------------------------------------------------------------
# TRAINING THRESHOLD — YOUDEN J
# ------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    train_y,
    train_score
)

youden = tpr - fpr

best_idx = np.nanargmax(youden)

LOCKED_THRESHOLD = thresholds[best_idx]

print("=" * 75)
print("FINAL LOCKED THREE-GENE THRESHOLD ANALYSIS")
print("=" * 75)

print()
print("Training ROC-AUC:",
      roc_auc_score(train_y, train_score))

print("Training PR-AUC:",
      average_precision_score(train_y, train_score))

print()
print("Locked threshold:",
      LOCKED_THRESHOLD)

print("Youden J:",
      youden[best_idx])

# ------------------------------------------------------------
# METRIC FUNCTION
# ------------------------------------------------------------

def evaluate(name, y, score):

    pred = (
        score >= LOCKED_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

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

    ppv = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    auc = roc_auc_score(
        y,
        score
    )

    pr_auc = average_precision_score(
        y,
        score
    )

    return {
        "Dataset": name,
        "N": len(y),
        "AD": int(y.sum()),
        "Control": int((y == 0).sum()),
        "Threshold": LOCKED_THRESHOLD,
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1_score(
            y,
            pred,
            zero_division=0
        ),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


rows = []

rows.append(
    evaluate(
        "GSE48350_Training",
        train_y,
        train_score
    )
)

rows.append(
    evaluate(
        "GSE48350_HeldOut_Test",
        test_y,
        test_score
    )
)

rows.append(
    evaluate(
        "GSE5281_External",
        external_y,
        external_score
    )
)

results = pd.DataFrame(rows)

print()
print("=" * 75)
print("LOCKED-THRESHOLD PERFORMANCE")
print("=" * 75)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

out = (
    OUTDIR +
    "/FINAL_LOCKED_MODEL_THRESHOLD_PERFORMANCE.csv"
)

results.to_csv(
    out,
    index=False
)

manuscript = (
    MANUSCRIPT_OUT +
    "/Table_final_locked_model_threshold_performance.csv"
)

results.to_csv(
    manuscript,
    index=False
)

print()
print("Saved:")
print(out)
print(manuscript)

print()
print("=" * 75)
print("LOCKED THRESHOLD ANALYSIS COMPLETE")
print("=" * 75)
