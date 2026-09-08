import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score
)

from sklearn.utils import resample


# ============================================================
# CONFIGURATION
# ============================================================

OUTDIR = "04_ML/Final_Model/Validation"
os.makedirs(OUTDIR, exist_ok=True)

FILES = {
    "GSE48350_Training":
        "04_ML/Final_Model/GSE48350_training_predictions.csv",

    "GSE48350_HeldOut_Test":
        "04_ML/Final_Model/GSE48350_test_predictions.csv",

    "GSE5281_External":
        "04_ML/Final_Model/GSE5281_external_predictions.csv"
}


# ============================================================
# LOAD DATA
# ============================================================

datasets = {}

for name, path in FILES.items():

    df = pd.read_csv(path)

    df["y"] = df["Diagnosis"].map({
        "Control": 0,
        "AD": 1
    })

    if df["y"].isna().any():
        raise ValueError(
            f"Unknown diagnosis labels found in {name}"
        )

    datasets[name] = df

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Samples:", len(df))
    print("AD:", int((df["y"] == 1).sum()))
    print("Control:", int((df["y"] == 0).sum()))


# ============================================================
# TRAINING-DERIVED YOUDEN THRESHOLD
# ============================================================

train = datasets["GSE48350_Training"]

y_train = train["y"].values
p_train = train["Predicted_probability_AD"].values

fpr_train, tpr_train, thresholds_train = roc_curve(
    y_train,
    p_train
)

youden = tpr_train - fpr_train

# Ignore infinite threshold
valid = np.isfinite(thresholds_train)

best_idx = np.argmax(
    np.where(valid, youden, -np.inf)
)

TRAIN_THRESHOLD = thresholds_train[best_idx]

print("\n" + "=" * 60)
print("TRAINING-DERIVED THRESHOLD")
print("=" * 60)

print("Youden-optimal threshold:",
      round(TRAIN_THRESHOLD, 6))

print("Training sensitivity:",
      round(tpr_train[best_idx], 4))

print("Training specificity:",
      round(1 - fpr_train[best_idx], 4))


# ============================================================
# BOOTSTRAP AUC
# ============================================================

def bootstrap_auc_ci(y, scores, n_boot=5000, seed=42):

    rng = np.random.default_rng(seed)

    y = np.asarray(y)
    scores = np.asarray(scores)

    aucs = []

    for _ in range(n_boot):

        idx = rng.integers(
            0,
            len(y),
            len(y)
        )

        y_b = y[idx]
        s_b = scores[idx]

        # Bootstrap sample must contain both classes
        if len(np.unique(y_b)) < 2:
            continue

        aucs.append(
            roc_auc_score(y_b, s_b)
        )

    aucs = np.array(aucs)

    return (
        roc_auc_score(y, scores),
        np.percentile(aucs, 2.5),
        np.percentile(aucs, 97.5),
        len(aucs)
    )


# ============================================================
# DATASET METRICS
# ============================================================

results = []

for name, df in datasets.items():

    y = df["y"].values
    scores = df["Predicted_probability_AD"].values

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    auc, auc_low, auc_high, n_boot = bootstrap_auc_ci(
        y,
        scores
    )

    fpr, tpr, thresholds = roc_curve(
        y,
        scores
    )

    # --------------------------------------------------------
    # PR
    # --------------------------------------------------------

    precision, recall, pr_thresholds = precision_recall_curve(
        y,
        scores
    )

    pr_auc = average_precision_score(
        y,
        scores
    )

    # --------------------------------------------------------
    # Classification at training-derived threshold
    # --------------------------------------------------------

    pred = (
        scores >= TRAIN_THRESHOLD
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

    accuracy = accuracy_score(
        y,
        pred
    )

    precision_value = precision_score(
        y,
        pred,
        zero_division=0
    )

    results.append({

        "Dataset": name,
        "N": len(y),

        "AD": int((y == 1).sum()),
        "Control": int((y == 0).sum()),

        "ROC_AUC": auc,
        "ROC_AUC_CI_low": auc_low,
        "ROC_AUC_CI_high": auc_high,

        "PR_AUC": pr_auc,

        "Threshold": TRAIN_THRESHOLD,

        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "Precision": precision_value,

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,

        "Bootstrap_samples": n_boot
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_file = (
    f"{OUTDIR}/GSE48350_3gene_validation_metrics.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL ROC / PR VALIDATION RESULTS")
print("=" * 60)

for _, row in results_df.iterrows():

    print("\n" + row["Dataset"])

    print(
        f"ROC-AUC: {row['ROC_AUC']:.4f} "
        f"(95% CI "
        f"{row['ROC_AUC_CI_low']:.4f}–"
        f"{row['ROC_AUC_CI_high']:.4f})"
    )

    print(
        f"PR-AUC: {row['PR_AUC']:.4f}"
    )

    print(
        f"Threshold: {row['Threshold']:.4f}"
    )

    print(
        f"Sensitivity: {row['Sensitivity']:.4f}"
    )

    print(
        f"Specificity: {row['Specificity']:.4f}"
    )

    print(
        f"Accuracy: {row['Accuracy']:.4f}"
    )

    print(
        f"Precision: {row['Precision']:.4f}"
    )

    print(
        "Confusion matrix: "
        f"TN={row['TN']} "
        f"FP={row['FP']} "
        f"FN={row['FN']} "
        f"TP={row['TP']}"
    )


# ============================================================
# ROC FIGURE
# ============================================================

plt.figure(figsize=(7, 6))

for name, df in datasets.items():

    y = df["y"].values
    scores = df["Predicted_probability_AD"].values

    fpr, tpr, _ = roc_curve(
        y,
        scores
    )

    auc = roc_auc_score(
        y,
        scores
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC={auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves — 3-Gene LASSO Signature"
)

plt.legend()

plt.tight_layout()

roc_file = (
    f"{OUTDIR}/GSE48350_3gene_ROC_curves.png"
)

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PR FIGURE
# ============================================================

plt.figure(figsize=(7, 6))

for name, df in datasets.items():

    y = df["y"].values
    scores = df["Predicted_probability_AD"].values

    precision, recall, _ = precision_recall_curve(
        y,
        scores
    )

    pr_auc = average_precision_score(
        y,
        scores
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{name} (AP={pr_auc:.3f})"
    )

plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision")

plt.title(
    "Precision–Recall Curves — 3-Gene LASSO Signature"
)

plt.legend()

plt.tight_layout()

pr_file = (
    f"{OUTDIR}/GSE48350_3gene_PR_curves.png"
)

plt.savefig(
    pr_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE ROC POINTS
# ============================================================

roc_rows = []

for name, df in datasets.items():

    y = df["y"].values
    scores = df["Predicted_probability_AD"].values

    fpr, tpr, thresholds = roc_curve(
        y,
        scores
    )

    for a, b, c in zip(
        fpr,
        tpr,
        thresholds
    ):

        roc_rows.append({
            "Dataset": name,
            "FPR": a,
            "TPR": b,
            "Threshold": c
        })

pd.DataFrame(roc_rows).to_csv(
    f"{OUTDIR}/GSE48350_3gene_ROC_points.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("ROC / PR VALIDATION COMPLETE")
print("=" * 60)

print("\nSaved:")
print(results_file)
print(roc_file)
print(pr_file)
print(
    f"{OUTDIR}/GSE48350_3gene_ROC_points.csv"
)

print("\nTraining-derived threshold:",
      round(TRAIN_THRESHOLD, 6))

print("=" * 60)

