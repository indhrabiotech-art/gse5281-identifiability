import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)

OUT = "07_Figures/Final_3gene"
os.makedirs(OUT, exist_ok=True)

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

BASE = "04_ML/Final_Characterization"

# ------------------------------------------------------------
# Load ROC / PR data
# ------------------------------------------------------------

datasets = {
    "GSE48350 Training":
        ("GSE48350_train_ROC.csv",
         "GSE48350_train_PR.csv"),

    "GSE48350 Internal Test":
        ("GSE48350_test_ROC.csv",
         "GSE48350_test_PR.csv"),

    "GSE5281 External":
        ("GSE5281_external_ROC.csv",
         "GSE5281_external_PR.csv")
}

# ------------------------------------------------------------
# ROC figure
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))

for name, (roc_file, _) in datasets.items():

    df = pd.read_csv(
        os.path.join(BASE, roc_file)
    )

    auc = np.trapz(
        df["tpr"],
        df["fpr"]
    )

    plt.plot(
        df["fpr"],
        df["tpr"],
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
plt.title("ROC Performance of the 3-Gene Signature")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUT}/Figure_ROC_3gene.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# Precision-recall figure
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))

for name, (_, pr_file) in datasets.items():

    df = pd.read_csv(
        os.path.join(BASE, pr_file)
    )

    plt.plot(
        df["recall"],
        df["precision"],
        linewidth=2,
        label=name
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Performance of the 3-Gene Signature")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUT}/Figure_PR_3gene.png",
    dpi=300
)

plt.close()

print("=" * 70)
print("FINAL FIGURES GENERATED")
print("=" * 70)

print(f"\nOutput directory:\n{OUT}")

print("\nFiles:")
for f in sorted(os.listdir(OUT)):
    print(f)

print("\n" + "=" * 70)
