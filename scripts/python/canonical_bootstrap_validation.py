import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path.home() / "project_ml"
FINAL = ROOT / "04_ML" / "Final_Model"
OUT = FINAL / "Validation"

OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "GSE48350_Training":
        FINAL / "GSE48350_training_predictions.csv",

    "GSE48350_HeldOut_Test":
        FINAL / "GSE48350_test_predictions.csv",

    "GSE5281_External":
        FINAL / "GSE5281_external_predictions.csv"
}

N_BOOT = 10000
SEED = 42

rng = np.random.default_rng(SEED)

results = []

print("=" * 90)
print("CANONICAL 3-GENE BOOTSTRAP VALIDATION")
print("=" * 90)

for dataset, path in FILES.items():

    df = pd.read_csv(path)

    y = (
        df["Diagnosis"]
        .map({
            "Control": 0,
            "AD": 1
        })
        .to_numpy()
    )

    score = df[
        "Predicted_probability_AD"
    ].to_numpy()

    observed_auc = roc_auc_score(
        y,
        score
    )

    observed_ap = average_precision_score(
        y,
        score
    )

    auc_boot = []
    ap_boot = []

    for _ in range(N_BOOT):

        idx = rng.integers(
            0,
            len(y),
            len(y)
        )

        y_b = y[idx]
        score_b = score[idx]

        if len(np.unique(y_b)) < 2:
            continue

        auc_boot.append(
            roc_auc_score(
                y_b,
                score_b
            )
        )

        ap_boot.append(
            average_precision_score(
                y_b,
                score_b
            )
        )

    auc_boot = np.asarray(auc_boot)
    ap_boot = np.asarray(ap_boot)

    auc_low, auc_high = np.percentile(
        auc_boot,
        [2.5, 97.5]
    )

    ap_low, ap_high = np.percentile(
        ap_boot,
        [2.5, 97.5]
    )

    results.append({
        "Dataset": dataset,
        "N": len(y),
        "AD": int(y.sum()),
        "Control": int((y == 0).sum()),
        "ROC_AUC": observed_auc,
        "ROC_AUC_CI_low": auc_low,
        "ROC_AUC_CI_high": auc_high,
        "PR_AUC": observed_ap,
        "PR_AUC_CI_low": ap_low,
        "PR_AUC_CI_high": ap_high,
        "Bootstrap_samples": len(auc_boot)
    })

    print("\n" + "-" * 90)
    print(dataset)
    print(f"N = {len(y)}")
    print(f"ROC-AUC = {observed_auc:.4f}")
    print(
        f"95% CI = "
        f"{auc_low:.4f}–{auc_high:.4f}"
    )
    print(f"PR-AUC = {observed_ap:.4f}")
    print(
        f"95% CI = "
        f"{ap_low:.4f}–{ap_high:.4f}"
    )

results_df = pd.DataFrame(results)

outfile = (
    OUT /
    "FINAL_CANONICAL_BOOTSTRAP_VALIDATION.csv"
)

results_df.to_csv(
    outfile,
    index=False
)

print("\n" + "=" * 90)
print("SAVED")
print(outfile)
print("=" * 90)
