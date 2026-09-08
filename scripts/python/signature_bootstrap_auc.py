import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


INPUT = (
    "04_ML/Final_Model/Validation/"
    "three_gene_signature_sample_scores.csv"
)

OUTDIR = "04_ML/Final_Model/Validation"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT)

datasets = [
    "GSE48350_Training",
    "GSE48350_HeldOut_Test",
    "GSE5281_External"
]

N_BOOT = 5000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

results = []

print("=" * 60)
print("BOOTSTRAP ROC-AUC CONFIDENCE INTERVALS")
print("=" * 60)


for dataset in datasets:

    sub = df[df["Dataset"] == dataset].copy()

    y = (
        sub["Diagnosis"]
        .map({
            "Control": 0,
            "AD": 1
        })
        .values
    )

    score = sub["Three_gene_signature"].values

    observed_auc = roc_auc_score(y, score)

    boot_auc = []

    for i in range(N_BOOT):

        idx = rng.integers(
            0,
            len(sub),
            len(sub)
        )

        y_boot = y[idx]
        score_boot = score[idx]

        # Bootstrap sample must contain both classes
        if len(np.unique(y_boot)) < 2:
            continue

        boot_auc.append(
            roc_auc_score(
                y_boot,
                score_boot
            )
        )

    boot_auc = np.array(boot_auc)

    ci_low = np.percentile(
        boot_auc,
        2.5
    )

    ci_high = np.percentile(
        boot_auc,
        97.5
    )

    print("\n" + "-" * 60)
    print(dataset)
    print("-" * 60)

    print("N:", len(sub))
    print("AD:", int(y.sum()))
    print("Control:", int((y == 0).sum()))
    print("Observed AUC:", round(observed_auc, 4))
    print("Bootstrap replicates:", len(boot_auc))
    print(
        "95% CI:",
        f"{ci_low:.4f} – {ci_high:.4f}"
    )

    results.append({
        "Dataset": dataset,
        "N": len(sub),
        "AD_n": int(y.sum()),
        "Control_n": int((y == 0).sum()),
        "ROC_AUC": observed_auc,
        "Bootstrap_N": len(boot_auc),
        "CI_95_lower": ci_low,
        "CI_95_upper": ci_high
    })


results_df = pd.DataFrame(results)

outfile = (
    f"{OUTDIR}/three_gene_signature_bootstrap_auc.csv"
)

results_df.to_csv(
    outfile,
    index=False
)

print("\n" + "=" * 60)
print("BOOTSTRAP ANALYSIS COMPLETE")
print("=" * 60)

print("\nResults:")
print(results_df.to_string(index=False))

print("\nSaved:")
print(outfile)

print("=" * 60)

