import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUTDIR = "04_ML/Final_Model/Validation/Locked_Ablation"
os.makedirs(OUTDIR, exist_ok=True)

COMPARISONS = [
    ("ABCA6+TNFRSF11B", ["ABCA6", "TNFRSF11B"],
     "ABCA6+CRLF1+TNFRSF11B",
     ["ABCA6", "CRLF1", "TNFRSF11B"])
]

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
    "04_ML/External_Validation/GSE5281_LASSO_candidate_matrix.csv",
    index_col=0
)

external_meta = pd.read_csv(
    "04_ML/External_Validation/GSE5281_validation_metadata.csv",
    index_col=0
)

y_train = (
    train_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .to_numpy()
)

y_test = (
    test_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .to_numpy()
)

y_external = (
    external_meta["diagnosis"]
    .map({"Control": 0, "AD": 1})
    .to_numpy()
)

rng = np.random.default_rng(42)
N_BOOT = 10000

results = []

print("=" * 85)
print("PAIRED BOOTSTRAP ΔAUC ANALYSIS")
print("=" * 85)

for better_name, better_genes, full_name, full_genes in COMPARISONS:

    models = {}

    for name, genes in [
        (better_name, better_genes),
        (full_name, full_genes)
    ]:

        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=0.3,
                    max_iter=5000,
                    random_state=42
                )
            )
        ])

        model.fit(
            train[genes],
            y_train
        )

        models[name] = model

    for dataset_name, data, y in [
        ("GSE48350_HeldOut_Test", test, y_test),
        ("GSE5281_External", external, y_external)
    ]:

        score_better = models[better_name].predict_proba(
            data[better_genes]
        )[:, 1]

        score_full = models[full_name].predict_proba(
            data[full_genes]
        )[:, 1]

        auc_better = roc_auc_score(
            y,
            score_better
        )

        auc_full = roc_auc_score(
            y,
            score_full
        )

        observed_delta = (
            auc_better - auc_full
        )

        deltas = []

        for _ in range(N_BOOT):

            idx = rng.integers(
                0,
                len(y),
                len(y)
            )

            y_b = y[idx]

            if len(np.unique(y_b)) < 2:
                continue

            auc_b = roc_auc_score(
                y_b,
                score_better[idx]
            )

            auc_f = roc_auc_score(
                y_b,
                score_full[idx]
            )

            deltas.append(
                auc_b - auc_f
            )

        deltas = np.asarray(deltas)

        ci_low = np.percentile(
            deltas,
            2.5
        )

        ci_high = np.percentile(
            deltas,
            97.5
        )

        probability_better = np.mean(
            deltas > 0
        )

        results.append({
            "Dataset": dataset_name,
            "Comparison": (
                better_name +
                "_vs_" +
                full_name
            ),
            "AUC_better_model": auc_better,
            "AUC_full_model": auc_full,
            "Observed_Delta_AUC": observed_delta,
            "Bootstrap_N": len(deltas),
            "Delta_AUC_CI_lower": ci_low,
            "Delta_AUC_CI_upper": ci_high,
            "Probability_Delta_gt_0":
                probability_better
        })

        print()
        print(dataset_name)
        print("-" * 60)
        print(
            f"{better_name} AUC: "
            f"{auc_better:.4f}"
        )
        print(
            f"{full_name} AUC: "
            f"{auc_full:.4f}"
        )
        print(
            f"Observed ΔAUC: "
            f"{observed_delta:.4f}"
        )
        print(
            f"95% CI: "
            f"{ci_low:.4f} to {ci_high:.4f}"
        )
        print(
            f"P(ΔAUC > 0): "
            f"{probability_better:.4f}"
        )

results_df = pd.DataFrame(results)

outfile = (
    f"{OUTDIR}/paired_ablation_delta_auc.csv"
)

results_df.to_csv(
    outfile,
    index=False
)

print()
print("=" * 85)
print("PAIRED ΔAUC ANALYSIS COMPLETE")
print("=" * 85)

print()
print(results_df.to_string(index=False))

print()
print("Saved:")
print(outfile)
