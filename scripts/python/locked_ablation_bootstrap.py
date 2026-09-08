import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUTDIR = "04_ML/Final_Model/Validation/Locked_Ablation"
os.makedirs(OUTDIR, exist_ok=True)

SIGNATURES = {
    "ABCA6": ["ABCA6"],
    "CRLF1": ["CRLF1"],
    "TNFRSF11B": ["TNFRSF11B"],
    "ABCA6+CRLF1": ["ABCA6", "CRLF1"],
    "ABCA6+TNFRSF11B": ["ABCA6", "TNFRSF11B"],
    "CRLF1+TNFRSF11B": ["CRLF1", "TNFRSF11B"],
    "ABCA6+CRLF1+TNFRSF11B": [
        "ABCA6",
        "CRLF1",
        "TNFRSF11B"
    ]
}

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

print("=" * 80)
print("LOCKED ABLATION BOOTSTRAP")
print("=" * 80)

for signature_name, genes in SIGNATURES.items():

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

    test_scores = model.predict_proba(
        test[genes]
    )[:, 1]

    external_scores = model.predict_proba(
        external[genes]
    )[:, 1]

    for dataset_name, y, scores in [
        ("GSE48350_HeldOut_Test", y_test, test_scores),
        ("GSE5281_External", y_external, external_scores)
    ]:

        observed = roc_auc_score(
            y,
            scores
        )

        boot_auc = []

        for _ in range(N_BOOT):

            idx = rng.integers(
                0,
                len(y),
                len(y)
            )

            y_b = y[idx]
            s_b = scores[idx]

            if len(np.unique(y_b)) < 2:
                continue

            boot_auc.append(
                roc_auc_score(
                    y_b,
                    s_b
                )
            )

        boot_auc = np.asarray(
            boot_auc
        )

        results.append({
            "Dataset": dataset_name,
            "Signature": signature_name,
            "N_genes": len(genes),
            "Observed_AUC": observed,
            "Bootstrap_N": len(boot_auc),
            "AUC_CI_lower": np.percentile(
                boot_auc,
                2.5
            ),
            "AUC_CI_upper": np.percentile(
                boot_auc,
                97.5
            )
        })

        print(
            f"{dataset_name:25s} "
            f"{signature_name:30s} "
            f"AUC={observed:.4f} "
            f"95% CI="
            f"{np.percentile(boot_auc,2.5):.4f}-"
            f"{np.percentile(boot_auc,97.5):.4f}"
        )

results_df = pd.DataFrame(results)

outfile = (
    f"{OUTDIR}/locked_ablation_bootstrap_results.csv"
)

results_df.to_csv(
    outfile,
    index=False
)

print()
print("=" * 80)
print("SAVED")
print(outfile)
print("=" * 80)
