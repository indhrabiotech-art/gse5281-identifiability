import os
import itertools
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


OUTDIR = "04_ML/Final_Model/Validation"
os.makedirs(OUTDIR, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

DATASETS = {
    "GSE48350_Training": (
        "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
        "04_ML/GSE48350_RMA_train_meta.csv",
        "Diagnosis"
    ),

    "GSE48350_HeldOut_Test": (
        "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
        "04_ML/GSE48350_RMA_test_meta.csv",
        "Diagnosis"
    ),

    "GSE5281_External": (
        "04_ML/External_Validation/GSE5281_LASSO_candidate_matrix.csv",
        "04_ML/External_Validation/GSE5281_validation_metadata.csv",
        "diagnosis"
    )
}


# ------------------------------------------------------------
# Gene combinations
# ------------------------------------------------------------

COMBINATIONS = [
    ("ABCA6",),
    ("CRLF1",),
    ("TNFRSF11B",),

    ("ABCA6", "CRLF1"),
    ("ABCA6", "TNFRSF11B"),
    ("CRLF1", "TNFRSF11B"),

    ("ABCA6", "CRLF1", "TNFRSF11B")
]


results = []


print("=" * 60)
print("THREE-GENE ABLATION / SENSITIVITY ANALYSIS")
print("=" * 60)


for dataset, (expr_file, meta_file, diagnosis_col) in DATASETS.items():

    print("\n" + "=" * 60)
    print(dataset)
    print("=" * 60)

    expr = pd.read_csv(
        expr_file,
        index_col=0
    )

    meta = pd.read_csv(
        meta_file,
        index_col=0
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    if list(expr.index) != list(meta.index):
        raise ValueError(
            f"{dataset}: expression/metadata alignment failed."
        )

    y = (
        meta[diagnosis_col]
        .map({
            "Control": 0,
            "AD": 1
        })
        .values
    )

    if np.isnan(y).any():
        raise ValueError(
            f"{dataset}: diagnosis mapping contains missing values."
        )

    print("Samples:", len(y))
    print("AD:", int(sum(y)))
    print("Control:", int(len(y) - sum(y)))

    # --------------------------------------------------------
    # Test every gene combination
    # --------------------------------------------------------

    for genes in COMBINATIONS:

        X = expr[list(genes)].copy()

        if X.isna().sum().sum() > 0:
            raise ValueError(
                f"{dataset}: missing values in {genes}"
            )

        # ----------------------------------------------------
        # Standardized signature score
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        signature = X_scaled.mean(axis=1)

        auc = roc_auc_score(y, signature)
        pr_auc = average_precision_score(y, signature)

        # ----------------------------------------------------
        # Logistic model using only selected genes
        # ----------------------------------------------------

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "logistic",
                LogisticRegression(
                    C=0.3,
                    solver="liblinear",
                    penalty="l1",
                    random_state=42
                )
            )
        ])

        model.fit(X, y)

        probability = model.predict_proba(X)[:, 1]

        model_auc = roc_auc_score(y, probability)
        model_pr_auc = average_precision_score(
            y,
            probability
        )

        results.append({
            "Dataset": dataset,
            "Gene_set": "+".join(genes),
            "N_genes": len(genes),
            "N": len(y),
            "AD_n": int(sum(y)),
            "Control_n": int(len(y) - sum(y)),
            "Signature_AUC": auc,
            "Signature_PR_AUC": pr_auc,
            "Logistic_AUC": model_auc,
            "Logistic_PR_AUC": model_pr_auc
        })

        print(
            f"{'+'.join(genes):32s} | "
            f"Signature AUC={auc:.4f} | "
            f"Logistic AUC={model_auc:.4f}"
        )


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

outfile = (
    f"{OUTDIR}/"
    "three_gene_ablation_results.csv"
)

results_df.to_csv(
    outfile,
    index=False
)


# ------------------------------------------------------------
# Identify best combination per dataset
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BEST GENE SETS")
print("=" * 60)

for dataset in results_df["Dataset"].unique():

    sub = results_df[
        results_df["Dataset"] == dataset
    ].sort_values(
        "Signature_AUC",
        ascending=False
    )

    print("\n" + dataset)

    print(
        sub[
            [
                "Gene_set",
                "Signature_AUC",
                "Logistic_AUC"
            ]
        ].to_string(index=False)
    )


print("\n" + "=" * 60)
print("ABLATION ANALYSIS COMPLETE")
print("=" * 60)

print("\nSaved:")
print(outfile)

GENE_SETS = [
    ("ABCA6",),
    ("CRLF1",),
    ("TNFRSF11B",),
    ("ABCA6", "CRLF1"),
    ("ABCA6", "TNFRSF11B"),
    ("CRLF1", "TNFRSF11B"),
    ("ABCA6", "CRLF1", "TNFRSF11B")
]


DATASETS = {
    "GSE48350_Training": (
        "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
        "04_ML/GSE48350_RMA_train_meta.csv",
        "Diagnosis"
    ),

    "GSE48350_HeldOut_Test": (
        "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
        "04_ML/GSE48350_RMA_test_meta.csv",
        "Diagnosis"
    ),

    "GSE5281_External": (
        "04_ML/External_Validation/GSE5281_LASSO_candidate_matrix.csv",
        "04_ML/External_Validation/GSE5281_validation_metadata.csv",
        "diagnosis"
    )
}


results = []


print("=" * 60)
print("THREE-GENE ABLATION / SENSITIVITY ANALYSIS")
print("=" * 60)


for dataset, (expr_file, meta_file, diagnosis_col) in DATASETS.items():

    print("\n" + "=" * 60)
    print(dataset)
    print("=" * 60)

    expr = pd.read_csv(
        expr_file,
        index_col=0
    )

    meta = pd.read_csv(
        meta_file,
        index_col=0
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    if list(expr.index) != list(meta.index):
        raise ValueError(
            f"{dataset}: expression/metadata alignment failed."
        )

    y = meta[diagnosis_col].map({
        "Control": 0,
        "AD": 1
    }).values

    if np.isnan(y).any():
        raise ValueError(
            f"{dataset}: diagnosis mapping failed."
        )

    print("Samples:", len(y))
    print("AD:", int(y.sum()))
    print("Control:", int(len(y) - y.sum()))

    # --------------------------------------------------------
    # Every gene set
    # --------------------------------------------------------

    for genes in GENE_SETS:

        genes = list(genes)

        missing = [
            gene for gene in genes
            if gene not in expr.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset}: missing genes: {missing}"
            )

        X = expr[genes].copy()

        if X.isna().sum().sum() > 0:
            raise ValueError(
                f"{dataset}: missing expression values."
            )

        # ----------------------------------------------------
        # Standardized equal-weight signature
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        signature = X_scaled.mean(axis=1)

        signature_auc = roc_auc_score(
            y,
            signature
        )

        signature_pr = average_precision_score(
            y,
            signature
        )

        # ----------------------------------------------------
        # Logistic regression using selected genes
        # ----------------------------------------------------

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "logistic",
                LogisticRegression(
                    C=0.3,
                    solver="liblinear",
                    penalty="l1",
                    random_state=42
                )
            )
        ])

        model.fit(X, y)

        probability = model.predict_proba(X)[:, 1]

        logistic_auc = roc_auc_score(
            y,
            probability
        )

        logistic_pr = average_precision_score(
            y,
            probability
        )

        gene_set = "+".join(genes)

        results.append({
            "Dataset": dataset,
            "Gene_set": gene_set,
            "N_genes": len(genes),
            "N": len(y),
            "AD_n": int(y.sum()),
            "Control_n": int(len(y) - y.sum()),
            "Signature_AUC": signature_auc,
            "Signature_PR_AUC": signature_pr,
            "Logistic_AUC": logistic_auc,
            "Logistic_PR_AUC": logistic_pr
        })

        print(
            f"{gene_set:32s} | "
            f"Signature AUC = {signature_auc:.4f} | "
            f"Logistic AUC = {logistic_auc:.4f}"
        )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

outfile = (
    f"{OUTDIR}/three_gene_ablation_results.csv"
)

results_df.to_csv(
    outfile,
    index=False
)


# ------------------------------------------------------------
# Best gene set per dataset
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BEST GENE SETS BY STANDARDIZED SIGNATURE AUC")
print("=" * 60)

for dataset in results_df["Dataset"].unique():

    sub = results_df[
        results_df["Dataset"] == dataset
    ].sort_values(
        "Signature_AUC",
        ascending=False
    )

    print("\n" + dataset)

    print(
        sub[
            [
                "Gene_set",
                "N_genes",
                "Signature_AUC",
                "Logistic_AUC"
            ]
        ].to_string(index=False)
    )


print("\n" + "=" * 60)
print("ABLATION ANALYSIS COMPLETE")
print("=" * 60)

print("\nSaved:")
print(outfile)

