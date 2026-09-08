import os
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score


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
        "04_ML/GSE48350_RMA_train_meta.csv"
    ),

    "GSE48350_HeldOut_Test": (
        "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
        "04_ML/GSE48350_RMA_test_meta.csv"
    ),

    "GSE5281_External": (
        "04_ML/External_Validation/GSE5281_LASSO_candidate_matrix.csv",
        "04_ML/External_Validation/GSE5281_validation_metadata.csv"
    )
}

results = []
sample_scores = []


print("=" * 60)
print("THREE-GENE STANDARDIZED SIGNATURE ANALYSIS")
print("=" * 60)


for dataset, (expr_file, meta_file) in DATASETS.items():

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

    diagnosis_col = (
        "Diagnosis"
        if "Diagnosis" in meta.columns
        else "diagnosis"
    )

    if list(expr.index) != list(meta.index):
        raise ValueError(
            f"Sample alignment failed: {dataset}"
        )

    # --------------------------------------------------------
    # Extract three genes
    # --------------------------------------------------------

    x = expr[GENES].astype(float).copy()

    # --------------------------------------------------------
    # Within-dataset Z-score
    # --------------------------------------------------------

    z = (
        x - x.mean(axis=0)
    ) / x.std(axis=0, ddof=1)

    # --------------------------------------------------------
    # Equal-weight signature
    # --------------------------------------------------------

    signature = z.mean(axis=1)

    y = (
        meta[diagnosis_col]
        .map({
            "Control": 0,
            "AD": 1
        })
        .values
    )

    ad_score = signature[y == 1]
    control_score = signature[y == 0]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    u, p = mannwhitneyu(
        ad_score,
        control_score,
        alternative="two-sided"
    )

    auc = roc_auc_score(
        y,
        signature.values
    )

    pr_auc = average_precision_score(
        y,
        signature.values
    )

    difference = (
        ad_score.mean()
        - control_score.mean()
    )

    print(
        f"Samples: {len(signature)}"
    )

    print(
        f"AD signature mean: "
        f"{ad_score.mean():.4f}"
    )

    print(
        f"Control signature mean: "
        f"{control_score.mean():.4f}"
    )

    print(
        f"AD - Control: "
        f"{difference:.4f}"
    )

    print(
        f"Mann-Whitney p: "
        f"{p:.6g}"
    )

    print(
        f"ROC-AUC: "
        f"{auc:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Save sample-level scores
    # --------------------------------------------------------

    temp = pd.DataFrame({
        "GSM": expr.index,
        "Dataset": dataset,
        "Diagnosis": meta[diagnosis_col].values,
        "ABCA6_z": z["ABCA6"].values,
        "CRLF1_z": z["CRLF1"].values,
        "TNFRSF11B_z": z["TNFRSF11B"].values,
        "Three_gene_signature": signature.values
    })

    sample_scores.append(temp)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results.append({
        "Dataset": dataset,
        "N": len(signature),
        "AD_n": int(sum(y == 1)),
        "Control_n": int(sum(y == 0)),
        "AD_mean_signature": ad_score.mean(),
        "Control_mean_signature": control_score.mean(),
        "AD_minus_Control": difference,
        "MannWhitney_U": u,
        "P_value": p,
        "ROC_AUC": auc,
        "PR_AUC": pr_auc
    })


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_file = (
    f"{OUTDIR}/three_gene_signature_validation.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

scores_df = pd.concat(
    sample_scores,
    ignore_index=True
)

scores_file = (
    f"{OUTDIR}/three_gene_signature_sample_scores.csv"
)

scores_df.to_csv(
    scores_file,
    index=False
)


print("\n" + "=" * 60)
print("THREE-GENE SIGNATURE ANALYSIS COMPLETE")
print("=" * 60)

print("\nResults:")
print(results_df.to_string(index=False))

print("\nSaved:")
print(results_file)
print(scores_file)

print("=" * 60)

