import os
import itertools
import numpy as np
import pandas as pd

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
        "04_ML/Final_Characterization/"
        "GSE5281_3gene_predictions_final.csv",
        None
    )
}


# ============================================================
# SIGNATURE DEFINITIONS
# ============================================================

SIGNATURES = {
    "ABCA6": ["ABCA6"],
    "CRLF1": ["CRLF1"],
    "TNFRSF11B": ["TNFRSF11B"],

    "ABCA6_CRLF1": [
        "ABCA6",
        "CRLF1"
    ],

    "ABCA6_TNFRSF11B": [
        "ABCA6",
        "TNFRSF11B"
    ],

    "CRLF1_TNFRSF11B": [
        "CRLF1",
        "TNFRSF11B"
    ],

    "ABCA6_CRLF1_TNFRSF11B": [
        "ABCA6",
        "CRLF1",
        "TNFRSF11B"
    ]
}


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(name, expr_file, meta_file):

    if name == "GSE5281_External":

        x = pd.read_csv(expr_file)

        x["GSM"] = x["GSM"].astype(str)

        y = (
            x["Diagnosis"]
            .map({
                "Control": 0,
                "AD": 1
            })
            .values
        )

        expr = x[GENES].astype(float).copy()

        return expr, y


    expr = pd.read_csv(
        expr_file,
        index_col=0
    )

    meta = pd.read_csv(
        meta_file
    )

    if len(expr) != len(meta):

        raise ValueError(
            f"{name}: expression/metadata size mismatch"
        )

    # Explicit alignment using the metadata index column
    if "index" in meta.columns:

        ids = (
            meta["index"]
            .astype(str)
            .values
        )

        if list(expr.index.astype(str)) != list(ids):

            raise ValueError(
                f"{name}: sample alignment failed"
            )

    diagnosis_col = (
        "Diagnosis"
        if "Diagnosis" in meta.columns
        else "diagnosis"
    )

    y = (
        meta[diagnosis_col]
        .map({
            "Control": 0,
            "AD": 1
        })
        .values
    )

    expr = expr[GENES].astype(float).copy()

    return expr, y


# ============================================================
# ANALYSIS
# ============================================================

rows = []


for dataset, files in DATASETS.items():

    expr, y = load_dataset(
        dataset,
        files[0],
        files[1]
    )

    print("\n" + "=" * 70)
    print(dataset)
    print("=" * 70)

    print(
        "N:",
        len(y),
        "| AD:",
        int(y.sum()),
        "| Control:",
        int((y == 0).sum())
    )

    # Within-dataset standardization
    z = (
        expr - expr.mean(axis=0)
    ) / expr.std(axis=0, ddof=1)


    for signature_name, genes in SIGNATURES.items():

        score = z[genes].mean(axis=1)

        auc = roc_auc_score(
            y,
            score
        )

        pr_auc = average_precision_score(
            y,
            score
        )

        rows.append({
            "Dataset": dataset,
            "Signature": signature_name,
            "N_genes": len(genes),
            "Genes": ";".join(genes),
            "ROC_AUC": auc,
            "PR_AUC": pr_auc
        })

        print(
            f"{signature_name:25s}"
            f" AUC={auc:.4f}"
            f" PR-AUC={pr_auc:.4f}"
        )


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(rows)


# ============================================================
# PERFORMANCE RELATIVE TO FULL MODEL
# ============================================================

full = results[
    results["Signature"]
    == "ABCA6_CRLF1_TNFRSF11B"
][
    [
        "Dataset",
        "ROC_AUC",
        "PR_AUC"
    ]
].rename(
    columns={
        "ROC_AUC": "Full_ROC_AUC",
        "PR_AUC": "Full_PR_AUC"
    }
)

results = results.merge(
    full,
    on="Dataset",
    how="left"
)

results["Delta_ROC_AUC"] = (
    results["ROC_AUC"]
    -
    results["Full_ROC_AUC"]
)

results["Delta_PR_AUC"] = (
    results["PR_AUC"]
    -
    results["Full_PR_AUC"]
)


# ============================================================
# SAVE
# ============================================================

outfile = (
    f"{OUTDIR}/"
    "final_three_gene_ablation_results.csv"
)

results.to_csv(
    outfile,
    index=False
)


# Manuscript copy

manuscript_file = (
    "06_Manuscript/Tables/"
    "Table_final_three_gene_ablation.csv"
)

os.makedirs(
    "06_Manuscript/Tables",
    exist_ok=True
)

results.to_csv(
    manuscript_file,
    index=False
)


print("\n" + "=" * 70)
print("FINAL THREE-GENE ABLATION COMPLETE")
print("=" * 70)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved:")
print(outfile)
print(manuscript_file)

