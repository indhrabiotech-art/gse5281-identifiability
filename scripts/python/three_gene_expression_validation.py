import os
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu


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


print("=" * 60)
print("THREE-GENE EXPRESSION VALIDATION")
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

    # --------------------------------------------------------
    # Make sure expression and metadata are aligned
    # --------------------------------------------------------

    if list(expr.index) != list(meta.index):
        raise ValueError(
            f"Alignment failed for {dataset}"
        )

    print("Samples:", len(expr))
    print("AD:", sum(meta["Diagnosis"].eq("AD"))
          if "Diagnosis" in meta.columns
          else sum(meta["diagnosis"].eq("AD")))

    diagnosis_col = (
        "Diagnosis"
        if "Diagnosis" in meta.columns
        else "diagnosis"
    )

    for gene in GENES:

        if gene not in expr.columns:
            raise ValueError(
                f"{gene} missing from {dataset}"
            )

        ad = expr.loc[
            meta[diagnosis_col] == "AD",
            gene
        ].astype(float)

        control = expr.loc[
            meta[diagnosis_col] == "Control",
            gene
        ].astype(float)

        # ----------------------------------------------------
        # Mann-Whitney U
        # ----------------------------------------------------

        stat, p = mannwhitneyu(
            ad,
            control,
            alternative="two-sided"
        )

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        ad_mean = ad.mean()
        control_mean = control.mean()

        difference = ad_mean - control_mean

        if difference > 0:
            direction = "AD_higher"
        elif difference < 0:
            direction = "Control_higher"
        else:
            direction = "Equal"

        # ----------------------------------------------------
        # Fold-like difference on log2 expression scale
        # ----------------------------------------------------

        results.append({

            "Dataset": dataset,
            "Gene": gene,

            "AD_n": len(ad),
            "Control_n": len(control),

            "AD_mean": ad_mean,
            "Control_mean": control_mean,

            "AD_median": ad.median(),
            "Control_median": control.median(),

            "AD_minus_Control": difference,

            "Direction": direction,

            "MannWhitney_U": stat,
            "P_value": p
        })

        print(
            f"{gene:12s} | "
            f"AD mean={ad_mean:.4f} | "
            f"Control mean={control_mean:.4f} | "
            f"Difference={difference:.4f} | "
            f"p={p:.5g} | "
            f"{direction}"
        )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

outfile = (
    f"{OUTDIR}/three_gene_expression_validation.csv"
)

results_df.to_csv(
    outfile,
    index=False
)


print("\n" + "=" * 60)
print("THREE-GENE EXPRESSION VALIDATION COMPLETE")
print("=" * 60)

print("\nSaved:")
print(outfile)

print("\nResults:")
print(
    results_df.to_string(index=False)
)

print("=" * 60)

