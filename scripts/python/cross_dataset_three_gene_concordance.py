import os
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


OUTDIR = "04_ML/Final_Model/Validation"
MANUSCRIPT_OUT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT_OUT, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled_sd = np.sqrt(
        (
            (len(a) - 1) * np.var(a, ddof=1)
            +
            (len(b) - 1) * np.var(b, ddof=1)
        )
        /
        (len(a) + len(b) - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (np.mean(a) - np.mean(b)) / pooled_sd


def load_gse48350():

    expr = pd.read_csv(
        "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
        index_col=0
    )

    meta = pd.read_csv(
        "04_ML/GSE48350_RMA_train_meta.csv"
    )

    meta = meta.rename(
        columns={"index": "GSM"}
    )

    expr.index = expr.index.astype(str)
    meta["GSM"] = meta["GSM"].astype(str)

    meta = meta.set_index("GSM")

    common = expr.index.intersection(meta.index)

    expr = expr.loc[common]
    meta = meta.loc[common]

    data = expr[GENES].copy()

    data["Diagnosis"] = meta["Diagnosis"].values

    return data


def load_gse5281():

    # Use the finalized locked 3-gene external-validation matrix.
    # This preserves the exact expression values used for the
    # final GSE5281 external validation.

    f = (
        "04_ML/Final_Characterization/"
        "GSE5281_3gene_predictions_final.csv"
    )

    data = pd.read_csv(f)

    data["GSM"] = data["GSM"].astype(str)

    data = data.set_index("GSM")

    required = [
        "Diagnosis",
        "ABCA6",
        "CRLF1",
        "TNFRSF11B"
    ]

    missing = [
        x for x in required
        if x not in data.columns
    ]

    if missing:
        raise ValueError(
            f"GSE5281 finalized matrix missing: {missing}"
        )

    data = data[
        required
    ].copy()

    data["Diagnosis"] = (
        data["Diagnosis"]
        .astype(str)
        .str.strip()
    )

    return data

def load_gse63060():

    expr = pd.read_csv(
        "09_CrossTissue/results/"
        "GSE63060_three_gene_genelevel_matrix.csv",
        index_col=0
    )

    meta = pd.read_csv(
        "09_CrossTissue/results/"
        "GSE63060_sample_metadata_raw.csv"
    )

    meta["GSM"] = meta["GSM"].astype(str)
    expr.index = expr.index.astype(str)

    meta = meta.set_index("GSM")

    common = expr.index.intersection(meta.index)

    expr = expr.loc[common]
    meta = meta.loc[common]

    data = expr[GENES].copy()

    # Parse the actual GSE63060 status labels.
    # The dataset uses CTL rather than Control.
    status = (
        meta["Characteristic_1"]
        .astype(str)
        .str.replace("status:", "", regex=False)
        .str.strip()
    )

    data["Diagnosis"] = status.map({
        "AD": "AD",
        "CTL": "Control",
        "MCI": "MCI"
    })

    # GSE63060 contains MCI; retain only AD vs Control
    # for direct case-control concordance.
    data = data[
        data["Diagnosis"].isin(["AD", "Control"])
    ].copy()

    return data


# ============================================================
# LOAD DATASETS
# ============================================================

datasets = {
    "GSE48350_Training": load_gse48350(),
    "GSE5281_External": load_gse5281(),
    "GSE63060_Blood": load_gse63060()
}


print("=" * 75)
print("THREE-GENE CROSS-DATASET CONCORDANCE")
print("=" * 75)


# ============================================================
# DATASET QC
# ============================================================

for name, data in datasets.items():

    print("\n" + "-" * 75)
    print(name)
    print("-" * 75)

    print("Shape:", data.shape)

    print(
        "Diagnosis counts:"
    )

    print(
        data["Diagnosis"].value_counts()
    )

    for gene in GENES:

        if gene not in data.columns:
            raise ValueError(
                f"{gene} missing from {name}"
            )

        print(
            f"{gene:12s} FOUND"
        )


# ============================================================
# DISCOVERY DIRECTION
# ============================================================

reference = datasets[
    "GSE48350_Training"
]

reference_directions = {}

for gene in GENES:

    ad = reference.loc[
        reference["Diagnosis"] == "AD",
        gene
    ].astype(float)

    control = reference.loc[
        reference["Diagnosis"] == "Control",
        gene
    ].astype(float)

    delta = ad.mean() - control.mean()

    if delta > 0:
        direction = "UP_AD"

    elif delta < 0:
        direction = "DOWN_AD"

    else:
        direction = "NO_CHANGE"

    reference_directions[gene] = direction

    print(
        f"\nREFERENCE {gene}"
    )
    print(
        "AD n:", len(ad),
        "Control n:", len(control)
    )
    print(
        "AD mean:", round(ad.mean(), 5)
    )
    print(
        "Control mean:", round(control.mean(), 5)
    )
    print(
        "Delta:", round(delta, 5)
    )
    print(
        "Direction:", direction
    )


# ============================================================
# GENE-LEVEL CONCORDANCE
# ============================================================

results = []

for dataset_name, data in datasets.items():

    for gene in GENES:

        ad = data.loc[
            data["Diagnosis"] == "AD",
            gene
        ].astype(float).values

        control = data.loc[
            data["Diagnosis"] == "Control",
            gene
        ].astype(float).values

        if len(ad) < 2 or len(control) < 2:

            results.append({
                "Dataset": dataset_name,
                "Gene": gene,
                "N_AD": len(ad),
                "N_Control": len(control),
                "AD_minus_Control": np.nan,
                "Cohens_d": np.nan,
                "P_value": np.nan,
                "Direction": "INSUFFICIENT_DATA",
                "Reference_Direction":
                    reference_directions[gene],
                "Direction_Concordant": False
            })

            continue

        delta = (
            np.mean(ad)
            -
            np.mean(control)
        )

        d = cohens_d(
            ad,
            control
        )

        U, p = mannwhitneyu(
            ad,
            control,
            alternative="two-sided"
        )

        if delta > 0:
            direction = "UP_AD"

        elif delta < 0:
            direction = "DOWN_AD"

        else:
            direction = "NO_CHANGE"

        concordant = (
            direction
            ==
            reference_directions[gene]
        )

        results.append({
            "Dataset": dataset_name,
            "Gene": gene,
            "N_AD": len(ad),
            "N_Control": len(control),
            "AD_minus_Control": delta,
            "Cohens_d": d,
            "P_value": p,
            "Direction": direction,
            "Reference_Direction":
                reference_directions[gene],
            "Direction_Concordant": concordant
        })


results_df = pd.DataFrame(results)


# ============================================================
# MULTIPLE-TEST CORRECTION
# ============================================================

valid = results_df["P_value"].notna()

results_df["FDR"] = np.nan

if valid.sum() > 0:

    results_df.loc[
        valid,
        "FDR"
    ] = multipletests(
        results_df.loc[valid, "P_value"],
        method="fdr_bh"
    )[1]


# ============================================================
# SUMMARY
# ============================================================

summary = []

for gene in GENES:

    sub = results_df[
        results_df["Gene"] == gene
    ]

    concordant = (
        sub["Direction_Concordant"]
        .sum()
    )

    valid_datasets = (
        sub["P_value"]
        .notna()
        .sum()
    )

    summary.append({
        "Gene": gene,
        "Reference_Direction":
            reference_directions[gene],
        "Datasets_Evaluated":
            valid_datasets,
        "Concordant_Datasets":
            concordant,
        "Concordance_Fraction":
            (
                concordant / valid_datasets
                if valid_datasets > 0
                else np.nan
            ),
        "Mean_Cohens_d":
            sub["Cohens_d"].mean(),
        "Min_Cohens_d":
            sub["Cohens_d"].min(),
        "Max_Cohens_d":
            sub["Cohens_d"].max()
    })


summary_df = pd.DataFrame(summary)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("GENE-LEVEL RESULTS")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 75)
print("DIRECTIONAL CONCORDANCE SUMMARY")
print("=" * 75)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

outfile1 = (
    f"{OUTDIR}/three_gene_cross_dataset_concordance.csv"
)

outfile2 = (
    f"{OUTDIR}/three_gene_directional_concordance_summary.csv"
)

outfile3 = (
    f"{MANUSCRIPT_OUT}/"
    "Table_three_gene_cross_dataset_concordance.csv"
)

results_df.to_csv(
    outfile1,
    index=False
)

summary_df.to_csv(
    outfile2,
    index=False
)

results_df.to_csv(
    outfile3,
    index=False
)


print("\n" + "=" * 75)
print("CONCORDANCE ANALYSIS COMPLETE")
print("=" * 75)

print("\nSaved:")
print(outfile1)
print(outfile2)
print(outfile3)

