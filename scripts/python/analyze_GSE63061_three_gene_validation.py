import os
import gzip
import csv
import numpy as np
import pandas as pd

from scipy.stats import (
    kruskal,
    mannwhitneyu,
    spearmanr
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIGURATION
# ============================================================

EXPR_FILE = "09_CrossTissue/GSE63061/GSE63061_normalized.txt.gz"

META_FILE = (
    "09_CrossTissue/results/"
    "GSE63061_sample_metadata_clean.csv"
)

OUTDIR = "09_CrossTissue/results"

os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# CANONICAL THREE-GENE PROBE MAPPING
# ============================================================

PROBES = {
    "ABCA6": "ILMN_1795507",
    "CRLF1": "ILMN_1681515",
    "TNFRSF11B": "ILMN_1676663"
}


# ============================================================
# READ PROBE EXPRESSION
# ============================================================

found = {}

inside = False

with gzip.open(EXPR_FILE, "rt") as fh:

    reader = csv.reader(fh, delimiter="\t")

    for row in reader:

        if not row:
            continue

        first = row[0].strip('"')

        if first == "!series_matrix_table_begin":
            inside = True
            continue

        if first == "!series_matrix_table_end":
            break

        if not inside:
            continue

        probe = first

        for gene, probe_id in PROBES.items():

            if probe == probe_id:

                found[gene] = np.array(
                    [float(v.strip('"')) for v in row[1:]],
                    dtype=float
                )


# ============================================================
# PROBE CHECK
# ============================================================

print("=" * 70)
print("GSE63061 THREE-GENE EXTERNAL VALIDATION")
print("=" * 70)

for gene, probe in PROBES.items():

    if gene not in found:

        raise ValueError(
            f"{gene} / {probe} not found in GSE63061 matrix."
        )

    print(
        f"{gene:12s} {probe:15s} FOUND"
    )


# ============================================================
# BUILD EXPRESSION MATRIX
# ============================================================

expr = pd.DataFrame(
    {
        gene: found[gene]
        for gene in PROBES
    }
)


# ============================================================
# LOAD METADATA
# ============================================================

meta = pd.read_csv(
    META_FILE
)

meta["GSM"] = meta["GSM"].astype(str)

if len(expr) != len(meta):

    raise ValueError(
        "Expression and metadata sample counts do not match."
    )

expr.index = meta["GSM"].values

meta.index = meta["GSM"].values


# ============================================================
# SAMPLE ALIGNMENT
# ============================================================

if not expr.index.equals(meta.index):

    raise ValueError(
        "Expression/metadata sample alignment failed."
    )


print()
print("Expression shape:", expr.shape)
print("Metadata shape:", meta.shape)

print()
print("Group counts:")
print(
    meta["Group"].value_counts(
        dropna=False
    )
)


# ============================================================
# REMOVE UNCLASSIFIED SAMPLES
# ============================================================

keep = meta["Group"].isin(
    ["Control", "MCI", "AD"]
)

expr = expr.loc[keep].copy()
meta = meta.loc[keep].copy()


print()
print(
    "Validation samples retained:",
    len(expr)
)


# ============================================================
# EXPRESSION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPRESSION SUMMARY")
print("=" * 70)

print(
    expr.describe().T[
        [
            "count",
            "mean",
            "std",
            "min",
            "50%",
            "max"
        ]
    ]
)


# ============================================================
# WITHIN-DATASET STANDARDIZATION
# ============================================================

z = (
    expr - expr.mean(axis=0)
) / expr.std(axis=0, ddof=1)


# ============================================================
# CANONICAL THREE-GENE EQUAL-WEIGHT SIGNATURE
# ============================================================

signature = z.mean(axis=1)


# ============================================================
# SAMPLE-LEVEL RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "GSM": expr.index,
        "Group": meta["Group"].values,

        "ABCA6": expr["ABCA6"].values,
        "CRLF1": expr["CRLF1"].values,
        "TNFRSF11B": expr["TNFRSF11B"].values,

        "ABCA6_z": z["ABCA6"].values,
        "CRLF1_z": z["CRLF1"].values,
        "TNFRSF11B_z": z["TNFRSF11B"].values,

        "Three_gene_signature":
            signature.values
    }
)


# ============================================================
# GROUP SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("THREE-GENE GROUP SUMMARY")
print("=" * 70)

summary = (
    results
    .groupby("Group")["Three_gene_signature"]
    .agg(
        N="count",
        Mean="mean",
        SD="std",
        Median="median",
        Min="min",
        Max="max"
    )
)

print(summary)

summary.to_csv(
    f"{OUTDIR}/GSE63061_three_gene_group_summary.csv"
)


# ============================================================
# KRUSKAL-WALLIS
# ============================================================

groups = {}

for group in [
    "Control",
    "MCI",
    "AD"
]:

    groups[group] = results.loc[
        results["Group"] == group,
        "Three_gene_signature"
    ].values


H, p_kw = kruskal(
    groups["Control"],
    groups["MCI"],
    groups["AD"]
)


print("\n" + "=" * 70)
print("KRUSKAL-WALLIS")
print("=" * 70)

print(
    "H statistic:",
    H
)

print(
    "P value:",
    p_kw
)


# ============================================================
# PAIRWISE MANN-WHITNEY TESTS
# ============================================================

comparisons = [
    ("Control", "MCI"),
    ("Control", "AD"),
    ("MCI", "AD")
]

pairwise = []

for g1, g2 in comparisons:

    a = groups[g1]
    b = groups[g2]

    U, p = mannwhitneyu(
        a,
        b,
        alternative="two-sided"
    )

    n1 = len(a)
    n2 = len(b)

    rank_biserial = (
        1 -
        (2 * U) /
        (n1 * n2)
    )

    pairwise.append(
        {
            "Group1": g1,
            "Group2": g2,
            "N1": n1,
            "N2": n2,
            "MannWhitney_U": U,
            "P_value": p,
            "Rank_biserial": rank_biserial
        }
    )


pairwise_df = pd.DataFrame(
    pairwise
)

pairwise_df["FDR"] = multipletests(
    pairwise_df["P_value"],
    method="fdr_bh"
)[1]


print("\n" + "=" * 70)
print("PAIRWISE COMPARISONS")
print("=" * 70)

print(
    pairwise_df.to_string(
        index=False
    )
)

pairwise_df.to_csv(
    f"{OUTDIR}/GSE63061_three_gene_pairwise_tests.csv",
    index=False
)


# ============================================================
# ROC / PR ANALYSIS
# ============================================================

def binary_auc(
    positive,
    negative
):

    sub = results[
        results["Group"].isin(
            [positive, negative]
        )
    ].copy()

    y = (
        sub["Group"] == positive
    ).astype(int)

    score = (
        sub["Three_gene_signature"]
    )

    auc = roc_auc_score(
        y,
        score
    )

    pr_auc = average_precision_score(
        y,
        score
    )

    return {
        "Positive": positive,
        "Negative": negative,
        "N": len(sub),
        "Positive_N": int(y.sum()),
        "Negative_N": int(
            (1 - y).sum()
        ),
        "ROC_AUC": auc,
        "PR_AUC": pr_auc
    }


roc_results = pd.DataFrame(
    [
        binary_auc(
            "AD",
            "Control"
        ),

        binary_auc(
            "MCI",
            "Control"
        ),

        binary_auc(
            "AD",
            "MCI"
        )
    ]
)


print("\n" + "=" * 70)
print("ROC / PR ANALYSIS")
print("=" * 70)

print(
    roc_results.to_string(
        index=False
    )
)

roc_results.to_csv(
    f"{OUTDIR}/GSE63061_three_gene_ROC_results.csv",
    index=False
)


# ============================================================
# DISEASE-STAGE TREND
# ============================================================

stage_map = {
    "Control": 0,
    "MCI": 1,
    "AD": 2
}

results["Disease_stage"] = (
    results["Group"]
    .map(stage_map)
)


trend_data = results.dropna(
    subset=[
        "Disease_stage",
        "Three_gene_signature"
    ]
)


rho, p_trend = spearmanr(
    trend_data["Disease_stage"],
    trend_data["Three_gene_signature"]
)


print("\n" + "=" * 70)
print("DISEASE-STAGE TREND")
print("=" * 70)

print(
    "Spearman rho:",
    rho
)

print(
    "P value:",
    p_trend
)


trend = pd.DataFrame(
    [
        {
            "Spearman_rho": rho,
            "P_value": p_trend,
            "Interpretation":
                "Control -> MCI -> AD"
        }
    ]
)

trend.to_csv(
    f"{OUTDIR}/GSE63061_three_gene_disease_stage_trend.csv",
    index=False
)


# ============================================================
# GENE-WISE GROUP STATISTICS
# ============================================================

gene_stats = []

for gene in PROBES:

    for group in [
        "Control",
        "MCI",
        "AD"
    ]:

        values = results.loc[
            results["Group"] == group,
            gene
        ]

        gene_stats.append(
            {
                "Gene": gene,
                "Group": group,
                "N": len(values),
                "Mean": values.mean(),
                "SD": values.std(),
                "Median": values.median()
            }
        )


gene_stats_df = pd.DataFrame(
    gene_stats
)

gene_stats_df.to_csv(
    f"{OUTDIR}/GSE63061_three_gene_gene_statistics.csv",
    index=False
)


# ============================================================
# SAVE SAMPLE SCORES
# ============================================================

results.to_csv(
    f"{OUTDIR}/GSE63061_three_gene_signature_scores.csv",
    index=False
)


# ============================================================
# VALIDATION SUMMARY
# ============================================================

ad_control = roc_results[
    (roc_results["Positive"] == "AD") &
    (roc_results["Negative"] == "Control")
].iloc[0]


validation_summary = pd.DataFrame(
    [
        {
            "Dataset": "GSE63061",
            "Signature": "ABCA6+CRLF1+TNFRSF11B",
            "N": len(results),
            "AD_n": int(
                (results["Group"] == "AD").sum()
            ),
            "Control_n": int(
                (results["Group"] == "Control").sum()
            ),
            "MCI_n": int(
                (results["Group"] == "MCI").sum()
            ),
            "AD_vs_Control_ROC_AUC":
                ad_control["ROC_AUC"],
            "AD_vs_Control_PR_AUC":
                ad_control["PR_AUC"],
            "Kruskal_Wallis_H": H,
            "Kruskal_Wallis_P": p_kw,
            "Disease_stage_Spearman_rho":
                rho,
            "Disease_stage_P":
                p_trend,
            "External_labels_used_for_training":
                False
        }
    ]
)

validation_summary.to_csv(
    f"{OUTDIR}/GSE63061_three_gene_validation_summary.csv",
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("GSE63061 THREE-GENE EXTERNAL VALIDATION COMPLETE")
print("=" * 70)

print()
print(
    "Canonical signature:"
)

print(
    "ABCA6 + CRLF1 + TNFRSF11B"
)

print()
print(
    "AD vs Control ROC-AUC:",
    f"{ad_control['ROC_AUC']:.6f}"
)

print(
    "AD vs Control PR-AUC:",
    f"{ad_control['PR_AUC']:.6f}"
)

print(
    "Disease-stage Spearman rho:",
    f"{rho:.6f}"
)

print(
    "Disease-stage P:",
    f"{p_trend:.6g}"
)

print()
print(
    "External labels used for training:",
    False
)

print()
print("Saved:")

for filename in [
    "GSE63061_three_gene_group_summary.csv",
    "GSE63061_three_gene_pairwise_tests.csv",
    "GSE63061_three_gene_ROC_results.csv",
    "GSE63061_three_gene_disease_stage_trend.csv",
    "GSE63061_three_gene_gene_statistics.csv",
    "GSE63061_three_gene_signature_scores.csv",
    "GSE63061_three_gene_validation_summary.csv"
]:

    print(
        f"{OUTDIR}/{filename}"
    )

print("=" * 70)
