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

META_FILE = "09_CrossTissue/results/GSE63061_sample_metadata_clean.csv"

OUTDIR = "09_CrossTissue/results"

os.makedirs(OUTDIR, exist_ok=True)


PROBES = {
    "ABCA6": "ILMN_1795507",
    "CRLF1": "ILMN_1681515"
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
# CHECK PROBES
# ============================================================

print("=" * 70)
print("GSE63061 TWO-GENE VALIDATION")
print("=" * 70)

for gene, probe in PROBES.items():

    if gene not in found:
        raise ValueError(
            f"{gene} / {probe} not found in GSE63061 matrix."
        )

    print(
        f"{gene:10s} {probe:15s} FOUND"
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

expr.index = meta["GSM"].values

meta.index = meta["GSM"].values


print()
print("Expression shape:", expr.shape)
print("Metadata shape:", meta.shape)


# ============================================================
# EXPRESSION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPRESSION SUMMARY")
print("=" * 70)

print(
    expr.describe().T[
        ["count", "mean", "std", "min", "50%", "max"]
    ]
)


# ============================================================
# WITHIN-DATASET Z-SCORE
# ============================================================

z = (
    expr - expr.mean(axis=0)
) / expr.std(axis=0, ddof=1)


# ============================================================
# TWO-GENE EQUAL-WEIGHT SCORE
# ============================================================

signature = z.mean(axis=1)


results = pd.DataFrame(
    {
        "GSM": expr.index,
        "Group": meta["Group"].values,
        "ABCA6": expr["ABCA6"].values,
        "CRLF1": expr["CRLF1"].values,
        "ABCA6_z": z["ABCA6"].values,
        "CRLF1_z": z["CRLF1"].values,
        "Two_gene_signature": signature.values
    }
)


# ============================================================
# GROUP SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GROUP SUMMARY")
print("=" * 70)

summary = (
    results
    .groupby("Group")["Two_gene_signature"]
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
    f"{OUTDIR}/GSE63061_two_gene_group_summary.csv"
)


# ============================================================
# KRUSKAL-WALLIS
# ============================================================

groups = {}

for group in ["Control", "MCI", "AD"]:

    groups[group] = results.loc[
        results["Group"] == group,
        "Two_gene_signature"
    ].values


H, p_kw = kruskal(
    groups["Control"],
    groups["MCI"],
    groups["AD"]
)

print("\n" + "=" * 70)
print("KRUSKAL-WALLIS")
print("=" * 70)

print("H statistic:", H)
print("P value:", p_kw)


# ============================================================
# PAIRWISE TESTS
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

    rbc = (
        1 - (2 * U) / (n1 * n2)
    )

    pairwise.append(
        {
            "Group1": g1,
            "Group2": g2,
            "N1": n1,
            "N2": n2,
            "MannWhitney_U": U,
            "P_value": p,
            "Rank_biserial": rbc
        }
    )


pairwise_df = pd.DataFrame(pairwise)

pairwise_df["FDR"] = multipletests(
    pairwise_df["P_value"],
    method="fdr_bh"
)[1]

print("\nPairwise comparisons:")
print(
    pairwise_df.to_string(index=False)
)

pairwise_df.to_csv(
    f"{OUTDIR}/GSE63061_two_gene_pairwise_tests.csv",
    index=False
)


# ============================================================
# ROC ANALYSIS
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

    score = sub["Two_gene_signature"]

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
        "Negative_N": int((1 - y).sum()),
        "ROC_AUC": auc,
        "PR_AUC": pr_auc
    }


roc_results = pd.DataFrame(
    [
        binary_auc("AD", "Control"),
        binary_auc("MCI", "Control"),
        binary_auc("AD", "MCI")
    ]
)

print("\n" + "=" * 70)
print("ROC ANALYSIS")
print("=" * 70)

print(
    roc_results.to_string(index=False)
)

roc_results.to_csv(
    f"{OUTDIR}/GSE63061_two_gene_ROC_results.csv",
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
    results["Group"].map(stage_map)
)

trend_data = results.dropna(subset=["Disease_stage", "Two_gene_signature"])

rho, p_trend = spearmanr(
    trend_data["Disease_stage"],
    trend_data["Two_gene_signature"]
)

print("\n" + "=" * 70)
print("DISEASE-STAGE TREND")
print("=" * 70)

print("Spearman rho:", rho)
print("P value:", p_trend)


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
    f"{OUTDIR}/GSE63061_two_gene_disease_stage_trend.csv",
    index=False
)


# ============================================================
# SAVE SAMPLE SCORES
# ============================================================

results.to_csv(
    f"{OUTDIR}/GSE63061_two_gene_signature_scores.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("GSE63061 TWO-GENE VALIDATION COMPLETE")
print("=" * 70)

print("\nSaved:")
print(
    f"{OUTDIR}/GSE63061_two_gene_group_summary.csv"
)
print(
    f"{OUTDIR}/GSE63061_two_gene_pairwise_tests.csv"
)
print(
    f"{OUTDIR}/GSE63061_two_gene_ROC_results.csv"
)
print(
    f"{OUTDIR}/GSE63061_two_gene_disease_stage_trend.csv"
)
print(
    f"{OUTDIR}/GSE63061_two_gene_signature_scores.csv"
)

