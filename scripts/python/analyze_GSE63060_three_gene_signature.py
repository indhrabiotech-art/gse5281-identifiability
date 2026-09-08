import os
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

EXPR_FILE = (
    "09_CrossTissue/results/"
    "GSE63060_three_gene_genelevel_matrix.csv"
)

META_FILE = (
    "09_CrossTissue/results/"
    "GSE63060_sample_metadata_raw.csv"
)

OUTDIR = "09_CrossTissue/results"
os.makedirs(OUTDIR, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GSE63060 THREE-GENE SIGNATURE ANALYSIS")
print("=" * 70)

expr = pd.read_csv(
    EXPR_FILE,
    index_col=0
)

meta = pd.read_csv(
    META_FILE
)

meta = meta.set_index("GSM")

print("\nExpression shape:", expr.shape)
print("Metadata shape:", meta.shape)


# ============================================================
# ALIGN SAMPLES
# ============================================================

common = expr.index.intersection(meta.index)

expr = expr.loc[common]
meta = meta.loc[common]

print("\nCommon samples:", len(common))

if len(common) != len(expr):
    raise ValueError("Sample alignment problem.")


# ============================================================
# EXTRACT DIAGNOSIS
# ============================================================

print("\nMetadata characteristics:")

for col in meta.columns:
    print(
        f"\n{col}:"
    )
    print(
        meta[col].value_counts(dropna=False)
    )


# ------------------------------------------------------------
# Find diagnosis/status column
# ------------------------------------------------------------

diagnosis = None

candidate_columns = [
    "Diagnosis",
    "diagnosis",
    "Status",
    "status",
    "Characteristic_1",
    "Characteristic_2",
    "Characteristic_3",
    "Characteristic_4",
    "Characteristic_5",
    "Characteristic_6"
]

for col in candidate_columns:

    if col not in meta.columns:
        continue

    values = (
        meta[col]
        .astype(str)
        .str.lower()
    )

    if (
        values.str.contains("ad", regex=False).any()
        or
        values.str.contains("control", regex=False).any()
        or
        values.str.contains("ctl", regex=False).any()
        or
        values.str.contains("mci", regex=False).any()
    ):

        diagnosis = col
        break


if diagnosis is None:
    raise ValueError(
        "Could not identify diagnosis column."
    )


print("\nUsing diagnosis column:", diagnosis)


# ============================================================
# NORMALIZE DIAGNOSIS LABELS
# ============================================================

raw = (
    meta[diagnosis]
    .astype(str)
    .str.strip()
    .str.lower()
)

# GSE63060 stores diagnosis as:
# "status: AD"
# "status: CTL"
# "status: MCI"
#
# Extract the diagnosis explicitly rather than relying on
# exact-string matching.

def classify(x):

    x = str(x).strip().lower()

    if "status:" in x:
        x = x.split("status:", 1)[1].strip()

    if x == "ad" or "alzheimer" in x:
        return "AD"

    if x == "mci" or "mild cognitive" in x:
        return "MCI"

    if x == "ctl" or x == "control" or "healthy" in x:
        return "Control"

    return np.nan


meta["Group"] = raw.map(classify)


print("\nRaw diagnosis values:")
print(raw.value_counts(dropna=False))

print("\nFinal groups:")
print(
    meta["Group"]
    .value_counts(dropna=False)
)

# Hard QC check
expected_counts = {
    "AD": 145,
    "Control": 104,
    "MCI": 80
}

observed_counts = meta["Group"].value_counts().to_dict()

for group, expected in expected_counts.items():

    observed = observed_counts.get(group, 0)

    if observed != expected:
        raise ValueError(
            f"Diagnosis parsing error: {group} "
            f"expected {expected}, observed {observed}"
        )

# ============================================================
# REMOVE UNCLASSIFIED SAMPLES
# ============================================================

keep = meta["Group"].notna()

expr = expr.loc[keep]
meta = meta.loc[keep]

print(
    "\nSamples retained:",
    len(expr)
)


# ============================================================
# WITHIN-DATASET STANDARDIZATION
# ============================================================

x = expr[GENES].astype(float)

z = (
    x - x.mean(axis=0)
) / x.std(axis=0, ddof=1)


# ============================================================
# EQUAL-WEIGHT THREE-GENE SIGNATURE
# ============================================================

signature = z.mean(axis=1)

results = pd.DataFrame({
    "GSM": expr.index,
    "Group": meta["Group"].values,
    "ABCA6": x["ABCA6"].values,
    "CRLF1": x["CRLF1"].values,
    "TNFRSF11B": x["TNFRSF11B"].values,
    "ABCA6_z": z["ABCA6"].values,
    "CRLF1_z": z["CRLF1"].values,
    "TNFRSF11B_z": z["TNFRSF11B"].values,
    "Three_gene_signature": signature.values
})


# ============================================================
# GROUP SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GROUP SUMMARY")
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
    f"{OUTDIR}/GSE63060_three_gene_group_summary.csv"
)


# ============================================================
# KRUSKAL-WALLIS
# ============================================================

groups = {}

for group in ["Control", "MCI", "AD"]:

    values = (
        results.loc[
            results["Group"] == group,
            "Three_gene_signature"
        ]
        .values
    )

    groups[group] = values


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

    # Rank-biserial correlation
    n1 = len(a)
    n2 = len(b)

    rbc = (
        1
        - (2 * U) / (n1 * n2)
    )

    pairwise.append({
        "Group1": g1,
        "Group2": g2,
        "N1": n1,
        "N2": n2,
        "MannWhitney_U": U,
        "P_value": p,
        "Rank_biserial": rbc
    })


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
    f"{OUTDIR}/GSE63060_three_gene_pairwise_tests.csv",
    index=False
)


# ============================================================
# ROC ANALYSIS
# ============================================================

def binary_auc(group_positive, group_negative):

    sub = results[
        results["Group"].isin(
            [group_positive, group_negative]
        )
    ].copy()

    y = (
        sub["Group"]
        == group_positive
    ).astype(int)

    score = sub["Three_gene_signature"]

    auc = roc_auc_score(
        y,
        score
    )

    pr_auc = average_precision_score(
        y,
        score
    )

    return {
        "Positive": group_positive,
        "Negative": group_negative,
        "N": len(sub),
        "Positive_N": int(y.sum()),
        "Negative_N": int((1-y).sum()),
        "ROC_AUC": auc,
        "PR_AUC": pr_auc
    }


roc_results = pd.DataFrame([
    binary_auc("AD", "Control"),
    binary_auc("MCI", "Control"),
    binary_auc("AD", "MCI")
])

print("\n" + "=" * 70)
print("ROC ANALYSIS")
print("=" * 70)

print(
    roc_results.to_string(index=False)
)

roc_results.to_csv(
    f"{OUTDIR}/GSE63060_three_gene_ROC_results.csv",
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

rho, p_trend = spearmanr(
    results["Disease_stage"],
    results["Three_gene_signature"]
)

print("\n" + "=" * 70)
print("DISEASE-STAGE TREND")
print("=" * 70)

print("Spearman rho:", rho)
print("P value:", p_trend)


trend = pd.DataFrame([{
    "Spearman_rho": rho,
    "P_value": p_trend,
    "Interpretation":
        "Control -> MCI -> AD"
}])

trend.to_csv(
    f"{OUTDIR}/GSE63060_three_gene_disease_stage_trend.csv",
    index=False
)


# ============================================================
# SAVE SAMPLE SCORES
# ============================================================

results.to_csv(
    f"{OUTDIR}/GSE63060_three_gene_signature_scores.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved:")
print(
    f"{OUTDIR}/GSE63060_three_gene_group_summary.csv"
)

print(
    f"{OUTDIR}/GSE63060_three_gene_pairwise_tests.csv"
)

print(
    f"{OUTDIR}/GSE63060_three_gene_ROC_results.csv"
)

print(
    f"{OUTDIR}/GSE63060_three_gene_disease_stage_trend.csv"
)

print(
    f"{OUTDIR}/GSE63060_three_gene_signature_scores.csv"
)

