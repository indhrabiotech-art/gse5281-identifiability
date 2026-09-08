import os
import numpy as np
import pandas as pd

from scipy.stats import (
    mannwhitneyu,
    kruskal,
    spearmanr
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

OUTDIR = "09_CrossTissue/results"
os.makedirs(OUTDIR, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

EXPR_FILE = (
    "09_CrossTissue/results/"
    "GSE63060_three_gene_genelevel_matrix.csv"
)

META_FILE = (
    "09_CrossTissue/results/"
    "GSE63060_sample_metadata_raw.csv"
)

expr = pd.read_csv(
    EXPR_FILE,
    index_col=0
)

meta = pd.read_csv(META_FILE)

expr.index = expr.index.astype(str)
meta["GSM"] = meta["GSM"].astype(str)

meta = meta.set_index("GSM")

common = expr.index.intersection(meta.index)

expr = expr.loc[common, GENES]
meta = meta.loc[common]

status = (
    meta["Characteristic_1"]
    .astype(str)
    .str.replace("status:", "", regex=False)
    .str.strip()
)

meta["Diagnosis"] = status.map({
    "AD": "AD",
    "CTL": "Control",
    "MCI": "MCI"
})

data = expr.copy()
data["Diagnosis"] = meta["Diagnosis"]

data = data.dropna(subset=GENES + ["Diagnosis"])

print("=" * 70)
print("GSE63060 THREE-GENE COMPOSITE VALIDATION")
print("=" * 70)

print("\nGroup counts:")
print(data["Diagnosis"].value_counts())

# ------------------------------------------------------------
# Within-dataset standardization
# ------------------------------------------------------------

z = (
    data[GENES] -
    data[GENES].mean(axis=0)
) / data[GENES].std(axis=0, ddof=1)

data["Three_gene_signature"] = z.mean(axis=1)

# ------------------------------------------------------------
# AD vs Control
# ------------------------------------------------------------

case = data[data["Diagnosis"] == "AD"]
control = data[data["Diagnosis"] == "Control"]

y = np.concatenate([
    np.ones(len(case)),
    np.zeros(len(control))
])

score = np.concatenate([
    case["Three_gene_signature"].values,
    control["Three_gene_signature"].values
])

auc = roc_auc_score(y, score)
pr_auc = average_precision_score(y, score)

u, p = mannwhitneyu(
    case["Three_gene_signature"],
    control["Three_gene_signature"],
    alternative="two-sided"
)

ad_mean = case["Three_gene_signature"].mean()
control_mean = control["Three_gene_signature"].mean()

delta = ad_mean - control_mean

pooled_sd = np.sqrt(
    (
        (len(case)-1) * case["Three_gene_signature"].var(ddof=1)
        +
        (len(control)-1) * control["Three_gene_signature"].var(ddof=1)
    )
    /
    (len(case) + len(control) - 2)
)

cohens_d = delta / pooled_sd

print("\n" + "=" * 70)
print("AD VS CONTROL")
print("=" * 70)

print("AD N:", len(case))
print("Control N:", len(control))
print("AD mean:", round(ad_mean, 6))
print("Control mean:", round(control_mean, 6))
print("AD-Control:", round(delta, 6))
print("Cohen's d:", round(cohens_d, 6))
print("Mann-Whitney p:", p)
print("ROC-AUC:", round(auc, 6))
print("PR-AUC:", round(pr_auc, 6))

# ------------------------------------------------------------
# Individual genes vs composite
# ------------------------------------------------------------

rows = []

for gene in GENES:

    gene_score = np.concatenate([
        case[gene].values,
        control[gene].values
    ])

    gene_auc = roc_auc_score(y, gene_score)

    rows.append({
        "Feature": gene,
        "ROC_AUC": gene_auc
    })

rows.append({
    "Feature": "Three_gene_composite",
    "ROC_AUC": auc
})

comparison = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("INDIVIDUAL GENES VS COMPOSITE")
print("=" * 70)

print(comparison.to_string(index=False))

comparison.to_csv(
    f"{OUTDIR}/GSE63060_three_gene_composite_comparison.csv",
    index=False
)

# ------------------------------------------------------------
# Three-group analysis
# ------------------------------------------------------------

groups = {
    group: data.loc[
        data["Diagnosis"] == group,
        "Three_gene_signature"
    ].values
    for group in ["Control", "MCI", "AD"]
}

H, p_kw = kruskal(
    groups["Control"],
    groups["MCI"],
    groups["AD"]
)

print("\n" + "=" * 70)
print("THREE-GROUP ANALYSIS")
print("=" * 70)

print("Kruskal-Wallis H:", H)
print("Kruskal-Wallis p:", p_kw)

summary = (
    data
    .groupby("Diagnosis")["Three_gene_signature"]
    .agg(
        N="count",
        Mean="mean",
        SD="std",
        Median="median"
    )
)

print("\nGroup summary:")
print(summary)

summary.to_csv(
    f"{OUTDIR}/GSE63060_three_gene_composite_group_summary.csv"
)

# ------------------------------------------------------------
# Disease-stage trend
# ------------------------------------------------------------

stage_map = {
    "Control": 0,
    "MCI": 1,
    "AD": 2
}

trend = data["Diagnosis"].map(stage_map)

rho, p_trend = spearmanr(
    trend,
    data["Three_gene_signature"]
)

print("\n" + "=" * 70)
print("DISEASE-STAGE TREND")
print("=" * 70)

print("Spearman rho:", rho)
print("P value:", p_trend)

pd.DataFrame([{
    "Spearman_rho": rho,
    "P_value": p_trend,
    "Interpretation": "Control -> MCI -> AD"
}]).to_csv(
    f"{OUTDIR}/GSE63060_three_gene_composite_stage_trend.csv",
    index=False
)

# ------------------------------------------------------------
# Save sample scores
# ------------------------------------------------------------

scores = pd.DataFrame({
    "GSM": data.index,
    "Diagnosis": data["Diagnosis"].values,
    "ABCA6_z": z["ABCA6"].values,
    "CRLF1_z": z["CRLF1"].values,
    "TNFRSF11B_z": z["TNFRSF11B"].values,
    "Three_gene_signature": data["Three_gene_signature"].values
})

scores.to_csv(
    f"{OUTDIR}/GSE63060_three_gene_composite_scores.csv",
    index=False
)

print("\n" + "=" * 70)
print("GSE63060 COMPOSITE VALIDATION COMPLETE")
print("=" * 70)
