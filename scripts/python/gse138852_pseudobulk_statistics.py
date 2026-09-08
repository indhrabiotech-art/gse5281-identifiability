import os
import itertools
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu

BASE = os.path.expanduser(
    "~/project_ml/11_SingleCell/GSE138852"
)

RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

INPUT = os.path.join(
    RESULTS,
    "GSE138852_ABCA6_CRLF1_pseudobulk.csv"
)

OUTPUT = os.path.join(
    RESULTS,
    "GSE138852_ABCA6_CRLF1_pseudobulk_statistics.csv"
)

GENES = ["ABCA6", "CRLF1"]

df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# Exact permutation test
# ------------------------------------------------------------

def exact_permutation_p(ad_values, ct_values):

    values = np.concatenate([
        ad_values,
        ct_values
    ])

    n_ad = len(ad_values)

    observed = (
        np.mean(ad_values)
        - np.mean(ct_values)
    )

    differences = []

    indices = range(len(values))

    for ad_idx in itertools.combinations(
        indices,
        n_ad
    ):

        ad_idx = set(ad_idx)

        perm_ad = np.array([
            values[i]
            for i in indices
            if i in ad_idx
        ])

        perm_ct = np.array([
            values[i]
            for i in indices
            if i not in ad_idx
        ])

        differences.append(
            np.mean(perm_ad)
            - np.mean(perm_ct)
        )

    differences = np.array(differences)

    p = np.mean(
        np.abs(differences)
        >= abs(observed)
    )

    return p


# ------------------------------------------------------------
# Cohen's d
# ------------------------------------------------------------

def cohens_d(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na = len(a)
    nb = len(b)

    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)

    pooled_sd = np.sqrt(
        (
            (na - 1) * va
            +
            (nb - 1) * vb
        )
        /
        (na + nb - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (
        np.mean(a)
        -
        np.mean(b)
    ) / pooled_sd


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------

results = []

for gene in GENES:

    value_col = (
        gene +
        "_mean_expression"
    )

    for cell_type in sorted(
        df["Cell_type"].unique()
    ):

        sub = df[
            df["Cell_type"] == cell_type
        ].copy()

        ad = sub[
            sub["Condition"] == "AD"
        ][value_col].dropna().values

        ct = sub[
            sub["Condition"] == "ct"
        ][value_col].dropna().values

        if len(ad) != 3 or len(ct) != 3:
            continue

        ad_mean = np.mean(ad)
        ct_mean = np.mean(ct)

        pseudocount = 1e-6

        fold_change = (
            (ad_mean + pseudocount)
            /
            (ct_mean + pseudocount)
        )

        log2_fc = np.log2(
            fold_change
        )

        d = cohens_d(ad, ct)

        exact_p = exact_permutation_p(
            ad,
            ct
        )

        try:

            mw = mannwhitneyu(
                ad,
                ct,
                alternative="two-sided",
                method="exact"
            )

            mw_p = mw.pvalue

        except Exception:

            mw_p = np.nan

        results.append({

            "Gene": gene,

            "Cell_type": cell_type,

            "N_AD_samples": len(ad),

            "N_Control_samples": len(ct),

            "AD_mean": ad_mean,

            "Control_mean": ct_mean,

            "Fold_change_AD_vs_Control":
                fold_change,

            "log2_FC_AD_vs_Control":
                log2_fc,

            "Cohens_d":
                d,

            "Exact_permutation_p":
                exact_p,

            "Mann_Whitney_exact_p":
                mw_p
        })


stats = pd.DataFrame(results)

# ------------------------------------------------------------
# FDR
# ------------------------------------------------------------

stats["FDR_BH"] = np.nan

valid = stats[
    "Exact_permutation_p"
].notna()

pvals = stats.loc[
    valid,
    "Exact_permutation_p"
].values

order = np.argsort(pvals)

ranked = pvals[order]

fdr = (
    ranked
    *
    len(ranked)
    /
    np.arange(
        1,
        len(ranked) + 1
    )
)

fdr = np.minimum.accumulate(
    fdr[::-1]
)[::-1]

fdr = np.minimum(
    fdr,
    1.0
)

stats.loc[
    valid,
    "FDR_BH"
] = fdr[
    np.argsort(order)
]

# ------------------------------------------------------------
# Sort by strongest absolute effect
# ------------------------------------------------------------

stats["Absolute_log2_FC"] = (
    stats["log2_FC_AD_vs_Control"]
    .abs()
)

stats = stats.sort_values(
    "Absolute_log2_FC",
    ascending=False
)

stats.to_csv(
    OUTPUT,
    index=False
)

# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("=" * 80)
print("GSE138852 — SAMPLE-LEVEL PSEUDOBULK STATISTICS")
print("=" * 80)

print("\nResults:")

print(
    stats.to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT)

print("\n" + "=" * 80)
print("PSEUDOBULK STATISTICS COMPLETE")
print("=" * 80)
