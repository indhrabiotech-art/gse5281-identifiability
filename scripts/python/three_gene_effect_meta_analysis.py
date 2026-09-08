import os
import numpy as np
import pandas as pd

from scipy.stats import chi2, norm


OUTDIR = "04_ML/Final_Model/Validation"
MANUSCRIPT_OUT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT_OUT, exist_ok=True)


# ============================================================
# INPUT
# ============================================================

INPUT = (
    "04_ML/Final_Model/Validation/"
    "three_gene_cross_dataset_concordance.csv"
)

df = pd.read_csv(INPUT)

# Only datasets with both AD and Control
df = df[
    (df["N_AD"] > 1) &
    (df["N_Control"] > 1)
].copy()


GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]


# ============================================================
# EFFECT-SIZE VARIANCE
# ============================================================

# Approximate sampling variance of Cohen's d
# v = (n1+n2)/(n1*n2) + d^2/[2*(n1+n2)]

df["Variance_d"] = (
    (df["N_AD"] + df["N_Control"])
    /
    (df["N_AD"] * df["N_Control"])
    +
    (df["Cohens_d"] ** 2)
    /
    (2 * (df["N_AD"] + df["N_Control"]))
)

df["SE_d"] = np.sqrt(df["Variance_d"])


# ============================================================
# META-ANALYSIS
# ============================================================

results = []


for gene in GENES:

    sub = df[
        df["Gene"] == gene
    ].copy()

    effects = sub["Cohens_d"].to_numpy(dtype=float)
    variances = sub["Variance_d"].to_numpy(dtype=float)

    k = len(effects)

    # --------------------------------------------------------
    # Fixed-effect
    # --------------------------------------------------------

    weights_fixed = 1 / variances

    fixed_d = np.sum(
        weights_fixed * effects
    ) / np.sum(weights_fixed)

    fixed_se = np.sqrt(
        1 / np.sum(weights_fixed)
    )

    fixed_ci_low = (
        fixed_d - 1.96 * fixed_se
    )

    fixed_ci_high = (
        fixed_d + 1.96 * fixed_se
    )

    # --------------------------------------------------------
    # Cochran Q
    # --------------------------------------------------------

    Q = np.sum(
        weights_fixed *
        (effects - fixed_d) ** 2
    )

    df_Q = k - 1

    Q_p = chi2.sf(
        Q,
        df_Q
    ) if df_Q > 0 else np.nan

    # --------------------------------------------------------
    # I2
    # --------------------------------------------------------

    if Q > 0:
        I2 = max(
            0,
            (Q - df_Q) / Q
        ) * 100
    else:
        I2 = 0.0

    # --------------------------------------------------------
    # DerSimonian-Laird tau2
    # --------------------------------------------------------

    if k > 1:

        C = (
            np.sum(weights_fixed)
            -
            (
                np.sum(weights_fixed ** 2)
                /
                np.sum(weights_fixed)
            )
        )

        tau2 = max(
            0,
            (Q - df_Q) / C
        )

    else:
        tau2 = 0.0

    # --------------------------------------------------------
    # Random-effects
    # --------------------------------------------------------

    weights_random = 1 / (
        variances + tau2
    )

    random_d = np.sum(
        weights_random * effects
    ) / np.sum(weights_random)

    random_se = np.sqrt(
        1 / np.sum(weights_random)
    )

    random_ci_low = (
        random_d - 1.96 * random_se
    )

    random_ci_high = (
        random_d + 1.96 * random_se
    )

    random_z = (
        random_d / random_se
    )

    random_p = (
        2 * norm.sf(abs(random_z))
    )

    # --------------------------------------------------------
    # Direction consistency
    # --------------------------------------------------------

    positive = int(
        np.sum(effects > 0)
    )

    negative = int(
        np.sum(effects < 0)
    )

    direction_fraction = (
        max(positive, negative) / k
    )

    results.append({

        "Gene": gene,

        "Datasets": k,

        "Positive_effects": positive,

        "Negative_effects": negative,

        "Direction_consistency": direction_fraction,

        "Fixed_effect_d": fixed_d,

        "Fixed_SE": fixed_se,

        "Fixed_CI_lower": fixed_ci_low,

        "Fixed_CI_upper": fixed_ci_high,

        "Random_effect_d": random_d,

        "Random_SE": random_se,

        "Random_CI_lower": random_ci_low,

        "Random_CI_upper": random_ci_high,

        "Random_p": random_p,

        "Q": Q,

        "Q_df": df_Q,

        "Q_p": Q_p,

        "I2_percent": I2,

        "Tau2": tau2
    })


results_df = pd.DataFrame(results)


# ============================================================
# PRINT
# ============================================================

print("=" * 80)
print("THREE-GENE CROSS-DATASET EFFECT-SIZE META-ANALYSIS")
print("=" * 80)

print("\nInput datasets:")
print(
    df[
        [
            "Dataset",
            "Gene",
            "N_AD",
            "N_Control",
            "Cohens_d",
            "P_value",
            "Direction"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 80)
print("META-ANALYSIS RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

for _, r in results_df.iterrows():

    heterogeneity = (
        "low"
        if r["I2_percent"] < 25
        else
        "moderate"
        if r["I2_percent"] < 50
        else
        "substantial"
        if r["I2_percent"] < 75
        else
        "considerable"
    )

    print(
        f"\n{r['Gene']}:"
    )

    print(
        f"  Random-effects d = "
        f"{r['Random_effect_d']:.4f} "
        f"("
        f"{r['Random_CI_lower']:.4f}, "
        f"{r['Random_CI_upper']:.4f}"
        f")"
    )

    print(
        f"  Direction consistency = "
        f"{r['Direction_consistency']:.2f}"
    )

    print(
        f"  I² = "
        f"{r['I2_percent']:.2f}% "
        f"({heterogeneity} heterogeneity)"
    )

    print(
        f"  Heterogeneity p = "
        f"{r['Q_p']:.4g}"
    )


# ============================================================
# SAVE
# ============================================================

outfile = (
    f"{OUTDIR}/three_gene_effect_meta_analysis.csv"
)

results_df.to_csv(
    outfile,
    index=False
)


manuscript_file = (
    f"{MANUSCRIPT_OUT}/"
    "Table_three_gene_effect_meta_analysis.csv"
)

results_df.to_csv(
    manuscript_file,
    index=False
)


print("\n" + "=" * 80)
print("META-ANALYSIS COMPLETE")
print("=" * 80)

print("\nSaved:")
print(outfile)
print(manuscript_file)

