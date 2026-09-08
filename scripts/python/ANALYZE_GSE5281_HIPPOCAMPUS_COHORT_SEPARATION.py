#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)

ROOT = Path.home() / "project_ml"

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

EXPR = (
    ROOT /
    "03_Preprocessing/"
    "GSE5281_RMA_genelevel.csv"
)

META = (
    ROOT /
    "04_ML/External_Validation/"
    "GSE5281_hippocampus_SOFT_complete_metadata.csv"
)

OUTDIR = (
    ROOT /
    "04_ML/External_Validation/"
)

FIGDIR = (
    ROOT /
    "07_Figures/RMA_QC"
)

OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("GSE5281 HIPPOCAMPAL COHORT-SEPARATION ANALYSIS")
print("=" * 90)

# ------------------------------------------------------------
# LOAD EXPRESSION
# ------------------------------------------------------------

print("\nLoading expression matrix:")

expr = pd.read_csv(EXPR, index_col=0)

print("Expression shape:", expr.shape)

# ------------------------------------------------------------
# NORMALIZE SAMPLE NAMES
# ------------------------------------------------------------

expr.columns = (
    expr.columns
    .astype(str)
    .str.replace(".CEL.gz", "", regex=False)
    .str.replace(".CEL", "", regex=False)
)

# If expression matrix is accidentally transposed
if expr.shape[1] != 23 and expr.shape[0] == 23:
    expr = expr.T

print("Final expression shape:", expr.shape)

# ------------------------------------------------------------
# KEEP EXACT HIPPOCAMPAL SUBSET
# ------------------------------------------------------------

controls = [
    f"GSM{i}"
    for i in range(119628, 119641)
]

ad = [
    f"GSM{i}"
    for i in range(238799, 238809)
]

samples = controls + ad

missing = [
    s for s in samples
    if s not in expr.columns
]

if missing:
    print("\nERROR: Missing samples:")
    print(missing)
    raise SystemExit(1)

expr = expr[samples]

print("\nHippocampal samples:", len(samples))
print("Controls:", len(controls))
print("AD:", len(ad))

# ------------------------------------------------------------
# COHORT LABEL
# ------------------------------------------------------------

cohort = np.array(
    [0] * len(controls) +
    [1] * len(ad)
)

cohort_name = np.array(
    ["2006_control_cohort"] * len(controls) +
    ["2007_AD_cohort"] * len(ad)
)

# ------------------------------------------------------------
# REMOVE NON-NUMERIC / CONSTANT FEATURES
# ------------------------------------------------------------

expr = expr.apply(pd.to_numeric, errors="coerce")

expr = expr.dropna(axis=0, how="any")

variance = expr.var(axis=1)

expr = expr.loc[variance > 0]

print("\nGenes after filtering:", expr.shape[0])

# ------------------------------------------------------------
# SAMPLE × GENE MATRIX
# ------------------------------------------------------------

X = expr.T

# ------------------------------------------------------------
# PCA
# ------------------------------------------------------------

print("\nRunning PCA...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(
    n_components=min(10, X.shape[0], X.shape[1])
)

PC = pca.fit_transform(X_scaled)

pc1 = PC[:, 0]
pc2 = PC[:, 1]

print(
    f"PC1 variance explained: "
    f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
)

print(
    f"PC2 variance explained: "
    f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
)

pc12 = (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100
print(f"PC1 + PC2 variance explained: {pc12:.2f}%")
# ------------------------------------------------------------
# PC / COHORT ASSOCIATION
# ------------------------------------------------------------

pc_summary = pd.DataFrame({
    "GSM": samples,
    "cohort": cohort_name,
    "cohort_binary": cohort,
    "PC1": pc1,
    "PC2": pc2
})

print("\nPC scores:")
print(pc_summary.to_string(index=False))

print("\nPC1 cohort means:")

for c in [0, 1]:

    label = (
        "Jul_10_2006"
        if c == 0
        else "Oct_19_2007"
    )

    vals = pc1[cohort == c]

    print(
        f"{label}: "
        f"mean={vals.mean():.4f}, "
        f"SD={vals.std(ddof=1):.4f}"
    )

# ------------------------------------------------------------
# COHORT CLASSIFICATION
#
# IMPORTANT:
# This is NOT disease prediction.
# It asks whether genome-wide expression can identify
# submission cohort.
# ------------------------------------------------------------

print("\nRunning leave-one-out cohort classification...")

classifier = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "logistic",
        LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=1.0,
            max_iter=10000,
            random_state=42
        )
    )
])

loo = LeaveOneOut()

prob = cross_val_predict(
    classifier,
    X,
    cohort,
    cv=loo,
    method="predict_proba"
)[:, 1]

pred = (
    prob >= 0.5
).astype(int)

auc = roc_auc_score(
    cohort,
    prob
)

accuracy = accuracy_score(
    cohort,
    pred
)

balanced_accuracy = balanced_accuracy_score(
    cohort,
    pred
)

cm = confusion_matrix(
    cohort,
    pred
)

print("\nCOHORT CLASSIFICATION RESULTS")
print("-" * 90)

print(f"LOOCV ROC-AUC:          {auc:.6f}")
print(f"LOOCV accuracy:         {accuracy:.6f}")
print(f"LOOCV balanced accuracy:{balanced_accuracy:.6f}")

print("\nConfusion matrix:")
print(cm)

# ------------------------------------------------------------
# PERMUTATION TEST
# ------------------------------------------------------------

print("\nRunning cohort-label permutation test...")

rng = np.random.default_rng(42)

n_perm = 5000

permuted_auc = np.zeros(n_perm)

for i in range(n_perm):

    shuffled = rng.permutation(cohort)

    try:

        perm_prob = cross_val_predict(
            classifier,
            X,
            shuffled,
            cv=loo,
            method="predict_proba"
        )[:, 1]

        permuted_auc[i] = roc_auc_score(
            shuffled,
            perm_prob
        )

    except Exception:

        permuted_auc[i] = np.nan

permuted_auc = (
    permuted_auc[
        np.isfinite(permuted_auc)
    ]
)

extreme = (
    np.abs(permuted_auc - 0.5)
    >=
    np.abs(auc - 0.5)
)

p_value = (
    np.sum(extreme) + 1
) / (
    len(permuted_auc) + 1
)

print(f"Permutations: {len(permuted_auc)}")
print(f"Empirical two-sided p-value: {p_value:.6f}")

# ------------------------------------------------------------
# COHORT-SPECIFIC EXPRESSION DISTRIBUTION
# ------------------------------------------------------------

global_effects = []

for gene in expr.index:

    x0 = expr.loc[gene, controls].values
    x1 = expr.loc[gene, ad].values

    pooled_sd = np.sqrt(
        (
            (len(x0) - 1) * np.var(x0, ddof=1)
            +
            (len(x1) - 1) * np.var(x1, ddof=1)
        )
        /
        (
            len(x0) + len(x1) - 2
        )
    )

    if pooled_sd > 0:

        d = (
            np.mean(x1) - np.mean(x0)
        ) / pooled_sd

    else:

        d = np.nan

    global_effects.append({
        "gene": gene,
        "mean_2006": np.mean(x0),
        "mean_2007": np.mean(x1),
        "cohort_effect_direction": d
    })

effects = pd.DataFrame(
    global_effects
)

effects["abs_effect"] = (
    effects["cohort_effect_direction"]
    .abs()
)

effects = effects.sort_values(
    "abs_effect",
    ascending=False
)

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

pc_summary.to_csv(
    OUTDIR /
    "GSE5281_hippocampus_cohort_PCA_scores.csv",
    index=False
)

effects.to_csv(
    OUTDIR /
    "GSE5281_hippocampus_cohort_expression_effects.csv",
    index=False
)

pd.DataFrame({
    "observed_auc": [auc],
    "accuracy": [accuracy],
    "balanced_accuracy": [balanced_accuracy],
    "n_samples": [len(samples)],
    "n_2006": [len(controls)],
    "n_2007": [len(ad)],
    "n_genes": [expr.shape[0]],
    "n_permutations": [len(permuted_auc)],
    "empirical_p_value": [p_value],
    "permutation_mean_auc": [
        np.mean(permuted_auc)
    ],
    "permutation_q025": [
        np.percentile(permuted_auc, 2.5)
    ],
    "permutation_q975": [
        np.percentile(permuted_auc, 97.5)
    ]
}).to_csv(
    OUTDIR /
    "GSE5281_hippocampus_cohort_separation_metrics.csv",
    index=False
)

pd.DataFrame({
    "permuted_auc": permuted_auc
}).to_csv(
    OUTDIR /
    "GSE5281_hippocampus_cohort_separation_permutation.csv",
    index=False
)

# ------------------------------------------------------------
# PLOT PCA
# ------------------------------------------------------------

try:

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    for label, marker in [
        ("2006_control_cohort", "o"),
        ("2007_AD_cohort", "s")
    ]:

        mask = (
            cohort_name == label
        )

        plt.scatter(
            pc1[mask],
            pc2[mask],
            marker=marker,
            s=70,
            label=label
        )

        for i in np.where(mask)[0]:

            plt.annotate(
                samples[i],
                (
                    pc1[i],
                    pc2[i]
                ),
                fontsize=7
            )

    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
    )

    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    )

    plt.title(
        "GSE5281 Hippocampal Cohort Separation"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGDIR /
        "GSE5281_hippocampus_cohort_PCA.png",
        dpi=300
    )

    plt.close()

    print(
        "\nSaved PCA figure:"
    )

    print(
        FIGDIR /
        "GSE5281_hippocampus_cohort_PCA.png"
    )

except Exception as e:

    print(
        "\nPCA figure could not be generated:"
    )

    print(e)

# ------------------------------------------------------------
# FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n")
print("=" * 90)
print("FINAL COHORT-SEPARATION INTERPRETATION")
print("=" * 90)

print(
    """
This analysis tests whether the 2006 versus 2007 submission cohort
is detectable from genome-wide expression in the 23 hippocampal samples.

IMPORTANT:

Strong cohort separation would demonstrate that submission cohort
corresponds to a detectable transcriptomic axis.

It would NOT, by itself, prove that the separation is entirely
technical or that the observed disease discrimination is entirely
an artifact.

Because diagnosis and submission cohort are perfectly confounded,
cohort separation and disease separation cannot be causally disentangled
within this hippocampal subset.

The result should therefore be interpreted as evidence regarding
dataset structure, not as a direct estimate of technical artifact.
"""
)

print("=" * 90)
print("COHORT-SEPARATION ANALYSIS COMPLETE")
print("=" * 90)

