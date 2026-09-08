import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = (
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv"
)

META_FILE = (
    "04_ML/GSE48350_RMA_train_meta.csv"
)

OUTDIR = "04_ML/LASSO"

os.makedirs(OUTDIR, exist_ok=True)

N_BOOTSTRAP = 2000
SEED = 42
C = 0.1

rng = np.random.default_rng(SEED)


# ============================================================
# LOAD DATA
# ============================================================

expr = pd.read_csv(
    TRAIN_FILE,
    index_col=0
)

meta = pd.read_csv(
    META_FILE
)


# ============================================================
# ALIGNMENT CHECK
# ============================================================

if list(expr.index) != list(meta["index"]):
    raise ValueError(
        "Expression samples and metadata are not aligned."
    )


X = expr.to_numpy(dtype=float)

genes = np.array(expr.columns)

y = (
    meta["Diagnosis"]
    .map({
        "Control": 0,
        "AD": 1
    })
    .to_numpy()
)


if np.isnan(X).any():
    raise ValueError(
        "Missing expression values detected."
    )

if np.isnan(y).any():
    raise ValueError(
        "Missing diagnosis labels detected."
    )


n_samples = X.shape[0]
n_genes = X.shape[1]


# ============================================================
# IDENTIFY TARGET GENES
# ============================================================

target_genes = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

target_indices = {
    gene: np.where(genes == gene)[0][0]
    for gene in target_genes
}


print("=" * 75)
print("GSE48350 — BOOTSTRAP LASSO FEATURE-SELECTION STABILITY")
print("=" * 75)

print()
print("Samples:", n_samples)
print("Genes:", n_genes)
print("AD:", np.sum(y == 1))
print("Control:", np.sum(y == 0))
print("Bootstrap resamples:", N_BOOTSTRAP)
print("LASSO C:", C)
print("Random seed:", SEED)


# ============================================================
# STORAGE
# ============================================================

selection_count = np.zeros(
    n_genes,
    dtype=int
)

positive_count = np.zeros(
    n_genes,
    dtype=int
)

negative_count = np.zeros(
    n_genes,
    dtype=int
)

coefficient_records = []


# ============================================================
# BOOTSTRAP
# ============================================================

for b in range(1, N_BOOTSTRAP + 1):

    # --------------------------------------------------------
    # Bootstrap sample
    # --------------------------------------------------------

    indices = rng.integers(
        low=0,
        high=n_samples,
        size=n_samples
    )

    X_boot = X[indices]
    y_boot = y[indices]

    # --------------------------------------------------------
    # Ensure both classes are represented
    # --------------------------------------------------------

    if len(np.unique(y_boot)) < 2:
        continue

    # --------------------------------------------------------
    # Leakage-safe preprocessing + LASSO
    # --------------------------------------------------------

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "lasso",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=C,
                    max_iter=5000,
                    random_state=SEED
                )
            )
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model.fit(
            X_boot,
            y_boot
        )

    coef = (
        model
        .named_steps["lasso"]
        .coef_[0]
    )

    selected = (
        np.abs(coef) > 1e-10
    )

    selection_count += selected.astype(int)

    positive_count += (
        coef > 1e-10
    ).astype(int)

    negative_count += (
        coef < -1e-10
    ).astype(int)

    coefficient_records.append(
        coef
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if b % 100 == 0:

        print(
            f"Completed bootstrap "
            f"{b}/{N_BOOTSTRAP}"
        )


# ============================================================
# EFFECTIVE BOOTSTRAP COUNT
# ============================================================

coef_matrix = np.vstack(
    coefficient_records
)

n_effective = coef_matrix.shape[0]


print()
print(
    "Effective bootstrap fits:",
    n_effective
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

selection_frequency = (
    selection_count / n_effective
)

positive_frequency = (
    positive_count / n_effective
)

negative_frequency = (
    negative_count / n_effective
)

mean_coefficient = (
    np.mean(
        coef_matrix,
        axis=0
    )
)

sd_coefficient = (
    np.std(
        coef_matrix,
        axis=0,
        ddof=1
    )
)

mean_absolute_coefficient = (
    np.mean(
        np.abs(coef_matrix),
        axis=0
    )
)

sign_consistency = np.maximum(
    positive_frequency,
    negative_frequency
)


# ============================================================
# FEATURE TABLE
# ============================================================

results = pd.DataFrame(
    {
        "gene": genes,

        "selection_count":
            selection_count,

        "selection_frequency":
            selection_frequency,

        "positive_frequency":
            positive_frequency,

        "negative_frequency":
            negative_frequency,

        "sign_consistency":
            sign_consistency,

        "mean_coefficient":
            mean_coefficient,

        "sd_coefficient":
            sd_coefficient,

        "mean_absolute_coefficient":
            mean_absolute_coefficient
    }
)


results = results.sort_values(
    [
        "selection_frequency",
        "sign_consistency"
    ],
    ascending=False
)


# ============================================================
# TARGET-GENE SUMMARY
# ============================================================

print()
print("=" * 75)
print("TARGET GENE BOOTSTRAP STABILITY")
print("=" * 75)

target_summary = results[
    results["gene"].isin(target_genes)
].copy()

print(
    target_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# STABILITY THRESHOLDS
# ============================================================

print()
print("=" * 75)
print("SELECTION-FREQUENCY THRESHOLDS")
print("=" * 75)

for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95
]:

    selected_n = (
        results["selection_frequency"]
        >= threshold
    ).sum()

    print(
        f"Genes selected >= "
        f"{int(threshold * 100)}%: "
        f"{selected_n}"
    )


# ============================================================
# SAVE
# ============================================================

outfile = (
    OUTDIR +
    "/GSE48350_lasso_bootstrap_selection_stability.csv"
)

results.to_csv(
    outfile,
    index=False
)


target_outfile = (
    OUTDIR +
    "/GSE48350_three_gene_bootstrap_stability.csv"
)

target_summary.to_csv(
    target_outfile,
    index=False
)


print()
print("=" * 75)
print("BOOTSTRAP LASSO STABILITY COMPLETE")
print("=" * 75)

print()
print("Saved:")
print(outfile)
print(target_outfile)
