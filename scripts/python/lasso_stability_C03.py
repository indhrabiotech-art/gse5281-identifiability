import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


print("=" * 60)
print("GSE48350 — LASSO STABILITY SELECTION")
print("C = 0.3")
print("=" * 60)


# ------------------------------------------------------------
# Load training data
# ------------------------------------------------------------

expr = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)


# ------------------------------------------------------------
# Alignment
# ------------------------------------------------------------

if list(expr.index) != list(meta["index"]):
    raise ValueError(
        "Sample alignment failed."
    )


X = expr.to_numpy(dtype=float)

y = (
    meta["Diagnosis"]
    .map({
        "Control": 0,
        "AD": 1
    })
    .to_numpy()
)

genes = np.array(expr.columns)


print("\nSamples:", X.shape[0])
print("Genes:", X.shape[1])

print("\nDiagnosis:")
print(meta["Diagnosis"].value_counts())


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

C = 0.3

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)


# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

selection_count = np.zeros(
    len(genes),
    dtype=int
)

coefficient_values = [
    []
    for _ in genes
]


# ------------------------------------------------------------
# Repeated CV
# ------------------------------------------------------------

total_folds = 50

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y),
    start=1
):

    X_train = X[train_idx]
    y_train = y[train_idx]


    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "lasso",
                LogisticRegression(
                    solver="liblinear",
                    C=C,
                    l1_ratio=1.0,
                    max_iter=5000,
                    random_state=42
                )
            )
        ]
    )


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model.fit(
            X_train,
            y_train
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


    for i, value in enumerate(coef):

        if abs(value) > 1e-10:
            coefficient_values[i].append(value)


    if fold % 10 == 0:

        print(
            f"Completed fold "
            f"{fold}/{total_folds} | "
            f"Selected genes: "
            f"{selected.sum()}"
        )


# ------------------------------------------------------------
# Build statistics
# ------------------------------------------------------------

selection_frequency = (
    selection_count /
    total_folds
)


mean_coefficient = np.zeros(
    len(genes)
)

sd_coefficient = np.zeros(
    len(genes)
)


for i, values in enumerate(
    coefficient_values
):

    if len(values) > 0:

        mean_coefficient[i] = np.mean(
            values
        )

        if len(values) > 1:

            sd_coefficient[i] = np.std(
                values,
                ddof=1
            )


stats = pd.DataFrame(
    {
        "gene": genes,

        "selection_count":
            selection_count,

        "selection_frequency":
            selection_frequency,

        "mean_coefficient":
            mean_coefficient,

        "sd_coefficient":
            sd_coefficient
    }
)


stats = stats.sort_values(
    [
        "selection_frequency",
        "selection_count"
    ],
    ascending=False
)


# ------------------------------------------------------------
# Stability summaries
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STABILITY SUMMARY")
print("=" * 60)


for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00
]:

    n = (
        stats["selection_frequency"]
        >= threshold
    ).sum()

    print(
        f"Genes selected >= "
        f"{int(threshold * 100)}% of folds: "
        f"{n}"
    )


# ------------------------------------------------------------
# Selected at least once
# ------------------------------------------------------------

selected_once = stats[
    stats["selection_count"] > 0
]


print(
    "\nGenes selected at least once:",
    len(selected_once)
)


# ------------------------------------------------------------
# Top stable genes
# ------------------------------------------------------------

print("\nTop 30 genes by selection frequency:")

print(
    stats[
        stats["selection_count"] > 0
    ]
    .head(30)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out_dir = "04_ML/LASSO"

os.makedirs(
    out_dir,
    exist_ok=True
)


stats.to_csv(
    f"{out_dir}/GSE48350_LASSO_C03_stability.csv",
    index=False
)


for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]:

    stable = stats[
        stats["selection_frequency"]
        >= threshold
    ]

    stable.to_csv(
        f"{out_dir}/GSE48350_LASSO_C03_stable_{int(threshold*100)}pct.csv",
        index=False
    )


print("\nSaved:")
print(
    "04_ML/LASSO/GSE48350_LASSO_C03_stability.csv"
)

print(
    "04_ML/LASSO/GSE48350_LASSO_C03_stable_*pct.csv"
)


print("\n" + "=" * 60)
print("LASSO C=0.3 STABILITY SELECTION COMPLETE")
print("=" * 60)
