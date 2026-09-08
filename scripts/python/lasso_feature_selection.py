import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score


# ============================================================
# LASSO FEATURE SELECTION — GSE48350
# ============================================================

print("=" * 60)
print("GSE48350 — LEAKAGE-SAFE LASSO FEATURE SELECTION")
print("=" * 60)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TRAIN_FILE = (
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv"
)

META_FILE = (
    "04_ML/GSE48350_RMA_train_meta.csv"
)

OUT_DIR = "04_ML/LASSO"


os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Load expression
# ------------------------------------------------------------

expr = pd.read_csv(
    TRAIN_FILE,
    index_col=0
)

meta = pd.read_csv(
    META_FILE
)


print("\nExpression matrix:")
print("Samples:", expr.shape[0])
print("Genes:", expr.shape[1])


# ------------------------------------------------------------
# Verify sample alignment
# ------------------------------------------------------------

if list(expr.index) != list(meta["index"]):

    raise ValueError(
        "Expression samples and metadata are not aligned."
    )


print("Sample alignment: VERIFIED")


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

y = (
    meta["Diagnosis"]
    .map({
        "Control": 0,
        "AD": 1
    })
    .to_numpy()
)


print("\nDiagnosis:")
print(
    meta["Diagnosis"].value_counts()
)


# ------------------------------------------------------------
# Expression matrix
# ------------------------------------------------------------

X = expr.to_numpy(dtype=float)

genes = np.array(expr.columns)


if np.isnan(X).any():

    raise ValueError(
        "Missing expression values detected."
    )


# ------------------------------------------------------------
# Repeated stratified CV
# ------------------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)


print("\nCV design:")
print("Splits per repeat: 5")
print("Repeats: 10")
print("Total folds: 50")


# ------------------------------------------------------------
# LASSO model
# ------------------------------------------------------------

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
                C=0.1,
                max_iter=5000,
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

selection_count = np.zeros(
    len(genes),
    dtype=int
)

coefficient_values = []


fold_results = []


# ------------------------------------------------------------
# CV
# ------------------------------------------------------------

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y),
    start=1
):

    X_train = X[train_idx]
    X_valid = X[valid_idx]

    y_train = y[train_idx]
    y_valid = y[valid_idx]


    # --------------------------------------------------------
    # Fit ONLY on CV training fold
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Extract LASSO coefficients
    # --------------------------------------------------------

    coef = (
        model
        .named_steps["lasso"]
        .coef_[0]
    )


    selected = (
        np.abs(coef) > 1e-10
    )


    selection_count += selected


    coefficient_values.append(
        coef
    )


    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    prob = model.predict_proba(
        X_valid
    )[:, 1]

    pred = (
        prob >= 0.5
    ).astype(int)


    auc = roc_auc_score(
        y_valid,
        prob
    )

    acc = accuracy_score(
        y_valid,
        pred
    )


    fold_results.append(
        {
            "fold": fold,
            "n_selected": int(
                selected.sum()
            ),
            "AUC": auc,
            "Accuracy": acc
        }
    )


    if fold % 10 == 0:

        print(
            f"Completed fold {fold}/50 | "
            f"Selected genes: {selected.sum()} | "
            f"AUC: {auc:.3f}"
        )


# ------------------------------------------------------------
# Selection frequency
# ------------------------------------------------------------

n_folds = len(fold_results)

selection_frequency = (
    selection_count / n_folds
)


# ------------------------------------------------------------
# Mean absolute coefficient
# ------------------------------------------------------------

coef_matrix = np.vstack(
    coefficient_values
)

mean_abs_coef = (
    np.mean(
        np.abs(coef_matrix),
        axis=0
    )
)


mean_coef = (
    np.mean(
        coef_matrix,
        axis=0
    )
)


# ------------------------------------------------------------
# Feature table
# ------------------------------------------------------------

feature_table = pd.DataFrame(
    {
        "gene": genes,

        "selection_count":
            selection_count,

        "selection_frequency":
            selection_frequency,

        "mean_coefficient":
            mean_coef,

        "mean_absolute_coefficient":
            mean_abs_coef
    }
)


feature_table = feature_table.sort_values(
    [
        "selection_frequency",
        "mean_absolute_coefficient"
    ],
    ascending=False
)


# ------------------------------------------------------------
# Save all feature statistics
# ------------------------------------------------------------

feature_table.to_csv(
    os.path.join(
        OUT_DIR,
        "GSE48350_LASSO_feature_statistics.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Stable features
# ------------------------------------------------------------

stable = feature_table[
    feature_table[
        "selection_frequency"
    ] >= 0.50
].copy()


stable.to_csv(
    os.path.join(
        OUT_DIR,
        "GSE48350_LASSO_stable_features.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# CV results
# ------------------------------------------------------------

fold_df = pd.DataFrame(
    fold_results
)

fold_df.to_csv(
    os.path.join(
        OUT_DIR,
        "GSE48350_LASSO_CV_results.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LASSO FEATURE SELECTION COMPLETE")
print("=" * 60)

print(
    "\nTotal genes:",
    len(genes)
)

print(
    "Total CV folds:",
    n_folds
)

print(
    "Genes selected at least once:",
    int(
        (selection_count > 0).sum()
    )
)

print(
    "Genes selected >=50% of folds:",
    len(stable)
)

print(
    "Genes selected >=75% of folds:",
    int(
        (
            selection_frequency >= 0.75
        ).sum()
    )
)

print(
    "Genes selected in all folds:",
    int(
        (
            selection_frequency == 1.0
        ).sum()
    )
)


print("\nCV performance:")

print(
    "Mean AUC:",
    f"{fold_df['AUC'].mean():.3f}"
)

print(
    "SD AUC:",
    f"{fold_df['AUC'].std():.3f}"
)

print(
    "Mean accuracy:",
    f"{fold_df['Accuracy'].mean():.3f}"
)

print(
    "SD accuracy:",
    f"{fold_df['Accuracy'].std():.3f}"
)


print("\nTop 30 stable features:")

print(
    stable[
        [
            "gene",
            "selection_count",
            "selection_frequency",
            "mean_coefficient"
        ]
    ].head(30).to_string(
        index=False
    )
)


print("\nSaved:")

print(
    os.path.join(
        OUT_DIR,
        "GSE48350_LASSO_feature_statistics.csv"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "GSE48350_LASSO_stable_features.csv"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "GSE48350_LASSO_CV_results.csv"
    )
)

print("=" * 60)
