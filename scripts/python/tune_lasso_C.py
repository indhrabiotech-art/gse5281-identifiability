import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score


print("=" * 60)
print("GSE48350 — LASSO REGULARIZATION TUNING")
print("=" * 60)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

expr = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)


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
# CV
# ------------------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)


# ------------------------------------------------------------
# C values
# ------------------------------------------------------------

C_values = [
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0
]


results = []


# ------------------------------------------------------------
# Tune C
# ------------------------------------------------------------

for C in C_values:

    print("\n" + "-" * 60)
    print(f"Testing C = {C}")
    print("-" * 60)

    auc_values = []
    acc_values = []
    n_selected_values = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X[train_idx]
        X_valid = X[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]


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


        prob = model.predict_proba(
            X_valid
        )[:, 1]


        pred = (
            prob >= 0.5
        ).astype(int)


        auc_values.append(
            roc_auc_score(
                y_valid,
                prob
            )
        )


        acc_values.append(
            accuracy_score(
                y_valid,
                pred
            )
        )


        n_selected_values.append(
            int(selected.sum())
        )


    results.append(
        {
            "C": C,

            "mean_AUC":
                np.mean(auc_values),

            "sd_AUC":
                np.std(auc_values, ddof=1),

            "mean_accuracy":
                np.mean(acc_values),

            "sd_accuracy":
                np.std(acc_values, ddof=1),

            "mean_selected_genes":
                np.mean(n_selected_values),

            "median_selected_genes":
                np.median(n_selected_values),

            "min_selected_genes":
                np.min(n_selected_values),

            "max_selected_genes":
                np.max(n_selected_values)
        }
    )


    print(
        f"Mean AUC: "
        f"{np.mean(auc_values):.3f}"
    )

    print(
        f"SD AUC: "
        f"{np.std(auc_values, ddof=1):.3f}"
    )

    print(
        f"Mean selected genes: "
        f"{np.mean(n_selected_values):.1f}"
    )

    print(
        f"Median selected genes: "
        f"{np.median(n_selected_values):.1f}"
    )


# ------------------------------------------------------------
# Results table
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "mean_AUC",
    ascending=False
)


os.makedirs(
    "04_ML/LASSO",
    exist_ok=True
)


results_df.to_csv(
    "04_ML/LASSO/GSE48350_LASSO_C_tuning.csv",
    index=False
)


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LASSO C TUNING COMPLETE")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\nSaved:")
print(
    "04_ML/LASSO/GSE48350_LASSO_C_tuning.csv"
)

print("=" * 60)
