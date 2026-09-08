import os
import itertools
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

# ============================================================
# CONFIG
# ============================================================

OUTDIR = "04_ML/Final_Model/Validation/Locked_Ablation"
os.makedirs(OUTDIR, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

COMBINATIONS = [
    ("ABCA6",),
    ("CRLF1",),
    ("TNFRSF11B",),
    ("ABCA6", "CRLF1"),
    ("ABCA6", "TNFRSF11B"),
    ("CRLF1", "TNFRSF11B"),
    ("ABCA6", "CRLF1", "TNFRSF11B")
]

# ============================================================
# LOAD TRAIN / TEST / EXTERNAL
# ============================================================

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

test = pd.read_csv(
    "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
    index_col=0
)

test_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_test_meta.csv"
)

external = pd.read_csv(
    "04_ML/External_Validation/GSE5281_LASSO_candidate_matrix.csv",
    index_col=0
)

external_meta = pd.read_csv(
    "04_ML/External_Validation/GSE5281_validation_metadata.csv",
    index_col=0
)

# ============================================================
# TARGETS
# ============================================================

y_train = (
    train_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .to_numpy()
)

y_test = (
    test_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .to_numpy()
)

y_external = (
    external_meta["diagnosis"]
    .map({"Control": 0, "AD": 1})
    .to_numpy()
)

# ============================================================
# ALIGNMENT CHECK
# ============================================================

assert list(train.index) == list(train_meta["index"])
assert list(test.index) == list(test_meta["index"])
assert list(external.index) == list(external_meta.index)

# ============================================================
# LOCKED ABLATION
# ============================================================

results = []

print("=" * 80)
print("LOCKED THREE-GENE ABLATION VALIDATION")
print("=" * 80)

print()
print("Training:", len(train), "samples")
print("Held-out test:", len(test), "samples")
print("External:", len(external), "samples")

for genes in COMBINATIONS:

    genes_name = "+".join(genes)

    X_train = train[list(genes)].to_numpy(dtype=float)
    X_test = test[list(genes)].to_numpy(dtype=float)
    X_external = external[list(genes)].to_numpy(dtype=float)

    # --------------------------------------------------------
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=0.3,
                max_iter=5000,
                random_state=42
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # FROZEN PREDICTIONS
    # --------------------------------------------------------

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    p_external = model.predict_proba(X_external)[:, 1]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    train_auc = roc_auc_score(
        y_train,
        p_train
    )

    test_auc = roc_auc_score(
        y_test,
        p_test
    )

    external_auc = roc_auc_score(
        y_external,
        p_external
    )

    train_ap = average_precision_score(
        y_train,
        p_train
    )

    test_ap = average_precision_score(
        y_test,
        p_test
    )

    external_ap = average_precision_score(
        y_external,
        p_external
    )

    results.append({
        "Gene_set": genes_name,
        "N_genes": len(genes),

        "Training_AUC": train_auc,
        "Test_AUC": test_auc,
        "External_AUC": external_auc,

        "Training_AP": train_ap,
        "Test_AP": test_ap,
        "External_AP": external_ap
    })

    print(
        f"{genes_name:35s} | "
        f"Train AUC={train_auc:.4f} | "
        f"Test AUC={test_auc:.4f} | "
        f"External AUC={external_auc:.4f}"
    )

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

outfile = (
    f"{OUTDIR}/locked_three_gene_ablation_results.csv"
)

results_df.to_csv(
    outfile,
    index=False
)

print()
print("=" * 80)
print("LOCKED ABLATION COMPLETE")
print("=" * 80)

print()
print(results_df.to_string(index=False))

print()
print("Saved:")
print(outfile)

