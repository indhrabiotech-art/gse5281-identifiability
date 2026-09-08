import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

ROOT = Path.home() / "project_ml"
OUT = ROOT / "04_ML" / "Reviewer_Robustness_V3"
OUT.mkdir(parents=True, exist_ok=True)

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

META = pd.read_csv(
    ROOT / "04_ML/GSE48350_RMA_train_meta.csv"
)

SCORES = pd.read_csv(
    OUT / "GSE48350_celltype_scores.csv"
)

# ------------------------------------------------------------
# ALIGN ONLY TRAINING SAMPLES
# ------------------------------------------------------------

META["_sample"] = META["index"].astype(str)
SCORES["_sample"] = SCORES.iloc[:, 0].astype(str)

# Keep only the score columns from the score table.
score_cols = [
    "_sample",
    "oligodendrocyte_score",
    "astrocyte_score",
    "microglia_score",
    "neuron_score",
    "endothelial_score",
    "OPC_score",
    "three_gene_score"
]

score_cols = [
    c for c in score_cols
    if c in SCORES.columns
]

SCORES_SMALL = SCORES[score_cols].copy()

df = META.merge(
    SCORES_SMALL,
    on="_sample",
    how="inner"
)

print("=" * 90)
print("TRAINING-ONLY OLIGODENDROCYTE CONFOUNDER ANALYSIS")
print("=" * 90)

print("Training samples:", len(df))
print("AD:", (df["Diagnosis"] == "AD").sum())
print("Control:", (df["Diagnosis"] == "Control").sum())

if len(df) != 49:
    raise RuntimeError(
        f"Expected 49 training samples, found {len(df)}"
    )

# ------------------------------------------------------------
# OUTCOME
# ------------------------------------------------------------

df["y"] = (
    df["Diagnosis"]
    .map({
        "Control": 0,
        "AD": 1
    })
)

# ------------------------------------------------------------
# THREE-GENE SCORE
# ------------------------------------------------------------

# Read expression matrix
expr = pd.read_csv(
    ROOT / "04_ML/GSE48350_RMA_ML_matrix.csv"
)

expr = expr.set_index(
    expr.columns[0]
)

expr.index = expr.index.astype(str)

train_ids = df["_sample"].tolist()

expr = expr.loc[train_ids]

missing = [
    g for g in GENES
    if g not in expr.columns
]

if missing:
    raise RuntimeError(
        f"Missing genes: {missing}"
    )

# Z-score each gene within TRAINING SET
gene_z = (
    expr[GENES]
    .subtract(expr[GENES].mean(axis=0))
    .divide(
        expr[GENES].std(axis=0)
    )
)

df["three_gene_score"] = gene_z.mean(
    axis=1
)

# ------------------------------------------------------------
# OLIGODENDROCYTE SCORE
# ------------------------------------------------------------

oligo_col = "oligodendrocyte_score"

if oligo_col not in df.columns:
    raise RuntimeError(
        "oligodendrocyte_score not found."
    )

# ------------------------------------------------------------
# BASIC ASSOCIATION
# ------------------------------------------------------------

rho, p = spearmanr(
    df["three_gene_score"],
    df[oligo_col]
)

print("\nThree-gene ↔ oligodendrocyte score")
print("Spearman rho:", rho)
print("P-value:", p)

# ------------------------------------------------------------
# MODEL FUNCTION
# ------------------------------------------------------------

def evaluate(features, name):

    d = df[
        features + ["y"]
    ].dropna()

    X = d[features]
    y = d["y"]

    model = Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        )
    ])

    model.fit(X, y)

    pred = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(
        y,
        pred
    )

    ap = average_precision_score(
        y,
        pred
    )

    return {
        "model": name,
        "features": ";".join(features),
        "n": len(y),
        "AUC": auc,
        "AP": ap
    }


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

rows = []

rows.append(
    evaluate(
        ["three_gene_score"],
        "3-gene"
    )
)

rows.append(
    evaluate(
        ["oligodendrocyte_score"],
        "Oligodendrocyte-only"
    )
)

rows.append(
    evaluate(
        [
            "three_gene_score",
            "oligodendrocyte_score"
        ],
        "3-gene + oligodendrocyte"
    )
)

rows.append(
    evaluate(
        [
            "three_gene_score",
            "Age"
        ],
        "3-gene + age"
    )
)

rows.append(
    evaluate(
        [
            "three_gene_score",
            "Age",
            "oligodendrocyte_score"
        ],
        "3-gene + age + oligodendrocyte"
    )
)

results = pd.DataFrame(rows)

print("\n" + "=" * 90)
print("TRAINING-ONLY MODEL RESULTS")
print("=" * 90)

print(
    results.to_string(
        index=False
    )
)

results.to_csv(
    OUT / "GSE48350_training_oligo_adjusted_models.csv",
    index=False
)

# ------------------------------------------------------------
# GENE ↔ OLIGO CORRELATIONS
# ------------------------------------------------------------

corr_rows = []

for gene in GENES:

    rho, p = spearmanr(
        expr[gene],
        df[oligo_col]
    )

    corr_rows.append({
        "gene": gene,
        "Spearman_rho": rho,
        "P_value": p
    })

corr = pd.DataFrame(corr_rows)

print("\n" + "=" * 90)
print("GENE ↔ OLIGODENDROCYTE CORRELATIONS")
print("=" * 90)

print(
    corr.to_string(
        index=False
    )
)

corr.to_csv(
    OUT / "GSE48350_training_gene_oligo_correlations.csv",
    index=False
)

# ------------------------------------------------------------
# RESIDUALIZED THREE-GENE SCORE
# ------------------------------------------------------------

x = df[oligo_col].values
y_score = df["three_gene_score"].values

coef = np.polyfit(
    x,
    y_score,
    1
)

predicted = (
    coef[0] * x
    + coef[1]
)

residual = (
    y_score
    - predicted
)

res_auc = roc_auc_score(
    df["y"],
    residual
)

res_ap = average_precision_score(
    df["y"],
    residual
)

print("\n" + "=" * 90)
print("RESIDUALIZED 3-GENE SCORE")
print("=" * 90)

print(
    "AUC:",
    res_auc
)

print(
    "AP :",
    res_ap
)

pd.DataFrame([{
    "analysis": "3-gene residualized for oligodendrocyte",
    "AUC": res_auc,
    "AP": res_ap,
    "Spearman_gene_score_oligo": rho
}]).to_csv(
    OUT / "GSE48350_training_oligo_residualized.csv",
    index=False
)

print("\nSaved results to:")
print(OUT)

