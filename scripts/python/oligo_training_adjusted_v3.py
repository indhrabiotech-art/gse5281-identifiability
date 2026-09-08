import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

ROOT = Path.home() / "project_ml"
OUT = ROOT / "04_ML" / "Reviewer_Robustness_V4"
OUT.mkdir(parents=True, exist_ok=True)


# ---- V3 PATCH: training-only, age-adjusted, full gene space ----
import numpy as _np, pandas as _pd
_full = _pd.read_csv(ROOT / "04_ML/GSE48350_RMA_ML_matrix.csv", index_col=0)
_full.index = _full.index.astype(str).str.strip()
_tm = _pd.read_csv(ROOT / "04_ML/GSE48350_RMA_train_meta.csv")
_tm[_tm.columns[0]] = _tm[_tm.columns[0]].astype(str).str.strip()
_ids = [s for s in _tm[_tm.columns[0]] if s in _full.index]
_full = _full.loc[_ids]
_age = _pd.to_numeric(_tm.set_index(_tm.columns[0]).loc[_ids, "Age"], errors="coerce").values
_ok = ~_np.isnan(_age)
_A = _np.column_stack([_np.ones(_ok.sum()), _age[_ok]])
_M = _full.values.astype(float)
_beta, *_ = _np.linalg.lstsq(_A, _M[_ok], rcond=None)
_M[_ok] = _M[_ok] - _np.outer(_age[_ok] - _age[_ok].mean(), _beta[1])
_adj = _pd.DataFrame(_M, index=_full.index, columns=_full.columns)
_adj.to_csv(ROOT / "04_ML/GSE48350_train49_age_adjusted_FULL.csv")
print(f"[V3] training-only age-adjusted matrix: {_adj.shape}")
# ----------------------------------------------------------------
EXPR_FILE = ROOT / "04_ML/GSE48350_train49_age_adjusted_FULL.csv"
META_FILE = ROOT / "04_ML/GSE48350_RMA_train_meta.csv"

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

OLIGO_MARKERS = [
    "MBP", "MOG", "MOBP", "PLP1",
    "OLIG1", "OLIG2", "MAG", "CLDN11",
    "CNP", "ERMN"
]

print("=" * 90)
print("TRAINING-ONLY OLIGODENDROCYTE CONFOUNDER ANALYSIS — V2")
print("=" * 90)

# ============================================================
# LOAD
# ============================================================

expr = pd.read_csv(EXPR_FILE)
meta = pd.read_csv(META_FILE)

sample_col = expr.columns[0]

expr = expr.set_index(sample_col)
expr.index = expr.index.astype(str)

meta["index"] = meta["index"].astype(str)

print("\nExpression:", expr.shape)
print("Metadata:", meta.shape)

# ============================================================
# ALIGN 49 TRAINING SAMPLES
# ============================================================

common = [
    x for x in meta["index"]
    if x in expr.index
]

if len(common) != 49:
    raise RuntimeError(
        f"Expected 49 matched training samples, found {len(common)}"
    )

expr = expr.loc[common]
meta = meta.set_index("index").loc[common]

print("\nAligned training samples:", len(common))
print("AD:", (meta["Diagnosis"] == "AD").sum())
print("Control:", (meta["Diagnosis"] == "Control").sum())

# ============================================================
# CHECK MARKERS
# ============================================================

missing_oligo = [
    g for g in OLIGO_MARKERS
    if g not in expr.columns
]

missing_genes = [
    g for g in GENES
    if g not in expr.columns
]

if missing_oligo:
    raise RuntimeError(
        f"Missing oligodendrocyte markers: {missing_oligo}"
    )

if missing_genes:
    raise RuntimeError(
        f"Missing signature genes: {missing_genes}"
    )

print("\nOligodendrocyte markers:",
      len(OLIGO_MARKERS))

print("Signature genes:",
      ", ".join(GENES))

# ============================================================
# OLIGODENDROCYTE SCORE
# ============================================================

oligo = expr[OLIGO_MARKERS].copy()

oligo_z = (
    oligo
    .subtract(oligo.mean(axis=0))
    .divide(oligo.std(axis=0))
)

oligo_score = oligo_z.mean(axis=1)

# ============================================================
# THREE-GENE SCORE
# ============================================================

three = expr[GENES].copy()

three_z = (
    three
    .subtract(three.mean(axis=0))
    .divide(three.std(axis=0))
)

three_gene_score = three_z.mean(axis=1)

# ============================================================
# ANALYSIS TABLE
# ============================================================

df = meta[
    ["Age", "Sex", "Diagnosis"]
].copy()

df["y"] = (
    df["Diagnosis"]
    .map({
        "Control": 0,
        "AD": 1
    })
)

df["oligodendrocyte_score"] = oligo_score
df["three_gene_score"] = three_gene_score

print("\nNaN CHECK")
print(
    df[
        [
            "Age",
            "oligodendrocyte_score",
            "three_gene_score",
            "y"
        ]
    ].isna().sum()
)

# ============================================================
# DIAGNOSTIC ASSOCIATION
# ============================================================

rho, p = spearmanr(
    df["three_gene_score"],
    df["oligodendrocyte_score"]
)

print("\n" + "=" * 90)
print("THREE-GENE ↔ OLIGODENDROCYTE")
print("=" * 90)

print(f"Spearman rho = {rho:.6f}")
print(f"P-value      = {p:.6g}")

# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate(features, name):

    d = df[
        features + ["y"]
    ].dropna()

    if len(d) == 0:
        raise RuntimeError(
            f"No valid samples for model: {name}"
        )

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

    return {
        "model": name,
        "features": ";".join(features),
        "n": len(y),
        "AUC": roc_auc_score(y, pred),
        "AP": average_precision_score(y, pred)
    }

# ============================================================
# MODELS
# ============================================================

rows = [
    evaluate(
        ["three_gene_score"],
        "3-gene"
    ),

    evaluate(
        ["oligodendrocyte_score"],
        "Oligodendrocyte-only"
    ),

    evaluate(
        [
            "three_gene_score",
            "oligodendrocyte_score"
        ],
        "3-gene + oligodendrocyte"
    ),

    evaluate(
        [
            "three_gene_score",
            "Age"
        ],
        "3-gene + age"
    ),

    evaluate(
        [
            "three_gene_score",
            "Age",
            "oligodendrocyte_score"
        ],
        "3-gene + age + oligodendrocyte"
    )
]

results = pd.DataFrame(rows)

print("\n" + "=" * 90)
print("TRAINING-ONLY RESULTS")
print("=" * 90)

print(
    results.to_string(index=False)
)

results.to_csv(
    OUT / "GSE48350_training_oligo_adjusted_models_v2.csv",
    index=False
)

# ============================================================
# GENE ↔ OLIGO CORRELATIONS
# ============================================================

corr_rows = []

for gene in GENES:

    r, pv = spearmanr(
        expr[gene],
        df["oligodendrocyte_score"]
    )

    corr_rows.append({
        "gene": gene,
        "Spearman_rho": r,
        "P_value": pv
    })

corr = pd.DataFrame(corr_rows)

print("\n" + "=" * 90)
print("GENE ↔ OLIGODENDROCYTE")
print("=" * 90)

print(
    corr.to_string(index=False)
)

corr.to_csv(
    OUT / "GSE48350_training_gene_oligo_correlations_v2.csv",
    index=False
)

# ============================================================
# RESIDUALIZED THREE-GENE SCORE
# ============================================================

x = df["oligodendrocyte_score"].values
z = df["three_gene_score"].values

coef = np.polyfit(
    x,
    z,
    1
)

fitted = (
    coef[0] * x +
    coef[1]
)

residual = z - fitted

res_auc = roc_auc_score(
    df["y"],
    residual
)

res_ap = average_precision_score(
    df["y"],
    residual
)

print("\n" + "=" * 90)
print("RESIDUALIZED THREE-GENE SCORE")
print("=" * 90)

print(f"AUC = {res_auc:.6f}")
print(f"AP  = {res_ap:.6f}")

pd.DataFrame([{
    "analysis":
        "3-gene residualized for oligodendrocyte",
    "AUC": res_auc,
    "AP": res_ap,
    "Spearman_rho":
        rho,
    "Spearman_p":
        p
}]).to_csv(
    OUT / "GSE48350_training_oligo_residualized_v2.csv",
    index=False
)

# ============================================================
# SAVE SAMPLE-LEVEL DATA
# ============================================================

df.reset_index().rename(
    columns={"index": "GSM"}
).to_csv(
    OUT / "GSE48350_training_oligo_analysis_samples_v2.csv",
    index=False
)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)

