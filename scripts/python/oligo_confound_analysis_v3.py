import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path.home() / "project_ml"

EXPR_FILE = ROOT / "04_ML/GSE48350_train49_age_adjusted_FULL.csv"
TRAIN_META = ROOT / "04_ML/GSE48350_RMA_train_meta.csv"
TEST_META = ROOT / "04_ML/GSE48350_RMA_test_meta.csv"

OUT = ROOT / "04_ML/Reviewer_Robustness_V4"
OUT.mkdir(parents=True, exist_ok=True)

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

MARKERS = {
    "oligodendrocyte": [
        "MBP", "MOG", "MOBP", "PLP1",
        "OLIG1", "OLIG2", "MAG", "CLDN11",
        "CNP", "ERMN"
    ],
    "astrocyte": [
        "GFAP", "AQP4", "ALDH1L1", "SLC1A3",
        "GJA1", "SOX9"
    ],
    "microglia": [
        "AIF1", "CX3CR1", "CSF1R", "C1QA",
        "C1QB", "C1QC", "P2RY12"
    ],
    "neuron": [
        "RBFOX3", "SNAP25", "SYT1",
        "MAP2", "NEFL", "ENO2"
    ],
    "endothelial": [
        "CLDN5", "PECAM1", "VWF",
        "KDR", "EMCN", "ESAM"
    ],
    "OPC": [
        "PDGFRA", "CSPG4", "SOX10",
        "OLIG1", "OLIG2"
    ]
}

print("=" * 90)
print("OLIGODENDROCYTE / CELL-COMPOSITION CONFOUNDER ANALYSIS")
print("=" * 90)

# ============================================================
# LOAD
# ============================================================

expr = pd.read_csv(EXPR_FILE)

sample_col = expr.columns[0]

expr = expr.set_index(sample_col)

print("\nExpression matrix:", expr.shape)

train_meta = pd.read_csv(TRAIN_META)
test_meta = pd.read_csv(TEST_META)

train_meta["_sample"] = train_meta["index"].astype(str)
test_meta["_sample"] = test_meta["index"].astype(str)

train_meta = train_meta[
    train_meta["_sample"].isin(expr.index)
].copy()

test_meta = test_meta[
    test_meta["_sample"].isin(expr.index)
].copy()

meta = pd.concat(
    [train_meta, test_meta],
    ignore_index=True
)

meta = meta.drop_duplicates(
    subset="_sample"
)

meta = meta.set_index("_sample")

# Ensure exact expression order
common = [
    s for s in expr.index
    if s in meta.index
]

expr = expr.loc[common]
meta = meta.loc[common]

print("Aligned samples:", len(common))
print("AD:", (meta["Diagnosis"] == "AD").sum())
print("Control:", (meta["Diagnosis"] == "Control").sum())

# ============================================================
# CELL-TYPE SCORES
# ============================================================

print("\n" + "=" * 90)
print("CALCULATING CELL-TYPE SCORES")
print("=" * 90)

score_df = pd.DataFrame(index=expr.index)

for celltype, markers in MARKERS.items():

    available = [
        g for g in markers
        if g in expr.columns
    ]

    if not available:
        raise RuntimeError(
            f"No markers available for {celltype}"
        )

    # Standardize each marker across samples.
    z = (
        expr[available]
        .subtract(expr[available].mean(axis=0))
        .divide(expr[available].std(axis=0).replace(0, np.nan))
    )

    score_df[f"{celltype}_score"] = z.mean(
        axis=1,
        skipna=True
    )

    print(
        f"{celltype:18s}: "
        f"{len(available)}/{len(markers)} markers"
    )

# ============================================================
# THREE-GENE SCORE
# ============================================================

missing_genes = [
    g for g in GENES
    if g not in expr.columns
]

if missing_genes:
    raise RuntimeError(
        f"Missing signature genes: {missing_genes}"
    )

gene_z = (
    expr[GENES]
    .subtract(expr[GENES].mean(axis=0))
    .divide(expr[GENES].std(axis=0).replace(0, np.nan))
)

score_df["three_gene_score"] = gene_z.mean(
    axis=1,
    skipna=True
)

# ============================================================
# METADATA + SCORES
# ============================================================

result = pd.concat(
    [
        meta[
            [
                "Age",
                "Sex",
                "Diagnosis"
            ]
        ],
        score_df
    ],
    axis=1
)

result["y"] = (
    result["Diagnosis"]
    .map(
        {
            "Control": 0,
            "AD": 1
        }
    )
)

# ============================================================
# SAVE SAMPLE-LEVEL SCORES
# ============================================================

score_file = OUT / "GSE48350_celltype_scores.csv"

result.to_csv(
    score_file,
    index=True
)

print("\nSaved:")
print(score_file)

# ============================================================
# CELL-TYPE SCORE DIAGNOSTICS
# ============================================================

diagnostics = []

for celltype in MARKERS:

    col = f"{celltype}_score"

    ad = result.loc[
        result["y"] == 1,
        col
    ].dropna()

    ctrl = result.loc[
        result["y"] == 0,
        col
    ].dropna()

    auc = roc_auc_score(
        result["y"],
        result[col]
    )

    mw = mannwhitneyu(
        ad,
        ctrl,
        alternative="two-sided"
    )

    rho, rho_p = spearmanr(
        result[col],
        result["y"]
    )

    diagnostics.append(
        {
            "cell_type": celltype,
            "AD_mean": ad.mean(),
            "Control_mean": ctrl.mean(),
            "AD_median": ad.median(),
            "Control_median": ctrl.median(),
            "AUC": auc,
            "Mann_Whitney_U": mw.statistic,
            "Mann_Whitney_p": mw.pvalue,
            "Spearman_rho": rho,
            "Spearman_p": rho_p
        }
    )

diagnostics = pd.DataFrame(diagnostics)

diagnostics.to_csv(
    OUT / "GSE48350_celltype_score_diagnostics.csv",
    index=False
)

print("\nCELL-TYPE DIAGNOSTICS")
print(
    diagnostics.to_string(
        index=False
    )
)

# ============================================================
# OLIGODENDROCYTE ↔ THREE-GENE ASSOCIATION
# ============================================================

oligo = result["oligodendrocyte_score"]
three = result["three_gene_score"]

rho, p = spearmanr(
    oligo,
    three
)

print("\n" + "=" * 90)
print("OLIGODENDROCYTE ↔ THREE-GENE SCORE")
print("=" * 90)

print(
    f"Spearman rho = {rho:.4f}"
)

print(
    f"P-value      = {p:.6g}"
)

# ============================================================
# THREE-GENE DISCRIMINATION
# ============================================================

auc_three = roc_auc_score(
    result["y"],
    result["three_gene_score"]
)

ap_three = average_precision_score(
    result["y"],
    result["three_gene_score"]
)

auc_oligo = roc_auc_score(
    result["y"],
    result["oligodendrocyte_score"]
)

ap_oligo = average_precision_score(
    result["y"],
    result["oligodendrocyte_score"]
)

print("\nTHREE-GENE SCORE")
print(
    f"AUC = {auc_three:.6f}"
)
print(
    f"AP  = {ap_three:.6f}"
)

print("\nOLIGODENDROCYTE SCORE")
print(
    f"AUC = {auc_oligo:.6f}"
)
print(
    f"AP  = {ap_oligo:.6f}"
)

# ============================================================
# CORRELATION OF EACH GENE WITH OLIGO SCORE
# ============================================================

gene_correlations = []

for gene in GENES:

    rho, p = spearmanr(
        result["oligodendrocyte_score"],
        expr.loc[result.index, gene]
    )

    gene_correlations.append(
        {
            "gene": gene,
            "Spearman_rho_with_oligo": rho,
            "Spearman_p": p
        }
    )

gene_correlations = pd.DataFrame(
    gene_correlations
)

gene_correlations.to_csv(
    OUT / "GSE48350_gene_oligodendrocyte_correlations.csv",
    index=False
)

print("\nGENE ↔ OLIGODENDROCYTE CORRELATIONS")
print(
    gene_correlations.to_string(
        index=False
    )
)

# ============================================================
# LOGISTIC REGRESSION MODELS
# ============================================================

print("\n" + "=" * 90)
print("CONFOUNDER-ADJUSTED LOGISTIC MODELS")
print("=" * 90)

models = {
    "three_gene": [
        "three_gene_score"
    ],

    "oligo_only": [
        "oligodendrocyte_score"
    ],

    "three_gene_plus_oligo": [
        "three_gene_score",
        "oligodendrocyte_score"
    ],

    "three_gene_plus_age": [
        "three_gene_score",
        "Age"
    ],

    "three_gene_plus_age_oligo": [
        "three_gene_score",
        "Age",
        "oligodendrocyte_score"
    ]
}

model_rows = []

for name, features in models.items():

    X = result[features].copy()
    y = result["y"].copy()

    valid = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid]
    yy = y.loc[valid]

    model = Pipeline(
        [
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
        ]
    )

    model.fit(
        X,
        yy
    )

    probability = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(
        yy,
        probability
    )

    ap = average_precision_score(
        yy,
        probability
    )

    model_rows.append(
        {
            "model": name,
            "features": ";".join(features),
            "n": len(yy),
            "AUC": auc,
            "AP": ap
        }
    )

models_df = pd.DataFrame(
    model_rows
)

models_df.to_csv(
    OUT / "GSE48350_oligodendrocyte_adjusted_models.csv",
    index=False
)

print(
    models_df.to_string(
        index=False
    )
)

# ============================================================
# PARTIAL / RESIDUALIZED THREE-GENE SCORE
# ============================================================

# Remove the linear component explained by
# oligodendrocyte score.

valid = (
    result[
        [
            "three_gene_score",
            "oligodendrocyte_score"
        ]
    ]
    .notna()
    .all(axis=1)
)

x_oligo = result.loc[
    valid,
    ["oligodendrocyte_score"]
]

y_three = result.loc[
    valid,
    "three_gene_score"
]

lm = np.polyfit(
    x_oligo.iloc[:, 0],
    y_three,
    deg=1
)

predicted = (
    lm[0] *
    x_oligo.iloc[:, 0]
    + lm[1]
)

residual_score = (
    y_three.values
    - predicted.values
)

res_auc = roc_auc_score(
    result.loc[valid, "y"],
    residual_score
)

res_ap = average_precision_score(
    result.loc[valid, "y"],
    residual_score
)

print("\n" + "=" * 90)
print("OLIGODENDROCYTE-RESIDUALIZED THREE-GENE SCORE")
print("=" * 90)

print(
    f"AUC = {res_auc:.6f}"
)

print(
    f"AP  = {res_ap:.6f}"
)

pd.DataFrame(
    {
        "GSM": result.index[valid],
        "Diagnosis": result.loc[
            valid,
            "Diagnosis"
        ].values,
        "three_gene_score": y_three.values,
        "oligodendrocyte_score": x_oligo.iloc[:, 0].values,
        "three_gene_residualized_for_oligo": residual_score
    }
).to_csv(
    OUT / "GSE48350_three_gene_residualized_oligo.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "analysis": "three_gene_raw",
            "AUC": auc_three,
            "AP": ap_three
        },
        {
            "analysis": "oligodendrocyte_score",
            "AUC": auc_oligo,
            "AP": ap_oligo
        },
        {
            "analysis": "three_gene_residualized_for_oligo",
            "AUC": res_auc,
            "AP": res_ap
        }
    ]
)

summary.to_csv(
    OUT / "GSE48350_oligodendrocyte_confounder_summary.csv",
    index=False
)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)

print(
    "\nOutput directory:",
    OUT
)
