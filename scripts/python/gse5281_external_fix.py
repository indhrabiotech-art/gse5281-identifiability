import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

ROOT = Path.home() / "project_ml"

TRAIN_EXPR = ROOT / "04_ML/GSE48350_RMA_ML_matrix.csv"
TRAIN_META = ROOT / "04_ML/GSE48350_RMA_train_meta.csv"

EXT_EXPR = ROOT / "04_ML/GSE5281_RMA_genelevel_harmonized.csv"
EXT_META = ROOT / "04_ML/External_Validation/GSE5281_validation_metadata.csv"

OUT = ROOT / "04_ML/Reviewer_Robustness_V3"
OUT.mkdir(parents=True, exist_ok=True)

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]


print("=" * 90)
print("GSE5281 EXTERNAL VALIDATION — FIXED")
print("=" * 90)


# ============================================================
# DISCOVERY EXPRESSION
# ============================================================

train_expr = pd.read_csv(TRAIN_EXPR)

train_expr = train_expr.set_index("Unnamed: 0")

train_expr.index = (
    train_expr.index.astype(str).str.strip()
)

train_meta = pd.read_csv(TRAIN_META)

train_meta["_sample"] = (
    train_meta["index"]
    .astype(str)
    .str.strip()
)

train_meta["_y"] = (
    train_meta["Diagnosis"]
    .astype(str)
    .str.lower()
    .str.strip()
    .map({
        "control": 0,
        "ad": 1
    })
)

train_meta = train_meta.set_index("_sample")

train_ids = [
    x for x in train_meta.index
    if x in train_expr.index
]

train_expr = train_expr.loc[train_ids]
train_meta = train_meta.loc[train_ids]

print("\nDiscovery cohort")
print("Samples:", len(train_expr))
print("AD:", int(train_meta["_y"].sum()))
print("Control:", int((train_meta["_y"] == 0).sum()))


# ============================================================
# CHECK THREE GENES
# ============================================================

missing_train = [
    g for g in GENES
    if g not in train_expr.columns
]

if missing_train:
    raise RuntimeError(
        f"Missing genes in GSE48350: {missing_train}"
    )


# ============================================================
# GSE5281 EXPRESSION
# ============================================================

ext = pd.read_csv(EXT_EXPR)

gene_col = "Unnamed: 0"

# Remove CEL filename suffix from sample IDs.
ext.columns = [
    str(c).replace(".CEL.gz", "")
    for c in ext.columns
]

ext = ext.set_index(gene_col)

ext.index = (
    ext.index.astype(str).str.strip()
)

# Current structure:
# genes × samples
#
# Convert to:
# samples × genes

ext = ext.T

ext = ext.apply(
    pd.to_numeric,
    errors="coerce"
)

ext = ext.replace(
    [np.inf, -np.inf],
    np.nan
)

ext = ext.fillna(
    ext.median()
)

ext.index = (
    ext.index.astype(str).str.strip()
)

print("\nGSE5281 expression")
print("Samples:", len(ext))
print("Genes:", len(ext.columns))


# ============================================================
# GSE5281 METADATA
# ============================================================

meta = pd.read_csv(EXT_META)

if "Unnamed: 0" in meta.columns:
    meta["_sample"] = (
        meta["Unnamed: 0"]
        .astype(str)
        .str.strip()
    )
elif "GSM" in meta.columns:
    meta["_sample"] = (
        meta["GSM"]
        .astype(str)
        .str.strip()
    )
else:
    raise RuntimeError(
        "Cannot identify GSE5281 sample ID column."
    )

meta["_y"] = (
    meta["diagnosis"]
    .astype(str)
    .str.lower()
    .str.strip()
    .map({
        "control": 0,
        "ad": 1
    })
)

meta["_age"] = pd.to_numeric(
    meta["Age_used_years"],
    errors="coerce"
)

meta = meta.set_index("_sample")


# ============================================================
# ALIGN
# ============================================================

common = [
    x for x in meta.index
    if x in ext.index
]

print("\nSample alignment")
print("Metadata samples:", len(meta))
print("Expression samples:", len(ext))
print("Matched samples:", len(common))

if len(common) != len(meta):
    missing = [
        x for x in meta.index
        if x not in ext.index
    ]

    print("Missing:", missing)

if len(common) == 0:
    raise RuntimeError(
        "No GSE5281 samples matched."
    )

ext = ext.loc[common]
meta = meta.loc[common]


# ============================================================
# CHECK GENES
# ============================================================

missing_ext = [
    g for g in GENES
    if g not in ext.columns
]

if missing_ext:
    raise RuntimeError(
        f"Missing genes in GSE5281: {missing_ext}"
    )

print("\nThree-gene availability")

for gene in GENES:
    v = ext[gene].astype(float)

    print(
        f"{gene}: "
        f"min={v.min():.4f}, "
        f"max={v.max():.4f}, "
        f"mean={v.mean():.4f}"
    )


# ============================================================
# LOCKED THREE-GENE MODEL
# ============================================================

X_train = train_expr[GENES].astype(float)

y_train = (
    train_meta["_y"]
    .astype(int)
    .values
)

X_ext = ext[GENES].astype(float)

y_ext = (
    meta["_y"]
    .astype(int)
    .values
)

model = Pipeline([
    (
        "scale",
        StandardScaler()
    ),
    (
        "lr",
        LogisticRegression(
            max_iter=5000,
            random_state=42
        )
    )
])

model.fit(
    X_train,
    y_train
)

p = model.predict_proba(
    X_ext
)[:, 1]


# ============================================================
# THREE-GENE METRICS
# ============================================================

three_auc = roc_auc_score(
    y_ext,
    p
)

three_ap = average_precision_score(
    y_ext,
    p
)

three_brier = brier_score_loss(
    y_ext,
    p
)


# ============================================================
# AGE BASELINE
# ============================================================

age = (
    meta["_age"]
    .astype(float)
    .values
)

valid_age = np.isfinite(age)

age_auc = roc_auc_score(
    y_ext[valid_age],
    age[valid_age]
)

age_ap = average_precision_score(
    y_ext[valid_age],
    age[valid_age]
)


# ============================================================
# CALIBRATION INTERCEPT / SLOPE
# ============================================================

cal_intercept = np.nan
cal_slope = np.nan

try:

    import statsmodels.api as sm

    p_clip = np.clip(
        p,
        1e-6,
        1 - 1e-6
    )

    logit_p = np.log(
        p_clip / (1 - p_clip)
    )

    X_cal = sm.add_constant(
        logit_p
    )

    calibration = sm.Logit(
        y_ext,
        X_cal
    ).fit(
        disp=False
    )

    cal_intercept = calibration.params[0]
    cal_slope = calibration.params[1]

except Exception as e:

    print(
        "\nCalibration model warning:",
        e
    )


# ============================================================
# RESULTS TABLE
# ============================================================

result = pd.DataFrame([{
    "cohort": "GSE5281",
    "n": len(y_ext),
    "AD": int(y_ext.sum()),
    "Control": int((y_ext == 0).sum()),
    "three_gene_AUC": three_auc,
    "three_gene_AP": three_ap,
    "three_gene_Brier": three_brier,
    "three_gene_calibration_intercept": cal_intercept,
    "three_gene_calibration_slope": cal_slope,
    "age_AUC": age_auc,
    "age_AP": age_ap
}])

result.to_csv(
    OUT / "GSE5281_external_fixed_metrics.csv",
    index=False
)


# ============================================================
# PREDICTIONS
# ============================================================

pred = meta.copy()

pred["predicted_probability"] = p
pred["outcome"] = y_ext

pred.to_csv(
    OUT / "GSE5281_external_fixed_predictions.csv"
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print("\n")
print("=" * 90)
print("FINAL GSE5281 RESULT")
print("=" * 90)

print(
    result.to_string(index=False)
)

print("\nSaved:")
print(
    OUT / "GSE5281_external_fixed_metrics.csv"
)

print(
    OUT / "GSE5281_external_fixed_predictions.csv"
)

print("\nDONE")
