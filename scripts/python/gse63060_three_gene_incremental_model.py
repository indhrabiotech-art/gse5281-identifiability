import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import roc_auc_score
from scipy.stats import chi2

OUTDIR = "09_CrossTissue/results"
os.makedirs(OUTDIR, exist_ok=True)

EXPR_FILE = (
    "09_CrossTissue/results/"
    "GSE63060_three_gene_genelevel_matrix.csv"
)

META_FILE = (
    "09_CrossTissue/results/"
    "GSE63060_sample_metadata_raw.csv"
)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

expr = pd.read_csv(
    EXPR_FILE,
    index_col=0
)

meta = pd.read_csv(
    META_FILE
)

expr.index = expr.index.astype(str)
meta["GSM"] = meta["GSM"].astype(str)
meta = meta.set_index("GSM")

common = expr.index.intersection(meta.index)

expr = expr.loc[common]
meta = meta.loc[common]

# ============================================================
# DIAGNOSIS
# ============================================================

meta["Diagnosis"] = (
    meta["Characteristic_1"]
    .astype(str)
    .str.replace(
        "status:",
        "",
        regex=False
    )
    .str.strip()
)

keep = meta["Diagnosis"].isin([
    "MCI",
    "AD"
])

expr = expr.loc[keep]
meta = meta.loc[keep]

# ============================================================
# AGE
# ============================================================

meta["Age"] = (
    meta["Characteristic_3"]
    .astype(str)
    .str.extract(
        r"(\d+(?:\.\d+)?)"
    )[0]
    .astype(float)
)

# ============================================================
# SEX
# ============================================================

meta["Sex"] = (
    meta["Characteristic_4"]
    .astype(str)
    .str.replace(
        "gender:",
        "",
        regex=False
    )
    .str.strip()
    .str.lower()
)

meta["Sex_male"] = (
    meta["Sex"] == "male"
).astype(int)

# ============================================================
# COMPLETE CASES
# ============================================================

valid = (
    meta["Age"].notna()
    &
    meta["Sex"].isin([
        "male",
        "female"
    ])
)

for gene in GENES:
    valid &= expr[gene].notna()

expr = expr.loc[valid]
meta = meta.loc[valid]

# ============================================================
# Z-SCORE GENES
# ============================================================

Z = pd.DataFrame(index=expr.index)

for gene in GENES:
    Z[gene + "_z"] = (
        expr[gene] - expr[gene].mean()
    ) / expr[gene].std(ddof=1)

# ============================================================
# OUTCOME
# ============================================================

y = (
    meta["Diagnosis"]
    .map({
        "MCI": 0,
        "AD": 1
    })
    .astype(float)
)

# ============================================================
# MODEL DEFINITIONS
# ============================================================

model_specs = {

    "Age_Sex": [
        "Age",
        "Sex_male"
    ],

    "Age_Sex_ABCA6": [
        "Age",
        "Sex_male",
        "ABCA6_z"
    ],

    "Age_Sex_ABCA6_CRLF1": [
        "Age",
        "Sex_male",
        "ABCA6_z",
        "CRLF1_z"
    ],

    "Age_Sex_ABCA6_CRLF1_TNFRSF11B": [
        "Age",
        "Sex_male",
        "ABCA6_z",
        "CRLF1_z",
        "TNFRSF11B_z"
    ]
}

models = {}
rows = []

# ============================================================
# FIT MODELS
# ============================================================

for name, variables in model_specs.items():

    X = pd.DataFrame(index=meta.index)

    for variable in variables:

        if variable in meta.columns:
            X[variable] = meta[variable]

        else:
            X[variable] = Z[variable]

    X = sm.add_constant(X)

    model = sm.Logit(
        y,
        X
    ).fit(
        disp=False
    )

    models[name] = model

    prediction = model.predict()

    auc = roc_auc_score(
        y,
        prediction
    )

    rows.append({
        "Model": name,
        "N": len(y),
        "AUC": auc,
        "AIC": model.aic,
        "LogLikelihood": model.llf,
        "McFadden_Pseudo_R2": model.prsquared
    })

results = pd.DataFrame(rows)

# ============================================================
# ΔAUC FROM AGE + SEX
# ============================================================

base_auc = results.loc[
    results["Model"] == "Age_Sex",
    "AUC"
].iloc[0]

results["Delta_AUC_vs_AgeSex"] = (
    results["AUC"] - base_auc
)

# ============================================================
# NESTED LIKELIHOOD-RATIO TESTS
# ============================================================

comparisons = [
    (
        "Age_Sex",
        "Age_Sex_ABCA6"
    ),
    (
        "Age_Sex_ABCA6",
        "Age_Sex_ABCA6_CRLF1"
    ),
    (
        "Age_Sex_ABCA6_CRLF1",
        "Age_Sex_ABCA6_CRLF1_TNFRSF11B"
    ),
    (
        "Age_Sex",
        "Age_Sex_ABCA6_CRLF1_TNFRSF11B"
    )
]

lrt_rows = []

for base, full in comparisons:

    base_model = models[base]
    full_model = models[full]

    lr_stat = 2 * (
        full_model.llf -
        base_model.llf
    )

    df_diff = (
        full_model.df_model -
        base_model.df_model
    )

    p = chi2.sf(
        lr_stat,
        df_diff
    )

    base_auc_value = results.loc[
        results["Model"] == base,
        "AUC"
    ].iloc[0]

    full_auc_value = results.loc[
        results["Model"] == full,
        "AUC"
    ].iloc[0]

    lrt_rows.append({
        "Base_Model": base,
        "Full_Model": full,
        "LR_statistic": lr_stat,
        "df": df_diff,
        "LR_p": p,
        "Base_AUC": base_auc_value,
        "Full_AUC": full_auc_value,
        "Delta_AUC": (
            full_auc_value -
            base_auc_value
        )
    })

lrt = pd.DataFrame(lrt_rows)

# ============================================================
# COEFFICIENT TABLE
# ============================================================

coef_rows = []

for model_name, model in models.items():

    for variable in model.params.index:

        if variable == "const":
            continue

        beta = model.params[variable]
        se = model.bse[variable]
        p = model.pvalues[variable]

        ci_low = (
            beta -
            1.96 * se
        )

        ci_high = (
            beta +
            1.96 * se
        )

        coef_rows.append({
            "Model": model_name,
            "Variable": variable,
            "Beta": beta,
            "SE": se,
            "OR": np.exp(beta),
            "OR_95CI_lower": np.exp(ci_low),
            "OR_95CI_upper": np.exp(ci_high),
            "P_value": p
        })

coefficients = pd.DataFrame(
    coef_rows
)

# ============================================================
# PRINT
# ============================================================

print("=" * 80)
print("GSE63060 THREE-GENE INCREMENTAL MODEL ANALYSIS")
print("=" * 80)

print("\nSample counts:")
print(
    meta["Diagnosis"].value_counts()
)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 80)
print("NESTED LIKELIHOOD-RATIO TESTS")
print("=" * 80)

print(
    lrt.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 80)
print("COEFFICIENTS")
print("=" * 80)

print(
    coefficients.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# SAVE
# ============================================================

results_file = (
    OUTDIR +
    "/GSE63060_three_gene_incremental_models.csv"
)

lrt_file = (
    OUTDIR +
    "/GSE63060_three_gene_incremental_LRT.csv"
)

coef_file = (
    OUTDIR +
    "/GSE63060_three_gene_incremental_coefficients.csv"
)

results.to_csv(
    results_file,
    index=False
)

lrt.to_csv(
    lrt_file,
    index=False
)

coefficients.to_csv(
    coef_file,
    index=False
)

print("\nSaved:")
print(results_file)
print(lrt_file)
print(coef_file)

print("\n" + "=" * 80)
print("THREE-GENE INCREMENTAL ANALYSIS COMPLETE")
print("=" * 80)
