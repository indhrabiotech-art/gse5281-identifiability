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

expr = pd.read_csv(EXPR_FILE, index_col=0)
meta = pd.read_csv(META_FILE)

expr.index = expr.index.astype(str)
meta["GSM"] = meta["GSM"].astype(str)
meta = meta.set_index("GSM")

common = expr.index.intersection(meta.index)

expr = expr.loc[common]
meta = meta.loc[common]

# Diagnosis
meta["Diagnosis"] = (
    meta["Characteristic_1"]
    .astype(str)
    .str.replace("status:", "", regex=False)
    .str.strip()
)

# MCI vs AD only
keep = meta["Diagnosis"].isin(["MCI", "AD"])

expr = expr.loc[keep]
meta = meta.loc[keep]

# Age
meta["Age"] = (
    meta["Characteristic_3"]
    .astype(str)
    .str.extract(r"(\d+(?:\.\d+)?)")[0]
    .astype(float)
)

# Sex
meta["Sex"] = (
    meta["Characteristic_4"]
    .astype(str)
    .str.replace("gender:", "", regex=False)
    .str.strip()
    .str.lower()
)

meta["Sex_male"] = (
    meta["Sex"] == "male"
).astype(int)

# Remove incomplete records
valid = (
    meta["Age"].notna()
    & meta["Sex"].isin(["male", "female"])
)

expr = expr.loc[valid]
meta = meta.loc[valid]

# ABCA6 Z-score
abca6 = expr["ABCA6"]

abca6_z = (
    abca6 - abca6.mean()
) / abca6.std(ddof=1)

# Outcome
y = (
    meta["Diagnosis"]
    .map({
        "MCI": 0,
        "AD": 1
    })
    .astype(float)
)

# ============================================================
# MODEL A: AGE + SEX
# ============================================================

XA = pd.DataFrame({
    "Age": meta["Age"],
    "Sex_male": meta["Sex_male"]
})

XA = sm.add_constant(XA)

model_A = sm.Logit(y, XA).fit(disp=False)

# ============================================================
# MODEL B: ABCA6 ONLY
# ============================================================

XB = pd.DataFrame({
    "ABCA6_z": abca6_z
})

XB = sm.add_constant(XB)

model_B = sm.Logit(y, XB).fit(disp=False)

# ============================================================
# MODEL C: AGE + SEX + ABCA6
# ============================================================

XC = pd.DataFrame({
    "Age": meta["Age"],
    "Sex_male": meta["Sex_male"],
    "ABCA6_z": abca6_z
})

XC = sm.add_constant(XC)

model_C = sm.Logit(y, XC).fit(disp=False)

models = {
    "Age_Sex": model_A,
    "ABCA6_only": model_B,
    "Age_Sex_ABCA6": model_C
}

rows = []

for name, model in models.items():

    pred = model.predict()

    auc = roc_auc_score(
        y,
        pred
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
# INCREMENTAL LIKELIHOOD-RATIO TEST
# ============================================================

lr_stat = 2 * (
    model_C.llf - model_A.llf
)

df_diff = (
    model_C.df_model -
    model_A.df_model
)

lr_p = chi2.sf(
    lr_stat,
    df_diff
)

auc_age_sex = results.loc[
    results["Model"] == "Age_Sex",
    "AUC"
].iloc[0]

auc_full = results.loc[
    results["Model"] == "Age_Sex_ABCA6",
    "AUC"
].iloc[0]

delta_auc = auc_full - auc_age_sex

print("=" * 75)
print("GSE63060 ABCA6 INCREMENTAL MODEL ANALYSIS")
print("=" * 75)

print("\nGroup counts:")
print(meta["Diagnosis"].value_counts())

print("\nModel performance:")
print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 75)
print("ABCA6 INCREMENTAL CONTRIBUTION")
print("=" * 75)

print("Age + Sex AUC:", f"{auc_age_sex:.6f}")
print("Age + Sex + ABCA6 AUC:", f"{auc_full:.6f}")
print("Delta AUC:", f"{delta_auc:.6f}")

print("Likelihood-ratio statistic:", f"{lr_stat:.6f}")
print("Degrees of freedom:", df_diff)
print("Likelihood-ratio p:", f"{lr_p:.6g}")

# Save model comparison
results["Delta_AUC_vs_AgeSex"] = (
    results["AUC"] - auc_age_sex
)

outfile = (
    OUTDIR +
    "/GSE63060_ABCA6_incremental_model.csv"
)

results.to_csv(
    outfile,
    index=False
)

# Save LR test separately
lr_out = pd.DataFrame([{
    "Base_Model": "Age + Sex",
    "Full_Model": "Age + Sex + ABCA6",
    "LR_statistic": lr_stat,
    "df": df_diff,
    "LR_p": lr_p,
    "Base_AUC": auc_age_sex,
    "Full_AUC": auc_full,
    "Delta_AUC": delta_auc
}])

lr_file = (
    OUTDIR +
    "/GSE63060_ABCA6_incremental_LRT.csv"
)

lr_out.to_csv(
    lr_file,
    index=False
)

print("\nSaved:")
print(outfile)
print(lr_file)

print("\n" + "=" * 75)
print("INCREMENTAL MODEL ANALYSIS COMPLETE")
print("=" * 75)
