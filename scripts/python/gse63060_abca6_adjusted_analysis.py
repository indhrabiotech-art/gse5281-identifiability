import os
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

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

expr = pd.read_csv(
    EXPR_FILE,
    index_col=0
)

meta = pd.read_csv(META_FILE)

expr.index = expr.index.astype(str)
meta["GSM"] = meta["GSM"].astype(str)

meta = meta.set_index("GSM")

common = expr.index.intersection(meta.index)

expr = expr.loc[common]
meta = meta.loc[common]

# ------------------------------------------------------------
# Diagnosis
# ------------------------------------------------------------

meta["Diagnosis"] = (
    meta["Characteristic_1"]
    .astype(str)
    .str.replace("status:", "", regex=False)
    .str.strip()
)

# Only MCI and AD
keep = meta["Diagnosis"].isin(["MCI", "AD"])

expr = expr.loc[keep]
meta = meta.loc[keep]

# ------------------------------------------------------------
# Parse age
# ------------------------------------------------------------

meta["Age"] = (
    meta["Characteristic_3"]
    .astype(str)
    .str.extract(r"(\d+(?:\.\d+)?)")[0]
    .astype(float)
)

# ------------------------------------------------------------
# Parse sex
# ------------------------------------------------------------

meta["Sex"] = (
    meta["Characteristic_4"]
    .astype(str)
    .str.replace("gender:", "", regex=False)
    .str.strip()
    .str.lower()
)

# Keep valid demographic records
valid = (
    meta["Age"].notna()
    &
    meta["Sex"].isin(["male", "female"])
)

expr = expr.loc[valid]
meta = meta.loc[valid]

print("=" * 75)
print("GSE63060 ABCA6 DEMOGRAPHIC-ADJUSTMENT ANALYSIS")
print("=" * 75)

print("\nGroup counts:")
print(meta["Diagnosis"].value_counts())

print("\nSex counts:")
print(meta["Sex"].value_counts())

print("\nAge summary:")
print(meta["Age"].describe())

# ------------------------------------------------------------
# ABCA6 z-score
# ------------------------------------------------------------

abca6 = expr["ABCA6"]

abca6_z = (
    abca6 - abca6.mean()
) / abca6.std(ddof=1)

# ------------------------------------------------------------
# Binary outcome
# ------------------------------------------------------------

y = (
    meta["Diagnosis"]
    .map({
        "MCI": 0,
        "AD": 1
    })
    .astype(float)
)

# ------------------------------------------------------------
# Model 1: ABCA6 only
# ------------------------------------------------------------

X1 = pd.DataFrame({
    "ABCA6_z": abca6_z
}, index=meta.index)

X1 = sm.add_constant(X1)

model1 = sm.Logit(y, X1).fit(disp=False)

# ------------------------------------------------------------
# Model 2: ABCA6 + Age
# ------------------------------------------------------------

X2 = pd.DataFrame({
    "ABCA6_z": abca6_z,
    "Age": meta["Age"]
}, index=meta.index)

X2 = sm.add_constant(X2)

model2 = sm.Logit(y, X2).fit(disp=False)

# ------------------------------------------------------------
# Model 3: ABCA6 + Age + Sex
# ------------------------------------------------------------

sex_male = (
    meta["Sex"]
    .eq("male")
    .astype(int)
)

X3 = pd.DataFrame({
    "ABCA6_z": abca6_z,
    "Age": meta["Age"],
    "Sex_male": sex_male
}, index=meta.index)

X3 = sm.add_constant(X3)

model3 = sm.Logit(y, X3).fit(disp=False)

# ------------------------------------------------------------
# Extract model statistics
# ------------------------------------------------------------

models = {
    "ABCA6_only": model1,
    "ABCA6_plus_Age": model2,
    "ABCA6_plus_Age_Sex": model3
}

rows = []

for name, model in models.items():

    coef = model.params["ABCA6_z"]
    se = model.bse["ABCA6_z"]
    p = model.pvalues["ABCA6_z"]

    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se

    OR = np.exp(coef)
    OR_low = np.exp(ci_low)
    OR_high = np.exp(ci_high)

    probability = model.predict()

    auc = roc_auc_score(
        y,
        probability
    )

    rows.append({
        "Model": name,
        "N": len(y),
        "ABCA6_beta": coef,
        "ABCA6_SE": se,
        "ABCA6_OR": OR,
        "OR_95CI_lower": OR_low,
        "OR_95CI_upper": OR_high,
        "ABCA6_p": p,
        "Model_AUC": auc,
        "AIC": model.aic,
        "Pseudo_R2": model.prsquared
    })

results = pd.DataFrame(rows)

# ------------------------------------------------------------
# MCI vs AD raw group comparison
# ------------------------------------------------------------

mci = abca6_z[
    meta["Diagnosis"] == "MCI"
]

ad = abca6_z[
    meta["Diagnosis"] == "AD"
]

U, p_raw = mannwhitneyu(
    mci,
    ad,
    alternative="two-sided"
)

print("\n" + "=" * 75)
print("RAW ABCA6 MCI VS AD")
print("=" * 75)

print("MCI N:", len(mci))
print("AD N:", len(ad))
print("MCI mean:", mci.mean())
print("AD mean:", ad.mean())
print("MCI - AD:", mci.mean() - ad.mean())
print("Mann-Whitney U:", U)
print("P:", p_raw)

print("\n" + "=" * 75)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 75)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

outfile = (
    OUTDIR +
    "/GSE63060_ABCA6_demographic_adjustment.csv"
)

results.to_csv(
    outfile,
    index=False
)

print("\nSaved:")
print(outfile)

print("=" * 75)
print("ABCA6 ADJUSTMENT ANALYSIS COMPLETE")
print("=" * 75)
