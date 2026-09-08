"""
Diagnostic: is the near-perfect classification driven by age
confounding rather than AD biology?

WHY: controls span age 20-99, AD cases cluster ~60-94 (per earlier
metadata QC). Age has a strong, independent effect on gene
expression. If a trivial age-based classifier separates the classes
almost as well as the gene-expression model, the "signal" the model
found may substantially reflect age, not AD-specific biology.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import stats

train_meta = pd.read_csv("04_ML/GSE48350_hippocampus_train_meta.csv")
test_meta  = pd.read_csv("04_ML/GSE48350_hippocampus_test_meta.csv")

# ------------------------------------------------------------
# Check 1: does age differ significantly between AD and Control?
# ------------------------------------------------------------
ad_ages = train_meta.loc[train_meta["Diagnosis"] == "AD", "Age"]
control_ages = train_meta.loc[train_meta["Diagnosis"] == "Control", "Age"]

print("Training set age by group:")
print("AD:      mean =", round(ad_ages.mean(), 1), " sd =", round(ad_ages.std(), 1),
      " range =", ad_ages.min(), "-", ad_ages.max())
print("Control: mean =", round(control_ages.mean(), 1), " sd =", round(control_ages.std(), 1),
      " range =", control_ages.min(), "-", control_ages.max())

t_stat, p_val = stats.ttest_ind(ad_ages, control_ages)
print(f"\nt-test AD vs Control age: t={t_stat:.2f}, p={p_val:.4f}")
if p_val < 0.05:
    print("Age differs significantly between groups - confounding is plausible.")
else:
    print("No significant age difference detected between groups.")

# ------------------------------------------------------------
# Check 2: how well does AGE ALONE classify AD vs Control?
# ------------------------------------------------------------
# If age alone gets a high CV AUC, it means a large part of any
# gene-expression model's "signal" could just be re-deriving age.
X_age_train = train_meta[["Age"]].values
y_train = (train_meta["Diagnosis"] == "AD").astype(int).values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
age_model = LogisticRegression()
age_auc_scores = cross_val_score(age_model, X_age_train, y_train, cv=cv, scoring="roc_auc")

print("\n" + "=" * 60)
print("AGE-ONLY BASELINE CLASSIFIER")
print("=" * 60)
print("CV ROC-AUC per fold (age alone):", np.round(age_auc_scores, 3))
print("Mean CV ROC-AUC (age alone):", round(age_auc_scores.mean(), 3))
print("\nCompare this to your gene-expression models:")
print("  Logistic Regression (genes): 0.870")
print("  Random Forest (genes):       0.938")
print("\nInterpretation guide:")
print("  - If age-alone AUC is LOW (~0.5-0.6): age is not a major")
print("    confounder, the gene-expression signal is likely real.")
print("  - If age-alone AUC is HIGH (~0.8+): a large share of what")
print("    the gene models are 'detecting' may just be age-related")
print("    expression changes, not AD-specific biology. Consider")
print("    age-matching the cohort or including age as a covariate")
print("    / regressing it out before re-running the ML models.")
