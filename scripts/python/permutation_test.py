"""
Permutation test for Random Forest's CV AUC (0.979 on age-adjusted data).

WHY: with 46 training samples and 5,342 features, a non-linear model
CAN find a high-AUC-looking pattern that reflects overfitting to
this specific sample set, not real biology. Shuffling the labels
many times and re-running the SAME CV procedure shows what AUC is
achievable by chance alone, given this exact sample/feature shape.
If the real (unshuffled) AUC sits well outside that null distribution,
that's genuine evidence of signal. If not, treat 0.979 as noise.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_STATE = 42
N_PERMUTATIONS = 200  # each one is a full 5-fold CV run - keep this a manageable number for an i3/8GB machine

full_matrix = pd.read_csv("04_ML/GSE48350_hippocampus_ML_matrix_AGEADJUSTED_v2.csv", index_col=0)
train_meta  = pd.read_csv("04_ML/GSE48350_hippocampus_train_meta.csv")

full_matrix_T = full_matrix.T
X_train = full_matrix_T.loc[train_meta["GSM"]].values
y_train_real = (train_meta["Diagnosis"] == "AD").astype(int).values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=RANDOM_STATE
)

# ------------------------------------------------------------
# Real (unshuffled) score - should match what you already saw (~0.979)
# ------------------------------------------------------------
real_scores = cross_val_score(rf, X_train, y_train_real, cv=cv, scoring="roc_auc")
real_mean_auc = real_scores.mean()
print(f"Real (unshuffled) CV mean AUC: {real_mean_auc:.3f}")

# ------------------------------------------------------------
# Null distribution: shuffle labels, repeat CV, record mean AUC
# ------------------------------------------------------------
print(f"\nRunning {N_PERMUTATIONS} label-shuffled permutations "
      f"(this may take a few minutes)...")

rng = np.random.RandomState(RANDOM_STATE)
null_aucs = []

for i in range(N_PERMUTATIONS):
    y_shuffled = rng.permutation(y_train_real)
    scores = cross_val_score(rf, X_train, y_shuffled, cv=cv, scoring="roc_auc")
    null_aucs.append(scores.mean())
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{N_PERMUTATIONS} permutations done...")

null_aucs = np.array(null_aucs)

# ------------------------------------------------------------
# Empirical p-value: fraction of shuffled runs that matched/beat
# the real score
# ------------------------------------------------------------
p_value = (np.sum(null_aucs >= real_mean_auc) + 1) / (N_PERMUTATIONS + 1)

print("\n" + "=" * 60)
print("PERMUTATION TEST RESULTS")
print("=" * 60)
print(f"Real CV mean AUC:        {real_mean_auc:.3f}")
print(f"Null distribution mean:  {null_aucs.mean():.3f}")
print(f"Null distribution sd:    {null_aucs.std():.3f}")
print(f"Null distribution max:   {null_aucs.max():.3f}")
print(f"Empirical p-value:       {p_value:.4f}")
print(f"  (fraction of {N_PERMUTATIONS} shuffled runs that matched or beat the real score)")

print("\nInterpretation:")
if p_value < 0.01:
    print("  p < 0.01: the real AUC is very unlikely to occur by chance given")
    print("  this sample/feature shape. Reasonable evidence of genuine signal,")
    print("  though this still doesn't prove the signal is AD-specific (could")
    print("  reflect other unmeasured confounds) - external validation on")
    print("  GSE5281 remains the real test.")
elif p_value < 0.05:
    print("  p < 0.05: some evidence of signal, but borderline given small")
    print("  sample size. Treat cautiously.")
else:
    print("  p >= 0.05: the real AUC is NOT clearly distinguishable from what")
    print("  random label shuffling achieves on this sample/feature shape.")
    print("  This suggests the 0.979 AUC likely reflects overfitting to small-")
    print("  sample noise rather than genuine AD-associated signal. Consider")
    print("  this a caution against over-interpreting the RF result as-is.")
