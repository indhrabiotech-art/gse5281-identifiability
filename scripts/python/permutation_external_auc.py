import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

# ------------------------------------------------------------
# Load training data
# ------------------------------------------------------------

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

# ------------------------------------------------------------
# Load harmonized external data
# ------------------------------------------------------------

external = pd.read_csv(
    "04_ML/GSE5281_RMA_genelevel_harmonized.csv",
    index_col=0
).T

external.index = (
    external.index
    .str.replace(".CEL.gz", "", regex=False)
)

external_meta = pd.read_csv(
    "02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv"
)

external_meta["GSM"] = (
    external_meta["GSM"]
    .astype(str)
    .str.strip()
)

external_meta = (
    external_meta
    .set_index("GSM")
    .loc[external.index]
    .reset_index()
)

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

y_train = (
    train_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .values
)

y_external = (
    external_meta["diagnosis"]
    .map({"Control": 0, "AD": 1})
    .values
)

# ------------------------------------------------------------
# Expression
# ------------------------------------------------------------

X_train = train[GENES].values
X_external = external[GENES].values

# ------------------------------------------------------------
# Scale using training data ONLY
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_external_scaled = scaler.transform(X_external)

# ------------------------------------------------------------
# Train locked model
# ------------------------------------------------------------

model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=0.3,
    max_iter=10000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

# ------------------------------------------------------------
# Observed external AUC
# ------------------------------------------------------------

prob = model.predict_proba(
    X_external_scaled
)[:, 1]

observed_auc = roc_auc_score(
    y_external,
    prob
)

print("=" * 70)
print("PERMUTATION TEST — EXTERNAL 3-GENE AUC")
print("=" * 70)

print(f"Observed AUC: {observed_auc:.4f}")

# ------------------------------------------------------------
# Permutation test
#
# Keep expression and locked model fixed.
# Randomize external labels only.
# ------------------------------------------------------------

n_perm = 10000

permuted_auc = np.zeros(n_perm)

for i in range(n_perm):

    shuffled_labels = np.random.permutation(
        y_external
    )

    permuted_auc[i] = roc_auc_score(
        shuffled_labels,
        prob
    )

# ------------------------------------------------------------
# Empirical two-sided p-value
# ------------------------------------------------------------

extreme = np.abs(
    permuted_auc - 0.5
) >= np.abs(
    observed_auc - 0.5
)

p_value = (
    np.sum(extreme) + 1
) / (
    n_perm + 1
)

print(f"Permutations: {n_perm}")
print(f"Empirical two-sided p-value: {p_value:.4f}")

print("\nPermutation AUC distribution:")
print(f"Mean:   {permuted_auc.mean():.4f}")
print(f"Median: {np.median(permuted_auc):.4f}")
print(f"2.5%:   {np.percentile(permuted_auc, 2.5):.4f}")
print(f"97.5%:  {np.percentile(permuted_auc, 97.5):.4f}")

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

summary = pd.DataFrame({
    "observed_auc": [observed_auc],
    "permutations": [n_perm],
    "empirical_p_value": [p_value],
    "permutation_mean_auc": [permuted_auc.mean()],
    "permutation_median_auc": [np.median(permuted_auc)],
    "permutation_q025": [np.percentile(permuted_auc, 2.5)],
    "permutation_q975": [np.percentile(permuted_auc, 97.5)]
})

summary.to_csv(
    "04_ML/External_Validation/"
    "GSE5281_3gene_permutation_test.csv",
    index=False
)

pd.DataFrame({
    "permuted_auc": permuted_auc
}).to_csv(
    "04_ML/External_Validation/"
    "GSE5281_3gene_permutation_auc_distribution.csv",
    index=False
)

print("\nSaved:")
print(
    "04_ML/External_Validation/"
    "GSE5281_3gene_permutation_test.csv"
)

print("=" * 70)
print("PERMUTATION TEST COMPLETE")
print("=" * 70)
