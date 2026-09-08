import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

ROOT = "."

# ============================================================
# TRAINING DATA
# ============================================================

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

y_train = (
    train_meta["Diagnosis"]
    .map({"Control": 0, "AD": 1})
    .values
)

X_train = train[GENES].values

# ============================================================
# RAW EXTERNAL GSE5281
# ============================================================

# IMPORTANT:
# This must be the RAW expression matrix, not the harmonized matrix.
#
# Search for the existing raw GSE5281 gene-level file if necessary.
# The script below tries the canonical raw candidates.

raw_candidates = [
    "03_Preprocessing/GSE5281_RMA_genelevel.csv",
    "04_ML/GSE5281_RMA_genelevel_raw.csv",
    "04_ML/External_Validation/GSE5281_RMA_genelevel.csv",
    "04_ML/External_Validation/GSE5281_RMA_genelevel_raw.csv",
]

raw_path = None

for candidate in raw_candidates:
    try:
        test = pd.read_csv(candidate, index_col=0, nrows=2)
        raw_path = candidate
        break
    except Exception:
        pass

if raw_path is None:
    raise FileNotFoundError(
        "\nCould not automatically locate the RAW GSE5281 "
        "gene-level expression matrix.\n\n"
        "Available likely files:\n"
        + "\n".join(
            str(x) for x in
            __import__("pathlib").Path("04_ML").rglob("*GSE5281*")
        )
    )

print("=" * 72)
print("RAW EXTERNAL GSE5281 PERMUTATION TEST")
print("=" * 72)

print(f"Raw expression file: {raw_path}")

external = pd.read_csv(
    raw_path,
    index_col=0
).T

external.index = (
    external.index
    .astype(str)
    .str.replace(".CEL.gz", "", regex=False)
    .str.strip()
)

# ============================================================
# EXTERNAL METADATA
# ============================================================

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

y_external = (
    external_meta["diagnosis"]
    .map({"Control": 0, "AD": 1})
    .values
)

# ============================================================
# CHECKS
# ============================================================

if len(y_external) != len(external):
    raise ValueError(
        f"Sample mismatch: expression={len(external)}, "
        f"metadata={len(y_external)}"
    )

if np.isnan(y_external).any():
    raise ValueError("Missing external diagnostic labels detected.")

missing_genes = [
    gene for gene in GENES
    if gene not in external.columns
]

if missing_genes:
    raise ValueError(
        f"Missing genes in raw external matrix: {missing_genes}"
    )

print(f"External samples: {len(external)}")
print(f"AD: {np.sum(y_external == 1)}")
print(f"Control: {np.sum(y_external == 0)}")

# ============================================================
# EXPRESSION
# ============================================================

X_external = external[GENES].values

# ============================================================
# LOCKED MODEL
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_external_scaled = scaler.transform(X_external)

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

# ============================================================
# OBSERVED RAW EXTERNAL AUC
# ============================================================

prob = model.predict_proba(
    X_external_scaled
)[:, 1]

observed_auc = roc_auc_score(
    y_external,
    prob
)

print()
print(f"Observed RAW external AUC: {observed_auc:.6f}")

# ============================================================
# VERIFY AGAINST EXPECTED CANONICAL VALUE
# ============================================================

EXPECTED = 0.6615384615384615

if not np.isclose(observed_auc, EXPECTED, atol=1e-10):
    raise RuntimeError(
        "\nSTOP.\n"
        f"Observed raw AUC = {observed_auc:.12f}\n"
        f"Expected canonical AUC = {EXPECTED:.12f}\n\n"
        "This file is NOT reproducing the canonical raw result.\n"
        "Do not continue until the raw expression preprocessing "
        "is verified."
    )

print("PASS: raw AUC reproduces canonical 0.661538.")

# ============================================================
# PERMUTATION TEST
# ============================================================

N_PERM = 10000

permuted_auc = np.zeros(N_PERM)

for i in range(N_PERM):

    shuffled_labels = np.random.permutation(
        y_external
    )

    permuted_auc[i] = roc_auc_score(
        shuffled_labels,
        prob
    )

extreme = (
    np.abs(permuted_auc - 0.5)
    >=
    np.abs(observed_auc - 0.5)
)

p_value = (
    np.sum(extreme) + 1
) / (
    N_PERM + 1
)

print()
print(f"Permutations: {N_PERM}")
print(f"RAW empirical two-sided p-value: {p_value:.6f}")

print()
print("Null distribution:")
print(f"Mean:    {permuted_auc.mean():.6f}")
print(f"Median:  {np.median(permuted_auc):.6f}")
print(f"2.5%:    {np.percentile(permuted_auc, 2.5):.6f}")
print(f"97.5%:   {np.percentile(permuted_auc, 97.5):.6f}")

# ============================================================
# SAVE RAW PERMUTATION RESULT
# ============================================================

out_dir = "04_ML/External_Validation"

summary = pd.DataFrame({
    "analysis": ["RAW_GSE5281"],
    "observed_auc": [observed_auc],
    "permutations": [N_PERM],
    "empirical_two_sided_p": [p_value],
    "permutation_mean_auc": [permuted_auc.mean()],
    "permutation_median_auc": [np.median(permuted_auc)],
    "permutation_q025": [np.percentile(permuted_auc, 2.5)],
    "permutation_q975": [np.percentile(permuted_auc, 97.5)]
})

summary.to_csv(
    f"{out_dir}/GSE5281_RAW_3gene_permutation_test.csv",
    index=False
)

pd.DataFrame({
    "permuted_auc": permuted_auc
}).to_csv(
    f"{out_dir}/GSE5281_RAW_3gene_permutation_auc_distribution.csv",
    index=False
)

print()
print("Saved:")
print(
    f"{out_dir}/GSE5281_RAW_3gene_permutation_test.csv"
)
print(
    f"{out_dir}/GSE5281_RAW_3gene_permutation_auc_distribution.csv"
)

print()
print("=" * 72)
print("RAW PERMUTATION TEST COMPLETE")
print("=" * 72)
