import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform

print("=" * 70)
print("GSE48350 → GSE5281 QUANTILE HARMONIZATION")
print("Training-reference based; NO external labels used")
print("=" * 70)

# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

TRAIN_FILE = "03_Preprocessing/GSE48350_RMA_genelevel.csv"
EXTERNAL_FILE = "03_Preprocessing/GSE5281_RMA_genelevel.csv"

OUT_FILE = "04_ML/GSE5281_RMA_genelevel_harmonized.csv"

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

train = pd.read_csv(
    TRAIN_FILE,
    index_col=0
)

external = pd.read_csv(
    EXTERNAL_FILE,
    index_col=0
)

print("\nTraining matrix:", train.shape)
print("External matrix:", external.shape)

# ------------------------------------------------------------
# Common genes
# ------------------------------------------------------------

common_genes = train.index.intersection(external.index)

print("Common genes:", len(common_genes))

if len(common_genes) == 0:
    raise ValueError("No common genes found.")

train = train.loc[common_genes]
external = external.loc[common_genes]

# ------------------------------------------------------------
# Verify candidate genes
# ------------------------------------------------------------

missing = [g for g in GENES if g not in common_genes]

if missing:
    raise ValueError(
        "Candidate genes missing from common gene set: "
        + ", ".join(missing)
    )

print("All 3 candidate genes present.")

# ------------------------------------------------------------
# IMPORTANT:
#
# We use GSE48350 as the reference distribution.
#
# Each GSE5281 gene is transformed according to the
# empirical distribution of that same gene in GSE48350.
#
# This uses NO diagnosis labels.
# ------------------------------------------------------------

def map_to_reference(reference, target):

    reference_sorted = np.sort(reference)

    n_ref = len(reference_sorted)
    n_target = len(target)

    # Target ranks
    target_order = np.argsort(target)
    target_sorted = target[target_order]

    # Percentile positions for target
    if n_target == 1:
        percentiles = np.array([0.5])
    else:
        percentiles = (
            np.arange(n_target) + 0.5
        ) / n_target

    # Corresponding reference quantiles
    ref_positions = (
        percentiles * n_ref - 0.5
    )

    ref_positions = np.clip(
        ref_positions,
        0,
        n_ref - 1
    )

    lower = np.floor(ref_positions).astype(int)
    upper = np.ceil(ref_positions).astype(int)

    weight = ref_positions - lower

    mapped_sorted = (
        reference_sorted[lower] * (1 - weight)
        + reference_sorted[upper] * weight
    )

    mapped = np.empty_like(target_sorted)
    mapped[target_order] = mapped_sorted

    return mapped


# ------------------------------------------------------------
# Harmonize gene-by-gene
# ------------------------------------------------------------

harmonized = pd.DataFrame(
    index=common_genes,
    columns=external.columns,
    dtype=float
)

print("\nHarmonizing genes...")

for i, gene in enumerate(common_genes, start=1):

    reference_values = (
        train.loc[gene].astype(float).values
    )

    external_values = (
        external.loc[gene].astype(float).values
    )

    harmonized.loc[gene] = map_to_reference(
        reference_values,
        external_values
    )

    if i % 2000 == 0:
        print(
            f"Processed {i} / {len(common_genes)} genes"
        )

# ------------------------------------------------------------
# QC
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HARMONIZATION QC")
print("=" * 70)

print("Output matrix:", harmonized.shape)

print(
    "Missing values:",
    harmonized.isna().sum().sum()
)

print("\nCandidate-gene distributions:")

for gene in GENES:

    print("\n" + gene)

    print(
        "GSE48350 reference:",
        f"mean={train.loc[gene].mean():.4f}",
        f"median={train.loc[gene].median():.4f}"
    )

    print(
        "GSE5281 before:",
        f"mean={external.loc[gene].mean():.4f}",
        f"median={external.loc[gene].median():.4f}"
    )

    print(
        "GSE5281 after:",
        f"mean={harmonized.loc[gene].mean():.4f}",
        f"median={harmonized.loc[gene].median():.4f}"
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

harmonized.to_csv(
    OUT_FILE
)

print("\nSaved:")
print(OUT_FILE)

print("\n" + "=" * 70)
print("HARMONIZATION COMPLETE")
print("=" * 70)
