import numpy as np
import pandas as pd

TRAIN = "03_Preprocessing/GSE48350_RMA_genelevel.csv"
EXT_RAW = "03_Preprocessing/GSE5281_RMA_genelevel.csv"
EXT_HARM = "04_ML/GSE5281_RMA_genelevel_harmonized.csv"

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

train = pd.read_csv(TRAIN, index_col=0)
raw = pd.read_csv(EXT_RAW, index_col=0)
harm = pd.read_csv(EXT_HARM, index_col=0)

common = train.index.intersection(raw.index).intersection(harm.index)

train = train.loc[common]
raw = raw.loc[common]
harm = harm.loc[common]

print("=" * 80)
print("HARMONIZATION DISTRIBUTION AUDIT")
print("=" * 80)

print("\nMatrix dimensions")
print("-" * 80)
print("Training :", train.shape)
print("Raw ext  :", raw.shape)
print("Harmonized:", harm.shape)

print("\nMissing values")
print("-" * 80)
print("Training :", int(train.isna().sum().sum()))
print("Raw ext  :", int(raw.isna().sum().sum()))
print("Harmonized:", int(harm.isna().sum().sum()))

print("\nCandidate-gene distribution comparison")
print("-" * 80)

for gene in GENES:

    tr = train.loc[gene].astype(float)
    rw = raw.loc[gene].astype(float)
    hm = harm.loc[gene].astype(float)

    print(f"\n{gene}")

    print(
        f"Training reference : "
        f"mean={tr.mean():.4f}, "
        f"median={tr.median():.4f}, "
        f"sd={tr.std():.4f}"
    )

    print(
        f"Raw external       : "
        f"mean={rw.mean():.4f}, "
        f"median={rw.median():.4f}, "
        f"sd={rw.std():.4f}"
    )

    print(
        f"Harmonized external: "
        f"mean={hm.mean():.4f}, "
        f"median={hm.median():.4f}, "
        f"sd={hm.std():.4f}"
    )

print("\nRank preservation check")
print("-" * 80)

for gene in GENES:

    rw = raw.loc[gene].astype(float)
    hm = harm.loc[gene].astype(float)

    rank_corr = rw.rank().corr(hm.rank(), method="spearman")

    print(
        f"{gene:12s} "
        f"Spearman raw-vs-harmonized = {rank_corr:.6f}"
    )

print("\nReference-scale deviation")
print("-" * 80)

rows = []

for gene in GENES:

    tr = train.loc[gene].astype(float)
    hm = harm.loc[gene].astype(float)

    rows.append({
        "gene": gene,
        "training_mean": tr.mean(),
        "harmonized_mean": hm.mean(),
        "mean_difference": hm.mean() - tr.mean(),
        "training_median": tr.median(),
        "harmonized_median": hm.median(),
        "median_difference": hm.median() - tr.median()
    })

audit = pd.DataFrame(rows)

print(audit.to_string(index=False))

audit.to_csv(
    "04_ML/External_Validation/GSE5281_harmonization_distribution_audit.csv",
    index=False
)

print("\nSaved:")
print(
    "04_ML/External_Validation/"
    "GSE5281_harmonization_distribution_audit.csv"
)

print("\n" + "=" * 80)
print("HARMONIZATION DISTRIBUTION AUDIT COMPLETE")
print("=" * 80)
