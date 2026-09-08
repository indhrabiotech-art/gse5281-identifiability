import numpy as np
import pandas as pd

print("=" * 70)
print("CROSS-DATASET GENE EFFECT-SIZE ANALYSIS")
print("=" * 70)

genes = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B",
    "PLA2G7",
    "SORBS1",
    "PMS2P2",
    "TAB2"
]

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

external = pd.read_csv(
    "04_ML/GSE5281_RMA_genelevel_age_adjusted.csv",
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

results = []

for gene in genes:

    train_ad = train.loc[
        train_meta["Diagnosis"].eq("AD").to_numpy(),
        gene
    ].to_numpy()

    train_control = train.loc[
        train_meta["Diagnosis"].eq("Control").to_numpy(),
        gene
    ].to_numpy()

    ext_ad = external.loc[
        external_meta["diagnosis"].eq("AD").to_numpy(),
        gene
    ].to_numpy()

    ext_control = external.loc[
        external_meta["diagnosis"].eq("Control").to_numpy(),
        gene
    ].to_numpy()

    train_mean_ad = np.mean(train_ad)
    train_mean_control = np.mean(train_control)

    ext_mean_ad = np.mean(ext_ad)
    ext_mean_control = np.mean(ext_control)

    train_delta = train_mean_ad - train_mean_control
    ext_delta = ext_mean_ad - ext_mean_control

    train_sd = np.sqrt(
        (
            (len(train_ad)-1)*np.var(train_ad, ddof=1)
            +
            (len(train_control)-1)*np.var(train_control, ddof=1)
        )
        /
        (len(train_ad)+len(train_control)-2)
    )

    ext_sd = np.sqrt(
        (
            (len(ext_ad)-1)*np.var(ext_ad, ddof=1)
            +
            (len(ext_control)-1)*np.var(ext_control, ddof=1)
        )
        /
        (len(ext_ad)+len(ext_control)-2)
    )

    train_d = train_delta / train_sd
    ext_d = ext_delta / ext_sd

    results.append({
        "gene": gene,

        "train_AD_mean": train_mean_ad,
        "train_Control_mean": train_mean_control,
        "train_delta": train_delta,
        "train_Cohens_d": train_d,

        "external_AD_mean": ext_mean_ad,
        "external_Control_mean": ext_mean_control,
        "external_delta": ext_delta,
        "external_Cohens_d": ext_d,

        "direction_same": (
            np.sign(train_delta) == np.sign(ext_delta)
        ),

        "effect_size_change": ext_d - train_d
    })

results = pd.DataFrame(results)

print("\n" + "=" * 70)
print("EFFECT SIZE COMPARISON")
print("=" * 70)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print("GENES WITH DIRECTION REVERSAL")
print("=" * 70)

print(
    results[
        ~results["direction_same"]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results.to_csv(
    "04_ML/External_Validation/"
    "GSE5281_cross_dataset_effect_sizes.csv",
    index=False
)

print("\nSaved:")
print(
    "04_ML/External_Validation/"
    "GSE5281_cross_dataset_effect_sizes.csv"
)

print("\n" + "=" * 70)
print("EFFECT-SIZE ANALYSIS COMPLETE")
print("=" * 70)
