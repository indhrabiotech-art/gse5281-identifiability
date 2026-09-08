import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

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

X_train = train[GENES].values
X_external = external[GENES].values

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

model.fit(X_train_scaled, y_train)

prob = model.predict_proba(X_external_scaled)[:, 1]

observed_auc = roc_auc_score(
    y_external,
    prob
)

print("=" * 70)
print("BOOTSTRAP CONFIDENCE INTERVAL — 3-GENE EXTERNAL AUC")
print("=" * 70)

print(f"Observed AUC: {observed_auc:.4f}")

n_bootstrap = 10000
boot_auc = []

n = len(y_external)

for i in range(n_bootstrap):

    idx = np.random.randint(
        0,
        n,
        n
    )

    y_b = y_external[idx]
    p_b = prob[idx]

    if len(np.unique(y_b)) < 2:
        continue

    boot_auc.append(
        roc_auc_score(y_b, p_b)
    )

boot_auc = np.array(boot_auc)

lower = np.percentile(
    boot_auc,
    2.5
)

upper = np.percentile(
    boot_auc,
    97.5
)

print(f"Valid bootstrap samples: {len(boot_auc)}")
print(f"95% bootstrap CI: {lower:.4f} – {upper:.4f}")

print("\nAUC > 0.5 bootstrap proportion:")

print(
    np.mean(boot_auc > 0.5)
)

print("\nAUC < 0.5 bootstrap proportion:")

print(
    np.mean(boot_auc < 0.5)
)

result = pd.DataFrame({
    "observed_auc": [observed_auc],
    "bootstrap_mean_auc": [boot_auc.mean()],
    "ci_lower_95": [lower],
    "ci_upper_95": [upper],
    "n_external": [n],
    "n_ad": [int(y_external.sum())],
    "n_control": [int((y_external == 0).sum())]
})

result.to_csv(
    "04_ML/External_Validation/"
    "GSE5281_3gene_bootstrap_auc.csv",
    index=False
)

print("\nSaved:")
print(
    "04_ML/External_Validation/"
    "GSE5281_3gene_bootstrap_auc.csv"
)

print("=" * 70)
print("BOOTSTRAP ANALYSIS COMPLETE")
print("=" * 70)
