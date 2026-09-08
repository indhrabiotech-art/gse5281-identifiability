"""
SHAP fold-stability check.

WHY THIS IS NOW ESSENTIAL, NOT OPTIONAL: the single-fit SHAP ranking
turned up mostly unexpected genes with none of the candidate
neuroinflammatory markers showing meaningful importance. Before
treating any of those genes as a real finding, we need to know: do
the SAME top genes show up when the model is retrained on different
subsets of the training data, or is the "top 30" list essentially
arbitrary noise specific to this one 46-sample fit?

METHOD: refit RF + SHAP separately on each of the 5 CV training
folds (using the SAME fold splits as your original cross-validation,
same seed=42), record each fold's top-20 genes, then measure overlap.
"""

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
TOP_N = 20

full_matrix = pd.read_csv("04_ML/GSE48350_hippocampus_ML_matrix_AGEADJUSTED_v2.csv", index_col=0)
train_meta  = pd.read_csv("04_ML/GSE48350_hippocampus_train_meta.csv")

full_matrix_T = full_matrix.T
X_train = full_matrix_T.loc[train_meta["GSM"]]
y_train = (train_meta["Diagnosis"] == "AD").astype(int).values
gene_names = X_train.columns.tolist()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_top_genes = []
fold_rankings = []

print("Running SHAP on each of the 5 CV folds separately...\n")

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train[train_idx]

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=RANDOM_STATE
    )
    rf.fit(X_fold_train, y_fold_train)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_fold_train)

    if isinstance(shap_values, list):
        shap_vals_ad = shap_values[1]
    elif shap_values.ndim == 3:
        shap_vals_ad = shap_values[:, :, 1]
    else:
        shap_vals_ad = shap_values

    mean_abs_shap = np.abs(shap_vals_ad).mean(axis=0)
    ranking = pd.DataFrame({
        "gene": gene_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    fold_rankings.append(ranking)
    top_genes_this_fold = set(ranking.head(TOP_N)["gene"])
    fold_top_genes.append(top_genes_this_fold)

    print(f"Fold {fold_idx + 1} (n={len(train_idx)} training samples) — "
          f"top 5 genes: {ranking.head(5)['gene'].tolist()}")

# ------------------------------------------------------------
# Stability analysis: how many folds does each gene appear in
# (within that fold's top 20)?
# ------------------------------------------------------------
from collections import Counter

gene_appearance_count = Counter()
for gene_set in fold_top_genes:
    for gene in gene_set:
        gene_appearance_count[gene] += 1

stability_df = pd.DataFrame(
    gene_appearance_count.items(), columns=["gene", "folds_appeared_in_top20"]
).sort_values("folds_appeared_in_top20", ascending=False)

print("\n" + "=" * 60)
print("STABILITY RESULTS")
print("=" * 60)

n_stable_5 = (stability_df["folds_appeared_in_top20"] == 5).sum()
n_stable_4plus = (stability_df["folds_appeared_in_top20"] >= 4).sum()
n_stable_3plus = (stability_df["folds_appeared_in_top20"] >= 3).sum()

print(f"Genes in top-{TOP_N} in ALL 5 folds:        {n_stable_5}")
print(f"Genes in top-{TOP_N} in 4+ folds:            {n_stable_4plus}")
print(f"Genes in top-{TOP_N} in 3+ folds (majority): {n_stable_3plus}")
print(f"Total distinct genes appearing in ANY fold's top-{TOP_N}: {len(stability_df)}")
print(f"  (maximum possible if every fold had a totally different top-{TOP_N}: {TOP_N * 5})")

print("\nGenes appearing in 3+ of 5 folds (candidate STABLE signature genes):")
stable_genes = stability_df[stability_df["folds_appeared_in_top20"] >= 3]
if len(stable_genes) > 0:
    print(stable_genes.to_string(index=False))
else:
    print("  NONE. No gene consistently appears across a majority of folds.")

print("\n" + "=" * 60)
print("INTERPRETATION GUIDE")
print("=" * 60)
if n_stable_3plus == 0:
    print("No genes are stable across folds. This strongly suggests the single-fit")
    print("SHAP ranking (ANKIB1, LINC02987, etc.) reflects noise/instability from")
    print("having too few samples relative to features, NOT a reproducible")
    print("biological signature. The permutation test result (real overall")
    print("discriminative signal) can still stand, but no individual gene from")
    print("this analysis should be reported as part of a 'signature' without much")
    print("stronger caveats, or without first confirming it independently on")
    print("GSE5281.")
elif n_stable_3plus < 5:
    print(f"Only {n_stable_3plus} gene(s) show cross-fold stability. Treat these as")
    print("the most credible candidates from this analysis - everything else in")
    print("the single-fit top-30 list should be treated as unstable/unreliable.")
else:
    print(f"{n_stable_3plus} genes show reasonable cross-fold stability - these are")
    print("your strongest candidates for a real signature. Proceed to check these")
    print("specific genes against the literature and GSE5281, rather than the")
    print("full original top-30 list.")

stability_df.to_csv("05_SHAP/GSE48350_hippocampus_SHAP_fold_stability.csv", index=False)
print("\nSaved: 05_SHAP/GSE48350_hippocampus_SHAP_fold_stability.csv")
