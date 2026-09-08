"""
Day 9: SHAP analysis on the age-adjusted Random Forest model.

Fits the FINAL model on the full training set (age-adjusted,
46 samples), then uses SHAP to identify which genes drive its
predictions and in which direction.

WHY the training set, not train+test: SHAP explains what the model
learned during training. The held-out test set was already used
once for final evaluation (per Rule 1/2) - using it again here
would not violate leakage rules (SHAP doesn't retrain), but keeping
the explanation scoped to the training fit keeps the analysis
consistent with "what did the model learn" rather than mixing in
data reserved for evaluation.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # no display available when running from console
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

# ------------------------------------------------------------
# Load age-adjusted data (same files used for the permutation test)
# ------------------------------------------------------------
full_matrix = pd.read_csv("04_ML/GSE48350_hippocampus_ML_matrix_AGEADJUSTED_v2.csv", index_col=0)
train_meta  = pd.read_csv("04_ML/GSE48350_hippocampus_train_meta.csv")

full_matrix_T = full_matrix.T
X_train = full_matrix_T.loc[train_meta["GSM"]]
y_train = (train_meta["Diagnosis"] == "AD").astype(int).values

gene_names = X_train.columns.tolist()

print("Training matrix shape:", X_train.shape)

# ------------------------------------------------------------
# Fit the final model (same hyperparameters as baseline_ml.py,
# for consistency with the CV/permutation results you already have)
# ------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)

# ------------------------------------------------------------
# SHAP explanation
# ------------------------------------------------------------
print("\nComputing SHAP values (TreeExplainer - fast for tree models)...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)

# shap_values shape handling: newer SHAP versions return a single
# array for binary classification (positive class), older versions
# return a list [class0_values, class1_values]
if isinstance(shap_values, list):
    shap_vals_ad = shap_values[1]  # class 1 = AD
elif shap_values.ndim == 3:
    shap_vals_ad = shap_values[:, :, 1]  # class 1 = AD (current SHAP API)
else:
    shap_vals_ad = shap_values

print("SHAP values computed. Shape:", np.array(shap_vals_ad).shape)

# ------------------------------------------------------------
# Global feature importance: mean absolute SHAP value per gene
# ------------------------------------------------------------
mean_abs_shap = np.abs(shap_vals_ad).mean(axis=0)

importance_df = pd.DataFrame({
    "gene": gene_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\n" + "=" * 60)
print("TOP 30 GENES BY SHAP IMPORTANCE")
print("=" * 60)
print(importance_df.head(30).to_string(index=False))

# ------------------------------------------------------------
# Cross-reference against your neuroinflammatory marker list
# (Section 3 of your master log) - informational only, does NOT
# change the ranking (Rule 6: no forcing genes in/out post-hoc)
# ------------------------------------------------------------
neuroinflam_genes = ["TREM2", "TYROBP", "NLRP3", "IL1B", "IL6", "TNF",
                      "C1QA", "C1QB", "C1QC", "C3", "CD68"]
top_50_genes = set(importance_df.head(50)["gene"])
overlap = [g for g in neuroinflam_genes if g in top_50_genes]

print(f"\nOf your {len(neuroinflam_genes)} candidate neuroinflammatory markers, "
      f"{len(overlap)} appear in the top 50 SHAP genes:")
print(overlap if overlap else "  (none in top 50 - check further down the ranking)")

# Show rank of each neuroinflammatory gene, even if not in top 50
print("\nRank of each candidate neuroinflammatory gene (out of",
      len(importance_df), "genes in the filtered matrix):")
importance_df_reset = importance_df.reset_index(drop=True)
for gene in neuroinflam_genes:
    match = importance_df_reset[importance_df_reset["gene"] == gene]
    if len(match) > 0:
        rank = match.index[0] + 1
        print(f"  {gene}: rank {rank}, mean|SHAP| = {match['mean_abs_shap'].values[0]:.4f}")
    else:
        print(f"  {gene}: not in filtered matrix (dropped at variance-filtering step)")

# ------------------------------------------------------------
# Save figures
# ------------------------------------------------------------
import os
os.makedirs("07_Figures", exist_ok=True)

plt.figure()
shap.summary_plot(shap_vals_ad, X_train, show=False, max_display=20)
plt.tight_layout()
plt.savefig("07_Figures/shap_beeswarm_top20.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved SHAP beeswarm plot: 07_Figures/shap_beeswarm_top20.png")

plt.figure()
shap.summary_plot(shap_vals_ad, X_train, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig("07_Figures/shap_bar_top20.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved SHAP bar plot: 07_Figures/shap_bar_top20.png")

# ------------------------------------------------------------
# Save the full ranked gene list
# ------------------------------------------------------------
os.makedirs("05_SHAP", exist_ok=True)
importance_df.to_csv("05_SHAP/GSE48350_hippocampus_SHAP_gene_ranking.csv", index=False)
print("\nSaved full ranked gene list: 05_SHAP/GSE48350_hippocampus_SHAP_gene_ranking.csv")

print("\nNext: cross-validate SHAP ranking stability across CV folds (are the")
print("top genes consistent, or do they change fold-to-fold?), then move to")
print("Phase 16 biological interpretation and STRING/PPI analysis (Phase 17)")
print("for the top-ranked genes.")
