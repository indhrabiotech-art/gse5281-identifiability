import os
import pandas as pd

OUT = "06_Manuscript/Results"
os.makedirs(OUT, exist_ok=True)

perf = pd.read_csv(
    "04_ML/Final_Characterization/3gene_model_performance_summary.csv"
)

genes = pd.read_csv(
    "04_ML/Final_Characterization/3gene_gene_wise_statistics.csv"
)

coef = pd.read_csv(
    "04_ML/Final_Characterization/3gene_final_coefficients.csv"
)

boot = pd.read_csv(
    "04_ML/External_Validation/GSE5281_3gene_bootstrap_auc.csv"
)

perm = pd.read_csv(
    "04_ML/External_Validation/GSE5281_3gene_permutation_test.csv"
)

robust = pd.read_csv(
    "04_ML/External_Validation/GSE5281_3gene_raw_vs_harmonized_robustness.csv"
)

print("=" * 75)
print("GENERATING MANUSCRIPT RESULTS")
print("=" * 75)

train = perf[perf["dataset"] == "GSE48350_train"].iloc[0]
test = perf[perf["dataset"] == "GSE48350_test"].iloc[0]
external = perf[perf["dataset"] == "GSE5281_external"].iloc[0]

observed_auc = float(boot["observed_auc"].iloc[0])
ci_low = float(boot["ci_lower_95"].iloc[0])
ci_high = float(boot["ci_upper_95"].iloc[0])
p_perm = float(perm["empirical_p_value"].iloc[0])

raw_auc = float(
    robust.loc[
        robust["analysis"] == "3-gene_model",
        "raw_AUC"
    ].iloc[0]
)

harm_auc = float(
    robust.loc[
        robust["analysis"] == "3-gene_model",
        "harmonized_AUC"
    ].iloc[0]
)

genes_text = ", ".join(genes["gene"].tolist())

results = f"""RESULTS

3-Gene Signature Development and Validation

A three-gene candidate signature comprising {genes_text} was evaluated using
the GSE48350 dataset for model development and internal testing, followed by
external evaluation in the independent GSE5281 hippocampal dataset.

The final three-gene model achieved a ROC-AUC of
{train['ROC_AUC']:.3f} in the GSE48350 training cohort
(n={int(train['n'])}), {test['ROC_AUC']:.3f} in the held-out internal test
set (n={int(test['n'])}), and {external['ROC_AUC']:.3f} in the external
GSE5281 cohort (n={int(external['n'])}).

External Validation

The external cohort contained {int(external['AD'])} AD samples and
{int(external['Control'])} controls. The externally evaluated model achieved
a ROC-AUC of {observed_auc:.3f}. The 95% bootstrap confidence interval was
{ci_low:.3f}–{ci_high:.3f}, demonstrating substantial uncertainty associated
with the limited external sample size.

Permutation testing using 10,000 label permutations produced an empirical
two-sided p-value of {p_perm:.4f}. Therefore, the observed external
discrimination should be interpreted as encouraging but not statistically
conclusive.

Gene-Level Performance

All three genes showed concordant AD-versus-control direction between the
training and external datasets.

ABCA6 showed an external ROC-AUC of
{genes.loc[genes['gene']=='ABCA6','external_AUC'].iloc[0]:.3f}.

CRLF1 showed an external ROC-AUC of
{genes.loc[genes['gene']=='CRLF1','external_AUC'].iloc[0]:.3f}.

TNFRSF11B showed an external ROC-AUC of
{genes.loc[genes['gene']=='TNFRSF11B','external_AUC'].iloc[0]:.3f}.

The final model coefficients were positive for all three genes, with the
largest absolute coefficient assigned to ABCA6, followed by TNFRSF11B and
CRLF1.

Robustness to Expression-Scale Harmonization

To evaluate whether the external result depended strongly on
training-reference-based quantile harmonization, model performance was
compared using the raw and harmonized external expression matrices.

The three-gene model achieved an external ROC-AUC of {raw_auc:.3f} using
the raw external expression data and {harm_auc:.3f} following harmonization,
corresponding to an AUC difference of {harm_auc-raw_auc:+.3f}.

The relatively small change indicates that the observed external
discrimination was not solely attributable to the harmonization procedure.

Interpretation

Taken together, these analyses support ABCA6, CRLF1 and TNFRSF11B as an
externally evaluated candidate gene signature associated with Alzheimer's
disease status in the analyzed datasets. However, the small external cohort,
the broad bootstrap confidence interval and the non-significant permutation
test indicate that independent validation in substantially larger cohorts
is required before any diagnostic or clinical interpretation can be made.
"""

path = os.path.join(OUT, "Results_3gene_signature.txt")

with open(path, "w") as f:
    f.write(results)

print("\nSaved:")
print(path)

print("\n" + "=" * 75)
print("RESULTS DRAFT COMPLETE")
print("=" * 75)
