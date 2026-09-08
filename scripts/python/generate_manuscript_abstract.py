import os
import pandas as pd

OUT = "06_Manuscript"
os.makedirs(OUT, exist_ok=True)

perf = pd.read_csv(
    "04_ML/Final_Characterization/3gene_model_performance_summary.csv"
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

train = perf[perf["dataset"] == "GSE48350_train"].iloc[0]
test = perf[perf["dataset"] == "GSE48350_test"].iloc[0]
external = perf[perf["dataset"] == "GSE5281_external"].iloc[0]

auc = float(boot["observed_auc"].iloc[0])
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

abstract = f"""
ABSTRACT

Background:
Alzheimer's disease is characterized by complex molecular alterations
involving multiple biological processes. Compact transcriptomic signatures
that retain cross-dataset discrimination may provide useful candidates for
further biological and diagnostic investigation.

Objective:
This study aimed to identify and externally evaluate a compact
gene-expression signature associated with Alzheimer's disease using
publicly available microarray datasets.

Methods:
GSE48350 was used for model development and held-out internal validation,
while GSE5281 was used as an independent external dataset. Raw CEL files
were subjected to RMA preprocessing followed by probe-to-gene annotation,
gene-level expression construction, age adjustment, variance-based feature
filtering and LASSO-based feature selection. The final candidate signature
was evaluated using ROC-AUC, bootstrap analysis and permutation testing.
Robustness to training-reference-based expression harmonization was also
examined. Functional enrichment and STRING first-shell network analyses
were performed to characterize the biological context of the selected genes.

Results:
The final signature comprised ABCA6, CRLF1 and TNFRSF11B. The three-gene
model achieved a ROC-AUC of {train['ROC_AUC']:.3f} in the GSE48350 training
cohort, {test['ROC_AUC']:.3f} in the held-out internal test cohort and
{external['ROC_AUC']:.3f} in the independent GSE5281 cohort. The external
cohort contained {int(external['AD'])} AD samples and
{int(external['Control'])} controls. Bootstrap analysis produced a 95%
confidence interval of {ci_low:.3f}–{ci_high:.3f}. Permutation testing
using 10,000 permutations produced an empirical two-sided p-value of
{p_perm:.4f}. The model achieved ROC-AUC values of {raw_auc:.3f} before and
{harm_auc:.3f} after expression-scale harmonization, indicating relatively
limited dependence on the harmonization procedure. Biological analyses
linked the three genes to lipid/transport, cytokine/neurotrophic and
TNF/TNFR-associated signaling modules.

Conclusion:
ABCA6, CRLF1 and TNFRSF11B represent an externally evaluated candidate
three-gene Alzheimer's disease signature with moderate cross-dataset
discrimination. However, the small external cohort, broad bootstrap
confidence interval and non-significant permutation test indicate that the
signature remains preliminary and requires validation in larger independent
cohorts before diagnostic or clinical application.
""".strip()

path = os.path.join(
    OUT,
    "Abstract_3gene_signature.txt"
)

with open(path, "w") as f:
    f.write(abstract + "\n")

print("=" * 75)
print("MANUSCRIPT ABSTRACT GENERATED")
print("=" * 75)
print()
print("Saved:")
print(path)
print()
print("=" * 75)
