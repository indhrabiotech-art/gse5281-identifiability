import os

OUT = "06_Manuscript/Methods"
os.makedirs(OUT, exist_ok=True)

methods = r"""
METHODS

Study Design and Overall Computational Workflow

A computational transcriptomic workflow was developed to identify and
evaluate a compact gene-expression signature associated with Alzheimer's
disease (AD). GSE48350 was used as the primary model-development dataset,
while GSE5281 was treated as an independent external validation dataset.
The workflow consisted of raw microarray preprocessing, RMA normalization,
probe-to-gene annotation, gene-level expression construction, quality
control, training/test partitioning, age adjustment, variance-based feature
filtering, machine-learning feature selection, internal validation,
independent external validation, expression-scale robustness analysis, and
biological characterization.

Microarray Data and RMA Preprocessing

Raw CEL files from GSE48350 and GSE5281 were processed using the Affymetrix
Human Genome U133 Plus 2.0 platform annotation. Raw CEL files were imported
and subjected to Robust Multi-array Average (RMA) preprocessing. RMA
processing generated log2-scale expression values and produced expression
matrices for downstream analysis.

Probe-to-Gene Annotation

RMA probe-level expression data were annotated using the
hgu133plus2.db Bioconductor annotation resource. Probes without an assigned
gene symbol were removed. Where multiple probes mapped to the same gene,
the probe with the highest mean RMA expression across samples was retained
to generate a single representative expression profile per gene.

The resulting gene-level matrices contained 21,367 annotated genes for both
datasets.

Training/Test Partitioning

The GSE48350 dataset was divided into a model-development training cohort
and an independent held-out internal test cohort. The final recorded
training cohort contained 49 samples, comprising 15 AD and 34 control
samples. The held-out test cohort contained 13 samples, comprising 4 AD and
9 control samples.

Feature Processing and Age Adjustment

Age adjustment was performed using training-derived information to avoid
using held-out test information during model development. Variance-based
feature filtering was subsequently applied within the training framework
to reduce the dimensionality of the transcriptomic feature space.

Machine-Learning Feature Selection

Regularized logistic regression/LASSO-based feature selection was used to
identify candidate genes associated with AD status. Candidate signatures
were evaluated using receiver operating characteristic area under the curve
(ROC-AUC) and related classification metrics.

Candidate Signature

Following model reduction and external robustness assessment, the final
candidate signature consisted of three genes:

ABCA6
CRLF1
TNFRSF11B

The final model coefficients were positive for all three genes.

Internal Validation

The held-out GSE48350 test cohort was not used during feature-selection
training and was evaluated independently. Model discrimination was assessed
using ROC-AUC together with classification performance metrics.

External Validation

GSE5281 was treated as an independent external dataset. The external cohort
contained 23 samples, including 10 AD samples and 13 control samples.

External expression data were evaluated both before and after
training-reference-based quantile harmonization. No external diagnostic
labels were used to calculate the expression transformation itself.

Expression-Scale Robustness

To determine whether external model performance was dependent on
harmonization, the three-gene model was evaluated using both raw and
harmonized GSE5281 expression values. The resulting ROC-AUC values were
compared directly.

Bootstrap Analysis

Uncertainty in external ROC-AUC was evaluated using 10,000 bootstrap
resamples. A 95% bootstrap confidence interval was calculated from the
resulting empirical AUC distribution.

Permutation Testing

Statistical evidence for external discrimination was evaluated using
10,000 permutations of the external diagnostic labels. The empirical
two-sided permutation p-value was calculated by comparing the observed AUC
with the null distribution generated from the permuted labels.

Functional Enrichment Analysis

Functional characterization of the final three-gene signature was performed
using gseapy/Enrichr-based enrichment analysis. Gene Ontology Biological
Process, Gene Ontology Molecular Function, KEGG and Reactome resources were
examined.

Protein-Interaction Network Analysis

STRING was queried to characterize functional network relationships
associated with the three signature genes. A first-shell network was also
constructed to identify neighboring proteins and functional associations
around the signature genes.

Statistical Interpretation

ROC-AUC was used as the primary measure of discrimination. Bootstrap
confidence intervals were used to quantify uncertainty, while permutation
testing was used to assess whether the observed external discrimination was
consistent with a label-randomized null distribution.

Because the external validation cohort was small and the permutation test
did not establish conventional statistical significance, the resulting
three-gene panel was considered an externally evaluated candidate gene
signature rather than a clinically validated diagnostic biomarker.
"""

path = os.path.join(OUT, "Methods_3gene_signature.txt")

with open(path, "w") as f:
    f.write(methods.strip() + "\n")

print("=" * 75)
print("MANUSCRIPT METHODS GENERATED")
print("=" * 75)
print()
print("Saved:")
print(path)
print()
print("=" * 75)
