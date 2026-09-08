from pathlib import Path

methods_file = Path(
    "06_Manuscript/Methods/Methods_3gene_signature.txt"
)

text = methods_file.read_text()

old = """External Dataset Harmonization

To reduce cross-dataset expression-scale differences, the GSE5281 external
dataset was harmonized to the GSE48350 training reference using a
training-reference-based quantile harmonization procedure. External
diagnostic labels were not used during the harmonization transformation.
"""

new = """External Dataset Harmonization

To reduce cross-dataset expression-scale differences, gene-wise empirical
quantile mapping was applied to the GSE5281 external expression matrix using
the corresponding GSE48350 training-gene distributions as reference
distributions. For each gene, GSE5281 sample ranks were converted to empirical
percentile positions and mapped to the corresponding quantiles of the
GSE48350 reference distribution using linear interpolation. The mapping was
performed independently for each common gene and did not use external
diagnostic labels.

As a quality-control assessment, the harmonized GSE5281 matrix contained
21,367 genes across 23 samples with no missing values. For the three final
signature genes, harmonized expression means closely matched the GSE48350
reference means, with mean differences of -0.0097 for ABCA6, -0.0069 for
CRLF1 and -0.0013 for TNFRSF11B. Spearman correlations between raw and
harmonized external expression ranks were 0.9990, 0.9998 and 0.9975,
respectively, indicating that the transformation primarily altered
expression scale while preserving sample-wise rank ordering.

As an additional robustness assessment, the three-gene model was evaluated
using both the raw and harmonized external expression matrices. External
ROC-AUC changed from 0.662 using the raw matrix to 0.685 after harmonization,
corresponding to an AUC difference of 0.023. This analysis was used to assess
whether the external classification signal was strongly dependent on the
expression-scale harmonization procedure.
"""

if old in text:
    text = text.replace(old, new)
else:
    marker = "\n\nBiological Characterization"
    if marker in text:
        text = text.replace(marker, "\n\n" + new + marker)
    else:
        text += "\n\n" + new

methods_file.write_text(text)

print("=" * 75)
print("MANUSCRIPT HARMONIZATION METHODS UPDATED")
print("=" * 75)

print("\nUpdated:")
print(methods_file)

print("\nNew harmonization QC included:")
print(" - 21,367 genes")
print(" - 23 external samples")
print(" - 0 missing values")
print(" - rank preservation")
print(" - raw vs harmonized AUC robustness")

print("\n" + "=" * 75)
