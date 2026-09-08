import os

OUT = "06_Manuscript/Discussion"
os.makedirs(OUT, exist_ok=True)

discussion = r"""
DISCUSSION

Principal Findings

This study developed and externally evaluated a compact three-gene
transcriptomic signature consisting of ABCA6, CRLF1 and TNFRSF11B for
Alzheimer's disease classification. The final model demonstrated progressive
performance from the training cohort to the held-out internal test cohort
and maintained discrimination in the independent GSE5281 external cohort.

The external ROC-AUC of approximately 0.685 is particularly relevant because
the three-gene model substantially improved upon the previously evaluated
seven-gene model, which showed poor external discrimination. The reduction
from seven genes to three therefore suggests that a smaller signature may
provide better cross-dataset robustness in this analysis.

Gene-Level Biological Interpretation

ABCA6 was the strongest component of the final model based on its absolute
model coefficient. Its expression showed concordant AD-associated direction
in the training, internal test and external datasets. Functional enrichment
and STRING first-shell analysis connected ABCA6 with ABC transporter and
lipid-transport-associated biology, including relationships with ABCA8,
ABCA9 and MAP2K6.

CRLF1 showed the highest first-shell network degree among the three
signature genes. The enrichment results associated CRLF1 with cytokine and
neurotrophic signaling, including CLCF1, CNTFR and IL27-related biology.
The GO and Reactome results further highlighted cytokine-mediated and
interleukin-associated signaling processes.

TNFRSF11B showed consistent AD-associated direction across datasets and
contributed substantially to the final model coefficient. Its functional
analysis connected it with TNF/TNFR signaling, osteoclast-related pathways
and immune-system-associated processes. STRING analysis identified
TNFRSF11A and MAP2K6 among its first-shell network neighbors.

Together, these observations indicate that the three-gene signature spans
multiple biological modules rather than representing a single canonical
pathway. The observed modules include lipid/transport-associated biology,
cytokine/neurotrophic signaling and TNF/TNFR-associated immune signaling.

Cross-Dataset Robustness

An important feature of the analysis was explicit evaluation of
cross-dataset robustness. The external three-gene model achieved an
ROC-AUC of approximately 0.662 using the raw GSE5281 expression matrix and
approximately 0.685 following training-reference-based harmonization. The
difference was small, indicating that the observed discrimination was not
created solely by the harmonization procedure.

All three genes also retained the same AD-versus-control direction between
the training and external datasets. This concordance provides additional
support for the stability of the candidate signature at the directionality
level.

Statistical Uncertainty

Despite encouraging external discrimination, the statistical evidence must
be interpreted cautiously. The external cohort consisted of only 23 samples,
with 10 AD and 13 control samples. The bootstrap 95% confidence interval for
external AUC was broad, approximately 0.442–0.886, demonstrating substantial
sampling uncertainty.

Furthermore, 10,000-label permutation testing produced an empirical
two-sided p-value of approximately 0.145. Therefore, the external result
does not provide statistically conclusive evidence of discrimination at the
conventional 0.05 threshold.

The appropriate interpretation is consequently that the three-gene panel
represents an externally evaluated candidate signature rather than a
clinically validated diagnostic biomarker.

Biological Integration

The biological analyses provide a mechanistic context for the statistical
signature. The enrichment results repeatedly highlighted cytokine signaling,
interleukin-associated processes, lipid transport and immune-related
pathways. The STRING first-shell network further demonstrated that individual
signature genes connect to biologically related neighboring proteins.

However, enrichment based on only three genes is intrinsically limited.
Individual genes can generate highly significant enrichment statistics when
they belong to small annotation categories, and these results should
therefore be considered hypothesis-generating rather than definitive
evidence of pathway activation.

Strengths

Several aspects strengthen the present analysis. First, the workflow used
independent internal and external evaluation rather than relying exclusively
on training performance. Second, probe-level expression was systematically
collapsed to gene-level expression using a predefined representative-probe
strategy. Third, external performance was evaluated using both raw and
harmonized expression data. Fourth, bootstrap and permutation analyses were
performed to explicitly quantify uncertainty. Finally, the computational
signature was complemented by functional enrichment and protein-interaction
network analyses.

Limitations

The principal limitation is the small external validation cohort. With only
23 samples, estimates of discrimination and classification performance are
subject to substantial sampling variability.

The permutation analysis also failed to establish conventional statistical
significance. Accordingly, the present study should not be interpreted as
establishing clinical diagnostic utility.

Additional limitations include potential biological and technical
heterogeneity between datasets, differences in cohort composition and the
limited size of the final signature. Although harmonization robustness was
examined, the analysis cannot completely eliminate all forms of
cross-platform or cross-cohort heterogeneity.

Future Validation

The next stage should involve independent validation using substantially
larger cohorts with standardized clinical phenotyping. External validation
across additional brain regions and, where biologically appropriate,
peripheral tissues would help determine whether the signature reflects
general AD-associated biology or tissue-specific transcriptomic effects.

Future studies should also evaluate calibration, sensitivity, specificity,
positive and negative predictive values, decision thresholds and clinical
covariates in larger cohorts. Independent experimental validation at the
RNA and/or protein level would provide an additional layer of evidence.

Conclusion

The present analysis identifies ABCA6, CRLF1 and TNFRSF11B as a compact
three-gene candidate signature that retained moderate discrimination in an
independent GSE5281 cohort. The signature integrates lipid/transport,
cytokine/neurotrophic and TNF/TNFR-associated biological signals.

The external result is encouraging but remains preliminary because of the
small validation cohort, broad bootstrap uncertainty and non-significant
permutation test. Therefore, the three genes should currently be regarded
as an externally evaluated candidate Alzheimer's disease gene signature
requiring validation in larger independent cohorts.
"""

path = os.path.join(OUT, "Discussion_3gene_signature.txt")

with open(path, "w") as f:
    f.write(discussion.strip() + "\n")

print("=" * 75)
print("MANUSCRIPT DISCUSSION GENERATED")
print("=" * 75)
print()
print("Saved:")
print(path)
print()
print("=" * 75)
