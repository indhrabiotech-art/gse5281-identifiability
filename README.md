# Diagnosis is confounded with submission cohort throughout GSE5281

Analysis code, derived results, and data-quality findings for a methodological
study of cross-dataset transfer in Alzheimer's disease transcriptomic
classification.

Archived at Zenodo: https://doi.org/10.5281/zenodo.22663479

## Summary

GSE5281 is a widely reused Alzheimer's disease expression resource. Every
control in the series was deposited in one GEO submission and every case in
another, across all six profiled brain regions. The separation is not merely
administrative: the two groups were scanned in different years and prepared
under different laboratory protocols, and the difference is visible in
quantities that disease cannot affect.

The consequence is that no case-control contrast in the dataset is separately
identifiable from submission cohort, and batch correction cannot recover it.

The same audit applied to GSE48350, a dataset frequently merged with GSE5281,
returns the opposite verdict. That contrast is what makes the audit a
diagnostic rather than a blanket warning.

## 1. Diagnosis is perfectly nested within submission cohort

All 74 samples in the July 2006 submission are controls; all 87 in the
October 2007 submission are cases. A design matrix containing intercept,
diagnosis and submission cohort is rank deficient (23 x 3, rank 2 in the
hippocampal subset).

## 2. The separation is technical, not administrative

| Measure | Added or determined | AD | Control | Cohort ROC-AUC |
|---|---|---|---|---|
| Poly-A spike-ins | to RNA, before reverse transcription | 8.364 | 2.723 | **1.000** |
| Hybridisation spike-ins | to cocktail, after labelling | 10.163 | 10.064 | 0.569 |
| GAPDH 3'/5' ratio | RNA integrity | 5.651 | 3.761 | 0.938 |
| ACTB 3'/5' ratio | RNA integrity | 6.634 | 7.954 | 0.992 |
| RNA degradation slope | RNA integrity | 10.690 | 8.456 | - |
| RLE median | array quality after fitting | 0.008 | -0.009 | n.s. |
| NUSE median | array quality after fitting | 1.003 | 1.002 | n.s. |

The dissociation is the point. Poly-A controls are pipetted into the RNA
sample at fixed concentration before reverse transcription and separate the
cohorts completely; hybridisation controls, added after labelling, do not.
RLE and NUSE show that the case arrays are not simply of poorer quality. The
difference therefore lies in RNA preparation and labelling, consistent with
the double-round amplification documented for the case submission and absent
from the control record.

CEL scan dates agree: none of the five GSE5281 scan dates carries samples of
both classes. Every control was scanned in 2004, every case in 2005 or 2006.

## 3. Batch correction cannot recover the contrast

Differentially expressed genes before ComBat (FDR < 0.05): 5,546.
Supplied with diagnosis as a covariate, ComBat halts with "The covariate is
confounded with batch". Without diagnosis it runs, and the count falls to 0.

The two options available to an analyst are a procedure that will not run and
a procedure that removes the effect of interest.

## 4. The discovery cohort is identifiable

GSE48350 also places cases and controls in disjoint accession blocks, so on
that evidence it would be judged confounded. Its arrays, however, were scanned
across 16 dates with both classes present on 9 of them, and the corresponding
design is of full rank (62 x 17, rank 17).

Accession structure alone therefore does not distinguish the two datasets.
Only the instrument record does.

## 5. Data-quality findings in GSE48350

A control array was re-deposited as a case. GSM1176215 and GSM300182 are
bit-identical: same md5 of the decompressed CEL, same file size, same scan
timestamp (12/05/05 14:24:37), Spearman 1.000000 across 21,367 genes. One is
recorded as male, 80, control; the other as female, 86, AD. Both are listed as
individual 15, and the duplication spans all four regions profiled for that
individual.

Both copies fall in the 49-sample training set with opposite labels. Removing
the mislabelled copy raises apparent AUC from 0.853 to 0.914 and five-fold
cross-validated AUC from 0.812 to 0.861.

Six samples have recorded sex contradicted by expression. In all six, XIST and
the Y-linked genes agree with each other and contradict the record. Five of
the six lie in the case submission block: 5 of 19 against 1 of 43 (Fisher
exact p = 0.0086). The local metadata matches the GEO record exactly, so the
discrepancy is in the deposited record. One of the six is explained by the
duplicate above; the other five are not.

## 6. What this means for a transferred signature

A three-gene logistic signature (ABCA6, CRLF1, TNFRSF11B) developed in
GSE48350 and applied unchanged to GSE5281 retained weak rank ordering while
its probability scale collapsed.

Transfer ROC-AUC 0.662 (permutation p = 0.206). Random three-gene panels
equalled or exceeded it in 50.6% of 1,000 draws. Mean predicted case
probability 0.003 against an observed prevalence of 0.435.

Locked preprocessing removes roughly 56% of the calibration discrepancy and
leaves the rest, while discrimination is unchanged across schemes (0.654,
0.662, 0.669). Rank ordering survives monotone rescaling; the probability
scale does not.

The signature is not offered as a biomarker. It is the vehicle for the
argument, and its ordinariness is what makes it useful.

## Reproducing the findings

Data are not redistributed. Raw CEL files and series matrices are available
from GEO under GSE48350, GSE5281, GSE63060 and GSE138852.

    Rscript scripts/R/audit_GSE48350_identifiability.R
    Rscript scripts/R/technical_qc_GSE5281.R
    Rscript scripts/R/data_quality_audit.R
    Rscript scripts/R/critique_runs.R
    Rscript scripts/R/frma_sensitivity.R
    Rscript scripts/R/figure_scan_timeline.R
    Rscript scripts/R/figure_spikein_panel.R
    python3 scripts/python/random_signature_null.py

Each prints a check block comparing computed values against those reported.

## Repository layout

    scripts/R/        preprocessing, annotation, age adjustment, audits, figures
    scripts/python/   feature selection, model fitting, transfer, nulls
    results/          derived tables, logs, per-sample predictions
    data_quality/     duplicate arrays, cross-block individuals, sensitivity
    figures/          submission figures, PDF and PNG
    manuscript/       manuscript text and tables

## Provenance

Model selection was not blind to the transfer cohort. Script timestamps place
the first GSE5281 evaluation before the creation of the train/test split, the
feature selection and the regularisation tuning. Transfer performance is
therefore selection-conditioned and is reported as such. The identifiability
findings do not depend on the modelling and would hold had no classifier been
built.

Superseded outputs are excluded. An earlier age adjustment fitted across all
62 samples including the held-out set; this was identified as a leak during
the project and corrected. An unadjusted matrix that produced a higher
apparent AUC has been renamed to make its status explicit. Only the corrected
lineage is deposited.

Nested cross-validation is leak-free. The variance filter and age adjustment
are re-estimated within each outer training fold.

## Environment

Python 3.12 with scikit-learn 1.9.0, numpy 2.5.2, pandas 2.3.3, scipy 1.17.1,
statsmodels 0.14.6, gseapy 1.3.1.

R 4.3.3 with affy 1.80.0, Biobase 2.62.0, AnnotationDbi 1.64.1,
hgu133plus2.db, sva, limma, frma, affyPLM, pROC.

Random seed 42 throughout.

## Citation

Manuscript in preparation. Please cite the Zenodo archive in the meantime:
https://doi.org/10.5281/zenodo.22663479

## Licence

Add a LICENSE file. MIT is conventional for research code.
