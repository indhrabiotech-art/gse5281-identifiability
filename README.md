# Alzheimer's disease transcriptomic signature transfer

Analysis code and derived results for a methodological study of cross-dataset
transfer in Alzheimer's disease transcriptomic classification.

## Summary

A three-gene logistic signature (ABCA6, CRLF1, TNFRSF11B) was developed in bulk
hippocampal tissue (GSE48350) and transferred, without refitting, to the
hippocampal subset of GSE5281. The transfer is treated as an object of study
rather than as validation.

Three findings:

1. In GSE5281, diagnosis is perfectly nested within GEO submission cohort across
   the entire 161-sample series. All 74 samples in the July 2006 submission are
   controls; all 87 in the October 2007 submission are cases. A design containing
   intercept, diagnosis and cohort is rank deficient, and a transcriptome-wide
   classifier separates the cohorts with leave-one-out ROC-AUC 1.000.
2. GSE5281 profiles laser-capture microdissected neurons with two-round
   amplification; GSE48350 profiles bulk tissue. Transfer crosses a measurement
   boundary as well as a cohort boundary.
3. Discrimination transported (ROC-AUC 0.662) while calibration did not: mean
   predicted case probability 0.003 against observed prevalence 0.435.

The signature is not offered as a biomarker.

## Data

Not redistributed here. Available from GEO:

| Accession | Role | Tissue |
|---|---|---|
| GSE48350 | Discovery | Bulk hippocampus |
| GSE5281 | Transfer test | LCM neurons, six regions |
| GSE63060 | Cross-tissue | Whole blood |
| GSE138852 | Single-cell | Entorhinal cortex |

## Pipeline order

1. `scripts/R/preprocess_raw_CEL.R` — RMA from CEL files
2. `scripts/R/annotate_RMA_genelevel.R` — probe to gene summarisation
3. `scripts/python/create_rma_train_test_split.py` — stratified 49/13 split
4. `scripts/R/training_top25_variance_filter.R` — variance filter, training only
5. `scripts/R/age_adjust_training_only.R` — per-gene age residualisation
6. `scripts/python/lasso_feature_selection.py` — L1 selection
7. `scripts/python/final_3gene_lasso_validation.py` — locked model
8. `scripts/python/validate_3gene_external.py` — transfer test
9. `scripts/R/manuscript_figures.R` — figures 2, 3, 4, S2

## Model

L1 logistic regression, liblinear, C = 0.3, seed 42. Standardisation fitted on
training data only. Coefficients: ABCA6 0.475, TNFRSF11B 0.439, CRLF1 0.380.
Decision threshold 0.3296990886390392, locked before transfer scoring.

## Important notes on provenance

**Model selection was not blind to the transfer cohort.** Script timestamps place
the first GSE5281 evaluation before the creation of the canonical train/test
split, the feature selection, and the regularisation tuning. Transfer performance
is therefore selection-conditioned and is reported as such in the manuscript.

**Two model specifications appear in this repository.** The canonical model is L1
at C = 0.3. Audit scripts under `13_Baselines` in the original tree used default
L2 and are not comparable; their outputs are excluded from `results/`.

**Superseded outputs have been removed** from `results/`. An earlier age
adjustment fitted across all 62 samples including the held-out set; this was
identified as a leak during the project and corrected. Only the corrected
lineage is included here.

## Reproducing the figures

```bash
Rscript scripts/R/manuscript_figures.R
```

Prints a check block comparing computed values against the manuscript text.

## Environment

Python 3 with scikit-learn 1.9.0, numpy 2.5.2, pandas 2.3.3, scipy 1.17.1,
statsmodels 0.14.6, xgboost 3.4.1, shap 0.52.0, gseapy 1.3.1.
R 4.3.3 with pROC and PRROC.

## Citation

[Manuscript reference to follow]

## Licence

[Choose one — MIT for code is conventional; CC BY 4.0 for data and text]
