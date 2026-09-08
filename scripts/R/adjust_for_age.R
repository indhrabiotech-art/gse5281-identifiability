## ============================================================
## Age-adjustment: remove the linear effect of Age from expression
## values before re-running ML, to isolate AD-specific signal from
## age-driven expression changes (confound confirmed: age-only AUC
## = 0.798, close to the gene-model AUCs of 0.870/0.938).
## ============================================================

library(limma)

setwd("~/project_ml")

gene_exprs_filtered <- read.csv("04_ML/GSE48350_hippocampus_ML_matrix_full.csv",
                                 row.names = 1, check.names = FALSE)
meta <- read.csv("03_Preprocessing/GSE48350_hippocampus_metadata_aligned.csv",
                  stringsAsFactors = FALSE)

stopifnot(identical(colnames(gene_exprs_filtered), meta$GSM))

## ------------------------------------------------------------
## Regress out Age using limma::removeBatchEffect with a continuous
## covariate. This fits, per gene, expression ~ Age and subtracts
## the fitted age effect - leaving residual variation not explained
## by age (which may include AD-specific signal, other biology, or
## noise, but NOT the linear age trend itself).
## ------------------------------------------------------------
## IMPORTANT: this does NOT remove Diagnosis-related variation -
## only Age. Diagnosis is intentionally left untouched since that's
## the thing you want the model to learn to predict.

gene_exprs_matrix <- as.matrix(gene_exprs_filtered)

age_adjusted <- removeBatchEffect(
  gene_exprs_matrix,
  covariates = meta$Age
)

cat("Age-adjusted matrix dimensions:", paste(dim(age_adjusted), collapse = " x "), "\n")

## ------------------------------------------------------------
## Rebuild the SAME frozen train/test split (same GSM assignments -
## do not re-randomize) using the age-adjusted values
## ------------------------------------------------------------
train_meta <- read.csv("04_ML/GSE48350_hippocampus_train_meta.csv", stringsAsFactors = FALSE)
test_meta  <- read.csv("04_ML/GSE48350_hippocampus_test_meta.csv", stringsAsFactors = FALSE)

train_exprs_adj <- age_adjusted[, train_meta$GSM]
test_exprs_adj  <- age_adjusted[, test_meta$GSM]

stopifnot(identical(colnames(train_exprs_adj), train_meta$GSM))
stopifnot(identical(colnames(test_exprs_adj), test_meta$GSM))

write.csv(age_adjusted, "04_ML/GSE48350_hippocampus_ML_matrix_AGEADJUSTED.csv")

cat("\nSaved: 04_ML/GSE48350_hippocampus_ML_matrix_AGEADJUSTED.csv\n")
cat("Train/test GSM assignments are UNCHANGED from the original frozen split -\n",
    "only the expression values are age-adjusted. Re-run baseline_ml.py against\n",
    "this file next to see whether the gene-expression signal holds up once\n",
    "the age effect is removed.\n")
