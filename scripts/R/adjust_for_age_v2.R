## ============================================================
## CORRECTED age-adjustment: fit the age effect on TRAINING DATA
## ONLY, then apply those same coefficients to both train and test.
##
## The previous version (adjust_for_age.R) fit removeBatchEffect()
## on all 62 samples, including the 16 locked test samples - that
## leaks test-set information into the "training" adjustment,
## violating the same principle Rule 1/2 protect for feature
## selection. This version fixes that.
## ============================================================

library(limma)

setwd("~/project_ml")

full_matrix <- read.csv("04_ML/GSE48350_hippocampus_ML_matrix_full.csv",
                         row.names = 1, check.names = FALSE)
train_meta <- read.csv("04_ML/GSE48350_hippocampus_train_meta.csv", stringsAsFactors = FALSE)
test_meta  <- read.csv("04_ML/GSE48350_hippocampus_test_meta.csv", stringsAsFactors = FALSE)

full_matrix <- as.matrix(full_matrix)
train_exprs <- full_matrix[, train_meta$GSM]
test_exprs  <- full_matrix[, test_meta$GSM]

stopifnot(identical(colnames(train_exprs), train_meta$GSM))
stopifnot(identical(colnames(test_exprs), test_meta$GSM))

## ------------------------------------------------------------
## Fit per-gene age effect using TRAINING samples only
## ------------------------------------------------------------
design_train <- model.matrix(~ Age, data = data.frame(Age = train_meta$Age))
fit <- lmFit(train_exprs, design_train)
beta_age <- fit$coefficients[, "Age"]   # one slope per gene, from TRAIN ONLY

train_mean_age <- mean(train_meta$Age)  # also derived from TRAIN ONLY

## ------------------------------------------------------------
## Apply the SAME train-derived slope/mean to both sets
## ------------------------------------------------------------
train_age_centered <- train_meta$Age - train_mean_age
test_age_centered  <- test_meta$Age  - train_mean_age   # note: train mean, not test mean

train_exprs_adj <- train_exprs - outer(beta_age, train_age_centered)
test_exprs_adj  <- test_exprs  - outer(beta_age, test_age_centered)

cat("Train adjusted matrix:", paste(dim(train_exprs_adj), collapse = " x "), "\n")
cat("Test adjusted matrix: ", paste(dim(test_exprs_adj), collapse = " x "), "\n")

## ------------------------------------------------------------
## Recombine in original sample order and save (for baseline_ml.py
## to consume exactly as before, no changes needed to that script
## other than the filename)
## ------------------------------------------------------------
combined_adj <- cbind(train_exprs_adj, test_exprs_adj)
combined_adj <- combined_adj[, colnames(full_matrix)]  # restore original order

write.csv(combined_adj, "04_ML/GSE48350_hippocampus_ML_matrix_AGEADJUSTED_v2.csv")

cat("\nSaved: 04_ML/GSE48350_hippocampus_ML_matrix_AGEADJUSTED_v2.csv\n")
cat("This version fits the age effect on TRAINING DATA ONLY - no test-set\n",
    "information was used to compute the adjustment. Point baseline_ml.py\n",
    "at this file (not the v1 AGEADJUSTED file, which had the leak).\n")
