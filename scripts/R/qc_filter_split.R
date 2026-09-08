## ============================================================
## STEP 13-15: Expression QC (PCA/clustering), feature filtering,
## and frozen stratified train/test split
## ============================================================

setwd("~/project_ml")

gene_exprs <- readRDS("03_Preprocessing/GSE48350_hippocampus_genelevel_log2.rds")
meta       <- read.csv("03_Preprocessing/GSE48350_hippocampus_metadata_aligned.csv",
                        stringsAsFactors = FALSE)

stopifnot(identical(colnames(gene_exprs), meta$GSM))

## ------------------------------------------------------------
## STEP 13: PCA + hierarchical clustering QC
## ------------------------------------------------------------
## WHY: the boxplot only checks per-sample distribution. PCA/
## clustering checks whether samples separate in ways that make
## biological sense (AD vs Control) or in ways that suggest a
## technical problem (e.g. one sample wildly off from all others).
## Per Section 25 of your log: do NOT "correct away" any AD/Control
## separation you see here - that's the biological signal you want.

dir.create("07_Figures", showWarnings = FALSE)

pca <- prcomp(t(gene_exprs), scale. = TRUE)
var_explained <- round(100 * pca$sdev^2 / sum(pca$sdev^2), 1)

png("07_Figures/hippocampus_pca.png", width = 900, height = 700)
plot(pca$x[, 1], pca$x[, 2],
     col = ifelse(meta$Diagnosis == "AD", "tomato", "steelblue"),
     pch = 19, cex = 1.3,
     xlab = paste0("PC1 (", var_explained[1], "%)"),
     ylab = paste0("PC2 (", var_explained[2], "%)"),
     main = "PCA - GSE48350 hippocampus (red=AD, blue=Control)")
legend("topright", legend = c("AD", "Control"),
       col = c("tomato", "steelblue"), pch = 19)
dev.off()
cat("Saved PCA plot to 07_Figures/hippocampus_pca.png\n")
cat("Check: do AD (red) and Control (blue) show ANY separation along\n",
    "PC1/PC2? Partial overlap is normal and expected - complete random\n",
    "mixing with zero AD/Control structure anywhere in early PCs would\n",
    "be a warning sign (though not necessarily fatal - ML models can\n",
    "still find non-linear/higher-dimensional structure PCA misses).\n")

sample_cor <- cor(gene_exprs)
png("07_Figures/hippocampus_sample_correlation_heatmap.png", width = 900, height = 800)
heatmap(sample_cor, main = "Sample-sample correlation - hippocampus",
        labRow = meta$Diagnosis, labCol = FALSE)
dev.off()
cat("Saved sample correlation heatmap to 07_Figures/hippocampus_sample_correlation_heatmap.png\n")

## Flag any sample whose average correlation to all others is unusually low
avg_cor <- rowMeans(sample_cor) 
low_cor_samples <- meta$GSM[avg_cor < (mean(avg_cor) - 3 * sd(avg_cor))]
if (length(low_cor_samples) > 0) {
  cat("\nWARNING: possible outlier sample(s) with unusually low average\n",
      "correlation to the rest of the cohort:\n")
  print(low_cor_samples)
  cat("Investigate before proceeding - do not silently drop without reason.\n")
} else {
  cat("\nNo samples flagged as correlation outliers.\n")
}

## ------------------------------------------------------------
## STEP 7 (feature filtering, part of PHASE 7): remove low-variance
## genes before feature selection
## ------------------------------------------------------------
## WHY: with 21,367 genes and only 62 samples (p >> n, Section
## "IMPORTANT SCIENTIFIC RISKS TO CONTROL" item 5), most genes carry
## no discriminative signal and only add noise/overfitting risk.
## Filtering on variance (not diagnosis-relatedness) avoids leaking
## label information into feature selection at this stage.

gene_var <- apply(gene_exprs, 1, var)
cat("\nGene variance summary (log2 scale):\n")
print(summary(gene_var))

## Keep top 25% most variable genes as a defensible, reproducible cutoff
var_cutoff <- quantile(gene_var, 0.75)
keep_genes <- names(gene_var[gene_var >= var_cutoff])

gene_exprs_filtered <- gene_exprs[keep_genes, ]
cat("\nGenes retained after variance filtering (top 25%):", nrow(gene_exprs_filtered), "\n")

## Confirm neuroinflammatory markers survived filtering (informational only -
## do NOT add them back in if filtered out; that would reintroduce the
## biological-circularity risk your log explicitly warns against in Rule 6)
check_genes <- c("TREM2", "TYROBP", "C1QA", "C1QB", "C1QC", "C3",
                  "NLRP3", "IL1B", "IL6", "TNF", "CD68")
survived <- check_genes[check_genes %in% rownames(gene_exprs_filtered)]
dropped  <- setdiff(check_genes, survived)
cat("Neuroinflammatory markers surviving variance filter:", paste(survived, collapse = ", "), "\n")
if (length(dropped) > 0) {
  cat("Neuroinflammatory markers DROPPED by variance filter (left out on purpose,\n",
      "per Rule 6 - not forced back in):", paste(dropped, collapse = ", "), "\n")
}

## ------------------------------------------------------------
## STEP 15: Frozen stratified train/test split
## ------------------------------------------------------------
## WHY: since hippocampus has exactly one sample per subject (confirmed
## last run), a stratified sample-level split is equivalent to a
## subject-level split here - no GroupKFold complexity needed for
## THIS cohort. This split must be frozen now and never touched again
## until final external validation, per Rule 1/Rule 2.

set.seed(42)  # fixed for reproducibility - document this seed in your methods

n <- ncol(gene_exprs_filtered)
ad_idx      <- which(meta$Diagnosis == "AD")
control_idx <- which(meta$Diagnosis == "Control")

train_frac <- 0.75
ad_train      <- sample(ad_idx, size = round(length(ad_idx) * train_frac))
control_train <- sample(control_idx, size = round(length(control_idx) * train_frac))

train_idx <- sort(c(ad_train, control_train))
test_idx  <- setdiff(seq_len(n), train_idx)

train_exprs <- gene_exprs_filtered[, train_idx]
test_exprs  <- gene_exprs_filtered[, test_idx]
train_meta  <- meta[train_idx, ]
test_meta   <- meta[test_idx, ]

cat("\n--- Frozen train/test split ---\n")
cat("Train set:", ncol(train_exprs), "samples -", 
    "AD:", sum(train_meta$Diagnosis == "AD"),
    "Control:", sum(train_meta$Diagnosis == "Control"), "\n")
cat("Test set:", ncol(test_exprs), "samples -",
    "AD:", sum(test_meta$Diagnosis == "AD"),
    "Control:", sum(test_meta$Diagnosis == "Control"), "\n")

dir.create("04_ML", showWarnings = FALSE)

saveRDS(train_exprs, "04_ML/GSE48350_hippocampus_train_exprs.rds")
saveRDS(test_exprs,  "04_ML/GSE48350_hippocampus_test_exprs.rds")
write.csv(train_meta, "04_ML/GSE48350_hippocampus_train_meta.csv", row.names = FALSE)
write.csv(test_meta,  "04_ML/GSE48350_hippocampus_test_meta.csv", row.names = FALSE)
write.csv(gene_exprs_filtered, "04_ML/GSE48350_hippocampus_ML_matrix_full.csv")

cat("\nSaved to 04_ML/:\n",
    " GSE48350_hippocampus_train_exprs.rds /  ..._test_exprs.rds\n",
    " GSE48350_hippocampus_train_meta.csv  /  ..._test_meta.csv\n",
    " GSE48350_hippocampus_ML_matrix_full.csv (pre-split, full filtered matrix)\n")
cat("\nFROZEN. Per Rule 1/Rule 2: the test set above must NOT be touched again\n",
    "until final model evaluation. Do not re-run this script with a different\n",
    "seed to 'improve' the split.\n")
