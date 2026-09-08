## ============================================================
## STEP 7-12: Log-transform, QC visuals, probe -> gene annotation
## Input:  03_Preprocessing/GSE48350_hippocampus_exprs_raw.rds
##         03_Preprocessing/GSE48350_hippocampus_metadata_aligned.csv
## ============================================================

library(hgu133plus2.db)
library(AnnotationDbi)

setwd("~/project_ml")  # safety net in case this is sourced fresh

hip_exprs   <- readRDS("03_Preprocessing/GSE48350_hippocampus_exprs_raw.rds")
hip_meta    <- read.csv("03_Preprocessing/GSE48350_hippocampus_metadata_aligned.csv",
                         stringsAsFactors = FALSE)

stopifnot(identical(colnames(hip_exprs), hip_meta$GSM))

## ------------------------------------------------------------
## STEP 7: Log2 transform
## ------------------------------------------------------------
## WHY: confirmed last run - max value 1438.76, min 0.01 = linear
## scale, not log-transformed. Log2 is the standard transform for
## microarray expression data; skipping it would leave a heavily
## right-skewed distribution that violates assumptions of most
## downstream models (esp. logistic regression) and distorts PCA.

cat("Pre-transform value range:\n")
print(summary(as.vector(hip_exprs)))

hip_exprs_log2 <- log2(hip_exprs)

cat("\nPost-log2-transform value range:\n")
print(summary(as.vector(hip_exprs_log2)))
cat("\n(Expect roughly -6 to 11 range now - typical for log2 microarray data.)\n")

## ------------------------------------------------------------
## STEP 8 (partial): Quick QC visuals, saved as PNG (no display
## available when sourcing from console)
## ------------------------------------------------------------
dir.create("07_Figures", showWarnings = FALSE)

png("07_Figures/hippocampus_boxplot_post_log2.png", width = 1400, height = 600)
boxplot(hip_exprs_log2[, 1:20],
        main = "Log2 expression distribution - first 20 hippocampus samples",
        las = 2, cex.axis = 0.7,
        col = ifelse(hip_meta$Diagnosis[1:20] == "AD", "tomato", "steelblue"))
dev.off()
cat("\nSaved boxplot QC figure to 07_Figures/hippocampus_boxplot_post_log2.png\n")
cat("Open it and check: are the boxes roughly similar height/position across\n",
    "samples? A few wildly different boxes would flag an outlier sample worth\n",
    "investigating before proceeding.\n")

## ------------------------------------------------------------
## STEP 9-11: Probe -> gene symbol annotation + duplicate handling
## ------------------------------------------------------------
## WHY: hgu133plus2.db is the correct annotation package for GPL570
## (confirmed last run). Multiple probes commonly map to the same
## gene symbol - Section 9/STEP 11 of your log requires resolving
## this before building the final gene-level matrix. Standard
## approach: keep the probe with highest mean expression per gene
## (most detectable signal), which is a common, defensible choice.

probe_ids <- rownames(hip_exprs_log2)

gene_symbols <- AnnotationDbi::select(
  hgu133plus2.db,
  keys    = probe_ids,
  columns = c("SYMBOL"),
  keytype = "PROBEID"
)

cat("\nProbe -> gene mapping result:\n")
cat("Total probes:", length(probe_ids), "\n")
cat("Probes with a mapped gene symbol:", sum(!is.na(gene_symbols$SYMBOL)), "\n")
cat("Probes with NO mapped gene symbol (will be dropped):",
    sum(is.na(gene_symbols$SYMBOL)), "\n")

## Remove probes with no gene symbol
gene_symbols <- gene_symbols[!is.na(gene_symbols$SYMBOL), ]
## AnnotationDbi::select can return >1 row per probe (rare multi-mapping) -
## keep first mapping per probe for simplicity/reproducibility
gene_symbols <- gene_symbols[!duplicated(gene_symbols$PROBEID), ]

## Compute mean expression per probe (used to pick the "best" probe per gene)
probe_mean_expr <- rowMeans(hip_exprs_log2)

annot_df <- data.frame(
  PROBEID   = rownames(hip_exprs_log2),
  mean_expr = probe_mean_expr,
  stringsAsFactors = FALSE
)
annot_df <- merge(annot_df, gene_symbols, by = "PROBEID")

## STEP 11: collapse - for each gene symbol, keep the probe with
## the highest mean expression
annot_df <- annot_df[order(annot_df$SYMBOL, -annot_df$mean_expr), ]
best_probe_per_gene <- annot_df[!duplicated(annot_df$SYMBOL), ]

cat("\nUnique gene symbols after collapsing duplicate probes:",
    nrow(best_probe_per_gene), "\n")

## ------------------------------------------------------------
## STEP 12: Build final gene-level expression matrix
## ------------------------------------------------------------
gene_level_exprs <- hip_exprs_log2[best_probe_per_gene$PROBEID, ]
rownames(gene_level_exprs) <- best_probe_per_gene$SYMBOL

stopifnot(identical(colnames(gene_level_exprs), hip_meta$GSM))

cat("\nFinal gene-level matrix dimensions (genes x samples):",
    paste(dim(gene_level_exprs), collapse = " x "), "\n")

## Quick sanity check: are known neuroinflammatory genes present?
check_genes <- c("TREM2", "TYROBP", "C1QA", "C1QB", "C1QC", "C3",
                  "NLRP3", "IL1B", "IL6", "TNF", "CD68")
present <- check_genes[check_genes %in% rownames(gene_level_exprs)]
missing <- setdiff(check_genes, present)
cat("\nNeuroinflammatory marker genes found in matrix:", paste(present, collapse = ", "), "\n")
if (length(missing) > 0) {
  cat("NOT found (check probe coverage / symbol naming):", paste(missing, collapse = ", "), "\n")
}

## ------------------------------------------------------------
## Save final preprocessed files (new filenames - provenance chain)
## ------------------------------------------------------------
saveRDS(gene_level_exprs, "03_Preprocessing/GSE48350_hippocampus_genelevel_log2.rds")
write.csv(gene_level_exprs, "03_Preprocessing/GSE48350_hippocampus_genelevel_log2.csv")

cat("\nSaved:\n",
    " 03_Preprocessing/GSE48350_hippocampus_genelevel_log2.rds\n",
    " 03_Preprocessing/GSE48350_hippocampus_genelevel_log2.csv\n")
cat("\nThis is the matrix to build the ML analysis matrix from next\n",
    "(feature filtering -> feature selection -> train/test split).\n")
