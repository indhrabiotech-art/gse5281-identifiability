cat("============================================\n")
cat("TRAINING-ONLY GENE VARIANCE FILTER\n")
cat("============================================\n")

# ------------------------------------------------------------
# Load RMA ML matrix
#
# IMPORTANT:
# Rows    = samples
# Columns = genes
# ------------------------------------------------------------

expr <- read.csv(
    "04_ML/GSE48350_RMA_ML_matrix.csv",
    row.names = 1,
    check.names = FALSE
)

cat("Full matrix:", nrow(expr), "samples x", ncol(expr), "genes\n")

# ------------------------------------------------------------
# Load fixed train/test metadata
# ------------------------------------------------------------

train_meta <- read.csv(
    "04_ML/GSE48350_RMA_train_meta.csv",
    check.names = FALSE
)

test_meta <- read.csv(
    "04_ML/GSE48350_RMA_test_meta.csv",
    check.names = FALSE
)

train_gsm <- train_meta$index
test_gsm  <- test_meta$index

# ------------------------------------------------------------
# Verify sample IDs
# ------------------------------------------------------------

missing_train <- setdiff(train_gsm, rownames(expr))
missing_test  <- setdiff(test_gsm, rownames(expr))

if (length(missing_train) > 0) {
    stop(
        "Training samples missing: ",
        paste(missing_train, collapse = ", ")
    )
}

if (length(missing_test) > 0) {
    stop(
        "Test samples missing: ",
        paste(missing_test, collapse = ", ")
    )
}

cat("Training samples:", length(train_gsm), "\n")
cat("Test samples:", length(test_gsm), "\n")

# ------------------------------------------------------------
# Extract TRAINING ONLY
# ------------------------------------------------------------

train_expr <- expr[train_gsm, , drop = FALSE]

cat("\nTraining matrix:",
    nrow(train_expr), "samples x",
    ncol(train_expr), "genes\n")

# ------------------------------------------------------------
# Calculate gene variance ONLY in training samples
# ------------------------------------------------------------

gene_variance <- apply(
    train_expr,
    2,
    var,
    na.rm = TRUE
)

cat("\nVariance summary:\n")
print(summary(gene_variance))

# ------------------------------------------------------------
# Remove zero-variance genes
# ------------------------------------------------------------

keep <- gene_variance > 0

cat("\nGenes before filtering:", length(keep), "\n")
cat("Genes retained:", sum(keep), "\n")
cat("Genes removed:", sum(!keep), "\n")

filtered_genes <- names(gene_variance)[keep]

# ------------------------------------------------------------
# Apply SAME genes to both train and test
# ------------------------------------------------------------

train_filtered <- train_expr[, filtered_genes, drop = FALSE]
test_filtered  <- expr[test_gsm, filtered_genes, drop = FALSE]

# ------------------------------------------------------------
# Save matrices
# ------------------------------------------------------------

dir.create("04_ML", showWarnings = FALSE)

write.csv(
    train_filtered,
    "04_ML/GSE48350_RMA_train_variance_filtered.csv",
    quote = FALSE
)

write.csv(
    test_filtered,
    "04_ML/GSE48350_RMA_test_variance_filtered.csv",
    quote = FALSE
)

# ------------------------------------------------------------
# Save gene variance table
# ------------------------------------------------------------

variance_df <- data.frame(
    gene = names(gene_variance),
    variance = as.numeric(gene_variance)
)

variance_df <- variance_df[
    order(variance_df$variance, decreasing = TRUE),
]

write.csv(
    variance_df,
    "04_ML/GSE48350_RMA_training_gene_variance.csv",
    row.names = FALSE,
    quote = FALSE
)

# ------------------------------------------------------------
# Final checks
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FINAL FILTER RESULTS\n")
cat("============================================\n")

cat("Training:", nrow(train_filtered),
    "samples x", ncol(train_filtered), "genes\n")

cat("Test:", nrow(test_filtered),
    "samples x", ncol(test_filtered), "genes\n")

cat("\nTraining diagnosis:\n")
print(table(train_meta$Diagnosis))

cat("\nTest diagnosis:\n")
print(table(test_meta$Diagnosis))

cat("\nMissing training values:",
    sum(is.na(train_filtered)), "\n")

cat("Missing test values:",
    sum(is.na(test_filtered)), "\n")

cat("\nSaved:\n")
cat("04_ML/GSE48350_RMA_train_variance_filtered.csv\n")
cat("04_ML/GSE48350_RMA_test_variance_filtered.csv\n")
cat("04_ML/GSE48350_RMA_training_gene_variance.csv\n")

cat("\n============================================\n")
cat("TRAINING VARIANCE FILTER COMPLETE\n")
cat("============================================\n")
