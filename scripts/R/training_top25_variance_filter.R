cat("============================================\n")
cat("TRAINING-ONLY TOP 25% VARIANCE FILTER\n")
cat("============================================\n")

expr <- read.csv(
    "04_ML/GSE48350_RMA_ML_matrix.csv",
    row.names = 1,
    check.names = FALSE
)

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

if (length(setdiff(train_gsm, rownames(expr))) > 0)
    stop("Training GSMs missing from expression matrix.")

if (length(setdiff(test_gsm, rownames(expr))) > 0)
    stop("Test GSMs missing from expression matrix.")

# ------------------------------------------------------------
# TRAINING ONLY
# ------------------------------------------------------------

train_expr <- expr[train_gsm, , drop = FALSE]

cat("Full dataset:",
    nrow(expr), "samples x", ncol(expr), "genes\n")

cat("Training:",
    nrow(train_expr), "samples x", ncol(train_expr), "genes\n")

cat("Test:",
    length(test_gsm), "samples\n")

# ------------------------------------------------------------
# Calculate variance ONLY in training data
# ------------------------------------------------------------

gene_variance <- apply(
    train_expr,
    2,
    var,
    na.rm = TRUE
)

# ------------------------------------------------------------
# Rank genes by variance
# ------------------------------------------------------------

variance_df <- data.frame(
    gene = names(gene_variance),
    variance = as.numeric(gene_variance),
    stringsAsFactors = FALSE
)

variance_df <- variance_df[
    order(variance_df$variance, decreasing = TRUE),
]

# ------------------------------------------------------------
# Top 25%
# ------------------------------------------------------------

n_genes <- nrow(variance_df)

n_keep <- ceiling(n_genes * 0.25)

selected_genes <- variance_df$gene[1:n_keep]

cat("\nGenes available:", n_genes, "\n")
cat("Top 25% genes retained:", n_keep, "\n")

# ------------------------------------------------------------
# Apply SAME selected genes to train and test
# ------------------------------------------------------------

train_filtered <- train_expr[
    ,
    selected_genes,
    drop = FALSE
]

test_filtered <- expr[
    test_gsm,
    selected_genes,
    drop = FALSE
]

# ------------------------------------------------------------
# Save matrices
# ------------------------------------------------------------

write.csv(
    train_filtered,
    "04_ML/GSE48350_RMA_train_top25var.csv",
    quote = FALSE
)

write.csv(
    test_filtered,
    "04_ML/GSE48350_RMA_test_top25var.csv",
    quote = FALSE
)

write.csv(
    variance_df,
    "04_ML/GSE48350_RMA_training_variance_ranked.csv",
    row.names = FALSE,
    quote = FALSE
)

writeLines(
    selected_genes,
    "04_ML/GSE48350_RMA_top25var_genes.txt"
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FINAL RESULTS\n")
cat("============================================\n")

cat("Training matrix:",
    nrow(train_filtered), "x",
    ncol(train_filtered), "\n")

cat("Test matrix:",
    nrow(test_filtered), "x",
    ncol(test_filtered), "\n")

cat("\nTraining diagnosis:\n")
print(table(train_meta$Diagnosis))

cat("\nTest diagnosis:\n")
print(table(test_meta$Diagnosis))

cat("\nMissing training values:",
    sum(is.na(train_filtered)), "\n")

cat("Missing test values:",
    sum(is.na(test_filtered)), "\n")

cat("\nTop 20 genes by training variance:\n")
print(head(variance_df, 20), row.names = FALSE)

cat("\nSaved:\n")
cat("04_ML/GSE48350_RMA_train_top25var.csv\n")
cat("04_ML/GSE48350_RMA_test_top25var.csv\n")
cat("04_ML/GSE48350_RMA_training_variance_ranked.csv\n")
cat("04_ML/GSE48350_RMA_top25var_genes.txt\n")

cat("\n============================================\n")
cat("TOP 25% VARIANCE FILTER COMPLETE\n")
cat("============================================\n")
