library(hgu133plus2.db)
library(AnnotationDbi)

input_file <- "03_Preprocessing/GSE5281_hippocampus_expression.csv"
output_file <- "03_Preprocessing/GSE5281_hippocampus_genelevel_log2.csv"

cat("Reading GSE5281 expression...\n")

expr <- read.csv(
  input_file,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

probe_ids <- expr$ID_REF

expr_mat <- as.matrix(expr[, -1])
rownames(expr_mat) <- probe_ids

storage.mode(expr_mat) <- "numeric"

cat("Raw matrix:", nrow(expr_mat), "probes x",
    ncol(expr_mat), "samples\n")

# --------------------------------------------------
# LOG2 TRANSFORMATION
# --------------------------------------------------

cat("\nApplying log2 transformation...\n")

if (max(expr_mat, na.rm = TRUE) > 50) {
    expr_log2 <- log2(expr_mat + 1)
} else {
    expr_log2 <- expr_mat
}

# --------------------------------------------------
# PROBE -> GENE SYMBOL
# --------------------------------------------------

cat("\nMapping GPL570 probes to gene symbols...\n")

gene_symbols <- AnnotationDbi::select(
    hgu133plus2.db,
    keys = rownames(expr_log2),
    columns = "SYMBOL",
    keytype = "PROBEID"
)

gene_symbols <- gene_symbols[
    !is.na(gene_symbols$SYMBOL) &
    gene_symbols$SYMBOL != "",
]

gene_symbols <- gene_symbols[
    !duplicated(gene_symbols$PROBEID),
]

cat("Mapped probes:",
    nrow(gene_symbols), "\n")

# --------------------------------------------------
# BUILD ANNOTATED MATRIX
# --------------------------------------------------

annot_df <- data.frame(
    PROBEID = rownames(expr_log2),
    expr_log2,
    check.names = FALSE
)

annot_df <- merge(
    gene_symbols,
    annot_df,
    by = "PROBEID"
)

# --------------------------------------------------
# SELECT ONE PROBE PER GENE
# SAME RULE AS DISCOVERY PIPELINE
# --------------------------------------------------

sample_cols <- colnames(expr_mat)

annot_df$mean_expression <- rowMeans(
    annot_df[, sample_cols, drop = FALSE],
    na.rm = TRUE
)

annot_df <- annot_df[
    order(
        annot_df$SYMBOL,
        -annot_df$mean_expression
    ),
]

best_probe_per_gene <- annot_df[
    !duplicated(annot_df$SYMBOL),
]

cat(
    "Unique genes after probe collapsing:",
    nrow(best_probe_per_gene),
    "\n"
)

# --------------------------------------------------
# CREATE GENE-LEVEL MATRIX
# --------------------------------------------------

gene_level <- best_probe_per_gene[, sample_cols, drop = FALSE]

rownames(gene_level) <- best_probe_per_gene$SYMBOL

# --------------------------------------------------
# SAVE
# --------------------------------------------------

write.csv(
    gene_level,
    output_file,
    quote = FALSE
)

cat("\nSaved:\n")
cat(output_file, "\n")

cat(
    "\nFinal matrix:",
    nrow(gene_level),
    "genes x",
    ncol(gene_level),
    "samples\n"
)
