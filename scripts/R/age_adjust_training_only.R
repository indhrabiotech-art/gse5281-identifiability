cat("============================================\n")
cat("TRAINING-ONLY AGE ADJUSTMENT\n")
cat("============================================\n")

# ------------------------------------------------------------
# Load matrices
# ------------------------------------------------------------

train <- read.csv(
    "04_ML/GSE48350_RMA_train_top25var.csv",
    row.names = 1,
    check.names = FALSE
)

test <- read.csv(
    "04_ML/GSE48350_RMA_test_top25var.csv",
    row.names = 1,
    check.names = FALSE
)

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

train_meta <- read.csv(
    "04_ML/GSE48350_RMA_train_meta.csv",
    stringsAsFactors = FALSE,
    check.names = FALSE
)

test_meta <- read.csv(
    "04_ML/GSE48350_RMA_test_meta.csv",
    stringsAsFactors = FALSE,
    check.names = FALSE
)

# ------------------------------------------------------------
# Verify dimensions
# ------------------------------------------------------------

cat("Training matrix:",
    nrow(train), "samples x", ncol(train), "genes\n")

cat("Test matrix:",
    nrow(test), "samples x", ncol(test), "genes\n")

cat("Training metadata:", nrow(train_meta), "samples\n")
cat("Test metadata:", nrow(test_meta), "samples\n")

if (nrow(train) != nrow(train_meta)) {
    stop("Training expression and metadata sample counts differ.")
}

if (nrow(test) != nrow(test_meta)) {
    stop("Test expression and metadata sample counts differ.")
}

# ------------------------------------------------------------
# Verify sample order
# ------------------------------------------------------------

if (!identical(
    rownames(train),
    train_meta$index
)) {
    stop("Training sample order does not match metadata.")
}

if (!identical(
    rownames(test),
    test_meta$index
)) {
    stop("Test sample order does not match metadata.")
}

cat("\nSample order: VERIFIED\n")

# ------------------------------------------------------------
# Verify gene identity
# ------------------------------------------------------------

if (!identical(
    colnames(train),
    colnames(test)
)) {
    stop("Training and test gene sets/order differ.")
}

genes <- colnames(train)

cat("Genes:", length(genes), "\n")

# ------------------------------------------------------------
# Extract age
# ------------------------------------------------------------

train_age <- as.numeric(train_meta$Age)
test_age  <- as.numeric(test_meta$Age)

if (any(is.na(train_age))) {
    stop("Missing training ages detected.")
}

if (any(is.na(test_age))) {
    stop("Missing test ages detected.")
}

cat("\nTraining age summary:\n")
print(summary(train_age))

cat("\nTest age summary:\n")
print(summary(test_age))

# ------------------------------------------------------------
# Allocate output matrices
# ------------------------------------------------------------

train_adj <- matrix(
    NA_real_,
    nrow = nrow(train),
    ncol = ncol(train),
    dimnames = dimnames(train)
)

test_adj <- matrix(
    NA_real_,
    nrow = nrow(test),
    ncol = ncol(test),
    dimnames = dimnames(test)
)

# ------------------------------------------------------------
# Store age-adjustment parameters
# ------------------------------------------------------------

age_coef <- numeric(length(genes))
age_pvalue <- numeric(length(genes))
age_r2 <- numeric(length(genes))

names(age_coef) <- genes
names(age_pvalue) <- genes
names(age_r2) <- genes

# ------------------------------------------------------------
# Gene-wise training-only adjustment
# ------------------------------------------------------------

cat("\nRunning gene-wise age adjustment...\n")

for (i in seq_along(genes)) {

    gene <- genes[i]

    y_train <- as.numeric(train[, i])

    # Training-only model
    fit <- lm(y_train ~ train_age)

    beta_age <- coef(fit)[2]

    # Store statistics
    age_coef[i] <- beta_age

    coef_table <- summary(fit)$coefficients

    if ("train_age" %in% rownames(coef_table)) {
        age_pvalue[i] <- coef_table["train_age", "Pr(>|t|)"]
    } else {
        age_pvalue[i] <- NA_real_
    }

    age_r2[i] <- summary(fit)$r.squared

    # --------------------------------------------------------
    # Remove age effect from TRAINING data
    #
    # residual = expression - beta_age * age
    #
    # We center the age variable around the TRAINING mean so
    # the adjusted expression remains on approximately the
    # original expression scale.
    # --------------------------------------------------------

    train_age_centered <- train_age - mean(train_age)

    train_adj[, i] <-
        y_train - beta_age * train_age_centered

    # --------------------------------------------------------
    # Apply TRAINING-derived beta to TEST data
    # --------------------------------------------------------

    test_age_centered <- test_age - mean(train_age)

    y_test <- as.numeric(test[, i])

    test_adj[, i] <-
        y_test - beta_age * test_age_centered

    if (i %% 500 == 0) {
        cat("Processed", i, "/", length(genes), "genes\n")
    }
}

# ------------------------------------------------------------
# Convert to data frames
# ------------------------------------------------------------

train_adj <- as.data.frame(train_adj, check.names = FALSE)
test_adj  <- as.data.frame(test_adj, check.names = FALSE)

# ------------------------------------------------------------
# Save adjustment statistics
# ------------------------------------------------------------

age_stats <- data.frame(
    gene = genes,
    age_coefficient = age_coef,
    age_pvalue = age_pvalue,
    age_r2 = age_r2,
    stringsAsFactors = FALSE
)

# ------------------------------------------------------------
# Final QC
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FINAL AGE-ADJUSTMENT QC\n")
cat("============================================\n")

cat("Training adjusted matrix:",
    nrow(train_adj), "x", ncol(train_adj), "\n")

cat("Test adjusted matrix:",
    nrow(test_adj), "x", ncol(test_adj), "\n")

cat("Missing training values:",
    sum(is.na(train_adj)), "\n")

cat("Missing test values:",
    sum(is.na(test_adj)), "\n")

cat("\nAge coefficient summary:\n")
print(summary(age_coef))

cat("\nGenes with nominal age association (p < 0.05):",
    sum(age_pvalue < 0.05, na.rm = TRUE), "\n")

cat("\nTop 10 genes by absolute age coefficient:\n")

top_age <- order(
    abs(age_stats$age_coefficient),
    decreasing = TRUE
)[1:10]

print(age_stats[top_age, ])

# ------------------------------------------------------------
# Save files
# ------------------------------------------------------------

write.csv(
    train_adj,
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    quote = FALSE
)

write.csv(
    test_adj,
    "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
    quote = FALSE
)

write.csv(
    age_stats,
    "04_ML/GSE48350_RMA_training_age_adjustment_stats.csv",
    row.names = FALSE,
    quote = FALSE
)

cat("\nSaved:\n")
cat("04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv\n")
cat("04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv\n")
cat("04_ML/GSE48350_RMA_training_age_adjustment_stats.csv\n")

cat("\n============================================\n")
cat("TRAINING-ONLY AGE ADJUSTMENT COMPLETE\n")
cat("============================================\n")
