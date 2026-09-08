library(limma)

setwd("~/project_ml")

# ------------------------------------------------------------
# Load the exact GSE48350 training data used for age adjustment
# ------------------------------------------------------------
train_exprs <- readRDS(
  "04_ML/GSE48350_hippocampus_train_exprs.rds"
)

train_meta <- read.csv(
  "04_ML/GSE48350_hippocampus_train_meta.csv",
  stringsAsFactors = FALSE
)

train_exprs <- as.matrix(train_exprs)

# Verify sample alignment
stopifnot(identical(colnames(train_exprs), train_meta$GSM))

cat("Training expression:", paste(dim(train_exprs), collapse = " x "), "\n")
cat("Training samples:", ncol(train_exprs), "\n")

# ------------------------------------------------------------
# Fit exactly the same age model as adjust_for_age_v2.R
# ------------------------------------------------------------
design_train <- model.matrix(
  ~ Age,
  data = data.frame(Age = train_meta$Age)
)

fit <- lmFit(train_exprs, design_train)

beta_age <- fit$coefficients[, "Age"]

train_mean_age <- mean(train_meta$Age)

# ------------------------------------------------------------
# Save coefficients
# ------------------------------------------------------------
beta_df <- data.frame(
  gene = rownames(train_exprs),
  beta_age = as.numeric(beta_age)
)

write.csv(
  beta_df,
  "06_Validation/GSE48350_age_coefficients.csv",
  row.names = FALSE
)

writeLines(
  as.character(train_mean_age),
  "06_Validation/GSE48350_train_mean_age.txt"
)

cat("\n========================================\n")
cat("GSE48350 AGE COEFFICIENTS\n")
cat("========================================\n")

cat("Training mean age:", train_mean_age, "\n")
cat("Number of genes:", length(beta_age), "\n")

cat("\nFirst coefficients:\n")
print(head(beta_df))

cat("\nSaved:\n")
cat("06_Validation/GSE48350_age_coefficients.csv\n")
cat("06_Validation/GSE48350_train_mean_age.txt\n")

