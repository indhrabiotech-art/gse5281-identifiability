#!/usr/bin/env Rscript
# ==============================================================================
# frma_sensitivity.R   --  critique item A5, locked preprocessing
#
# The manuscript scores GSE5281 after RMA run independently on that dataset.
# RMA is a multi-array method, so the resulting values depend on the samples
# processed together and are not comparable with a training reference. This is
# a plausible alternative explanation for the calibration collapse.
#
# Frozen RMA (fRMA) uses precomputed probe effects and a fixed reference
# distribution, so each array is normalised to the same scale regardless of
# what else is in the batch. That is the correct preprocessing for
# single-sample transfer.
#
# Three preprocessing schemes are compared, scoring the same locked model:
#   1. RMA  (current manuscript)
#   2. fRMA (locked reference)
#   3. quantile normalisation of RMA values to the training reference
#
# For each: transfer ROC-AUC, mean predicted probability, Brier score,
# calibration-in-the-large.
#
# Usage:  cd ~/project_ml && Rscript R_scripts/frma_sensitivity.R
# ==============================================================================

rm(list = ls()); options(stringsAsFactors = FALSE, warn = 1)
setwd(path.expand("~/project_ml"))
OUT <- "04_ML/fRMA_Sensitivity"; dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
sink(file.path(OUT, "frma_sensitivity_log.txt"), split = TRUE)
cat("=========== fRMA SENSITIVITY (A5) ===========\n", format(Sys.time()), "\n\n")

bioc <- function(p) if (!requireNamespace(p, quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  BiocManager::install(p, ask = FALSE, update = FALSE) }
for (p in c("affy","frma","hgu133plus2frmavecs","hgu133plus2.db",
            "AnnotationDbi","preprocessCore")) bioc(p)
if (!requireNamespace("pROC", quietly = TRUE))
  install.packages("pROC", repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({library(affy); library(frma); library(pROC)})

SIG   <- c("ABCA6","CRLF1","TNFRSF11B")
COEF  <- c(ABCA6 = 0.475, TNFRSF11B = 0.439, CRLF1 = 0.380)
key   <- function(x) sub("\\.CEL.*$","",sub("_.*$","",x))

# ---------------------------------------------------- locked model parameters
# Refit on the canonical training matrix so scaler and intercept are exact.
tr <- read.csv("04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
               row.names = 1, check.names = FALSE)
if (nrow(tr) > ncol(tr)) tr <- as.data.frame(t(tr))
rownames(tr) <- key(rownames(tr))
mt <- read.csv("04_ML/GSE48350_RMA_train_meta.csv")
gcol <- names(mt)[which(sapply(mt, function(c) any(grepl("^GSM", as.character(c)))))[1]]
dcol <- grep("diagnos|group|status", names(mt), ignore.case = TRUE, value = TRUE)[1]
mt[[gcol]] <- key(as.character(mt[[gcol]])); mt <- mt[match(rownames(tr), mt[[gcol]]), ]
ytr <- as.integer(grepl("^ad$|alzh|affect", mt[[dcol]], ignore.case = TRUE))

lk <- c("04_ML/Final_Model/GSE48350_final_3gene_coefficients.csv",
        "MANUSCRIPT_DATA/Table_3gene_coefficients.csv",
        "04_ML/Final_Model/GSE5281_locked_model_coefficients.csv")
lk <- lk[file.exists(lk)][1]
cat("locked coefficient file:", lk, "\n"); print(read.csv(lk))
ctr <- colMeans(tr[, SIG]); sdv <- apply(tr[, SIG], 2, sd)
Z   <- scale(tr[, SIG], center = ctr, scale = sdv)
fit <- glm(ytr ~ Z, family = binomial)
cat("locked model refit on canonical training matrix\n")
cat("  intercept", sprintf("%.4f", coef(fit)[1]), " coefficients",
    paste(sprintf("%.4f", coef(fit)[-1]), collapse = ", "), "\n")
cat("  training scaler centre:", paste(sprintf("%.4f", ctr), collapse=", "), "\n")
cat("  training scaler scale :", paste(sprintf("%.4f", sdv), collapse=", "), "\n\n")

score <- function(E, lab) {
  miss <- setdiff(SIG, rownames(E))
  if (length(miss)) { cat(lab, ": missing", paste(miss, collapse=","), "\n"); return(NULL) }
  X <- t(E[SIG, , drop = FALSE])
  Zx <- scale(X, center = ctr, scale = sdv)
  p  <- as.numeric(plogis(cbind(1, Zx) %*% coef(fit)))
  a  <- as.numeric(auc(roc(yex, p, quiet = TRUE)))
  br <- mean((p - yex)^2)
  lg <- log(pmax(pmin(p, 1-1e-6), 1e-6) / (1 - pmax(pmin(p, 1-1e-6), 1e-6)))
  cil <- tryCatch(coef(glm(yex ~ offset(lg), family = binomial))[1],
                  error = function(e) NA)
  cat(sprintf("%-28s AUC %.4f | mean p %.5f | Brier %.4f | CITL %+.3f\n",
              lab, a, mean(p), br, cil))
  data.frame(scheme = lab, auc = a, mean_pred = mean(p), min_pred = min(p),
             max_pred = max(p), brier = br, citl = as.numeric(cil))
}

# --------------------------------------------------------------- transfer set
me <- read.csv("02_Metadata/GSE5281_hippocampus_metadata.csv")

# 1. RMA as used in the manuscript
E_rma <- as.matrix(read.csv("03_Preprocessing/GSE5281_RMA_genelevel.csv",
                            row.names = 1, check.names = FALSE))
gx  <- key(colnames(E_rma))
yex <- as.integer(grepl("^ad$|alzh|affect", me$diagnosis[match(gx, me$GSM)],
                        ignore.case = TRUE))
cat("transfer cohort:", ncol(E_rma), "samples,", sum(yex), "cases\n")
cat("observed prevalence", sprintf("%.4f", mean(yex)), "\n\n")
cat("--- transfer performance by preprocessing scheme ---\n")
res <- list(); res$rma <- score(E_rma, "1. RMA (manuscript)")

# 2. fRMA
cat("\nrunning fRMA (several minutes)\n")
dirs <- list.dirs(".", recursive = TRUE); dirs <- dirs[grepl("5281", dirs)]
cels <- unlist(lapply(dirs, function(d)
  list.files(d, pattern = "\\.CEL(\\.gz)?$", full.names = TRUE, ignore.case = TRUE)))
cels <- cels[!duplicated(basename(cels))]
cels <- cels[key(basename(cels)) %in% gx]
cat("  CEL files matched:", length(cels), "of", length(gx), "\n")
if (length(cels) == length(gx)) {
  ab <- ReadAffy(filenames = cels); sampleNames(ab) <- key(basename(cels))
  ef <- exprs(frma(ab, summarize = "robust_weighted_average"))
  ef <- ef[, gx]
  sym <- AnnotationDbi::mapIds(hgu133plus2.db::hgu133plus2.db, rownames(ef),
                               "SYMBOL", "PROBEID", multiVals = "first")
  ok <- !is.na(sym); ef <- ef[ok, ]; sym <- sym[ok]
  # same collapsing rule as the manuscript: highest mean probe per symbol
  o <- order(sym, -rowMeans(ef)); ef <- ef[o, ]; sym <- sym[o]
  ef <- ef[!duplicated(sym), ]; rownames(ef) <- sym[!duplicated(sym)]
  cat("  fRMA genes:", nrow(ef), "\n")
  write.csv(ef, file.path(OUT, "GSE5281_fRMA_genelevel.csv"))
  res$frma <- score(ef, "2. fRMA (locked reference)")
} else cat("  CEL files incomplete; fRMA skipped\n")

# 3. quantile normalise RMA values to the training reference
ref <- read.csv("03_Preprocessing/GSE48350_RMA_genelevel.csv",
                row.names = 1, check.names = FALSE)
common <- intersect(rownames(E_rma), rownames(ref))
target <- sort(rowMeans(as.matrix(ref[common, ])))
Eq <- apply(E_rma[common, ], 2, function(v) target[rank(v, ties.method = "first")])
rownames(Eq) <- common
res$qn <- score(Eq, "3. quantile-normalised to training")

# --------------------------------------------------------------------- output
tab <- do.call(rbind, Filter(Negate(is.null), res))
write.csv(tab, file.path(OUT, "frma_sensitivity_results.csv"), row.names = FALSE)

cat("\n=========== INTERPRETATION ===========\n")
if (nrow(tab) > 1) {
  base <- tab$mean_pred[1]; best <- max(tab$mean_pred)
  cat(sprintf("mean predicted probability: %.5f under RMA, %.5f at best\n", base, best))
  cat(sprintf("observed prevalence %.4f\n\n", mean(yex)))
  if (best > 0.10) {
    cat("CALIBRATION LARGELY RECOVERS under locked preprocessing.\n")
    cat("The collapse is then substantially a preprocessing artefact rather\n")
    cat("than a property of the datasets. Section 3.7 must be rewritten and\n")
    cat("this analysis reported as a main result.\n")
  } else if (best > 0.02) {
    cat("CALIBRATION IMPROVES BUT DOES NOT RECOVER. Report both: locked\n")
    cat("preprocessing removes part of the collapse, and a substantial\n")
    cat("discrepancy remains that preprocessing cannot explain.\n")
  } else {
    cat("CALIBRATION DOES NOT RECOVER. The collapse survives locked\n")
    cat("preprocessing, so it is not an artefact of separate normalisation.\n")
    cat("This strengthens Section 3.7 and pre-empts the obvious objection.\n")
  }
  cat("\nAUC is expected to change little: rank-preserving transformations\n")
  cat("cannot alter it, which is the point of Section 4.4.\n")
}
cat("\nOutputs in", OUT, "\n")
sink()
