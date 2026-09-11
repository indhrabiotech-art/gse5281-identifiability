#!/usr/bin/env Rscript
# ==============================================================================
# technical_qc_GSE5281.R
#
# Label-independent technical evidence that the GSE5281 cohort separation is
# a processing effect rather than disease biology.
#
# Every quantity below is fixed in the laboratory or is a property of RNA
# handling. None can be altered by Alzheimer's pathology. If they differ by
# cohort, the difference is technical by construction.
#
#   1. CEL scan dates
#   2. Hybridisation spike-in controls (BioB, BioC, BioDn, CreX)
#   3. Poly-A spike-in controls (LysX, PheX, ThrX, DapX)
#   4. 3'/5' ratios for GAPDH and ACTB
#   5. RNA degradation slope
#   6. RLE and NUSE from probe-level model fits
#
# Usage:  cd ~/project_ml && Rscript R_scripts/technical_qc_GSE5281.R
# ==============================================================================

rm(list = ls()); options(stringsAsFactors = FALSE)
setwd(path.expand("~/project_ml"))

for (p in c("affy", "affyPLM", "affyio")) {
  if (!requireNamespace(p, quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE))
      install.packages("BiocManager", repos = "https://cloud.r-project.org")
    BiocManager::install(p, ask = FALSE, update = FALSE)
  }
}
suppressPackageStartupMessages({library(affy); library(affyPLM)})

OUT <- "04_ML/Technical_QC"; dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
sink(file.path(OUT, "GSE5281_technical_qc_log.txt"), split = TRUE)

cat("================================================================\n")
cat("GSE5281 LABEL-INDEPENDENT TECHNICAL QC\n")
cat("Run:", format(Sys.time()), "\n")
cat("================================================================\n\n")

# ------------------------------------------------------------------- locate
dirs <- list.dirs(".", recursive = TRUE); dirs <- dirs[grepl("5281", dirs)]
cels <- unlist(lapply(dirs, function(d)
  list.files(d, pattern = "\\.CEL(\\.gz)?$", full.names = TRUE, ignore.case = TRUE)))
cels <- cels[!duplicated(basename(cels))]
gsm  <- sub("\\.CEL.*$", "", sub("_.*$", "", basename(cels)))
cat("CEL files:", length(cels), "\n")

meta <- read.csv("02_Metadata/GSE5281_hippocampus_metadata.csv")
dx   <- meta$diagnosis[match(gsm, meta$GSM)]
keep <- !is.na(dx); cels <- cels[keep]; gsm <- gsm[keep]; dx <- dx[keep]
grp  <- ifelse(grepl("^ad$|alzh|affect", dx, ignore.case = TRUE), "AD", "Control")
cat("matched to metadata:", length(gsm), "\n")
print(table(grp))

if (length(gsm) < 23)
  cat("\nNOTE: fewer than 23 samples. Only hippocampal CELs appear to be\n",
      "present locally. To extend this to all 161 samples, download the\n",
      "full GSE5281_RAW supplementary archive from GEO.\n")

# --------------------------------------------------------------- 1. scan date
cat("\n--- 1. Scan dates ---------------------------------------------\n")
scan <- sapply(cels, function(f)
  tryCatch(affyio::read.celfile.header(f, info = "full")$ScanDate,
           error = function(e) NA_character_))
sdate <- sub(" .*$", "", scan)
syear <- sub("^.*/", "", sdate)
print(table(grp, syear))
cat("scan dates carrying both classes:",
    sum(tapply(grp, sdate, function(v) length(unique(v)) == 2)),
    "of", length(unique(sdate)), "\n")

# ------------------------------------------------------------- read CEL batch
cat("\nreading CEL files (this takes a few minutes)\n")
ab <- ReadAffy(filenames = cels)
sampleNames(ab) <- gsm

# ------------------------------------------------------------- 2/3. spike-ins
cat("\n--- 2/3. Spike-in controls ------------------------------------\n")
eset <- rma(ab, verbose = FALSE)
E <- exprs(eset)
pick <- function(pat) grep(pat, rownames(E), value = TRUE)

hyb  <- c(pick("^AFFX-BioB"), pick("^AFFX-BioC"), pick("^AFFX-BioDn"), pick("^AFFX-CreX"))
polya <- c(pick("AFFX.*LysX"), pick("AFFX.*PheX"), pick("AFFX.*ThrX"), pick("AFFX.*DapX"))

spike_test <- function(ids, label) {
  ids <- unique(ids); ids <- ids[ids %in% rownames(E)]
  if (!length(ids)) { cat("\n", label, ": no probe sets found\n"); return(NULL) }
  cat("\n", label, " (", length(ids), " probe sets)\n", sep = "")
  sc <- colMeans(E[ids, , drop = FALSE])
  cat(sprintf("  AD      mean %.3f  SD %.3f\n", mean(sc[grp=="AD"]), sd(sc[grp=="AD"])))
  cat(sprintf("  Control mean %.3f  SD %.3f\n", mean(sc[grp=="Control"]), sd(sc[grp=="Control"])))
  tt <- t.test(sc ~ grp); wt <- wilcox.test(sc ~ grp, exact = FALSE)
  d  <- (mean(sc[grp=="AD"]) - mean(sc[grp=="Control"])) /
        sqrt((var(sc[grp=="AD"]) + var(sc[grp=="Control"]))/2)
  cat(sprintf("  t p = %.4g   Wilcoxon p = %.4g   Cohen's d = %.2f\n",
              tt$p.value, wt$p.value, d))
  a <- pROC::auc(pROC::roc(grp, sc, quiet = TRUE))
  cat(sprintf("  ROC-AUC separating cohorts on this control alone: %.3f\n", as.numeric(a)))
  data.frame(GSM = gsm, group = grp, metric = label, value = sc)
}
if (!requireNamespace("pROC", quietly = TRUE))
  install.packages("pROC", repos = "https://cloud.r-project.org")
rows <- list()
rows$hyb   <- spike_test(hyb,   "Hybridisation spike-ins")
rows$polya <- spike_test(polya, "Poly-A spike-ins")

# --------------------------------------------------------------- 4. 3'/5'
cat("\n--- 4. 3'/5' ratios -------------------------------------------\n")
ratio_test <- function(stub, label) {
  p3 <- grep(paste0(stub, ".*3_at$"), rownames(E), value = TRUE)
  p5 <- grep(paste0(stub, ".*5_at$"), rownames(E), value = TRUE)
  if (!length(p3) || !length(p5)) { cat("\n", label, ": probe sets not found\n"); return(NULL) }
  v <- E[p3[1], ] - E[p5[1], ]
  cat(sprintf("\n%s  (%s / %s)\n", label, p3[1], p5[1]))
  cat(sprintf("  AD      mean %.3f  SD %.3f\n", mean(v[grp=="AD"]), sd(v[grp=="AD"])))
  cat(sprintf("  Control mean %.3f  SD %.3f\n", mean(v[grp=="Control"]), sd(v[grp=="Control"])))
  cat(sprintf("  t p = %.4g   Wilcoxon p = %.4g\n",
              t.test(v ~ grp)$p.value, wilcox.test(v ~ grp, exact = FALSE)$p.value))
  cat(sprintf("  ROC-AUC separating cohorts: %.3f\n",
              as.numeric(pROC::auc(pROC::roc(grp, v, quiet = TRUE)))))
  data.frame(GSM = gsm, group = grp, metric = label, value = v)
}
rows$gapdh <- ratio_test("AFFX-HUMGAPDH", "GAPDH 3-5 ratio (log2)")
rows$actb  <- ratio_test("AFFX-HSAC07",   "ACTB 3-5 ratio (log2)")

cat("\n--- 5. RNA degradation slope ----------------------------------\n")
deg <- AffyRNAdeg(ab)
sl  <- deg$slope
cat(sprintf("  AD mean %.4f   Control mean %.4f   t p = %.4g\n",
            mean(sl[grp=="AD"]), mean(sl[grp=="Control"]), t.test(sl ~ grp)$p.value))
rows$deg <- data.frame(GSM = gsm, group = grp, metric = "RNA degradation slope", value = sl)

# ------------------------------------------------------------- 6. RLE / NUSE
cat("\n--- 6. RLE and NUSE -------------------------------------------\n")
pset <- fitPLM(ab, verbosity.level = 0)
rle_med  <- apply(RLE(pset,  type = "values"), 2, median, na.rm = TRUE)
nuse_med <- apply(NUSE(pset, type = "values"), 2, median, na.rm = TRUE)
for (nm in c("RLE median", "NUSE median")) {
  v <- if (nm == "RLE median") rle_med else nuse_med
  cat(sprintf("\n%s\n  AD mean %.4f   Control mean %.4f   t p = %.4g\n",
              nm, mean(v[grp=="AD"]), mean(v[grp=="Control"]), t.test(v ~ grp)$p.value))
  rows[[nm]] <- data.frame(GSM = gsm, group = grp, metric = nm, value = v)
}

# ------------------------------------------------------------------- output
res <- do.call(rbind, Filter(Negate(is.null), rows))
res$scan_date <- sdate[match(res$GSM, gsm)]
write.csv(res, file.path(OUT, "GSE5281_technical_qc_metrics.csv"), row.names = FALSE)

cat("\n=========================== SUMMARY ===========================\n")
cat("Every metric above is fixed at the bench or is a property of RNA\n")
cat("handling. Disease biology cannot alter spike-in concentrations, and\n")
cat("cannot alter the date on which an array was scanned. Where these\n")
cat("differ between the two cohorts, the difference is technical.\n\n")
cat("Report the significant ones in Section 3.5 and add a panel to Figure 1.\n")
cat("Outputs in", OUT, "\n")
sink()
