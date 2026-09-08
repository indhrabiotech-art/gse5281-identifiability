## ============================================================
## STEP 5-6: Extract hippocampus-only expression submatrix
## Uses the validated manifest from build_subject_group_id.py
## (02_Metadata/GSE48350_metadata_grouped.csv)
## ============================================================

library(GEOquery)
library(Biobase)

## ------------------------------------------------------------
## Load full series matrix (skip re-downloading; GEOquery caches
## locally if the .gz file is already in the working directory)
## ------------------------------------------------------------
gse <- getGEO(filename = "01_GEO_Data/GSE48350_series_matrix.txt.gz", getGPL = FALSE)
eset_full <- gse
cat("Full expression matrix dimensions (probes x samples):",
    paste(dim(exprs(eset_full)), collapse = " x "), "\n")

## ------------------------------------------------------------
## Load the validated hippocampus manifest (from Python step)
## ------------------------------------------------------------
manifest <- read.csv("02_Metadata/GSE48350_metadata_grouped.csv",
                      stringsAsFactors = FALSE)

hip_manifest <- manifest[manifest$Brain_region == "hippocampus", ]
cat("Hippocampus samples in manifest:", nrow(hip_manifest), "\n")
cat("Diagnosis breakdown:\n")
print(table(hip_manifest$Diagnosis))

## ------------------------------------------------------------
## Subset the expression matrix to exactly these GSM IDs
## ------------------------------------------------------------
## WHY: this is the alignment step your log flags in Section 22 -
## expression matrix columns must match metadata rows exactly,
## same order, same set. Doing this via GSM ID (not position) is
## what guarantees correctness.

full_gsm <- colnames(exprs(eset_full))
missing_gsm <- setdiff(hip_manifest$GSM, full_gsm)

if (length(missing_gsm) > 0) {
  stop("ERROR: ", length(missing_gsm),
       " GSM IDs from the manifest are not found in the expression ",
       "matrix. Check for ID formatting mismatches before continuing:\n",
       paste(missing_gsm, collapse = ", "))
}

hip_exprs <- exprs(eset_full)[, hip_manifest$GSM]

## Re-order manifest to guarantee identical order (belt-and-braces,
## even though GSM-based subsetting above already preserves it)
hip_manifest <- hip_manifest[match(colnames(hip_exprs), hip_manifest$GSM), ]

stopifnot(identical(colnames(hip_exprs), hip_manifest$GSM))

cat("\nHippocampus expression submatrix dimensions (probes x samples):",
    paste(dim(hip_exprs), collapse = " x "), "\n")

## ------------------------------------------------------------
## STEP 6 checks: value scale, log-transform, normalization state
## ------------------------------------------------------------
## WHY: your log (Section 24) explicitly warns against re-normalizing
## an already-processed GEO matrix. This determines whether you skip
## or apply that step before probe annotation.

cat("\nExpression value summary (raw check for log-transform state):\n")
print(summary(as.vector(hip_exprs)))

cat("\nMax value:", max(hip_exprs), " -- ",
    "if this is roughly 12-16, data is very likely already log2-transformed.\n",
    "If this is in the thousands/tens of thousands, it is likely raw/linear scale.\n")

## ------------------------------------------------------------
## Save the frozen hippocampus expression matrix + aligned metadata
## ------------------------------------------------------------
dir.create("03_Preprocessing", showWarnings = FALSE)

saveRDS(hip_exprs, "03_Preprocessing/GSE48350_hippocampus_exprs_raw.rds")
write.csv(hip_manifest,
          "03_Preprocessing/GSE48350_hippocampus_metadata_aligned.csv",
          row.names = FALSE)

cat("\nSaved:\n",
    " 03_Preprocessing/GSE48350_hippocampus_exprs_raw.rds\n",
    " 03_Preprocessing/GSE48350_hippocampus_metadata_aligned.csv\n")
cat("\nDo NOT overwrite the original series matrix or metadata_grouped.csv -\n",
    "these are new, separately named files per the provenance-chain rule.\n")

## ------------------------------------------------------------
## Platform confirmation (needed for STEP 8-9: correct annotation package)
## ------------------------------------------------------------
cat("\nPlatform (GPL) for this series:\n")
print(annotation(eset_full))

