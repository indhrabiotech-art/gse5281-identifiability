## ============================================================
## RAW CEL PREPROCESSING
## GSE48350 + GSE5281
## ============================================================

library(affy)
library(Biobase)

setwd("~/project_ml")

## ------------------------------------------------------------
## Function: process one dataset
## ------------------------------------------------------------

process_CEL <- function(
    cel_dir,
    output_rds,
    output_csv,
    dataset_name
) {

    cat("\n============================================\n")
    cat(dataset_name, "\n")
    cat("============================================\n")

    cel_files <- list.files(
        cel_dir,
        pattern = "\\.CEL\\.gz$",
        full.names = TRUE,
        ignore.case = TRUE
    )

    cel_files <- sort(cel_files)

    cat("CEL files found:", length(cel_files), "\n")

    if (length(cel_files) == 0) {
        stop("No CEL files found in: ", cel_dir)
    }

    ## --------------------------------------------------------
    ## Read raw CEL files
    ## --------------------------------------------------------

    cat("\nReading CEL files...\n")

    raw_data <- ReadAffy(
        filenames = cel_files
    )

    cat("Arrays:", ncol(raw_data), "\n")
    cat("Raw probes:", nrow(exprs(raw_data)), "\n")

    ## --------------------------------------------------------
    ## RMA preprocessing
    ## --------------------------------------------------------

    cat("\nRunning RMA preprocessing...\n")
    cat("This may take several minutes.\n")

    eset_rma <- rma(raw_data)

    ## --------------------------------------------------------
    ## Extract expression matrix
    ## --------------------------------------------------------

    expr_matrix <- exprs(eset_rma)

    cat("\nRMA complete.\n")
    cat(
        "Expression matrix:",
        nrow(expr_matrix),
        "probes x",
        ncol(expr_matrix),
        "samples\n"
    )

    cat("\nExpression summary:\n")
    print(summary(as.vector(expr_matrix)))

    ## --------------------------------------------------------
    ## Save
    ## --------------------------------------------------------

    saveRDS(
        eset_rma,
        output_rds
    )

    write.csv(
        expr_matrix,
        output_csv
    )

    cat("\nSaved:\n")
    cat(output_rds, "\n")
    cat(output_csv, "\n")

    return(eset_rma)
}

## ============================================================
## GSE48350
## ============================================================

gse48350 <- process_CEL(
    cel_dir = "01_GEO_Data/CEL/GSE48350",
    output_rds = "03_Preprocessing/GSE48350_rawCEL_RMA.rds",
    output_csv = "03_Preprocessing/GSE48350_rawCEL_RMA.csv",
    dataset_name = "GSE48350 — 62 hippocampus samples"
)

## ============================================================
## GSE5281
## ============================================================

gse5281 <- process_CEL(
    cel_dir = "01_GEO_Data/CEL/GSE5281",
    output_rds = "03_Preprocessing/GSE5281_rawCEL_RMA.rds",
    output_csv = "03_Preprocessing/GSE5281_rawCEL_RMA.csv",
    dataset_name = "GSE5281 — 23 validation samples"
)

cat("\n============================================\n")
cat("RAW CEL PREPROCESSING COMPLETE\n")
cat("============================================\n")
