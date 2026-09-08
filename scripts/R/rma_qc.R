library(Biobase)

setwd("~/project_ml")

dir.create(
    "07_Figures/RMA_QC",
    recursive = TRUE,
    showWarnings = FALSE
)

run_qc <- function(
    rds_file,
    metadata_file,
    dataset_name
) {

    cat("\n============================================\n")
    cat("RMA QC:", dataset_name, "\n")
    cat("============================================\n")

    # --------------------------------------------------------
    # Load RMA expression data
    # --------------------------------------------------------

    eset <- readRDS(rds_file)
    expr <- exprs(eset)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    meta <- read.csv(
        metadata_file,
        stringsAsFactors = FALSE,
        check.names = FALSE
    )

    # --------------------------------------------------------
    # Extract GSM IDs from RMA sample names
    # --------------------------------------------------------

    gsm <- sub(
        "^(GSM[0-9]+).*",
        "\\1",
        sampleNames(eset),
        ignore.case = TRUE
    )

    # --------------------------------------------------------
    # Check metadata GSM column
    # --------------------------------------------------------

    if (!"GSM" %in% colnames(meta)) {
        stop(
            "Metadata file has no GSM column: ",
            metadata_file
        )
    }

    # Align metadata to expression matrix
    meta <- meta[match(gsm, meta$GSM), , drop = FALSE]

    if (any(is.na(meta$GSM))) {
        stop(
            "Some RMA GSM IDs were not found in metadata for ",
            dataset_name
        )
    }

    stopifnot(
        identical(
            gsm,
            meta$GSM
        )
    )

    # --------------------------------------------------------
    # Find diagnosis column
    # --------------------------------------------------------

    diagnosis_column <- NULL

    if ("Diagnosis" %in% colnames(meta)) {
        diagnosis_column <- "Diagnosis"
    } else if ("diagnosis" %in% colnames(meta)) {
        diagnosis_column <- "diagnosis"
    }

    if (is.null(diagnosis_column)) {
        stop(
            "No Diagnosis/diagnosis column found in metadata for ",
            dataset_name
        )
    }

    diagnosis <- as.character(
        meta[[diagnosis_column]]
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    cat(
        "Expression matrix:",
        paste(dim(expr), collapse = " x "),
        "\n"
    )

    cat(
        "Metadata rows:",
        nrow(meta),
        "\n"
    )

    cat(
        "Diagnosis column:",
        diagnosis_column,
        "\n"
    )

    cat("\nDiagnosis:\n")
    print(table(diagnosis))

    # ========================================================
    # 1. BOX PLOT
    # ========================================================

    png(
        paste0(
            "07_Figures/RMA_QC/",
            dataset_name,
            "_boxplot.png"
        ),
        width = 1600,
        height = 900
    )

    boxplot(
        expr,
        outline = FALSE,
        las = 2,
        cex.axis = 0.6,
        main = paste0(
            dataset_name,
            " — RMA expression distribution"
        ),
        ylab = "RMA log2 expression"
    )

    dev.off()

    # ========================================================
    # 2. DENSITY PLOT
    # ========================================================

    png(
        paste0(
            "07_Figures/RMA_QC/",
            dataset_name,
            "_density.png"
        ),
        width = 1200,
        height = 800
    )

    d <- density(expr[, 1])

    plot(
        d,
        main = paste0(
            dataset_name,
            " — RMA density"
        ),
        xlab = "RMA log2 expression"
    )

    if (ncol(expr) > 1) {
        for (i in 2:ncol(expr)) {
            lines(
                density(expr[, i])
            )
        }
    }

    dev.off()

    # ========================================================
    # 3. SAMPLE-SAMPLE CORRELATION
    # ========================================================

    sample_cor <- cor(expr)

    png(
        paste0(
            "07_Figures/RMA_QC/",
            dataset_name,
            "_sample_correlation.png"
        ),
        width = 1000,
        height = 900
    )

    heatmap(
        sample_cor,
        symm = TRUE,
        main = paste0(
            dataset_name,
            " — sample correlation"
        ),
        labRow = diagnosis,
        labCol = FALSE
    )

    dev.off()

    # ========================================================
    # 4. PCA
    # ========================================================

    pca <- prcomp(
        t(expr),
        scale. = TRUE
    )

    variance <- 100 *
        pca$sdev^2 /
        sum(pca$sdev^2)

    diagnosis_factor <- factor(diagnosis)

    png(
        paste0(
            "07_Figures/RMA_QC/",
            dataset_name,
            "_PCA.png"
        ),
        width = 1000,
        height = 800
    )

    plot(
        pca$x[, 1],
        pca$x[, 2],
        pch = 19,
        cex = 1.3,
        col = ifelse(
            diagnosis_factor == "AD",
            "tomato",
            "steelblue"
        ),
        xlab = paste0(
            "PC1 (",
            round(variance[1], 1),
            "%)"
        ),
        ylab = paste0(
            "PC2 (",
            round(variance[2], 1),
            "%)"
        ),
        main = paste0(
            dataset_name,
            " — RMA PCA"
        )
    )

    legend(
        "topright",
        legend = c(
            "AD",
            "Control"
        ),
        col = c(
            "tomato",
            "steelblue"
        ),
        pch = 19
    )

    dev.off()

    # ========================================================
    # 5. SAMPLE CORRELATION OUTLIER CHECK
    # ========================================================

    avg_cor <- rowMeans(
        sample_cor
    )

    threshold <- mean(avg_cor) -
        3 * sd(avg_cor)

    outliers <- meta$GSM[
        avg_cor < threshold
    ]

    cat("\nAverage sample correlation:\n")
    print(summary(avg_cor))

    cat(
        "\nCorrelation outlier threshold:",
        round(threshold, 4),
        "\n"
    )

    if (length(outliers) == 0) {

        cat(
            "No correlation outliers detected.\n"
        )

    } else {

        cat(
            "Potential correlation outliers:\n"
        )

        print(outliers)

    }

    # ========================================================
    # 6. SAVE PCA SCORES
    # ========================================================

    pca_df <- data.frame(
        GSM = gsm,
        Diagnosis = diagnosis,
        PC1 = pca$x[, 1],
        PC2 = pca$x[, 2]
    )

    write.csv(
        pca_df,
        paste0(
            "07_Figures/RMA_QC/",
            dataset_name,
            "_PCA_scores.csv"
        ),
        row.names = FALSE
    )

    cat(
        "\nQC figures saved under:",
        "07_Figures/RMA_QC/\n"
    )
}


# ============================================================
# GSE48350
# ============================================================

run_qc(
    "03_Preprocessing/GSE48350_rawCEL_RMA.rds",
    "03_Preprocessing/GSE48350_hippocampus_metadata_aligned.csv",
    "GSE48350"
)


# ============================================================
# GSE5281
# ============================================================

run_qc(
    "03_Preprocessing/GSE5281_rawCEL_RMA.rds",
    "02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv",
    "GSE5281"
)


cat(
    "\n============================================\n",
    "RMA QC COMPLETE\n",
    "============================================\n"
)
