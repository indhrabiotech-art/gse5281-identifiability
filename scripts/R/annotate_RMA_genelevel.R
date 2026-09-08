library(Biobase)
library(hgu133plus2.db)
library(AnnotationDbi)

setwd("~/project_ml")

dir.create(
    "03_Preprocessing",
    showWarnings = FALSE
)

annotate_rma <- function(
    rds_file,
    output_rds,
    output_csv,
    dataset_name
) {

    cat("\n============================================\n")
    cat("RMA PROBE -> GENE:", dataset_name, "\n")
    cat("============================================\n")

    # --------------------------------------------------------
    # Load RMA ExpressionSet
    # --------------------------------------------------------

    eset <- readRDS(rds_file)
    expr <- exprs(eset)

    cat(
        "Input:",
        paste(dim(expr), collapse = " x "),
        "\n"
    )

    # --------------------------------------------------------
    # Probe IDs
    # --------------------------------------------------------

    probe_ids <- rownames(expr)

    # --------------------------------------------------------
    # Probe -> gene symbol
    # --------------------------------------------------------

    cat("\nMapping GPL570 probes to gene symbols...\n")

    mapping <- AnnotationDbi::select(
        hgu133plus2.db,
        keys = probe_ids,
        columns = "SYMBOL",
        keytype = "PROBEID"
    )

    cat(
        "Mapping rows returned:",
        nrow(mapping),
        "\n"
    )

    # --------------------------------------------------------
    # Remove unmapped probes
    # --------------------------------------------------------

    mapping <- mapping[
        !is.na(mapping$SYMBOL) &
        mapping$SYMBOL != "",
        ,
        drop = FALSE
    ]

    cat(
        "Mapped probe records:",
        nrow(mapping),
        "\n"
    )

    # --------------------------------------------------------
    # One mapping per probe
    # --------------------------------------------------------

    mapping <- mapping[
        !duplicated(mapping$PROBEID),
        ,
        drop = FALSE
    ]

    # --------------------------------------------------------
    # Keep only probes that have expression data
    # --------------------------------------------------------

    mapping <- mapping[
        mapping$PROBEID %in% rownames(expr),
        ,
        drop = FALSE
    ]

    # --------------------------------------------------------
    # Build annotation dataframe
    # --------------------------------------------------------

    annot <- data.frame(
        PROBEID = mapping$PROBEID,
        SYMBOL = mapping$SYMBOL,
        stringsAsFactors = FALSE
    )

    annot$mean_expression <- rowMeans(
        expr[annot$PROBEID, , drop = FALSE],
        na.rm = TRUE
    )

    # --------------------------------------------------------
    # Collapse multiple probes per gene
    #
    # Rule:
    # retain probe with highest mean RMA expression
    # --------------------------------------------------------

    annot <- annot[
        order(
            annot$SYMBOL,
            -annot$mean_expression
        ),
        ,
        drop = FALSE
    ]

    best_probe <- annot[
        !duplicated(annot$SYMBOL),
        ,
        drop = FALSE
    ]

    cat(
        "\nUnique genes after probe collapsing:",
        nrow(best_probe),
        "\n"
    )

    # --------------------------------------------------------
    # Build gene-level matrix
    # --------------------------------------------------------

    gene_expr <- expr[
        best_probe$PROBEID,
        ,
        drop = FALSE
    ]

    rownames(gene_expr) <- best_probe$SYMBOL

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    stopifnot(
        nrow(gene_expr) == nrow(best_probe)
    )

    stopifnot(
        !anyDuplicated(rownames(gene_expr))
    )

    stopifnot(
        identical(
            colnames(gene_expr),
            sampleNames(eset)
        )
    )

    cat(
        "\nFinal gene-level matrix:",
        paste(dim(gene_expr), collapse = " x "),
        "\n"
    )

    # --------------------------------------------------------
    # Check important neuroinflammatory genes
    # --------------------------------------------------------

    markers <- c(
        "TREM2",
        "TYROBP",
        "C1QA",
        "C1QB",
        "C1QC",
        "C3",
        "NLRP3",
        "IL1B",
        "IL6",
        "TNF",
        "CD68"
    )

    present <- markers[
        markers %in% rownames(gene_expr)
    ]

    missing <- setdiff(
        markers,
        present
    )

    cat(
        "\nNeuroinflammatory markers present:\n"
    )

    print(present)

    if (length(missing) > 0) {

        cat(
            "\nNot present:\n"
        )

        print(missing)

    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    saveRDS(
        gene_expr,
        output_rds
    )

    write.csv(
        gene_expr,
        output_csv,
        quote = FALSE
    )

    cat("\nSaved:\n")
    cat(output_rds, "\n")
    cat(output_csv, "\n")
}


# ============================================================
# GSE48350
# ============================================================

annotate_rma(
    "03_Preprocessing/GSE48350_rawCEL_RMA.rds",
    "03_Preprocessing/GSE48350_RMA_genelevel.rds",
    "03_Preprocessing/GSE48350_RMA_genelevel.csv",
    "GSE48350"
)


# ============================================================
# GSE5281
# ============================================================

annotate_rma(
    "03_Preprocessing/GSE5281_rawCEL_RMA.rds",
    "03_Preprocessing/GSE5281_RMA_genelevel.rds",
    "03_Preprocessing/GSE5281_RMA_genelevel.csv",
    "GSE5281"
)


cat(
    "\n============================================\n",
    "RMA GENE-LEVEL ANNOTATION COMPLETE\n",
    "============================================\n"
)

