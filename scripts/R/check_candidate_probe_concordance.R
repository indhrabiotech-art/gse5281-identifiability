library(Biobase)
library(hgu133plus2.db)
library(AnnotationDbi)

setwd("~/project_ml")

genes <- c(
    "ABCA6",
    "CRLF1",
    "TNFRSF11B",
    "PLA2G7",
    "SORBS1",
    "PMS2P2",
    "TAB2"
)

check_dataset <- function(rds_file, dataset_name) {

    cat("\n")
    cat("============================================================\n")
    cat(dataset_name, "\n")
    cat("============================================================\n")

    eset <- readRDS(rds_file)

    expr <- exprs(eset)

    probe_ids <- rownames(expr)

    mapping <- AnnotationDbi::select(
        hgu133plus2.db,
        keys = probe_ids,
        columns = "SYMBOL",
        keytype = "PROBEID"
    )

    mapping <- mapping[
        !is.na(mapping$SYMBOL) &
        mapping$SYMBOL != "",
        ,
        drop = FALSE
    ]

    mapping <- mapping[
        !duplicated(mapping$PROBEID),
        ,
        drop = FALSE
    ]

    mapping <- mapping[
        mapping$SYMBOL %in% genes,
        ,
        drop = FALSE
    ]

    mapping$mean_expression <- rowMeans(
        expr[mapping$PROBEID, , drop = FALSE],
        na.rm = TRUE
    )

    mapping <- mapping[
        order(
            mapping$SYMBOL,
            -mapping$mean_expression
        ),
        ,
        drop = FALSE
    ]

    for (gene in genes) {

        x <- mapping[
            mapping$SYMBOL == gene,
            ,
            drop = FALSE
        ]

        cat("\nGene:", gene, "\n")

        if (nrow(x) == 0) {
            cat("  NO PROBE FOUND\n")
            next
        }

        print(
            x[
                c(
                    "PROBEID",
                    "SYMBOL",
                    "mean_expression"
                )
            ],
            row.names = FALSE
        )

        cat(
            "  SELECTED PROBE:",
            x$PROBEID[1],
            "\n"
        )
    }

    return(mapping)
}


gse48350 <- check_dataset(
    "03_Preprocessing/GSE48350_rawCEL_RMA.rds",
    "GSE48350"
)

gse5281 <- check_dataset(
    "03_Preprocessing/GSE5281_rawCEL_RMA.rds",
    "GSE5281"
)


# ------------------------------------------------------------
# Compare selected probes
# ------------------------------------------------------------

selected_48350 <- gse48350[
    !duplicated(gse48350$SYMBOL),
    c("SYMBOL", "PROBEID")
]

selected_5281 <- gse5281[
    !duplicated(gse5281$SYMBOL),
    c("SYMBOL", "PROBEID")
]

comparison <- merge(
    selected_48350,
    selected_5281,
    by = "SYMBOL",
    suffixes = c("_GSE48350", "_GSE5281")
)

comparison$probe_same <- (
    comparison$PROBEID_GSE48350 ==
    comparison$PROBEID_GSE5281
)

cat("\n")
cat("============================================================\n")
cat("PROBE CONCORDANCE SUMMARY\n")
cat("============================================================\n")

print(comparison)

cat("\nSame probe:", sum(comparison$probe_same), "/", nrow(comparison), "\n")

cat(
    "Different probe:",
    sum(!comparison$probe_same),
    "/",
    nrow(comparison),
    "\n"
)

write.csv(
    comparison,
    "04_ML/External_Validation/"
    "GSE48350_GSE5281_candidate_probe_concordance.csv",
    row.names = FALSE
)

cat("\nSaved:\n")
cat(
    "04_ML/External_Validation/"
    "GSE48350_GSE5281_candidate_probe_concordance.csv\n"
)

cat("\n============================================================\n")
cat("PROBE CONCORDANCE CHECK COMPLETE\n")
cat("============================================================\n")
