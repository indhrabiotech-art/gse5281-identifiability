import os
import pandas as pd

OUT = "05_Biology/Final_Summary"
os.makedirs(OUT, exist_ok=True)

genes = ["ABCA6", "CRLF1", "TNFRSF11B"]

# ------------------------------------------------------------
# ML statistics
# ------------------------------------------------------------

ml = pd.read_csv(
    "05_Biology/final_3gene_biology_input.csv"
)

# ------------------------------------------------------------
# PPI degree
# ------------------------------------------------------------

ppi = pd.read_csv(
    "05_Biology/PPI_FirstShell/signature_gene_network_degree.csv"
)

summary = ml.merge(
    ppi,
    on="gene",
    how="left"
)

# ------------------------------------------------------------
# Biological interpretation based strictly on
# results already generated in this project
# ------------------------------------------------------------

modules = {
    "ABCA6":
        "ABC transporter / lipid transport module; "
        "connected to ABCA8, ABCA9 and MAP2K6",

    "CRLF1":
        "Cytokine and neurotrophic signaling module; "
        "associated with CLCF1, CNTFR and IL27",

    "TNFRSF11B":
        "TNF/TNFR and osteoclast-related signaling module; "
        "connected to TNFRSF11A and MAP2K6"
}

summary["biological_module"] = summary["gene"].map(modules)

summary["interpretation"] = [
    "Candidate lipid/transport-associated signature component",
    "Candidate cytokine/neurotrophic signaling-associated component",
    "Candidate TNF/TNFR and immune-remodeling-associated component"
]

# ------------------------------------------------------------
# Save integrated table
# ------------------------------------------------------------

summary.to_csv(
    f"{OUT}/final_3gene_biological_evidence_table.csv",
    index=False
)

# ------------------------------------------------------------
# Create concise manuscript-style table
# ------------------------------------------------------------

manuscript = summary[
    [
        "gene",
        "coefficient",
        "train_AUC",
        "test_AUC",
        "external_AUC",
        "external_delta",
        "network_degree",
        "biological_module"
    ]
].copy()

manuscript.to_csv(
    f"{OUT}/manuscript_3gene_biology_table.csv",
    index=False
)

# ------------------------------------------------------------
# Text summary
# ------------------------------------------------------------

with open(
    f"{OUT}/final_3gene_biology_summary.txt",
    "w"
) as f:

    f.write(
        "FINAL 3-GENE BIOLOGICAL CHARACTERIZATION\n"
        "=========================================\n\n"
    )

    f.write(
        "Candidate signature:\n"
        "ABCA6, CRLF1, TNFRSF11B\n\n"
    )

    f.write(
        "The three genes show distinct but biologically "
        "complementary network associations.\n\n"
    )

    for gene in genes:

        row = summary[
            summary["gene"] == gene
        ].iloc[0]

        f.write(f"{gene}\n")
        f.write("-" * 50 + "\n")

        f.write(
            f"Model coefficient: {row['coefficient']:.4f}\n"
        )

        f.write(
            f"Internal test AUC: {row['test_AUC']:.4f}\n"
        )

        f.write(
            f"External AUC: {row['external_AUC']:.4f}\n"
        )

        f.write(
            f"External AD-Control delta: "
            f"{row['external_delta']:.4f}\n"
        )

        f.write(
            f"STRING first-shell degree: "
            f"{int(row['network_degree'])}\n"
        )

        f.write(
            f"Biological module: "
            f"{row['biological_module']}\n\n"
        )

    f.write(
        "Overall interpretation\n"
        "----------------------\n"
        "The final 3-gene signature should be treated as an "
        "externally evaluated candidate Alzheimer's disease "
        "gene signature rather than a clinically validated "
        "diagnostic biomarker. The ML analysis demonstrates "
        "cross-dataset discrimination, while enrichment and "
        "STRING analyses provide complementary biological "
        "context. Because the external cohort contains only "
        "23 samples, biological and predictive conclusions "
        "require independent validation in larger cohorts.\n"
    )

print("=" * 70)
print("FINAL BIOLOGICAL SUMMARY GENERATED")
print("=" * 70)

print("\nIntegrated evidence:")
print(summary.to_string(index=False))

print("\nSaved:")
print(f"{OUT}/final_3gene_biological_evidence_table.csv")
print(f"{OUT}/manuscript_3gene_biology_table.csv")
print(f"{OUT}/final_3gene_biology_summary.txt")

print("\n" + "=" * 70)
print("BIOLOGICAL INTEGRATION COMPLETE")
print("=" * 70)
