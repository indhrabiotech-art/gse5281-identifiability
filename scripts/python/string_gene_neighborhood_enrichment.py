import os
import requests
import pandas as pd

OUTDIR = "10_STRING/results"
os.makedirs(OUTDIR, exist_ok=True)

SPECIES = 9606

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

INPUT = (
    "10_STRING/results/"
    "STRING_three_gene_first_shell_network.csv"
)

print("=" * 80)
print("STRING GENE-SPECIFIC FIRST-SHELL FUNCTIONAL ENRICHMENT")
print("=" * 80)

network = pd.read_csv(INPUT)

required = [
    "preferredName_A",
    "preferredName_B",
    "score"
]

missing = [c for c in required if c not in network.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

all_results = []

for gene in GENES:

    print("\n" + "=" * 80)
    print(f"ANALYSING: {gene}")
    print("=" * 80)

    # --------------------------------------------------------
    # Identify first-shell neighbors
    # --------------------------------------------------------

    neighbors_a = network.loc[
        network["preferredName_A"] == gene,
        "preferredName_B"
    ].tolist()

    neighbors_b = network.loc[
        network["preferredName_B"] == gene,
        "preferredName_A"
    ].tolist()

    neighbors = sorted(
        set(neighbors_a + neighbors_b)
    )

    print(f"First-shell neighbors: {len(neighbors)}")

    if not neighbors:
        print("No neighbors found.")
        continue

    print("Neighbors:")
    for n in neighbors:
        print(" ", n)

    # Include seed gene itself
    query_genes = [gene] + neighbors

    identifiers = "%0d".join(query_genes)

    url = (
        "https://string-db.org/api/tsv/enrichment"
        f"?identifiers={identifiers}"
        f"&species={SPECIES}"
        "&caller_identity=AD_three_gene_project"
    )

    print("\nQuerying STRING enrichment...")

    response = requests.get(
        url,
        timeout=60
    )

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("STRING request failed.")
        continue

    if not response.text.strip():
        print("No enrichment returned.")
        continue

    from io import StringIO

    result = pd.read_csv(
        StringIO(response.text),
        sep="\t"
    )

    if result.empty:
        print("No enrichment terms returned.")
        continue

    result["Seed_Gene"] = gene
    result["Neighborhood_Size"] = len(query_genes)

    all_results.append(result)

    safe_name = gene

    outfile = os.path.join(
        OUTDIR,
        f"STRING_{safe_name}_neighborhood_enrichment.csv"
    )

    result.to_csv(
        outfile,
        index=False
    )

    print(
        f"Saved: {outfile}"
    )

    # --------------------------------------------------------
    # Display strongest terms
    # --------------------------------------------------------

    pvalue_column = None

    for candidate in [
        "fdr",
        "p_value"
    ]:
        if candidate in result.columns:
            pvalue_column = candidate
            break

    if pvalue_column:

        display_cols = [
            c for c in [
                "category",
                "term",
                "description",
                "number_of_genes",
                "number_of_genes_in_background",
                "p_value",
                "fdr"
            ]
            if c in result.columns
        ]

        print("\nTop enrichment terms:")

        print(
            result
            .sort_values(pvalue_column)
            [display_cols]
            .head(10)
            .to_string(index=False)
        )

# ============================================================
# COMBINE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("COMBINING NEIGHBORHOOD ENRICHMENT RESULTS")
print("=" * 80)

if all_results:

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined_file = os.path.join(
        OUTDIR,
        "STRING_gene_specific_neighborhood_enrichment.csv"
    )

    combined.to_csv(
        combined_file,
        index=False
    )

    print(
        f"Combined results saved:\n{combined_file}"
    )

    # --------------------------------------------------------
    # Significant terms
    # --------------------------------------------------------

    if "fdr" in combined.columns:

        significant = combined[
            combined["fdr"] < 0.05
        ].copy()

        significant_file = os.path.join(
            OUTDIR,
            "STRING_gene_specific_significant_enrichment.csv"
        )

        significant.to_csv(
            significant_file,
            index=False
        )

        print(
            f"Significant enrichment saved:\n"
            f"{significant_file}"
        )

        print("\nSignificant terms:")

        if significant.empty:
            print(
                "No terms reached FDR < 0.05."
            )
        else:
            print(
                significant[
                    [
                        c for c in [
                            "Seed_Gene",
                            "category",
                            "term",
                            "description",
                            "fdr"
                        ]
                        if c in significant.columns
                    ]
                ]
                .sort_values("fdr")
                .to_string(index=False)
            )

else:

    print(
        "\nNo neighborhood enrichment results "
        "were returned for any seed gene."
    )

print("\n" + "=" * 80)
print("STRING GENE-SPECIFIC ENRICHMENT COMPLETE")
print("=" * 80)

