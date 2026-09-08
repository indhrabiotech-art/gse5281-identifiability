import os
import requests
import pandas as pd

OUTDIR = "10_STRING/results"
os.makedirs(OUTDIR, exist_ok=True)

SPECIES = 9606

SEED_GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

# ------------------------------------------------------------
# Load first-shell network
# ------------------------------------------------------------

network_file = (
    "10_STRING/results/"
    "STRING_three_gene_first_shell_network.csv"
)

network = pd.read_csv(network_file)

print("=" * 75)
print("STRING FUNCTIONAL ENRICHMENT ANALYSIS")
print("=" * 75)

print("\nSeed genes:")
for gene in SEED_GENES:
    print(" ", gene)

# ------------------------------------------------------------
# Build expanded protein set
# ------------------------------------------------------------

proteins = set(SEED_GENES)

for col in ["preferredName_A", "preferredName_B"]:
    if col in network.columns:
        proteins.update(
            network[col]
            .dropna()
            .astype(str)
            .tolist()
        )

proteins = sorted(proteins)

print("\nExpanded protein set:")
print("N =", len(proteins))

for p in proteins:
    print(" ", p)

# ------------------------------------------------------------
# STRING enrichment API
# ------------------------------------------------------------

identifiers = "%0d".join(proteins)

url = (
    "https://string-db.org/api/tsv/enrichment"
    f"?identifiers={identifiers}"
    f"&species={SPECIES}"
    "&caller_identity=AD_three_gene_enrichment"
)

print("\nQuerying STRING enrichment API...")

response = requests.get(
    url,
    timeout=60
)

print("HTTP status:", response.status_code)

if response.status_code != 200:
    raise RuntimeError(
        f"STRING enrichment request failed: "
        f"HTTP {response.status_code}\n"
        f"{response.text[:1000]}"
    )

raw_file = os.path.join(
    OUTDIR,
    "STRING_three_gene_first_shell_enrichment_raw.tsv"
)

with open(raw_file, "w") as f:
    f.write(response.text)

print("\nRaw enrichment saved:")
print(raw_file)

# ------------------------------------------------------------
# Parse enrichment
# ------------------------------------------------------------

from io import StringIO

enrich = pd.read_csv(
    StringIO(response.text),
    sep="\t"
)

print("\nEnrichment shape:", enrich.shape)

if enrich.empty:
    print("\nNo enrichment terms returned.")
    raise SystemExit(0)

print("\nAvailable columns:")
print(enrich.columns.tolist())

# ------------------------------------------------------------
# Save complete enrichment
# ------------------------------------------------------------

complete_file = os.path.join(
    OUTDIR,
    "STRING_three_gene_first_shell_enrichment.csv"
)

enrich.to_csv(
    complete_file,
    index=False
)

# ------------------------------------------------------------
# Display strongest biological terms
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("TOP FUNCTIONAL ENRICHMENT TERMS")
print("=" * 75)

# STRING enrichment normally provides:
# category, term, description, number_of_genes,
# number_of_genes_in_background, p_value, fdr

preferred = [
    "category",
    "term",
    "description",
    "number_of_genes",
    "number_of_genes_in_background",
    "p_value",
    "fdr"
]

available = [
    c for c in preferred
    if c in enrich.columns
]

if "fdr" in enrich.columns:

    display = (
        enrich
        .sort_values("fdr")
        .head(30)
    )

elif "p_value" in enrich.columns:

    display = (
        enrich
        .sort_values("p_value")
        .head(30)
    )

else:

    display = enrich.head(30)

print(
    display[available]
    .to_string(index=False)
)

# ------------------------------------------------------------
# FDR-significant terms
# ------------------------------------------------------------

if "fdr" in enrich.columns:

    significant = enrich[
        pd.to_numeric(
            enrich["fdr"],
            errors="coerce"
        ) < 0.05
    ].copy()

else:

    significant = pd.DataFrame()

print("\n" + "=" * 75)
print("FDR < 0.05 ENRICHMENT")
print("=" * 75)

if significant.empty:

    print(
        "No terms reached FDR < 0.05."
    )

else:

    print(
        significant[available]
        .sort_values("fdr")
        .to_string(index=False)
    )

significant.to_csv(
    os.path.join(
        OUTDIR,
        "STRING_significant_enrichment_FDR05.csv"
    ),
    index=False
)

print("\nSaved:")
print(complete_file)

print(
    "10_STRING/results/"
    "STRING_significant_enrichment_FDR05.csv"
)

print("\n" + "=" * 75)
print("STRING FUNCTIONAL ENRICHMENT COMPLETE")
print("=" * 75)
