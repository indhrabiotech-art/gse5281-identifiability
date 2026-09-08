import os
import requests
import pandas as pd

OUTDIR = "10_STRING/results"
os.makedirs(OUTDIR, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

SPECIES = 9606

print("=" * 80)
print("STRING THREE-GENE FUNCTIONAL ENRICHMENT")
print("=" * 80)

print("\nInput genes:")
for g in GENES:
    print(" ", g)

# ------------------------------------------------------------
# STRING enrichment API
# ------------------------------------------------------------

identifiers = "%0d".join(GENES)

url = (
    "https://string-db.org/api/tsv/enrichment"
    f"?identifiers={identifiers}"
    f"&species={SPECIES}"
    "&caller_identity=AD_three_gene_project"
)

print("\nQuerying STRING enrichment...")
response = requests.get(url, timeout=60)

print("HTTP status:", response.status_code)

response.raise_for_status()

out = pd.read_csv(
    pd.io.common.StringIO(response.text),
    sep="\t"
)

print("\nReturned rows:", len(out))

# ------------------------------------------------------------
# Save raw enrichment
# ------------------------------------------------------------

raw_file = os.path.join(
    OUTDIR,
    "STRING_three_gene_functional_enrichment_raw.tsv"
)

out.to_csv(
    raw_file,
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# Display important columns
# ------------------------------------------------------------

preferred = [
    "category",
    "term",
    "description",
    "number_of_genes",
    "number_of_genes_in_background",
    "ncbiTaxonId",
    "p_value",
    "fdr",
    "genes"
]

available = [
    c for c in preferred
    if c in out.columns
]

if available:
    summary = out[available].copy()
else:
    summary = out.copy()

# Sort by FDR if available
if "fdr" in summary.columns:
    summary = summary.sort_values("fdr")

summary_file = os.path.join(
    OUTDIR,
    "STRING_three_gene_functional_enrichment.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print("\n" + "=" * 80)
print("FUNCTIONAL ENRICHMENT RESULTS")
print("=" * 80)

if len(summary) == 0:
    print("No enrichment terms returned.")

else:
    print(
        summary.head(30).to_string(index=False)
    )

    if "fdr" in summary.columns:
        significant = summary[
            summary["fdr"] < 0.05
        ]

        print("\n" + "=" * 80)
        print("FDR < 0.05 TERMS")
        print("=" * 80)

        if len(significant) == 0:
            print(
                "No terms reached FDR < 0.05."
            )
        else:
            print(
                significant.to_string(index=False)
            )

print("\nSaved:")
print(raw_file)
print(summary_file)

print("\n" + "=" * 80)
print("STRING FUNCTIONAL ENRICHMENT COMPLETE")
print("=" * 80)
