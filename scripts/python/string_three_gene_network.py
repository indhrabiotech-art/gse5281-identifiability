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
REQUIRED_SCORE = 700

print("=" * 75)
print("STRING THREE-GENE NETWORK ANALYSIS")
print("=" * 75)

print("\nGenes:")
for gene in GENES:
    print(" ", gene)

# ------------------------------------------------------------
# STRING network API
# ------------------------------------------------------------

identifiers = "%0d".join(GENES)

url = (
    "https://string-db.org/api/tsv/network"
    f"?identifiers={identifiers}"
    f"&species={SPECIES}"
    f"&required_score={REQUIRED_SCORE}"
    "&network_type=functional"
    "&caller_identity=AD_three_gene_project"
)

print("\nQuerying STRING...")
print("Species: Homo sapiens (9606)")
print("Required confidence score:", REQUIRED_SCORE)
print("Network type: functional")

response = requests.get(url, timeout=60)

print("HTTP status:", response.status_code)

if response.status_code != 200:
    raise RuntimeError(
        f"STRING API request failed: HTTP {response.status_code}\n"
        f"{response.text[:1000]}"
    )

outfile = os.path.join(
    OUTDIR,
    "STRING_three_gene_network.tsv"
)

with open(outfile, "w") as f:
    f.write(response.text)

print("\nSaved raw STRING network:")
print(outfile)

# ------------------------------------------------------------
# Read and summarize
# ------------------------------------------------------------

from io import StringIO

df = pd.read_csv(
    StringIO(response.text),
    sep="\t"
)

print("\nNetwork shape:", df.shape)

if df.empty:
    print("\nWARNING: STRING returned no associations at this threshold.")
    print("This is itself an important biological result.")
else:

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nNetwork edges:")

    cols = [
        "preferredName_A",
        "preferredName_B",
        "score"
    ]

    available = [
        c for c in cols
        if c in df.columns
    ]

    print(
        df[available]
        .sort_values("score", ascending=False)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Internal three-gene edges
    # --------------------------------------------------------

    gene_set = set(GENES)

    internal = df[
        df["preferredName_A"].isin(gene_set)
        &
        df["preferredName_B"].isin(gene_set)
    ].copy()

    print("\n" + "=" * 75)
    print("INTERNAL THREE-GENE ASSOCIATIONS")
    print("=" * 75)

    if internal.empty:
        print("No direct STRING associations among the three genes")
        print("at the selected confidence threshold.")
    else:
        print(
            internal[
                [
                    "preferredName_A",
                    "preferredName_B",
                    "score"
                ]
            ]
            .sort_values("score", ascending=False)
            .to_string(index=False)
        )

    internal.to_csv(
        os.path.join(
            OUTDIR,
            "STRING_three_gene_internal_edges.csv"
        ),
        index=False
    )

print("\n" + "=" * 75)
print("STRING NETWORK ANALYSIS COMPLETE")
print("=" * 75)
