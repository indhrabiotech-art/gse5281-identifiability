import os
import requests
import pandas as pd

from io import StringIO

OUTDIR = "10_STRING/results"
os.makedirs(OUTDIR, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

SPECIES = 9606

# Moderate-high confidence for neighborhood discovery
REQUIRED_SCORE = 400

# Number of additional interactors per seed
ADDITIONAL_NODES = 10

print("=" * 75)
print("STRING FIRST-SHELL NETWORK ANALYSIS")
print("=" * 75)

print("\nSeed genes:")
for gene in GENES:
    print(" ", gene)

print("\nSpecies: Homo sapiens")
print("STRING score threshold:", REQUIRED_SCORE)
print("Additional interactors per seed:", ADDITIONAL_NODES)

# ------------------------------------------------------------
# STRING API
# ------------------------------------------------------------

identifiers = "%0d".join(GENES)

url = (
    "https://string-db.org/api/tsv/network"
    f"?identifiers={identifiers}"
    f"&species={SPECIES}"
    f"&required_score={REQUIRED_SCORE}"
    f"&add_nodes={ADDITIONAL_NODES}"
    "&network_type=functional"
    "&caller_identity=AD_three_gene_first_shell"
)

print("\nQuerying STRING...")

response = requests.get(
    url,
    timeout=60
)

print("HTTP status:", response.status_code)

if response.status_code != 200:
    raise RuntimeError(
        f"STRING API request failed: HTTP {response.status_code}\n"
        f"{response.text[:1000]}"
    )

raw_file = os.path.join(
    OUTDIR,
    "STRING_three_gene_first_shell_raw.tsv"
)

with open(raw_file, "w") as f:
    f.write(response.text)

print("\nSaved:")
print(raw_file)

# ------------------------------------------------------------
# Parse network
# ------------------------------------------------------------

df = pd.read_csv(
    StringIO(response.text),
    sep="\t"
)

print("\nNetwork shape:", df.shape)

if df.empty:
    print("\nWARNING: STRING returned no network edges.")
    raise SystemExit(0)

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# Save complete network
# ------------------------------------------------------------

network_file = os.path.join(
    OUTDIR,
    "STRING_three_gene_first_shell_network.csv"
)

df.to_csv(
    network_file,
    index=False
)

print("\nComplete network saved:")
print(network_file)

# ------------------------------------------------------------
# Identify seed-containing edges
# ------------------------------------------------------------

seed_set = set(GENES)

seed_edges = df[
    df["preferredName_A"].isin(seed_set)
    |
    df["preferredName_B"].isin(seed_set)
].copy()

seed_edges = seed_edges.sort_values(
    "score",
    ascending=False
)

print("\n" + "=" * 75)
print("SEED-GENE ASSOCIATIONS")
print("=" * 75)

print(
    seed_edges[
        [
            "preferredName_A",
            "preferredName_B",
            "score"
        ]
    ].to_string(index=False)
)

seed_edges.to_csv(
    os.path.join(
        OUTDIR,
        "STRING_seed_gene_edges.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# Neighbor frequency
# ------------------------------------------------------------

neighbors = []

for gene in GENES:

    subset = df[
        (
            df["preferredName_A"] == gene
        )
        |
        (
            df["preferredName_B"] == gene
        )
    ]

    for _, row in subset.iterrows():

        if row["preferredName_A"] == gene:
            neighbor = row["preferredName_B"]
        else:
            neighbor = row["preferredName_A"]

        neighbors.append(
            {
                "Seed_gene": gene,
                "Neighbor": neighbor,
                "Score": row["score"]
            }
        )

neighbors_df = pd.DataFrame(neighbors)

if not neighbors_df.empty:

    neighbor_summary = (
        neighbors_df
        .groupby("Neighbor")
        .agg(
            Seed_count=("Seed_gene", "nunique"),
            Mean_score=("Score", "mean"),
            Max_score=("Score", "max")
        )
        .reset_index()
        .sort_values(
            ["Seed_count", "Mean_score"],
            ascending=False
        )
    )

else:

    neighbor_summary = pd.DataFrame(
        columns=[
            "Neighbor",
            "Seed_count",
            "Mean_score",
            "Max_score"
        ]
    )

print("\n" + "=" * 75)
print("SHARED NEIGHBOR SUMMARY")
print("=" * 75)

print(
    neighbor_summary.head(30).to_string(
        index=False
    )
)

neighbor_summary.to_csv(
    os.path.join(
        OUTDIR,
        "STRING_shared_neighbor_summary.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# Shared neighbors
# ------------------------------------------------------------

shared = neighbor_summary[
    neighbor_summary["Seed_count"] >= 2
].copy()

print("\n" + "=" * 75)
print("SHARED NEIGHBORS")
print("=" * 75)

if shared.empty:
    print(
        "No proteins were shared by >=2 seed genes "
        "at this threshold."
    )
else:
    print(
        shared.to_string(index=False)
    )

shared.to_csv(
    os.path.join(
        OUTDIR,
        "STRING_shared_neighbors.csv"
    ),
    index=False
)

print("\n" + "=" * 75)
print("FIRST-SHELL STRING ANALYSIS COMPLETE")
print("=" * 75)

print("\nGenerated files:")
print("1.", raw_file)
print("2.", network_file)
print("3. 10_STRING/results/STRING_seed_gene_edges.csv")
print("4. 10_STRING/results/STRING_shared_neighbor_summary.csv")
print("5. 10_STRING/results/STRING_shared_neighbors.csv")
