import os
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

OUT = "05_Biology/PPI_FirstShell"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("STRING FIRST-SHELL NETWORK — FINAL 3-GENE SIGNATURE")
print("=" * 70)

url = "https://string-db.org/api/tsv/network"

params = {
    "identifiers": "\r".join(GENES),
    "species": 9606,
    "required_score": 200,
    "add_nodes": 10,
    "network_type": "functional"
}

print("\nQuerying STRING first-shell network...")

response = requests.get(
    url,
    params=params,
    timeout=60
)

response.raise_for_status()

ppi = pd.read_csv(
    StringIO(response.text),
    sep="\t"
)

print("\nSTRING records:", len(ppi))

if len(ppi) == 0:
    print("\nNo STRING records returned.")
    print("Try a broader network query later.")
    raise SystemExit(0)

print("\nColumns:")
print(ppi.columns.tolist())

# ------------------------------------------------------------
# Save complete STRING output
# ------------------------------------------------------------

ppi.to_csv(
    f"{OUT}/STRING_first_shell_raw.csv",
    index=False
)

# ------------------------------------------------------------
# Simplified interaction table
# ------------------------------------------------------------

cols = [
    "preferredName_A",
    "preferredName_B",
    "score",
    "nscore",
    "fscore",
    "pscore",
    "ascore",
    "escore",
    "dscore",
    "tscore"
]

available = [c for c in cols if c in ppi.columns]

simple = ppi[available].copy()

simple = simple.sort_values(
    "score",
    ascending=False
)

simple.to_csv(
    f"{OUT}/STRING_first_shell_interactions.csv",
    index=False
)

print("\nTop STRING interactions:")
print(simple.head(30).to_string(index=False))

# ------------------------------------------------------------
# Identify signature genes in network
# ------------------------------------------------------------

nodes = set()

for col in ["preferredName_A", "preferredName_B"]:
    if col in ppi.columns:
        nodes.update(
            ppi[col].dropna().astype(str).tolist()
        )

signature_present = [
    g for g in GENES
    if g in nodes
]

print("\nSignature genes represented:")
for g in signature_present:
    print(" -", g)

# ------------------------------------------------------------
# Network construction
# ------------------------------------------------------------

G = nx.Graph()

for _, row in ppi.iterrows():

    a = row["preferredName_A"]
    b = row["preferredName_B"]

    if pd.isna(a) or pd.isna(b):
        continue

    G.add_edge(
        str(a),
        str(b),
        score=float(row["score"])
    )

print("\nNetwork:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# ------------------------------------------------------------
# Degree of signature genes
# ------------------------------------------------------------

degree_rows = []

for gene in GENES:

    degree_rows.append({
        "gene": gene,
        "network_degree": G.degree(gene)
        if gene in G
        else 0
    })

degree_df = pd.DataFrame(degree_rows)

print("\nSignature-gene network degree:")
print(degree_df.to_string(index=False))

degree_df.to_csv(
    f"{OUT}/signature_gene_network_degree.csv",
    index=False
)

# ------------------------------------------------------------
# Shared neighboring genes
# ------------------------------------------------------------

neighbor_sets = {}

for gene in GENES:

    if gene in G:
        neighbor_sets[gene] = set(G.neighbors(gene))
    else:
        neighbor_sets[gene] = set()

shared = (
    neighbor_sets[GENES[0]]
    & neighbor_sets[GENES[1]]
    & neighbor_sets[GENES[2]]
)

print("\nShared first-shell neighbors:")
print(sorted(shared))

pd.DataFrame({
    "shared_neighbor": sorted(shared)
}).to_csv(
    f"{OUT}/shared_first_shell_neighbors.csv",
    index=False
)

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

plt.figure(figsize=(12, 10))

pos = nx.spring_layout(
    G,
    seed=42,
    k=0.8
)

# Draw all network nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=700
)

# Draw edges
nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.45
)

# Label signature genes
label_nodes = {
    gene: gene
    for gene in GENES
    if gene in G
}

nx.draw_networkx_labels(
    G,
    pos,
    labels=label_nodes,
    font_size=12,
    font_weight="bold"
)

plt.title(
    "STRING First-Shell Network of Final 3-Gene Signature"
)

plt.axis("off")
plt.tight_layout()

fig = f"{OUT}/Figure_STRING_first_shell_3gene.png"

plt.savefig(
    fig,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(f"{OUT}/STRING_first_shell_raw.csv")
print(f"{OUT}/STRING_first_shell_interactions.csv")
print(f"{OUT}/signature_gene_network_degree.csv")
print(f"{OUT}/shared_first_shell_neighbors.csv")
print(fig)

print("\n" + "=" * 70)
print("FIRST-SHELL STRING ANALYSIS COMPLETE")
print("=" * 70)
