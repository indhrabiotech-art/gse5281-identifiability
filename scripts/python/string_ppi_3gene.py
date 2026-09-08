import os
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

OUT = "05_Biology/PPI"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("STRING PPI / NETWORK ANALYSIS — FINAL 3-GENE SIGNATURE")
print("=" * 70)

print("\nGenes:")
for g in GENES:
    print(" -", g)

# ------------------------------------------------------------
# STRING API
# ------------------------------------------------------------

url = "https://string-db.org/api/tsv/network"

params = {
    "identifiers": "\r".join(GENES),
    "species": 9606,
    "required_score": 400,
    "network_type": "functional"
}

print("\nQuerying STRING...")

response = requests.get(
    url,
    params=params,
    timeout=60
)

response.raise_for_status()

out_file = f"{OUT}/STRING_3gene_network.tsv"

with open(out_file, "w") as f:
    f.write(response.text)

# ------------------------------------------------------------
# Read network
# ------------------------------------------------------------

from io import StringIO

ppi = pd.read_csv(
    StringIO(response.text),
    sep="\t"
)

print("\nSTRING records:", len(ppi))

print("\nAvailable columns:")
print(ppi.columns.tolist())

# ------------------------------------------------------------
# Save simplified interaction table
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

available = [
    c for c in cols
    if c in ppi.columns
]

ppi[available].to_csv(
    f"{OUT}/STRING_3gene_interactions.csv",
    index=False
)

print("\nInteraction table:")
print(
    ppi[available].to_string(index=False)
)

# ------------------------------------------------------------
# Direct interactions among the 3 signature genes
# ------------------------------------------------------------

G = nx.Graph()

G.add_nodes_from(GENES)

for _, row in ppi.iterrows():

    a = row.get("preferredName_A")
    b = row.get("preferredName_B")
    score = row.get("score")

    if a in GENES and b in GENES:
        G.add_edge(
            a,
            b,
            score=float(score)
        )

print("\n" + "=" * 70)
print("DIRECT 3-GENE NETWORK")
print("=" * 70)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

if G.number_of_edges() > 0:

    for a, b, data in G.edges(data=True):
        print(
            f"{a} -- {b} | STRING score = {data['score']:.3f}"
        )

else:
    print("No direct STRING interaction detected among the 3 genes.")

# ------------------------------------------------------------
# Network figure
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))

pos = nx.spring_layout(
    G,
    seed=42
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=1800
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=11,
    font_weight="bold"
)

if G.number_of_edges() > 0:

    widths = [
        1 + 4 * data["score"]
        for _, _, data in G.edges(data=True)
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths
    )

plt.title(
    "STRING Network — ABCA6, CRLF1, TNFRSF11B"
)

plt.axis("off")

plt.tight_layout()

figure_file = (
    f"{OUT}/Figure_STRING_3gene_network.png"
)

plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Save network summary
# ------------------------------------------------------------

summary = pd.DataFrame({
    "metric": [
        "number_of_signature_genes",
        "number_of_direct_STRING_edges"
    ],
    "value": [
        G.number_of_nodes(),
        G.number_of_edges()
    ]
})

summary.to_csv(
    f"{OUT}/STRING_3gene_network_summary.csv",
    index=False
)

print("\nSaved:")
print(out_file)
print(f"{OUT}/STRING_3gene_interactions.csv")
print(f"{OUT}/STRING_3gene_network_summary.csv")
print(figure_file)

print("\n" + "=" * 70)
print("STRING PPI ANALYSIS COMPLETE")
print("=" * 70)
