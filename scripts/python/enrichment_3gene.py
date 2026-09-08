import os
import pandas as pd
import gseapy as gp

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

OUT = "05_Biology/Enrichment"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("FINAL 3-GENE FUNCTIONAL ENRICHMENT")
print("=" * 70)

print("\nGenes:")
for gene in GENES:
    print(" -", gene)

# ------------------------------------------------------------
# GO Biological Process
# ------------------------------------------------------------

print("\nRunning GO Biological Process enrichment...")

go = gp.enrichr(
    gene_list=GENES,
    gene_sets=["GO_Biological_Process_2023"],
    organism="human",
    outdir=OUT,
    cutoff=1.0
)

go_results = go.results

if go_results is not None and len(go_results) > 0:
    go_results.to_csv(
        f"{OUT}/GO_Biological_Process_3gene.csv",
        index=False
    )
    print("GO results:", len(go_results))
else:
    print("No GO terms returned.")

# ------------------------------------------------------------
# GO Molecular Function
# ------------------------------------------------------------

print("\nRunning GO Molecular Function enrichment...")

gomf = gp.enrichr(
    gene_list=GENES,
    gene_sets=["GO_Molecular_Function_2023"],
    organism="human",
    outdir=OUT,
    cutoff=1.0
)

if gomf.results is not None and len(gomf.results) > 0:
    gomf.results.to_csv(
        f"{OUT}/GO_Molecular_Function_3gene.csv",
        index=False
    )
    print("GO-MF results:", len(gomf.results))
else:
    print("No GO-MF terms returned.")

# ------------------------------------------------------------
# KEGG
# ------------------------------------------------------------

print("\nRunning KEGG enrichment...")

kegg = gp.enrichr(
    gene_list=GENES,
    gene_sets=["KEGG_2021_Human"],
    organism="human",
    outdir=OUT,
    cutoff=1.0
)

if kegg.results is not None and len(kegg.results) > 0:
    kegg.results.to_csv(
        f"{OUT}/KEGG_3gene.csv",
        index=False
    )
    print("KEGG results:", len(kegg.results))
else:
    print("No KEGG pathways returned.")

# ------------------------------------------------------------
# Reactome
# ------------------------------------------------------------

print("\nRunning Reactome enrichment...")

reactome = gp.enrichr(
    gene_list=GENES,
    gene_sets=["Reactome_2022"],
    organism="human",
    outdir=OUT,
    cutoff=1.0
)

if reactome.results is not None and len(reactome.results) > 0:
    reactome.results.to_csv(
        f"{OUT}/Reactome_3gene.csv",
        index=False
    )
    print("Reactome results:", len(reactome.results))
else:
    print("No Reactome pathways returned.")

print("\n" + "=" * 70)
print("ENRICHMENT COMPLETE")
print("=" * 70)

print("\nOutput directory:")
print(OUT)
