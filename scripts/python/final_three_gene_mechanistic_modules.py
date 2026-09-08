import os
import pandas as pd

OUTDIR = "10_STRING/results"
MANUSCRIPT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT, exist_ok=True)

INPUT = (
    "10_STRING/results/"
    "STRING_gene_specific_significant_enrichment.csv"
)

print("=" * 80)
print("FINAL THREE-GENE MECHANISTIC MODULE INTEGRATION")
print("=" * 80)

df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# Select biologically informative STRING categories
# ------------------------------------------------------------

keywords = [
    "cytokine",
    "interleukin",
    "JAK-STAT",
    "STAT",
    "neurotrophic",
    "neuron",
    "apoptotic",
    "ABC transporter",
    "ABC-type transporter",
    "lipid",
    "RANKL",
    "RANK",
    "osteoclast",
    "TNF receptor"
]

mask = df["description"].astype(str).str.contains(
    "|".join(keywords),
    case=False,
    regex=True,
    na=False
)

selected = df.loc[mask].copy()

# ------------------------------------------------------------
# Assign mechanistic module
# ------------------------------------------------------------

def assign_module(description):

    d = str(description).lower()

    if any(x in d for x in [
        "abc transporter",
        "abc-type transporter",
        "lipid transporter",
        "lipid homeostasis"
    ]):
        return "Lipid_transport_homeostasis"

    if any(x in d for x in [
        "il-6",
        "interleukin",
        "jak-stat",
        "stat",
        "cytokine",
        "neurotrophic",
        "neuron apoptotic",
        "neuron"
    ]):
        return "Cytokine_neurotrophic_signaling"

    if any(x in d for x in [
        "rankl",
        "rank signaling",
        "osteoclast",
        "tnfr/ngfr",
        "tnf receptor"
    ]):
        return "TNFR_RANKL_signaling"

    return "Other"

selected["Mechanistic_Module"] = selected[
    "description"
].apply(assign_module)

selected = selected[
    selected["Mechanistic_Module"] != "Other"
].copy()

# ------------------------------------------------------------
# Keep strongest evidence per gene/module
# ------------------------------------------------------------

selected = selected.sort_values(
    ["Seed_Gene", "Mechanistic_Module", "fdr"]
)

module_summary = (
    selected
    .groupby(
        ["Seed_Gene", "Mechanistic_Module"],
        as_index=False
    )
    .agg(
        Best_FDR=("fdr", "min"),
        Evidence_Count=("description", "count"),
        Representative_Term=("description", "first")
    )
)

# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("THREE-GENE MECHANISTIC MODULES")
print("=" * 80)

print(module_summary.to_string(index=False))

print("\n" + "=" * 80)
print("GENE-LEVEL BIOLOGICAL INTERPRETATION")
print("=" * 80)

for gene in ["ABCA6", "CRLF1", "TNFRSF11B"]:

    x = module_summary[
        module_summary["Seed_Gene"] == gene
    ]

    print(f"\n{gene}")

    if len(x) == 0:
        print("  No selected mechanistic module.")
    else:
        for _, r in x.iterrows():
            print(
                f"  - {r['Mechanistic_Module']} "
                f"(best FDR={r['Best_FDR']:.3e})"
            )
            print(
                f"    Evidence: {r['Representative_Term']}"
            )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

selected.to_csv(
    f"{OUTDIR}/FINAL_THREE_GENE_MECHANISTIC_EVIDENCE.csv",
    index=False
)

module_summary.to_csv(
    f"{OUTDIR}/FINAL_THREE_GENE_MECHANISTIC_MODULES.csv",
    index=False
)

module_summary.to_csv(
    f"{MANUSCRIPT}/Table_final_three_gene_mechanistic_modules.csv",
    index=False
)

print("\n" + "=" * 80)
print("MECHANISTIC MODULE ANALYSIS COMPLETE")
print("=" * 80)

print("\nSaved:")
print(
    f"{OUTDIR}/FINAL_THREE_GENE_MECHANISTIC_EVIDENCE.csv"
)
print(
    f"{OUTDIR}/FINAL_THREE_GENE_MECHANISTIC_MODULES.csv"
)
print(
    f"{MANUSCRIPT}/Table_final_three_gene_mechanistic_modules.csv"
)

