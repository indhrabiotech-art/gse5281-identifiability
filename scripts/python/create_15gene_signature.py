import pandas as pd

INPUT = "04_ML/GSE48350_hippocampus_ML_matrix_AGEADJUSTED_v2.csv"
OUTPUT = "04_ML/GSE48350_hippocampus_15gene_signature.csv"

genes = [
    "SLC25A46",
    "FAM170A",
    "CD5",
    "LINC02987",
    "RAE1",
    "HSPA12A",
    "ANKIB1",
    "BTK",
    "ZNF621",
    "SPDEF",
    "KCNJ5",
    "TRIM49",
    "PART1",
    "MEFV",
    "P3H3"
]

df = pd.read_csv(INPUT, index_col=0)

missing = [g for g in genes if g not in df.index]

if missing:
    print("ERROR: Missing genes:")
    print(missing)
    raise SystemExit(1)

signature = df.loc[genes]

print("15-gene signature matrix")
print("Shape:", signature.shape)
print("\nGenes:")
print("\n".join(signature.index))

signature.to_csv(OUTPUT)

print("\nSaved:")
print(OUTPUT)
