import pandas as pd
import os

genes = ["ABCA6", "CRLF1", "TNFRSF11B"]

outdir = "05_Biology"
os.makedirs(outdir, exist_ok=True)

df = pd.DataFrame({
    "gene": genes,
    "role_in_model": [
        "candidate_signature_gene",
        "candidate_signature_gene",
        "candidate_signature_gene"
    ]
})

# Add final model coefficients
coef = pd.read_csv(
    "04_ML/Final_Characterization/3gene_final_coefficients.csv"
)

df = df.merge(
    coef[["gene", "coefficient", "abs_coefficient"]],
    on="gene",
    how="left"
)

# Add gene-wise validation statistics
stats = pd.read_csv(
    "04_ML/Final_Characterization/3gene_gene_wise_statistics.csv"
)

df = df.merge(
    stats[
        [
            "gene",
            "train_delta",
            "train_AUC",
            "test_delta",
            "test_AUC",
            "external_delta",
            "external_AUC",
            "train_external_direction_same"
        ]
    ],
    on="gene",
    how="left"
)

df.to_csv(
    f"{outdir}/final_3gene_biology_input.csv",
    index=False
)

with open(
    f"{outdir}/final_3gene_list.txt",
    "w"
) as f:
    for gene in genes:
        f.write(gene + "\n")

print("=" * 70)
print("FINAL 3-GENE BIOLOGY INPUT PREPARED")
print("=" * 70)

print(df.to_string(index=False))

print("\nSaved:")
print(f"{outdir}/final_3gene_biology_input.csv")
print(f"{outdir}/final_3gene_list.txt")

print("=" * 70)
