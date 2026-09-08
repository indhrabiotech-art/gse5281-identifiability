import os

OUT = "06_Manuscript/Results"
os.makedirs(OUT, exist_ok=True)

text = r"""
FIGURE LEGENDS

Figure 1. Receiver operating characteristic analysis of the final three-gene
signature.

ROC curves showing discrimination of the ABCA6, CRLF1 and TNFRSF11B
three-gene model in the GSE48350 training cohort, held-out GSE48350 internal
test cohort and independent GSE5281 external validation cohort. ROC-AUC was
0.855 in the training cohort, 0.694 in the internal test cohort and 0.685
in the external cohort.

Figure 2. Precision-recall analysis of the final three-gene signature.

Precision-recall curves for the three-gene model across the GSE48350 training,
GSE48350 internal test and GSE5281 external validation cohorts. Average
precision was 0.828, 0.694 and 0.635, respectively.

Figure 3. Expression heatmap of the final three-gene signature.

Standardized expression values for ABCA6, CRLF1 and TNFRSF11B across the
GSE48350 training samples. Each gene was standardized for visualization.

Figure 4. STRING first-shell functional network of the three-gene signature.

STRING-based first-shell network surrounding ABCA6, CRLF1 and TNFRSF11B.
The network contains the three signature genes together with their detected
first-shell neighboring proteins. ABCA6, CRLF1 and TNFRSF11B showed network
degrees of 3, 6 and 2, respectively. No shared first-shell neighbor was
identified among all three signature genes in the generated network.
"""

path = os.path.join(
    OUT,
    "Figure_Legends_3gene_signature.txt"
)

with open(path, "w") as f:
    f.write(text.strip() + "\n")

print("=" * 75)
print("FIGURE LEGENDS GENERATED")
print("=" * 75)
print()
print("Saved:")
print(path)
print()
print("=" * 75)
