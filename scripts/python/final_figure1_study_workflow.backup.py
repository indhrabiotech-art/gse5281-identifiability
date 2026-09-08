import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTDIR = "06_Manuscript/Figures"
os.makedirs(OUTDIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 15))

ax.set_xlim(0, 12)
ax.set_ylim(0, 17)
ax.axis("off")

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def box(x, y, w, h, title, text):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=1.5,
        fill=False
    )
    ax.add_patch(patch)

    ax.text(
        x + w/2,
        y + h - 0.35,
        title,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold"
    )

    ax.text(
        x + w/2,
        y + h/2 - 0.15,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        linespacing=1.35
    )


def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=16,
            linewidth=1.3
        )
    )


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

ax.text(
    6,
    16.65,
    "Computational Framework for Three-Gene Alzheimer's Disease Signature Discovery",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold"
)

ax.text(
    6,
    16.15,
    "Integrated machine-learning, cross-cohort validation and biological interpretation",
    ha="center",
    va="center",
    fontsize=10
)

# ------------------------------------------------------------
# PHASE 1
# ------------------------------------------------------------

box(
    1.0, 14.2, 10.0, 1.25,
    "1. PUBLIC TRANSCRIPTOMIC DATA",
    "AD brain discovery cohort: GSE48350\n"
    "Independent hippocampal cohort: GSE5281\n"
    "Blood cohort: GSE63060"
)

arrow(6, 14.2, 6, 13.65)

# ------------------------------------------------------------
# PHASE 2
# ------------------------------------------------------------

box(
    1.0, 12.25, 10.0, 1.4,
    "2. DATA PROCESSING & FEATURE DISCOVERY",
    "Expression preprocessing / normalization\n"
    "Variance-based feature reduction and age adjustment\n"
    "Candidate-gene identification"
)

arrow(6, 12.25, 6, 11.7)

# ------------------------------------------------------------
# PHASE 3
# ------------------------------------------------------------

box(
    1.0, 10.15, 10.0, 1.45,
    "3. THREE-GENE SIGNATURE",
    "ABCA6  +  CRLF1  +  TNFRSF11B\n"
    "Final compact molecular signature\n"
    "Discovery model established in GSE48350"
)

arrow(6, 10.15, 6, 9.55)

# ------------------------------------------------------------
# PHASE 4 — ML
# ------------------------------------------------------------

box(
    0.6, 7.0, 5.0, 2.2,
    "4A. MACHINE-LEARNING CHARACTERIZATION",
    "Logistic regression\n"
    "LASSO feature-selection stability\n"
    "Random Forest\n"
    "XGBoost\n"
    "SHAP interpretability"
)

# ------------------------------------------------------------
# PHASE 4 — VALIDATION
# ------------------------------------------------------------

box(
    6.4, 7.0, 5.0, 2.2,
    "4B. MODEL VALIDATION",
    "Held-out GSE48350 samples\n"
    "Independent GSE5281 hippocampus\n"
    "ROC-AUC and PR-AUC\n"
    "Locked threshold and calibration"
)

arrow(6, 9.55, 3.1, 9.2)
arrow(6, 9.55, 8.9, 9.2)

arrow(3.1, 7.0, 6, 6.35)
arrow(8.9, 7.0, 6, 6.35)

# ------------------------------------------------------------
# PHASE 5
# ------------------------------------------------------------

box(
    1.0, 4.95, 10.0, 1.4,
    "5. ROBUSTNESS & STATISTICAL VALIDATION",
    "Three-gene ablation and 5,000-bootstrap analysis\n"
    "Cross-model importance concordance\n"
    "Effect-size meta-analysis and heterogeneity\n"
    "Age/sex-adjusted incremental modelling in blood"
)

arrow(6, 4.95, 6, 4.35)

# ------------------------------------------------------------
# PHASE 6
# ------------------------------------------------------------

box(
    1.0, 2.85, 10.0, 1.4,
    "6. CROSS-COHORT & CROSS-TISSUE BIOLOGY",
    "Training → hippocampal external validation → blood\n"
    "Effect-size concordance and directional transportability\n"
    "Identification of tissue-specific limitations"
)

arrow(6, 2.85, 6, 2.25)

# ------------------------------------------------------------
# PHASE 7
# ------------------------------------------------------------

box(
    1.0, 0.75, 10.0, 1.4,
    "7. BIOLOGICAL MECHANISTIC INTERPRETATION",
    "STRING first-shell network and functional enrichment\n"
    "ABCA6 → lipid transport/homeostasis\n"
    "CRLF1 → cytokine/IL-6-type signaling\n"
    "TNFRSF11B → inflammatory/TNFR-RANKL signaling"
)

# ------------------------------------------------------------
# FOOTNOTE
# ------------------------------------------------------------

ax.text(
    6,
    0.15,
    "Final interpretation: a compact, independently evaluated three-gene AD-associated signature with "
    "multi-model, cross-cohort and mechanistic evidence.",
    ha="center",
    va="center",
    fontsize=8.5
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

png = os.path.join(
    OUTDIR,
    "Figure_1_Study_Workflow.png"
)

pdf = os.path.join(
    OUTDIR,
    "Figure_1_Study_Workflow.pdf"
)

plt.savefig(
    png,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    pdf,
    bbox_inches="tight"
)

plt.close()

print("=" * 80)
print("FIGURE 1 — STUDY WORKFLOW COMPLETE")
print("=" * 80)
print()
print("Saved:")
print(png)
print(pdf)

