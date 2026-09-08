#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path.home() / "project_ml"

OUTDIR = ROOT / "16_Figures_Final" / "Main"
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "Figure_1.png"
PDF = OUTDIR / "Figure_1.pdf"

fig, ax = plt.subplots(figsize=(12, 15))

ax.set_xlim(0, 12)
ax.set_ylim(0, 17)
ax.axis("off")


# ============================================================
# HELPERS
# ============================================================

def box(x, y, w, h, title, text, fontsize=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.5,
        fill=False
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h - 0.28,
        title,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold"
    )

    ax.text(
        x + w / 2,
        y + h / 2 - 0.10,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
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


def section_label(x, y, text):
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold"
    )


# ============================================================
# TITLE
# ============================================================

ax.text(
    6,
    16.65,
    "Study design for transcriptomic signature transfer",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold"
)

ax.text(
    6,
    16.20,
    "Discovery → model locking → transfer testing → identifiability audit → biological interpretation",
    ha="center",
    va="center",
    fontsize=10
)


# ============================================================
# DISCOVERY
# ============================================================

section_label(0.8, 15.65, "DISCOVERY")

box(
    0.8, 14.15, 10.4, 1.25,
    "GSE48350 — DISCOVERY DATASET",
    "Alzheimer's disease transcriptomic dataset\n"
    "Expression preprocessing, quality control and feature reduction",
    fontsize=9.5
)

arrow(6, 14.15, 6, 13.55)


box(
    0.8, 12.25, 10.4, 1.30,
    "FEATURE SELECTION",
    "LASSO logistic regression\n"
    "Candidate feature selection within the discovery framework",
    fontsize=9.5
)

arrow(6, 12.25, 6, 11.60)


# ============================================================
# LOCKED MODEL
# ============================================================

box(
    2.0, 10.00, 8.0, 1.35,
    "LOCKED THREE-GENE SIGNATURE",
    "ABCA6     +     CRLF1     +     TNFRSF11B\n"
    "Model coefficients fixed before transfer testing",
    fontsize=10
)

arrow(6, 10.00, 6, 9.35)


# ============================================================
# INTERNAL VS TRANSFER TEST
# ============================================================

section_label(0.8, 9.05, "MODEL EVALUATION")

box(
    0.7, 6.85, 5.0, 2.0,
    "INTERNAL EVALUATION",
    "GSE48350 held-out test set\n"
    "Repeated nested cross-validation\n"
    "ROC-AUC / PR-AUC\n"
    "Feature-selection stability",
    fontsize=9.3
)

box(
    6.3, 6.85, 5.0, 2.0,
    "GSE5281 — CROSS-DATASET TRANSFER TEST",
    "23 hippocampal samples\n"
    "13 Control  |  10 AD\n"
    "Locked model; no disease-model refitting\n"
    "Raw and harmonized transfer performance",
    fontsize=9.1
)

arrow(6, 9.35, 3.2, 8.85)
arrow(6, 9.35, 8.8, 8.85)


# ============================================================
# IDENTIFIABILITY AUDIT
# ============================================================

arrow(8.8, 6.85, 8.8, 6.20)

box(
    1.0, 4.65, 10.0, 1.40,
    "GSE5281 DESIGN IDENTIFIABILITY AUDIT",
    "Diagnosis perfectly nested within submission cohort\n"
    "13 Controls → Jul 10 2006     |     10 AD → Oct 19 2007\n"
    "Intercept + Diagnosis + Cohort: 23 × 3 design, rank = 2",
    fontsize=9.2
)

arrow(6, 4.65, 6, 4.00)


# ============================================================
# COHORT SEPARATION
# ============================================================

box(
    1.0, 2.55, 10.0, 1.35,
    "TRANSCRIPTOME-WIDE COHORT SEPARATION",
    "PCA identifies strong cohort-associated expression structure\n"
    "LOOCV cohort classifier: ROC-AUC = 1.000; accuracy = 1.000",
    fontsize=9.5
)

arrow(6, 2.55, 6, 1.95)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

box(
    0.8, 0.45, 10.4, 1.20,
    "METHODS-PAPER INTERPRETATION",
    "External discrimination is interpreted as cross-dataset transfer-test performance,\n"
    "not clean independent biological validation, because diagnosis and cohort effects\n"
    "cannot be independently identified within the GSE5281 hippocampal subset.",
    fontsize=9.2
)


# ============================================================
# SIDE NOTE
# ============================================================

ax.text(
    11.35,
    8.0,
    "MODEL\nLOCKED\n\n↓\n\nNO\nREFITTING",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    rotation=90
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    PNG,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    PDF,
    bbox_inches="tight"
)

plt.close()

print("=" * 80)
print("FIGURE 1 — FINAL STUDY DESIGN")
print("=" * 80)
print(f"PNG: {PNG}")
print(f"PDF: {PDF}")
