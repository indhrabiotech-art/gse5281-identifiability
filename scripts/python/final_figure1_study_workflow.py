#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path.home() / "project_ml"
OUTDIR = ROOT / "16_Figures_Final" / "Main"
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "Figure_1.png"
PDF = OUTDIR / "Figure_1.pdf"


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(14, 9))

ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")


# ============================================================
# HELPERS
# ============================================================

def box(x, y, w, h, title, lines,
        title_size=13, text_size=10.5,
        linewidth=1.6):

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.10",
        linewidth=linewidth,
        fill=False
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h - 0.28,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold"
    )

    text = "\n".join(lines)

    ax.text(
        x + w / 2,
        y + h / 2 - 0.12,
        text,
        ha="center",
        va="center",
        fontsize=text_size,
        linespacing=1.45
    )


def arrow(x1, y1, x2, y2):

    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.4
        )
    )


# ============================================================
# TITLE
# ============================================================

ax.text(
    7,
    8.65,
    "Study design and analytical workflow",
    ha="center",
    va="center",
    fontsize=19,
    fontweight="bold"
)

ax.text(
    7,
    8.27,
    "Transcriptomic Alzheimer's disease signature discovery and cross-dataset transfer testing",
    ha="center",
    va="center",
    fontsize=11
)


# ============================================================
# A — DISCOVERY
# ============================================================

ax.text(
    3.5,
    7.75,
    "A  DISCOVERY",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold"
)

box(
    0.6, 5.85, 5.8, 1.35,
    "GSE48350",
    [
        "AD brain transcriptomic cohort",
        "Preprocessing and feature reduction"
    ]
)

box(
    0.6, 3.95, 5.8, 1.35,
    "FEATURE SELECTION",
    [
        "LASSO-based candidate selection",
        "Model development and internal evaluation"
    ]
)

box(
    0.6, 2.05, 5.8, 1.35,
    "FINAL SIGNATURE",
    [
        "ABCA6   •   CRLF1   •   TNFRSF11B",
        "Locked three-gene logistic model"
    ]
)

arrow(3.5, 5.85, 3.5, 5.30)
arrow(3.5, 3.95, 3.5, 3.40)


# ============================================================
# B — TRANSFER TESTING
# ============================================================

ax.text(
    10.5,
    7.75,
    "B  MODEL EVALUATION",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold"
)

box(
    7.6, 5.85, 5.8, 1.35,
    "INTERNAL EVALUATION",
    [
        "Held-out test set",
        "Nested cross-validation"
    ]
)

box(
    7.6, 3.95, 5.8, 1.35,
    "GSE5281",
    [
        "Hippocampal cross-dataset transfer test",
        "Raw and harmonized performance"
    ]
)

box(
    7.6, 2.05, 5.8, 1.35,
    "CROSS-TISSUE TEST",
    [
        "Blood cohort analysis",
        "Direction and transportability"
    ]
)

arrow(6.4, 2.72, 7.55, 2.72)

arrow(10.5, 5.85, 10.5, 5.30)
arrow(10.5, 3.95, 10.5, 3.40)


# ============================================================
# C — METHODological AUDIT
# ============================================================

ax.text(
    7,
    1.48,
    "C  METHODOLOGICAL AUDIT",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold"
)

audit_text = (
    "Discrimination     •     Calibration     •     Cohort confounding     •     "
    "Design identifiability     •     Expression separation"
)

ax.text(
    7,
    1.02,
    audit_text,
    ha="center",
    va="center",
    fontsize=10.5
)


# ============================================================
# D — INTERPRETATION
# ============================================================

box(
    2.0, -0.25, 10.0, 0.85,
    "FINAL INTERPRETATION",
    [
        "External discrimination is reported as cross-dataset transfer-test performance,",
        "not as clean independent biological validation."
    ],
    title_size=12,
    text_size=9.5
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    PDF,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("=" * 80)
print("FIGURE 1 — READABLE JOURNAL-STYLE VERSION")
print("=" * 80)
print(f"PNG: {PNG}")
print(f"PDF: {PDF}")
