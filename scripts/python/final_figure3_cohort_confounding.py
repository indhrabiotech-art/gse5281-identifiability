#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTDIR = "06_Manuscript/Figures"
os.makedirs(OUTDIR, exist_ok=True)

ROOT = "04_ML/External_Validation"

PCA_FILE = f"{ROOT}/GSE5281_cohort_PCA_scores_audit.csv"
ASSOC_FILE = f"{ROOT}/GSE5281_PC_diagnosis_association_audit.csv"
SUMMARY_FILE = f"{ROOT}/GSE5281_cohort_separation_MASTER_summary.csv"
DESIGN_FILE = f"{ROOT}/GSE5281_design_identifiability_audit.csv"


# ============================================================
# LOAD DATA
# ============================================================

pca = pd.read_csv(PCA_FILE)
assoc = pd.read_csv(ASSOC_FILE)
summary = pd.read_csv(SUMMARY_FILE)

pca = pca.rename(columns={pca.columns[0]: "GSM"})

control = pca[pca["Group"] == "Control"].copy()
ad = pca[pca["Group"] == "AD"].copy()

n_control = len(control)
n_ad = len(ad)

pc1_var = summary.loc[0, "PC1_variance"] * 100
pc2_var = (
    assoc.loc[assoc["PC"] == "PC2", "variance_explained"].iloc[0] * 100
)
pc12_var = pc1_var + pc2_var

pc2_auc = (
    assoc.loc[assoc["PC"] == "PC2", "PC_AUC_for_diagnosis"].iloc[0]
)

classifier_auc = summary.loc[0, "PC_classifier_LOOCV_AUC"]
classifier_acc = summary.loc[0, "PC_classifier_LOOCV_accuracy"]


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(figsize=(13, 8.5))

gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.05, 1.25],
    height_ratios=[1.0, 1.0],
    hspace=0.32,
    wspace=0.28
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])


# ============================================================
# PANEL A — SAMPLE / COHORT STRUCTURE
# ============================================================

ax1.axis("off")

ax1.text(
    0.02, 0.96,
    "A. Diagnosis–submission structure",
    transform=ax1.transAxes,
    fontsize=13,
    fontweight="bold",
    va="top"
)

# Controls
ax1.add_patch(
    FancyBboxPatch(
        (0.05, 0.53),
        0.90,
        0.27,
        boxstyle="round,pad=0.02",
        fill=False,
        linewidth=1.5
    )
)

ax1.text(
    0.50, 0.70,
    "CONTROL",
    transform=ax1.transAxes,
    ha="center",
    fontsize=12,
    fontweight="bold"
)

ax1.text(
    0.50, 0.59,
    "n = 13\nSubmission: Jul 10 2006",
    transform=ax1.transAxes,
    ha="center",
    va="center",
    fontsize=10
)

# AD
ax1.add_patch(
    FancyBboxPatch(
        (0.05, 0.15),
        0.90,
        0.27,
        boxstyle="round,pad=0.02",
        fill=False,
        linewidth=1.5
    )
)

ax1.text(
    0.50, 0.32,
    "ALZHEIMER'S DISEASE",
    transform=ax1.transAxes,
    ha="center",
    fontsize=12,
    fontweight="bold"
)

ax1.text(
    0.50, 0.21,
    "n = 10\nSubmission: Oct 19 2007",
    transform=ax1.transAxes,
    ha="center",
    va="center",
    fontsize=10
)

ax1.text(
    0.50, 0.03,
    "Diagnosis is perfectly nested within submission cohort",
    transform=ax1.transAxes,
    ha="center",
    fontsize=9.5
)


# ============================================================
# PANEL B — PCA
# ============================================================

ax2.scatter(
    control["PC1"],
    control["PC2"],
    s=55,
    marker="o",
    label="Control"
)

ax2.scatter(
    ad["PC1"],
    ad["PC2"],
    s=65,
    marker="^",
    label="AD"
)

ax2.set_xlabel(f"PC1 ({pc1_var:.2f}% variance)")
ax2.set_ylabel(f"PC2 ({pc2_var:.2f}% variance)")

ax2.set_title(
    "B. Transcriptome-wide expression structure",
    fontsize=13,
    fontweight="bold"
)

ax2.legend(frameon=False)

ax2.text(
    0.03,
    0.04,
    f"PC1 + PC2 = {pc12_var:.2f}%",
    transform=ax2.transAxes,
    fontsize=10
)


# ============================================================
# PANEL C — DIAGNOSIS ASSOCIATION
# ============================================================

pc2_control = control["PC2"].values
pc2_ad = ad["PC2"].values

positions = [1, 2]

bp = ax3.boxplot(
    [pc2_control, pc2_ad],
    positions=positions,
    widths=0.50,
    patch_artist=False,
    showfliers=True
)

# Overlay points
rng = np.random.default_rng(42)

ax3.scatter(
    np.ones(len(pc2_control)) + rng.normal(0, 0.04, len(pc2_control)),
    pc2_control,
    s=28,
    alpha=0.7
)

ax3.scatter(
    np.ones(len(pc2_ad)) * 2 + rng.normal(0, 0.04, len(pc2_ad)),
    pc2_ad,
    s=32,
    alpha=0.7
)

ax3.set_xticks([1, 2])
ax3.set_xticklabels(["Control", "AD"])

ax3.set_ylabel("PC2 score")

ax3.set_title(
    "C. PC2 strongly tracks diagnosis",
    fontsize=13,
    fontweight="bold"
)

ax3.text(
    0.50,
    0.94,
    f"Diagnosis AUC = {pc2_auc:.3f}",
    transform=ax3.transAxes,
    ha="center",
    va="top",
    fontsize=10
)

ax3.grid(axis="y", alpha=0.20)


# ============================================================
# PANEL D — IDENTIFIABILITY / CLASSIFIER
# ============================================================

ax4.axis("off")

ax4.text(
    0.02, 0.96,
    "D. Design identifiability",
    transform=ax4.transAxes,
    fontsize=13,
    fontweight="bold",
    va="top"
)

# Design
ax4.add_patch(
    FancyBboxPatch(
        (0.05, 0.60),
        0.90,
        0.25,
        boxstyle="round,pad=0.02",
        fill=False,
        linewidth=1.5
    )
)

ax4.text(
    0.50,
    0.77,
    "Design: Intercept + Diagnosis + Cohort",
    transform=ax4.transAxes,
    ha="center",
    fontsize=11,
    fontweight="bold"
)

ax4.text(
    0.50,
    0.66,
    "23 × 3 matrix    |    rank = 2",
    transform=ax4.transAxes,
    ha="center",
    fontsize=10
)

# Classifier
ax4.add_patch(
    FancyBboxPatch(
        (0.05, 0.27),
        0.90,
        0.22,
        boxstyle="round,pad=0.02",
        fill=False,
        linewidth=1.5
    )
)

ax4.text(
    0.50,
    0.42,
    "Transcriptome cohort classifier",
    transform=ax4.transAxes,
    ha="center",
    fontsize=11,
    fontweight="bold"
)

ax4.text(
    0.50,
    0.33,
    f"LOOCV AUC = {classifier_auc:.3f}    |    Accuracy = {classifier_acc:.3f}",
    transform=ax4.transAxes,
    ha="center",
    fontsize=10
)

# Conclusion
ax4.text(
    0.50,
    0.12,
    "Diagnosis and submission cohort cannot be\n"
    "independently estimated in this subset.",
    transform=ax4.transAxes,
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold"
)


# ============================================================
# FIGURE TITLE
# ============================================================

fig.suptitle(
    "GSE5281 Hippocampal Cohort Confounding Limits Independent Validation",
    fontsize=16,
    fontweight="bold",
    y=0.985
)

fig.text(
    0.50,
    0.005,
    "Interpretation: GSE5281 provides cross-dataset transfer-test performance, "
    "not a clean independent estimate of biological validation.",
    ha="center",
    fontsize=9.5
)

# ============================================================
# SAVE
# ============================================================

png = f"{OUTDIR}/Figure_3_Cohort_Confounding.png"
pdf = f"{OUTDIR}/Figure_3_Cohort_Confounding.pdf"

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
print("FIGURE 3 — COHORT CONFOUNDING")
print("=" * 80)

print(f"Controls: {n_control}")
print(f"AD: {n_ad}")
print(f"PC1 variance: {pc1_var:.4f}%")
print(f"PC2 variance: {pc2_var:.4f}%")
print(f"PC1 + PC2 variance: {pc12_var:.4f}%")
print(f"PC2 diagnosis AUC: {pc2_auc:.4f}")
print(f"Cohort classifier LOOCV AUC: {classifier_auc:.4f}")
print(f"Cohort classifier accuracy: {classifier_acc:.4f}")

print("\nSaved:")
print(png)
print(pdf)
