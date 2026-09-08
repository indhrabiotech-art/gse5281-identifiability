#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score

from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path.home() / "project_ml"

EXPR = ROOT / "03_Preprocessing/GSE5281_RMA_genelevel.csv"

META = (
    ROOT
    / "04_ML/External_Validation/"
      "GSE5281_hippocampus_SOFT_complete_metadata.csv"
)

OUTDIR = ROOT / "04_ML/External_Validation"
OUTDIR.mkdir(parents=True, exist_ok=True)


print("=" * 90)
print("GSE5281 COHORT / EXPRESSION SEPARATION AUDIT")
print("=" * 90)


# ============================================================
# 1. LOAD EXPRESSION MATRIX
# ============================================================

print("\n[1] Loading expression matrix")

expr = pd.read_csv(
    EXPR,
    index_col=0
)

print("Original shape:", expr.shape)

print("First expression columns:")
print(expr.columns[:5].tolist())


# ============================================================
# 2. NORMALIZE SAMPLE IDS
# ============================================================

print("\n[2] Normalizing sample identifiers")

def clean_sample_id(x):

    x = str(x)

    x = x.replace(".CEL.gz", "")
    x = x.replace(".cel.gz", "")
    x = x.replace(".CEL", "")
    x = x.replace(".cel", "")

    return x.strip()


expr.columns = [
    clean_sample_id(x)
    for x in expr.columns
]


print("Cleaned expression columns:")
print(expr.columns.tolist())


# ============================================================
# 3. TARGET HIPPOCAMPAL SAMPLES
# ============================================================

controls = [
    f"GSM{i}"
    for i in range(119628, 119641)
]

ad = [
    f"GSM{i}"
    for i in range(238799, 238809)
]

targets = controls + ad


print("\nExpected samples:", len(targets))

missing = [
    x
    for x in targets
    if x not in expr.columns
]

if missing:

    print("\nERROR: Missing expression samples:")
    print(missing)

    raise SystemExit(
        "Expression/sample mapping incomplete."
    )


expr = expr[
    targets
]


print(
    "Final expression matrix:",
    expr.shape,
    "(genes x samples)"
)


# ============================================================
# 4. TRANSPOSE TO SAMPLE x GENE
# ============================================================

X_df = expr.T

X_df.index.name = "GSM"

print(
    "Sample x gene matrix:",
    X_df.shape
)


# ============================================================
# 5. LOAD COMPLETE SOFT METADATA
# ============================================================

print("\n[3] Loading SOFT metadata")

meta = pd.read_csv(
    META
)

meta["GSM"] = (
    meta["GSM"]
    .astype(str)
    .str.strip()
)

meta = meta.set_index("GSM")


missing_meta = [
    x
    for x in targets
    if x not in meta.index
]

if missing_meta:

    print(
        "ERROR: Missing metadata samples:",
        missing_meta
    )

    raise SystemExit(
        "Metadata/sample mapping incomplete."
    )


meta = meta.loc[
    targets
]


print(
    meta[
        [
            "Group",
            "submission_date",
            "extract_protocol_ch1",
            "label_ch1",
            "label_protocol_ch1",
            "hyb_protocol",
            "scan_protocol",
            "data_processing"
        ]
    ].to_string()
)


# ============================================================
# 6. CREATE DIAGNOSIS LABEL
# ============================================================

y = (
    meta["Group"]
    .map(
        {
            "Control": 0,
            "AD": 1
        }
    )
    .values
)


if np.isnan(y).any():

    raise SystemExit(
        "Diagnosis labels contain missing values."
    )


print("\nSample counts:")
print("Controls:", int(np.sum(y == 0)))
print("AD:", int(np.sum(y == 1)))


# ============================================================
# 7. CLEAN EXPRESSION
# ============================================================

print("\n[4] Cleaning expression matrix")

X_df = X_df.apply(
    pd.to_numeric,
    errors="coerce"
)

# Remove genes with missing values
X_df = X_df.dropna(
    axis=1,
    how="any"
)

# Remove zero-variance genes
X_df = X_df.loc[
    :,
    X_df.var(axis=0) > 0
]

print(
    "Cleaned sample x gene matrix:",
    X_df.shape
)


# ============================================================
# 8. PCA
# ============================================================

print("\n" + "=" * 90)
print("[5] PCA — EXPRESSION COHORT SEPARATION")
print("=" * 90)


# Standardize genes across samples
X_scaled = StandardScaler().fit_transform(
    X_df
)


n_components = min(
    10,
    X_scaled.shape[0] - 1
)


pca = PCA(
    n_components=n_components
)

PC = pca.fit_transform(
    X_scaled
)


pc_names = [
    f"PC{i}"
    for i in range(1, n_components + 1)
]


pc_df = pd.DataFrame(
    PC,
    index=targets,
    columns=pc_names
)

pc_df["Group"] = meta["Group"]


print("\nVariance explained:")

for i, variance in enumerate(
    pca.explained_variance_ratio_,
    start=1
):

    print(
        f"PC{i}: {variance * 100:.3f}%"
    )


pc_df.to_csv(
    OUTDIR
    / "GSE5281_cohort_PCA_scores_audit.csv"
)


# ============================================================
# 9. PC vs DIAGNOSIS
# ============================================================

print("\n" + "=" * 90)
print("[6] ASSOCIATION BETWEEN PRINCIPAL COMPONENTS AND DIAGNOSIS")
print("=" * 90)


pc_results = []


for i, pc in enumerate(
    pc_names
):

    control_values = (
        pc_df.loc[
            meta["Group"] == "Control",
            pc
        ]
        .values
    )

    ad_values = (
        pc_df.loc[
            meta["Group"] == "AD",
            pc
        ]
        .values
    )


    mw_stat, mw_p = mannwhitneyu(
        control_values,
        ad_values,
        alternative="two-sided"
    )


    pc_auc = roc_auc_score(
        y,
        pc_df[pc].values
    )


    rho, rho_p = spearmanr(
        pc_df[pc].values,
        y
    )


    result = {

        "PC": pc,

        "variance_explained":
            pca.explained_variance_ratio_[i],

        "control_mean":
            np.mean(control_values),

        "AD_mean":
            np.mean(ad_values),

        "mann_whitney_U":
            mw_stat,

        "mann_whitney_p":
            mw_p,

        "PC_AUC_for_diagnosis":
            pc_auc,

        "spearman_rho":
            rho,

        "spearman_p":
            rho_p
    }


    pc_results.append(
        result
    )


    print(
        f"{pc:5s} | "
        f"variance="
        f"{result['variance_explained']*100:7.3f}% | "
        f"AUC="
        f"{pc_auc:.4f} | "
        f"MW p="
        f"{mw_p:.6g} | "
        f"rho="
        f"{rho:.4f}"
    )


pc_results = pd.DataFrame(
    pc_results
)


pc_results.to_csv(
    OUTDIR
    / "GSE5281_PC_diagnosis_association_audit.csv",
    index=False
)


# ============================================================
# 10. EXPRESSION-BASED CLASSIFIER USING PCs
# ============================================================

print("\n" + "=" * 90)
print("[7] EXPRESSION-BASED COHORT CLASSIFIER")
print("=" * 90)


n_pcs_for_classifier = min(
    5,
    PC.shape[1]
)


X_pc = PC[
    :,
    :n_pcs_for_classifier
]


loo = LeaveOneOut()


predictions = np.zeros(
    len(y)
)


for train_idx, test_idx in loo.split(
    X_pc
):

    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=10000,
        random_state=42
    )


    classifier.fit(
        X_pc[train_idx],
        y[train_idx]
    )


    predictions[test_idx] = (
        classifier
        .predict_proba(
            X_pc[test_idx]
        )[:, 1]
    )


pc_classifier_auc = roc_auc_score(
    y,
    predictions
)


pc_classifier_accuracy = accuracy_score(
    y,
    predictions >= 0.5
)


print(
    f"PCs used: {n_pcs_for_classifier}"
)

print(
    f"LOOCV AUC: {pc_classifier_auc:.4f}"
)

print(
    f"LOOCV accuracy: {pc_classifier_accuracy:.4f}"
)


pd.DataFrame({

    "GSM": targets,

    "Group":
        meta["Group"].values,

    "cohort_classifier_probability":
        predictions

}).to_csv(

    OUTDIR
    / "GSE5281_expression_cohort_classifier_predictions.csv",

    index=False
)


# ============================================================
# 11. SUBMISSION DATE NESTING
# ============================================================

print("\n" + "=" * 90)
print("[8] SUBMISSION DATE / DIAGNOSIS NESTING")
print("=" * 90)


submission_table = pd.crosstab(
    meta["submission_date"],
    meta["Group"]
)


print(
    submission_table
)


submission_table.to_csv(
    OUTDIR
    / "GSE5281_submission_date_diagnosis_table.csv"
)


submission_nested = (
    meta
    .groupby(
        "submission_date"
    )["Group"]
    .nunique()
    .max()
    == 1
)


if submission_nested:

    print(
        "\nRESULT: "
        "PERFECT SUBMISSION-DATE / DIAGNOSIS NESTING"
    )

else:

    print(
        "\nRESULT: "
        "Submission date is not perfectly nested."
    )


# ============================================================
# 12. EXPERIMENTAL WORKFLOW DIFFERENCES
# ============================================================

print("\n" + "=" * 90)
print("[9] EXPERIMENTAL WORKFLOW DIFFERENCES")
print("=" * 90)


workflow_cols = [

    "extract_protocol_ch1",

    "label_ch1",

    "label_protocol_ch1",

    "hyb_protocol",

    "scan_protocol",

    "data_processing"
]


workflow_results = []


for col in workflow_cols:

    control_values = (

        meta.loc[
            meta["Group"] == "Control",
            col
        ]

        .fillna("MISSING")

        .astype(str)

        .unique()

    )


    ad_values = (

        meta.loc[
            meta["Group"] == "AD",
            col
        ]

        .fillna("MISSING")

        .astype(str)

        .unique()

    )


    identical = (
        set(control_values)
        ==
        set(ad_values)
    )


    workflow_results.append({

        "variable": col,

        "control_values":
            " || ".join(control_values),

        "AD_values":
            " || ".join(ad_values),

        "identical_between_groups":
            identical

    })


    print(
        f"\n{col}"
    )

    print(
        "CONTROL:",
        control_values
    )

    print(
        "AD:",
        ad_values
    )

    print(
        "IDENTICAL:",
        identical
    )


workflow_results_df = pd.DataFrame(
    workflow_results
)


workflow_results_df.to_csv(
    OUTDIR
    / "GSE5281_workflow_group_difference_audit.csv",
    index=False
)


workflow_differences = int(
    (
        ~workflow_results_df[
            "identical_between_groups"
        ]
    ).sum()
)


# ============================================================
# 13. MASTER SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("[10] MASTER AUDIT SUMMARY")
print("=" * 90)


pc1_auc = pc_results.loc[
    pc_results["PC"] == "PC1",
    "PC_AUC_for_diagnosis"
].iloc[0]


pc2_auc = pc_results.loc[
    pc_results["PC"] == "PC2",
    "PC_AUC_for_diagnosis"
].iloc[0]


summary = pd.DataFrame([{

    "n_samples":
        len(targets),

    "n_control":
        int(np.sum(y == 0)),

    "n_AD":
        int(np.sum(y == 1)),

    "submission_date_perfectly_nested":
        submission_nested,

    "workflow_variables_different":
        workflow_differences,

    "PC1_variance":
        pca.explained_variance_ratio_[0],

    "PC1_AUC":
        pc1_auc,

    "PC2_AUC":
        pc2_auc,

    "PC_classifier_LOOCV_AUC":
        pc_classifier_auc,

    "PC_classifier_LOOCV_accuracy":
        pc_classifier_accuracy

}])


print(
    summary.to_string(
        index=False
    )
)


summary.to_csv(
    OUTDIR
    / "GSE5281_cohort_separation_MASTER_summary.csv",
    index=False
)


# ============================================================
# 14. SCIENTIFIC INTERPRETATION
# ============================================================

print("\n" + "=" * 90)
print("[11] SCIENTIFIC INTERPRETATION")
print("=" * 90)


print(
    "\nThe audit tests whether the GSE5281 hippocampal "
    "AD/control distinction is structurally associated "
    "with cohort-level expression variation."
)


if submission_nested:

    print(
        "\nPASS: Diagnosis is perfectly nested within "
        "submission-date cohort."
    )

else:

    print(
        "\nWARNING: Submission date is not perfectly nested."
    )


print(
    f"\nPC1 explains "
    f"{pca.explained_variance_ratio_[0]*100:.3f}% "
    f"of total standardized expression variance."
)


print(
    f"PC1 AUC for diagnosis = "
    f"{pc1_auc:.4f}"
)


print(
    f"LOOCV AUC using first "
    f"{n_pcs_for_classifier} PCs = "
    f"{pc_classifier_auc:.4f}"
)


print(
    f"\nDocumented workflow variables differing "
    f"between groups: "
    f"{workflow_differences}/{len(workflow_cols)}"
)


print(
    "\nIMPORTANT:"
)


print(
    "Strong expression separation would demonstrate "
    "cohort-associated molecular structure, but it would "
    "not by itself prove that the separation is entirely "
    "technical batch effect."
)


print(
    "The appropriate manuscript language must distinguish "
    "observed cohort confounding from proof of artifact."
)


print("\nFiles written to:")
print(OUTDIR)


print("\n" + "=" * 90)
print("GSE5281 COHORT SEPARATION AUDIT COMPLETE")
print("=" * 90)

