import os
import pandas as pd

OUTDIR = "04_ML/Final_Model/Validation"
MANUSCRIPT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT, exist_ok=True)

rows = [

    {
        "Evidence_ID": "E01",
        "Domain": "Discovery",
        "Analysis": "Three-gene discovery model",
        "Dataset": "GSE48350",
        "Primary_metric": "ROC-AUC",
        "Value": 0.8549,
        "Interpretation": "Strong discovery discrimination",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E02",
        "Domain": "Discovery",
        "Analysis": "Three-gene discovery model",
        "Dataset": "GSE48350",
        "Primary_metric": "PR-AUC",
        "Value": 0.8257,
        "Interpretation": "Strong precision-recall performance",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E03",
        "Domain": "Internal validation",
        "Analysis": "Held-out test",
        "Dataset": "GSE48350",
        "Primary_metric": "ROC-AUC",
        "Value": 0.6389,
        "Interpretation": "Moderate discrimination in held-out samples",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E04",
        "Domain": "External validation",
        "Analysis": "Independent hippocampal validation",
        "Dataset": "GSE5281",
        "Primary_metric": "ROC-AUC",
        "Value": 0.6615,
        "Interpretation": "Moderate independent discrimination",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E05",
        "Domain": "Robustness",
        "Analysis": "Three-gene ablation",
        "Dataset": "GSE48350",
        "Primary_metric": "Full model ROC-AUC",
        "Value": 0.8549,
        "Interpretation": "Multigene signature improves over individual components",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E06",
        "Domain": "Robustness",
        "Analysis": "Ablation bootstrap",
        "Dataset": "GSE48350 / GSE5281",
        "Primary_metric": "5000 bootstrap iterations",
        "Value": 5000,
        "Interpretation": "Uncertainty of ablation comparisons quantified",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E07",
        "Domain": "Feature stability",
        "Analysis": "LASSO bootstrap",
        "Dataset": "GSE48350",
        "Primary_metric": "Maximum selection frequency",
        "Value": 0.1515,
        "Interpretation": "Individual gene selection stability is low",
        "Manuscript_role": "Caveat"
    },

    {
        "Evidence_ID": "E08",
        "Domain": "Cross-cohort biology",
        "Analysis": "ABCA6 directional concordance",
        "Dataset": "Brain / blood",
        "Primary_metric": "Direction concordance",
        "Value": 1.0,
        "Interpretation": "ABCA6 preserves direction across evaluated cohorts",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E09",
        "Domain": "Cross-cohort biology",
        "Analysis": "CRLF1 directional concordance",
        "Dataset": "Brain / blood",
        "Primary_metric": "Direction concordance",
        "Value": 1.0,
        "Interpretation": "CRLF1 preserves direction across evaluated cohorts",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E10",
        "Domain": "Cross-cohort biology",
        "Analysis": "TNFRSF11B directional concordance",
        "Dataset": "Brain / blood",
        "Primary_metric": "Direction concordance",
        "Value": 0.6667,
        "Interpretation": "Direction reverses in blood",
        "Manuscript_role": "Caveat"
    },

    {
        "Evidence_ID": "E11",
        "Domain": "Clinical adjustment",
        "Analysis": "ABCA6 age/sex-adjusted model",
        "Dataset": "GSE63060",
        "Primary_metric": "ABCA6 p-value",
        "Value": 0.000760,
        "Interpretation": "ABCA6 remains associated after adjustment",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E12",
        "Domain": "Incremental value",
        "Analysis": "ABCA6 added to age + sex",
        "Dataset": "GSE63060",
        "Primary_metric": "Delta AUC",
        "Value": 0.0512,
        "Interpretation": "ABCA6 adds discrimination beyond demographics",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E13",
        "Domain": "Incremental value",
        "Analysis": "ABCA6 added to age + sex",
        "Dataset": "GSE63060",
        "Primary_metric": "Likelihood-ratio p-value",
        "Value": 0.000423,
        "Interpretation": "Significant incremental model contribution",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E14",
        "Domain": "Incremental value",
        "Analysis": "CRLF1 added after ABCA6",
        "Dataset": "GSE63060",
        "Primary_metric": "Delta AUC",
        "Value": 0.0081,
        "Interpretation": "Limited incremental discrimination",
        "Manuscript_role": "Caveat"
    },

    {
        "Evidence_ID": "E15",
        "Domain": "Incremental value",
        "Analysis": "TNFRSF11B added after ABCA6 + CRLF1",
        "Dataset": "GSE63060",
        "Primary_metric": "Delta AUC",
        "Value": -0.0010,
        "Interpretation": "No detectable incremental contribution",
        "Manuscript_role": "Caveat"
    },

    {
        "Evidence_ID": "E16",
        "Domain": "Meta-analysis",
        "Analysis": "ABCA6 random-effects meta-analysis",
        "Dataset": "Three cohorts",
        "Primary_metric": "Random-effects Cohen's d",
        "Value": 0.5413,
        "Interpretation": "Positive pooled effect with substantial heterogeneity",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E17",
        "Domain": "Meta-analysis",
        "Analysis": "CRLF1 random-effects meta-analysis",
        "Dataset": "Three cohorts",
        "Primary_metric": "Random-effects Cohen's d",
        "Value": 0.3862,
        "Interpretation": "Positive pooled effect with substantial heterogeneity",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E18",
        "Domain": "Meta-analysis",
        "Analysis": "TNFRSF11B random-effects meta-analysis",
        "Dataset": "Three cohorts",
        "Primary_metric": "Random-effects Cohen's d",
        "Value": 0.3692,
        "Interpretation": "High heterogeneity and inconsistent direction",
        "Manuscript_role": "Caveat"
    },

    {
        "Evidence_ID": "E19",
        "Domain": "Threshold",
        "Analysis": "Locked Youden threshold",
        "Dataset": "GSE48350",
        "Primary_metric": "Threshold",
        "Value": 0.3297,
        "Interpretation": "Threshold determined only from discovery cohort",
        "Manuscript_role": "Primary"
    },

    {
        "Evidence_ID": "E20",
        "Domain": "Calibration",
        "Analysis": "External calibration",
        "Dataset": "GSE5281",
        "Primary_metric": "Calibration slope",
        "Value": 0.5210,
        "Interpretation": "Probability calibration attenuated",
        "Manuscript_role": "Caveat"
    },

    {
        "Evidence_ID": "E21",
        "Domain": "Clinical translation",
        "Analysis": "Clinical validation",
        "Dataset": "None",
        "Primary_metric": "Evidence status",
        "Value": 0,
        "Interpretation": "No clinical validation performed",
        "Manuscript_role": "Caveat"
    }
]


df = pd.DataFrame(rows)

print("=" * 80)
print("FINAL RESULTS–EVIDENCE MATRIX")
print("=" * 80)

print(
    df.to_string(
        index=False
    )
)

out = (
    OUTDIR +
    "/FINAL_RESULTS_EVIDENCE_MATRIX.csv"
)

manuscript = (
    MANUSCRIPT +
    "/Table_final_results_evidence_matrix.csv"
)

df.to_csv(
    out,
    index=False
)

df.to_csv(
    manuscript,
    index=False
)

print()
print("Saved:")
print(out)
print(manuscript)

print()
print("=" * 80)
print("RESULTS–EVIDENCE MATRIX COMPLETE")
print("=" * 80)
