#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "project_ml"

META = (
    ROOT
    / "04_ML/External_Validation/"
      "GSE5281_hippocampus_SOFT_complete_metadata.csv"
)

OUT = (
    ROOT
    / "04_ML/External_Validation/"
      "GSE5281_design_identifiability_audit.csv"
)

print("=" * 90)
print("GSE5281 HIPPOCAMPAL DESIGN IDENTIFIABILITY AUDIT")
print("=" * 90)

meta = pd.read_csv(META)

meta["Diagnosis"] = (
    meta["Group"]
    .map({
        "Control": 0,
        "AD": 1
    })
)

meta["Submission"] = (
    meta["submission_date"]
    .astype(str)
)

print("\nSamples:")
print(meta[[
    "GSM",
    "Group",
    "submission_date"
]].to_string(index=False))

print("\n" + "=" * 90)
print("CROSS-TABULATION")
print("=" * 90)

ct = pd.crosstab(
    meta["submission_date"],
    meta["Group"]
)

print(ct)

print("\n" + "=" * 90)
print("DESIGN IDENTIFIABILITY")
print("=" * 90)

# Dummy-coded submission date
batch = pd.get_dummies(
    meta["submission_date"],
    drop_first=True,
    dtype=float
)

X = pd.DataFrame({
    "Intercept": 1,
    "Diagnosis": meta["Diagnosis"].astype(float)
})

X = pd.concat(
    [X, batch],
    axis=1
)

print("\nDesign matrix:")
print(X)

rank = np.linalg.matrix_rank(
    X.to_numpy()
)

n_columns = X.shape[1]

print(
    "\nDesign matrix shape:",
    X.shape
)

print(
    "Matrix rank:",
    rank
)

print(
    "Number of columns:",
    n_columns
)

if rank < n_columns:

    print(
        "\nRESULT: NON-IDENTIFIABLE / RANK DEFICIENT"
    )

    print(
        "Diagnosis and submission cohort cannot be "
        "estimated as independent effects."
    )

else:

    print(
        "\nRESULT: FULL RANK"
    )

    print(
        "Diagnosis and submission cohort are "
        "statistically identifiable."
    )


# Direct equivalence test
same = (
    meta["Diagnosis"].to_numpy()
    ==
    batch.iloc[:, 0].to_numpy()
)

print(
    "\nDiagnosis == submission cohort indicator:",
    bool(np.all(same))
)


# Save
summary = pd.DataFrame([{
    "n_samples": len(meta),
    "n_columns": n_columns,
    "design_rank": rank,
    "rank_deficient": rank < n_columns,
    "diagnosis_equals_submission_indicator": bool(np.all(same))
}])

summary.to_csv(
    OUT,
    index=False
)

print(
    "\nSaved:",
    OUT
)

print(
    "\n" + "=" * 90
)
print(
    "AUDIT COMPLETE"
)
print(
    "=" * 90
)

