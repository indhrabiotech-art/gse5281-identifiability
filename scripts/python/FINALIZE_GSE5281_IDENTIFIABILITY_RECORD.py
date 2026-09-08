#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path.home() / "project_ml"

META = ROOT / "04_ML/External_Validation/GSE5281_hippocampus_SOFT_complete_metadata.csv"

OUT = ROOT / "04_ML/External_Validation/GSE5281_FINAL_IDENTIFIABILITY_RECORD.txt"

meta = pd.read_csv(META)

meta["Diagnosis"] = meta["Group"].map({
    "Control": 0,
    "AD": 1
})

batch = pd.get_dummies(
    meta["submission_date"],
    drop_first=True,
    dtype=float
)

X = pd.DataFrame({
    "Intercept": 1,
    "Diagnosis": meta["Diagnosis"].astype(float)
})

X = pd.concat([X, batch], axis=1)

rank = np.linalg.matrix_rank(X.to_numpy())

with open(OUT, "w") as f:

    f.write("=" * 90 + "\n")
    f.write("GSE5281 HIPPOCAMPAL DESIGN IDENTIFIABILITY — FINAL RECORD\n")
    f.write("=" * 90 + "\n\n")

    f.write(
        "Generated: "
        + datetime.now().isoformat()
        + "\n\n"
    )

    f.write("1. DATASET STRUCTURE\n")
    f.write("-" * 90 + "\n")
    f.write(f"Hippocampal samples: {len(meta)}\n")
    f.write(
        f"Controls: {(meta['Group'] == 'Control').sum()}\n"
    )
    f.write(
        f"AD: {(meta['Group'] == 'AD').sum()}\n\n"
    )

    f.write("2. SAMPLE-LEVEL STRUCTURE\n")
    f.write("-" * 90 + "\n")
    f.write(
        meta[
            ["GSM", "Group", "submission_date"]
        ].to_string(index=False)
    )
    f.write("\n\n")

    f.write("3. DIAGNOSIS × SUBMISSION DATE\n")
    f.write("-" * 90 + "\n")
    f.write(
        pd.crosstab(
            meta["submission_date"],
            meta["Group"]
        ).to_string()
    )
    f.write("\n\n")

    f.write("4. DESIGN MATRIX\n")
    f.write("-" * 90 + "\n")
    f.write(
        X.to_string(index=False)
    )
    f.write("\n\n")

    f.write("5. MATRIX IDENTIFIABILITY\n")
    f.write("-" * 90 + "\n")
    f.write(
        f"Design matrix shape: {X.shape}\n"
    )
    f.write(
        f"Number of columns: {X.shape[1]}\n"
    )
    f.write(
        f"Matrix rank: {rank}\n"
    )
    f.write(
        f"Rank deficient: {rank < X.shape[1]}\n"
    )

    same = np.all(
        meta["Diagnosis"].to_numpy()
        ==
        batch.iloc[:, 0].to_numpy()
    )

    f.write(
        f"Diagnosis equals submission indicator: {same}\n\n"
    )

    f.write("6. FORMAL CONCLUSION\n")
    f.write("-" * 90 + "\n")

    if rank < X.shape[1] and same:

        f.write(
            "Diagnosis and submission cohort are perfectly confounded "
            "within the GSE5281 hippocampal subset.\n\n"
        )

        f.write(
            "The design is rank deficient, preventing independent "
            "estimation of diagnosis and submission-cohort effects.\n\n"
        )

        f.write(
            "Therefore the GSE5281 hippocampal data cannot provide a "
            "clean independent estimate of biological disease signal "
            "separate from submission-cohort effects.\n\n"
        )

    f.write("7. APPROPRIATE MANUSCRIPT TERMINOLOGY\n")
    f.write("-" * 90 + "\n")

    f.write(
        "Preferred:\n"
        "\"cross-dataset transfer test\"\n\n"
    )

    f.write(
        "Avoid:\n"
        "\"independent external validation\"\n\n"
    )

    f.write("8. CLAIM BOUNDARIES\n")
    f.write("-" * 90 + "\n")

    f.write(
        "Supported:\n"
        "- Complete diagnosis/submission-cohort confounding.\n"
        "- Rank deficiency of diagnosis + submission-cohort design.\n"
        "- Inability to independently estimate both effects.\n\n"
    )

    f.write(
        "Not established:\n"
        "- That all observed external discrimination is technical artifact.\n"
        "- That the signature has no biological signal.\n"
        "- That batch effects explain the entire observed AUC.\n"
    )

print("=" * 90)
print("FINAL IDENTIFIABILITY RECORD CREATED")
print("=" * 90)
print(OUT)

