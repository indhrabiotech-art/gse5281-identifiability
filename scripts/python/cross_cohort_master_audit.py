import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "project_ml"

OUTDIR = ROOT / "12_MetaAnalysis"
RESULTS = OUTDIR / "results"
LOGS = OUTDIR / "logs"

RESULTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

DATASET_NAMES = [
    "GSE48350",
    "GSE5281",
    "GSE63060",
    "GSE63061",
    "GSE138852"
]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def read_table(path, nrows=None):

    try:

        if path.suffix.lower() == ".tsv":
            return pd.read_csv(
                path,
                sep="\t",
                nrows=nrows
            )

        return pd.read_csv(
            path,
            nrows=nrows
        )

    except Exception:

        try:

            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                nrows=nrows
            )

        except Exception:

            return None


def identify_dataset(path):

    name = path.name.upper()
    full = str(path).upper()

    for ds in DATASET_NAMES:

        if ds in name or ds in full:
            return ds

    return "UNKNOWN"


def find_gene_columns(columns):

    found = []

    normalized = {
        str(c).strip().upper(): c
        for c in columns
    }

    for gene in GENES:

        if gene.upper() in normalized:
            found.append(
                normalized[gene.upper()]
            )

    return found


def find_diagnosis_columns(columns):

    keywords = [
        "diagnosis",
        "diagnose",
        "condition",
        "group",
        "status",
        "phenotype",
        "disease"
    ]

    result = []

    for c in columns:

        c_lower = str(c).lower()

        if any(
            k in c_lower
            for k in keywords
        ):
            result.append(str(c))

    return result


def classify_file(path, columns):

    name = path.name.lower()

    if any(
        x in name
        for x in [
            "coefficient",
            "effect",
            "cohen",
            "differential",
            "deg",
            "limma",
            "de_result",
            "statistics"
        ]
    ):
        return "EFFECT_OR_DE_FILE"

    if any(
        x in name
        for x in [
            "prediction",
            "probability",
            "model",
            "random_forest",
            "xgboost",
            "lasso"
        ]
    ):
        return "MODEL_FILE"

    if any(
        x in name
        for x in [
            "meta",
            "metadata",
            "phenotype",
            "clinical",
            "covariate"
        ]
    ):
        return "METADATA_FILE"

    gene_columns = find_gene_columns(
        columns
    )

    if len(gene_columns) >= 1:
        return "EXPRESSION_FILE"

    return "OTHER"


# ------------------------------------------------------------
# Find files
# ------------------------------------------------------------

print("=" * 90)
print("CROSS-COHORT THREE-GENE MASTER AUDIT")
print("=" * 90)

files = []

for path in ROOT.rglob("*"):

    if not path.is_file():
        continue

    # Avoid environments and caches
    path_string = str(path)

    if any(
        skip in path_string
        for skip in [
            "/.venv_ml/",
            "/venv/",
            "/__pycache__/",
            "/site-packages/"
        ]
    ):
        continue

    if path.suffix.lower() not in [
        ".csv",
        ".tsv"
    ]:
        continue

    files.append(path)

files = sorted(files)

print("\nTotal CSV/TSV files found:", len(files))

# ------------------------------------------------------------
# Audit
# ------------------------------------------------------------

records = []

for path in files:

    dataset = identify_dataset(path)

    if dataset == "UNKNOWN":
        continue

    try:

        df = read_table(
            path,
            nrows=5
        )

        if df is None:
            continue

        columns = df.columns.tolist()

        gene_columns = find_gene_columns(
            columns
        )

        diagnosis_columns = find_diagnosis_columns(
            columns
        )

        classification = classify_file(
            path,
            columns
        )

        # Full size when possible
        try:

            full_df = read_table(path)

            if full_df is not None:

                n_rows = len(full_df)
                n_columns = len(full_df.columns)

            else:

                n_rows = np.nan
                n_columns = len(columns)

        except Exception:

            n_rows = np.nan
            n_columns = len(columns)

        records.append({

            "Dataset": dataset,

            "File":
                str(path.relative_to(ROOT)),

            "File_type":
                classification,

            "Rows":
                n_rows,

            "Columns":
                n_columns,

            "ABCA6_present":
                "ABCA6" in [
                    str(x).upper()
                    for x in gene_columns
                ],

            "CRLF1_present":
                "CRLF1" in [
                    str(x).upper()
                    for x in gene_columns
                ],

            "TNFRSF11B_present":
                "TNFRSF11B" in [
                    str(x).upper()
                    for x in gene_columns
                ],

            "Gene_columns":
                ";".join(
                    map(str, gene_columns)
                ),

            "Diagnosis_columns":
                ";".join(
                    diagnosis_columns
                ),

            "All_columns":
                " | ".join(
                    map(str, columns)
                )
        })

    except Exception as e:

        records.append({

            "Dataset": dataset,

            "File":
                str(path.relative_to(ROOT)),

            "File_type":
                "READ_ERROR",

            "Rows": np.nan,

            "Columns": np.nan,

            "ABCA6_present": False,

            "CRLF1_present": False,

            "TNFRSF11B_present": False,

            "Gene_columns": "",

            "Diagnosis_columns": "",

            "All_columns":
                "ERROR: " + str(e)
        })


audit = pd.DataFrame(records)

# ------------------------------------------------------------
# Save complete audit
# ------------------------------------------------------------

audit_file = (
    RESULTS /
    "cross_cohort_file_audit.csv"
)

audit.to_csv(
    audit_file,
    index=False
)

# ------------------------------------------------------------
# Relevant files only
# ------------------------------------------------------------

relevant = audit[
    (
        audit["ABCA6_present"]
        |
        audit["CRLF1_present"]
        |
        audit["TNFRSF11B_present"]
    )
    |
    (
        audit["File_type"]
        == "EFFECT_OR_DE_FILE"
    )
].copy()

relevant_file = (
    RESULTS /
    "cross_cohort_relevant_files.csv"
)

relevant.to_csv(
    relevant_file,
    index=False
)

# ------------------------------------------------------------
# Dataset summary
# ------------------------------------------------------------

summary_rows = []

for dataset in DATASET_NAMES:

    sub = audit[
        audit["Dataset"] == dataset
    ]

    summary_rows.append({

        "Dataset":
            dataset,

        "Files_found":
            len(sub),

        "Three_gene_expression_files":
            int(
                (
                    sub["File_type"]
                    == "EXPRESSION_FILE"
                ).sum()
            ),

        "Effect_or_DE_files":
            int(
                (
                    sub["File_type"]
                    == "EFFECT_OR_DE_FILE"
                ).sum()
            ),

        "Metadata_files":
            int(
                (
                    sub["File_type"]
                    == "METADATA_FILE"
                ).sum()
            ),

        "ABCA6_files":
            int(
                sub["ABCA6_present"].sum()
            ),

        "CRLF1_files":
            int(
                sub["CRLF1_present"].sum()
            ),

        "TNFRSF11B_files":
            int(
                sub["TNFRSF11B_present"].sum()
            )
        })

summary = pd.DataFrame(
    summary_rows
)

summary_file = (
    RESULTS /
    "cross_cohort_dataset_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

# ------------------------------------------------------------
# Human-readable report
# ------------------------------------------------------------

report_file = (
    LOGS /
    "cross_cohort_master_audit.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "CROSS-COHORT THREE-GENE MASTER AUDIT\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Project root: {ROOT}\n"
    )

    f.write(
        f"CSV/TSV files scanned: {len(files)}\n\n"
    )

    f.write(
        "DATASET SUMMARY\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "RELEVANT FILES\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for dataset in DATASET_NAMES:

        f.write(
            f"\n[{dataset}]\n"
        )

        sub = relevant[
            relevant["Dataset"] == dataset
        ]

        if len(sub) == 0:

            f.write(
                "No relevant files found.\n"
            )

            continue

        for _, row in sub.iterrows():

            f.write(
                f"\nFile: {row['File']}\n"
            )

            f.write(
                f"Type: {row['File_type']}\n"
            )

            f.write(
                f"Rows: {row['Rows']}\n"
            )

            f.write(
                f"Columns: {row['Columns']}\n"
            )

            f.write(
                "Genes: "
                f"{row['Gene_columns']}\n"
            )

            f.write(
                "Diagnosis columns: "
                f"{row['Diagnosis_columns']}\n"
            )

    f.write(
        "\n\n"
        "NEXT ANALYTICAL REQUIREMENT\n"
        + "-" * 80
        + "\n"
    )

    f.write(
        "Before meta-analysis, determine for each cohort:\n"
        "1. Independent biological sample unit.\n"
        "2. AD/control definition.\n"
        "3. Gene-level effect estimate.\n"
        "4. Standard error or variance, if available.\n"
        "5. Tissue/platform.\n"
        "6. Whether the effect is directly comparable with other cohorts.\n"
    )

# ------------------------------------------------------------
# Print report
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("DATASET SUMMARY")
print("=" * 90)

print(
    summary.to_string(
        index=False
    )
)

print("\n" + "=" * 90)
print("RELEVANT FILES")
print("=" * 90)

for dataset in DATASET_NAMES:

    sub = relevant[
        relevant["Dataset"] == dataset
    ]

    print(
        f"\n===== {dataset} ====="
    )

    if len(sub) == 0:

        print(
            "No relevant files found."
        )

        continue

    for _, row in sub.iterrows():

        print(
            f"\n{row['File']}"
        )

        print(
            "Type:",
            row["File_type"]
        )

        print(
            "Rows:",
            row["Rows"],
            "| Columns:",
            row["Columns"]
        )

        print(
            "Genes:",
            row["Gene_columns"]
        )

        print(
            "Diagnosis columns:",
            row["Diagnosis_columns"]
        )

print("\n" + "=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)

print("\nSaved:")
print(audit_file)
print(relevant_file)
print(summary_file)
print(report_file)

print("\nIMPORTANT:")
print(
    "No meta-analysis was performed yet."
)
print(
    "The next step is to extract comparable effect sizes "
    "from the audited cohorts."
)
