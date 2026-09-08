import os
from pathlib import Path
import pandas as pd
import numpy as np

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

# ============================================================
# CANDIDATE FILES IDENTIFIED BY MASTER AUDIT
# ============================================================

CANDIDATES = {

    "GSE48350_3gene_coefficients":
        ROOT /
        "04_ML/Final_Model/GSE48350_final_3gene_coefficients.csv",

    "GSE48350_LASSO":
        ROOT /
        "04_ML/LASSO/GSE48350_LASSO_feature_statistics.csv",

    "GSE5281_effect_sizes":
        ROOT /
        "04_ML/External_Validation/GSE5281_cross_dataset_effect_sizes.csv",

    "GSE5281_locked_coefficients":
        ROOT /
        "04_ML/External_Validation/GSE5281_locked_model_coefficients.csv",

    "GSE63060_incremental":
        ROOT /
        "09_CrossTissue/results/GSE63060_three_gene_incremental_coefficients.csv",

    "GSE138852_pseudobulk":
        ROOT /
        "11_SingleCell/GSE138852/results/GSE138852_ABCA6_CRLF1_pseudobulk_statistics.csv",

    "GSE138852_publication":
        ROOT /
        "11_SingleCell/GSE138852/results/GSE138852_ABCA6_CRLF1_publication_effects.csv"
}

# ============================================================
# HELPERS
# ============================================================

def load_file(path):

    if not path.exists():

        return None

    try:

        if path.suffix.lower() == ".tsv":

            return pd.read_csv(
                path,
                sep="\t"
            )

        return pd.read_csv(path)

    except Exception as e:

        print(
            f"ERROR reading {path}: {e}"
        )

        return None


def find_gene_column(df):

    for col in df.columns:

        values = (
            df[col]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        if values.isin(
            [g.upper() for g in GENES]
        ).any():

            return col

    return None


# ============================================================
# MAIN
# ============================================================

print("=" * 90)
print("THREE-GENE CROSS-COHORT EFFECT EXTRACTION")
print("=" * 90)

all_records = []

log_lines = []

for label, path in CANDIDATES.items():

    print("\n" + "=" * 90)
    print(label)
    print(path)

    df = load_file(path)

    if df is None:

        print("FILE NOT AVAILABLE")

        log_lines.append(
            f"{label}: FILE NOT AVAILABLE"
        )

        continue

    print(
        "Shape:",
        df.shape
    )

    print(
        "Columns:",
        df.columns.tolist()
    )

    log_lines.append(
        f"\n{label}\n{path}\n"
    )

    # --------------------------------------------------------
    # Identify gene column
    # --------------------------------------------------------

    gene_col = find_gene_column(df)

    # --------------------------------------------------------
    # Case 1: long-format gene table
    # --------------------------------------------------------

    if gene_col is not None:

        sub = df[
            df[gene_col]
            .astype(str)
            .str.upper()
            .isin(
                [g.upper() for g in GENES]
            )
        ].copy()

        print(
            "\nThree-gene rows:"
        )

        if len(sub):

            print(
                sub.to_string(
                    index=False
                )
            )

            for _, row in sub.iterrows():

                gene = str(
                    row[gene_col]
                ).strip().upper()

                record = {

                    "Source":
                        label,

                    "Dataset":
                        label.split("_")[0],

                    "Gene":
                        gene,

                    "Source_file":
                        str(
                            path.relative_to(ROOT)
                        )
                }

                for col in df.columns:

                    if col == gene_col:
                        continue

                    record[
                        str(col)
                    ] = row[col]

                all_records.append(
                    record
                )

        else:

            print(
                "No three-gene rows found."
            )

    # --------------------------------------------------------
    # Case 2: wide-format expression table
    # --------------------------------------------------------

    else:

        found = []

        for gene in GENES:

            if gene in df.columns:

                found.append(gene)

        if found:

            print(
                "\nWide-format genes found:",
                found
            )

            print(
                df[found]
                .head(10)
                .to_string(
                    index=False
                )
            )

            log_lines.append(
                f"Wide-format genes: {found}\n"
            )

        else:

            print(
                "\nNo direct three-gene columns."
            )

# ============================================================
# CONSOLIDATED TABLE
# ============================================================

if all_records:

    consolidated = pd.DataFrame(
        all_records
    )

else:

    consolidated = pd.DataFrame()

output = (
    RESULTS /
    "three_gene_cross_cohort_raw_effect_extraction.csv"
)

consolidated.to_csv(
    output,
    index=False
)

# ============================================================
# SAVE HUMAN-READABLE LOG
# ============================================================

log_file = (
    LOGS /
    "three_gene_cross_cohort_effect_extraction.txt"
)

with open(
    log_file,
    "w"
) as f:

    f.write(
        "THREE-GENE CROSS-COHORT EFFECT EXTRACTION\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    for line in log_lines:

        f.write(
            str(line) + "\n"
        )

    f.write(
        "\n\nCONSOLIDATED RECORDS: "
        + str(len(consolidated))
        + "\n"
    )

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("EXTRACTION SUMMARY")
print("=" * 90)

print(
    "Total extracted records:",
    len(consolidated)
)

if len(consolidated):

    print(
        "\nGenes represented:"
    )

    if "Gene" in consolidated.columns:

        print(
            consolidated[
                "Gene"
            ].value_counts()
        )

    print(
        "\nSources represented:"
    )

    if "Source" in consolidated.columns:

        print(
            consolidated[
                "Source"
            ].value_counts()
        )

    print(
        "\nConsolidated table preview:"
    )

    print(
        consolidated.head(30)
        .to_string(index=False)
    )

print("\nSaved:")
print(output)
print(log_file)

print("\n" + "=" * 90)
print("EFFECT EXTRACTION COMPLETE")
print("=" * 90)

print(
    "\nIMPORTANT:"
)

print(
    "No meta-analysis was performed."
)

print(
    "Effect estimates must be checked for "
    "comparability before statistical pooling."
)
