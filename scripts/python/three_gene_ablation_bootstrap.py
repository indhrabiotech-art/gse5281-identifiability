import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


INPUT = (
    "04_ML/Final_Model/Validation/"
    "final_three_gene_ablation_results.csv"
)

OUTDIR = "04_ML/Final_Model/Validation"

os.makedirs(OUTDIR, exist_ok=True)

N_BOOT = 5000
SEED = 42

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------
# Load original expression data
# ------------------------------------------------------------

DATASETS = {
    "GSE48350_Training": (
        "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
        "04_ML/GSE48350_RMA_train_meta.csv"
    ),

    "GSE48350_HeldOut_Test": (
        "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
        "04_ML/GSE48350_RMA_test_meta.csv"
    ),

    "GSE5281_External": (
        "04_ML/Final_Characterization/"
        "GSE5281_3gene_predictions_final.csv",
        None
    )
}


SIGNATURES = {
    "ABCA6": ["ABCA6"],
    "CRLF1": ["CRLF1"],
    "TNFRSF11B": ["TNFRSF11B"],

    "ABCA6_CRLF1": [
        "ABCA6",
        "CRLF1"
    ],

    "ABCA6_TNFRSF11B": [
        "ABCA6",
        "TNFRSF11B"
    ],

    "CRLF1_TNFRSF11B": [
        "CRLF1",
        "TNFRSF11B"
    ],

    "ABCA6_CRLF1_TNFRSF11B": [
        "ABCA6",
        "CRLF1",
        "TNFRSF11B"
    ]
}


def load_dataset(name, expr_file, meta_file):

    if name == "GSE5281_External":

        x = pd.read_csv(expr_file)

        y = (
            x["Diagnosis"]
            .map({
                "Control": 0,
                "AD": 1
            })
            .values
        )

        expr = x[
            [
                "ABCA6",
                "CRLF1",
                "TNFRSF11B"
            ]
        ].astype(float)

        return expr, y


    expr = pd.read_csv(
        expr_file,
        index_col=0
    )

    meta = pd.read_csv(
        meta_file
    )

    y = (
        meta["Diagnosis"]
        .map({
            "Control": 0,
            "AD": 1
        })
        .values
    )

    expr = expr[
        [
            "ABCA6",
            "CRLF1",
            "TNFRSF11B"
        ]
    ].astype(float)

    return expr, y


# ------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------

all_results = []


for dataset, files in DATASETS.items():

    print("\n" + "=" * 70)
    print(dataset)
    print("=" * 70)

    expr, y = load_dataset(
        dataset,
        files[0],
        files[1]
    )

    # Same within-dataset standardization as the ablation analysis
    z = (
        expr - expr.mean(axis=0)
    ) / expr.std(axis=0, ddof=1)

    scores = {}

    for name, genes in SIGNATURES.items():

        scores[name] = (
            z[genes]
            .mean(axis=1)
            .values
        )

    boot = {
        name: []
        for name in SIGNATURES
    }

    winner_counts = {
        name: 0
        for name in SIGNATURES
    }

    comparison_counts = {
        name: 0
        for name in SIGNATURES
    }

    for i in range(N_BOOT):

        while True:

            idx = rng.integers(
                0,
                len(y),
                len(y)
            )

            if len(np.unique(y[idx])) == 2:
                break

        aucs = {}

        for name in SIGNATURES:

            auc = roc_auc_score(
                y[idx],
                scores[name][idx]
            )

            aucs[name] = auc
            boot[name].append(auc)

        # Best signature in this bootstrap replicate
        winner = max(
            aucs,
            key=aucs.get
        )

        winner_counts[winner] += 1

    # --------------------------------------------------------
    # Summarize
    # --------------------------------------------------------

    full = np.array(
        boot["ABCA6_CRLF1_TNFRSF11B"]
    )

    for name in SIGNATURES:

        values = np.array(
            boot[name]
        )

        delta = values - full

        result = {
            "Dataset": dataset,
            "Signature": name,
            "Observed_AUC": roc_auc_score(
                y,
                scores[name]
            ),
            "Bootstrap_N": len(values),
            "Median_AUC": np.median(values),
            "CI_95_lower": np.percentile(
                values,
                2.5
            ),
            "CI_95_upper": np.percentile(
                values,
                97.5
            ),
            "Mean_Delta_AUC_vs_Full": np.mean(
                delta
            ),
            "Delta_CI_95_lower": np.percentile(
                delta,
                2.5
            ),
            "Delta_CI_95_upper": np.percentile(
                delta,
                97.5
            ),
            "Probability_Better_Than_Full": np.mean(
                delta > 0
            ),
            "Probability_Winner": (
                winner_counts[name]
                / N_BOOT
            )
        }

        all_results.append(result)

        print(
            f"{name:25s}"
            f" observed={result['Observed_AUC']:.4f}"
            f" median={result['Median_AUC']:.4f}"
            f" CI="
            f"{result['CI_95_lower']:.4f}"
            f"-"
            f"{result['CI_95_upper']:.4f}"
            f" P>full="
            f"{result['Probability_Better_Than_Full']:.3f}"
        )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results = pd.DataFrame(
    all_results
)

outfile = (
    f"{OUTDIR}/"
    "three_gene_ablation_bootstrap.csv"
)

results.to_csv(
    outfile,
    index=False
)

manuscript = (
    "06_Manuscript/Tables/"
    "Table_three_gene_ablation_bootstrap.csv"
)

results.to_csv(
    manuscript,
    index=False
)


print("\n" + "=" * 70)
print("ABLATION BOOTSTRAP COMPLETE")
print("=" * 70)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved:")
print(outfile)
print(manuscript)

