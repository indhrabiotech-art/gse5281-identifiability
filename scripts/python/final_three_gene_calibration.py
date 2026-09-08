import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression


OUTDIR = "04_ML/Final_Model/Validation"
MANUSCRIPT = "06_Manuscript/Tables"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MANUSCRIPT, exist_ok=True)


GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

test = pd.read_csv(
    "04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv",
    index_col=0
)

test_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_test_meta.csv"
)

external = pd.read_csv(
    "04_ML/Final_Characterization/"
    "GSE5281_3gene_predictions_final.csv"
)


# ============================================================
# LABELS
# ============================================================

train_y = (
    train_meta["Diagnosis"]
    .astype(str)
    .str.strip()
    .eq("AD")
    .astype(int)
    .to_numpy()
)

test_y = (
    test_meta["Diagnosis"]
    .astype(str)
    .str.strip()
    .eq("AD")
    .astype(int)
    .to_numpy()
)

external_y = (
    external["Diagnosis"]
    .astype(str)
    .str.strip()
    .eq("AD")
    .astype(int)
    .to_numpy()
)


# ============================================================
# LOCKED THREE-GENE SCORE
# ============================================================

def score_dataset(df):

    z = pd.DataFrame(index=df.index)

    for gene in GENES:

        z[gene] = (
            df[gene] - train[gene].mean()
        ) / train[gene].std(ddof=1)

    return z[GENES].mean(axis=1).to_numpy()


train_score = score_dataset(train)

test_score = score_dataset(test)

external_score = score_dataset(
    external[GENES]
)


# ============================================================
# CONVERT SCORE TO PROBABILITY
# ============================================================
#
# Fit probability calibration ONLY on training data.
# Then freeze this mapping for test/external validation.
#
# ============================================================

calibrator = LogisticRegression(
    solver="lbfgs"
)

calibrator.fit(
    train_score.reshape(-1, 1),
    train_y
)


train_prob = calibrator.predict_proba(
    train_score.reshape(-1, 1)
)[:, 1]

test_prob = calibrator.predict_proba(
    test_score.reshape(-1, 1)
)[:, 1]

external_prob = calibrator.predict_proba(
    external_score.reshape(-1, 1)
)[:, 1]


# ============================================================
# CALIBRATION FUNCTION
# ============================================================

def calibration_metrics(
    name,
    y,
    probability
):

    # Calibration intercept/slope
    eps = 1e-6

    p = np.clip(
        probability,
        eps,
        1 - eps
    )

    logit_p = np.log(
        p / (1 - p)
    )

    X = sm.add_constant(
        logit_p
    )

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial()
    ).fit()

    intercept = model.params[0]
    slope = model.params[1]

    # Brier score
    brier = brier_score_loss(
        y,
        probability
    )

    auc = roc_auc_score(
        y,
        probability
    )

    return {
        "Dataset": name,
        "N": len(y),
        "AD": int(y.sum()),
        "Control": int((y == 0).sum()),
        "ROC_AUC": auc,
        "Brier_score": brier,
        "Calibration_intercept": intercept,
        "Calibration_slope": slope,
        "Mean_predicted_probability": np.mean(
            probability
        ),
        "Observed_AD_fraction": np.mean(y)
    }


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame([

    calibration_metrics(
        "GSE48350_Training",
        train_y,
        train_prob
    ),

    calibration_metrics(
        "GSE48350_HeldOut_Test",
        test_y,
        test_prob
    ),

    calibration_metrics(
        "GSE5281_External",
        external_y,
        external_prob
    )

])


# ============================================================
# PRINT
# ============================================================

print("=" * 80)
print("FINAL THREE-GENE CALIBRATION ANALYSIS")
print("=" * 80)

print()

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# CALIBRATION DECISION
# ============================================================

print()
print("=" * 80)
print("CALIBRATION INTERPRETATION")
print("=" * 80)

for _, r in results.iterrows():

    print()
    print(r["Dataset"])

    print(
        "Brier score:",
        f"{r['Brier_score']:.4f}"
    )

    print(
        "Calibration intercept:",
        f"{r['Calibration_intercept']:.4f}"
    )

    print(
        "Calibration slope:",
        f"{r['Calibration_slope']:.4f}"
    )

    if (
        abs(r["Calibration_intercept"]) <= 0.25
        and
        0.75 <= r["Calibration_slope"] <= 1.25
    ):

        print("Calibration status: ACCEPTABLE")

    else:

        print(
            "Calibration status: "
            "ATTENUATED / NEEDS CAUTION"
        )


# ============================================================
# SAVE
# ============================================================

out = (
    OUTDIR +
    "/FINAL_THREE_GENE_CALIBRATION.csv"
)

results.to_csv(
    out,
    index=False
)

manuscript = (
    MANUSCRIPT +
    "/Table_final_three_gene_calibration.csv"
)

results.to_csv(
    manuscript,
    index=False
)


print()
print("Saved:")
print(out)
print(manuscript)

print()
print("=" * 80)
print("CALIBRATION ANALYSIS COMPLETE")
print("=" * 80)
