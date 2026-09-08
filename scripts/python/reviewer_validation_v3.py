import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

SEED = 42

N_REPEATS = 50
OUTER_FOLDS = 5
INNER_FOLDS = 5

N_RANDOM = 1000

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

ROOT = Path.home() / "project_ml"

OUT = ROOT / "04_ML" / "Reviewer_Robustness_V3"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_EXPR = (
    ROOT / "04_ML" / "GSE48350_RMA_ML_matrix.csv"
)

TRAIN_META = (
    ROOT / "04_ML" / "GSE48350_RMA_train_meta.csv"
)

TEST_META = (
    ROOT / "04_ML" / "GSE48350_RMA_test_meta.csv"
)

EXT_EXPR = (
    ROOT / "04_ML" / "GSE5281_RMA_genelevel_harmonized.csv"
)

EXT_META = (
    ROOT
    / "04_ML"
    / "External_Validation"
    / "GSE5281_validation_metadata.csv"
)

# ============================================================
# LOGGING
# ============================================================

LOG = OUT / "run.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")


# ============================================================
# LOAD DISCOVERY EXPRESSION
# ============================================================

def load_discovery_expression():

    log("")
    log("=" * 90)
    log("LOADING GSE48350 EXPRESSION")
    log("=" * 90)

    df = pd.read_csv(TRAIN_EXPR)

    if "Unnamed: 0" not in df.columns:
        raise RuntimeError(
            "Expected Unnamed: 0 sample-ID column."
        )

    df["Unnamed: 0"] = (
        df["Unnamed: 0"]
        .astype(str)
        .str.strip()
    )

    # samples x genes
    expr = df.set_index("Unnamed: 0")

    expr = expr.apply(
        pd.to_numeric,
        errors="coerce"
    )

    expr = expr.replace(
        [np.inf, -np.inf],
        np.nan
    )

    expr = expr.dropna(
        axis=1,
        how="all"
    )

    # Median-impute any remaining missing gene values.
    expr = expr.fillna(
        expr.median()
    )

    log(
        f"Expression shape: {expr.shape}"
    )

    log(
        f"Samples: {expr.shape[0]}"
    )

    log(
        f"Genes: {expr.shape[1]}"
    )

    return expr


# ============================================================
# LOAD DISCOVERY METADATA
# ============================================================

def load_discovery_metadata(path):

    df = pd.read_csv(path)

    df["_sample"] = (
        df["index"]
        .astype(str)
        .str.strip()
    )

    df["_age"] = pd.to_numeric(
        df["Age"],
        errors="coerce"
    )

    df["_sex"] = (
        df["Sex"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["_y"] = (
        df["Diagnosis"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map({
            "ad": 1,
            "control": 0
        })
    )

    if df["_y"].isna().any():
        raise RuntimeError(
            "Unknown diagnosis labels."
        )

    return df


# ============================================================
# LOAD EXTERNAL METADATA
# ============================================================

def load_external_metadata():

    df = pd.read_csv(
        EXT_META
    )

    if "Unnamed: 0" in df.columns:
        sample_col = "Unnamed: 0"
    elif "GSM" in df.columns:
        sample_col = "GSM"
    else:
        raise RuntimeError(
            "Cannot identify GSE5281 sample ID column."
        )

    df["_sample"] = (
        df[sample_col]
        .astype(str)
        .str.strip()
    )

    df["_age"] = pd.to_numeric(
        df["Age_used_years"],
        errors="coerce"
    )

    # GSE5281 metadata has no sex field.
    df["_sex"] = np.nan

    df["_y"] = (
        df["diagnosis"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map({
            "ad": 1,
            "control": 0
        })
    )

    return df


# ============================================================
# ALIGN TRAINING MATRIX TO METADATA
# ============================================================

def align_discovery(expr, meta):

    common = [
        s
        for s in meta["_sample"]
        if s in expr.index
    ]

    log("")
    log(
        f"Common discovery samples: {len(common)}"
    )

    if len(common) != len(meta):
        missing = [
            s
            for s in meta["_sample"]
            if s not in expr.index
        ]

        log(
            "Missing metadata samples:"
        )

        log(
            str(missing)
        )

    aligned_expr = expr.loc[
        common
    ].copy()

    aligned_meta = (
        meta
        .set_index("_sample")
        .loc[common]
        .copy()
    )

    return aligned_expr, aligned_meta



# ============================================================
# ALIGN EXPRESSION MATRIX TO METADATA
# ============================================================

def align(expr, meta):

    common = [
        s for s in meta["_sample"]
        if s in expr.index
    ]

    log("")
    log(
        f"Common samples found: {len(common)}"
    )

    if len(common) == 0:
        raise RuntimeError(
            "No matching sample IDs between expression "
            "matrix and metadata."
        )

    missing = [
        s for s in meta["_sample"]
        if s not in expr.index
    ]

    if missing:
        log(
            f"WARNING: {len(missing)} metadata samples "
            f"were not found in expression matrix."
        )
        log(
            f"Missing IDs: {missing}"
        )

    aligned_expr = expr.loc[common].copy()

    aligned_meta = (
        meta
        .set_index("_sample")
        .loc[common]
        .copy()
    )

    return aligned_expr, aligned_meta


# ============================================================
# BASIC METRICS
# ============================================================

def calc_auc(y, p):

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(
        y,
        p
    )


def calc_ap(y, p):

    if len(np.unique(y)) < 2:
        return np.nan

    return average_precision_score(
        y,
        p
    )


# ============================================================
# AGE / SEX BASELINES
# ============================================================

def demographic_baseline(meta, cohort):

    y = meta["_y"].astype(int).values

    age = (
        meta["_age"]
        .astype(float)
        .values
    )

    rows = []

    rows.append({
        "cohort": cohort,
        "model": "Age",
        "n": len(y),
        "AD": int(y.sum()),
        "Control": int((y == 0).sum()),
        "AUC": calc_auc(y, age),
        "AP": calc_ap(y, age)
    })

    if meta["_sex"].notna().all():

        sex = (
            meta["_sex"]
            .map({
                "female": 0,
                "male": 1
            })
            .astype(float)
            .values
        )

        rows.append({
            "cohort": cohort,
            "model": "Sex",
            "n": len(y),
            "AD": int(y.sum()),
            "Control": int((y == 0).sum()),
            "AUC": calc_auc(y, sex),
            "AP": calc_ap(y, sex)
        })

        X = np.column_stack([
            age,
            sex
        ])

        model = Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=5000,
                    random_state=SEED
                )
            )
        ])

        model.fit(
            X,
            y
        )

        p = model.predict_proba(
            X
        )[:, 1]

        rows.append({
            "cohort": cohort,
            "model": "Age+Sex",
            "n": len(y),
            "AD": int(y.sum()),
            "Control": int((y == 0).sum()),
            "AUC": calc_auc(y, p),
            "AP": calc_ap(y, p)
        })

    return pd.DataFrame(rows)


# ============================================================
# CONTROLS-ONLY AGE CORRECTION
# ============================================================

def fit_age_models(
    X_train,
    meta_train
):

    control_mask = (
        meta_train["_y"].values == 0
    )

    ages = (
        meta_train["_age"]
        .astype(float)
        .values
    )

    age_control = ages[
        control_mask
    ]

    center = age_control.mean()

    models = {}

    for gene in X_train.columns:

        y_control = X_train.loc[
            meta_train.index[
                control_mask
            ],
            gene
        ].astype(float).values

        lr = LinearRegression()

        lr.fit(
            (
                age_control - center
            ).reshape(-1, 1),
            y_control
        )

        models[gene] = lr

    return models, center


def apply_age_models(
    X,
    meta,
    models,
    center
):

    ages = (
        meta["_age"]
        .astype(float)
        .values
    )

    out = pd.DataFrame(
        index=X.index,
        columns=X.columns,
        dtype=float
    )

    for gene in X.columns:

        expected = models[gene].predict(
            (
                ages - center
            ).reshape(-1, 1)
        )

        out[gene] = (
            X[gene].values
            - expected
        )

    return out


# ============================================================
# LASSO FEATURE SELECTION
# ============================================================

def select_features(
    X,
    y,
    C
):

    model = Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "lasso",
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=C,
                max_iter=5000,
                random_state=SEED
            )
        )
    ])

    model.fit(
        X,
        y
    )

    coef = (
        model
        .named_steps["lasso"]
        .coef_[0]
    )

    genes = X.columns.tolist()

    selected = [
        genes[i]
        for i, c in enumerate(coef)
        if abs(c) > 1e-10
    ]

    return selected


# ============================================================
# NESTED CV
# ============================================================

def run_nested_cv(
    expr,
    meta
):

    log("")
    log("=" * 90)
    log("STARTING 50 x 5-FOLD NESTED CV")
    log("=" * 90)

    X = expr.copy()

    y = (
        meta["_y"]
        .astype(int)
        .values
    )

    # Use the existing filtered gene background:
    # 5,342 highest-variance genes.
    variances = X.var(
        axis=0
    )

    background_genes = (
        variances
        .sort_values(
            ascending=False
        )
        .head(
            min(
                5342,
                len(variances)
            )
        )
        .index
        .tolist()
    )

    X = X[
        background_genes
    ].copy()

    log(
        f"Nested-CV background genes: "
        f"{X.shape[1]}"
    )

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS,
        n_repeats=N_REPEATS,
        random_state=SEED
    )

    results = []

    feature_counts = {
        g: 0
        for g in GENES
    }

    all_feature_counts = {}

    iteration = 0

    for train_idx, test_idx in outer.split(
        X,
        y
    ):

        iteration += 1

        Xtr = X.iloc[
            train_idx
        ].copy()

        Xte = X.iloc[
            test_idx
        ].copy()

        mtr = meta.iloc[
            train_idx
        ].copy()

        mte = meta.iloc[
            test_idx
        ].copy()

        ytr = y[
            train_idx
        ]

        yte = y[
            test_idx
        ]

        # --------------------------------------------------------
        # AGE CORRECTION
        # --------------------------------------------------------

        age_models, age_center = fit_age_models(
            Xtr,
            mtr
        )

        Xtr_adj = apply_age_models(
            Xtr,
            mtr,
            age_models,
            age_center
        )

        Xte_adj = apply_age_models(
            Xte,
            mte,
            age_models,
            age_center
        )

        # --------------------------------------------------------
        # INNER CV FOR LASSO C
        # --------------------------------------------------------

        inner = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + iteration
        )

        best_C = None
        best_score = -np.inf

        for C in [
            0.01,
            0.03,
            0.05,
            0.1,
            0.2,
            0.5,
            1.0
        ]:

            scores = []

            for itr, iva in inner.split(
                Xtr_adj,
                ytr
            ):

                model = Pipeline([
                    (
                        "scale",
                        StandardScaler()
                    ),
                    (
                        "lasso",
                        LogisticRegression(
                            penalty="l1",
                            solver="liblinear",
                            C=C,
                            max_iter=5000,
                            random_state=SEED
                        )
                    )
                ])

                model.fit(
                    Xtr_adj.iloc[itr],
                    ytr[itr]
                )

                p = model.predict_proba(
                    Xtr_adj.iloc[iva]
                )[:, 1]

                scores.append(
                    calc_auc(
                        ytr[iva],
                        p
                    )
                )

            score = np.nanmean(
                scores
            )

            if score > best_score:

                best_score = score
                best_C = C

        # --------------------------------------------------------
        # FINAL FEATURE SELECTION
        # --------------------------------------------------------

        selected = select_features(
            Xtr_adj,
            ytr,
            best_C
        )

        # Record all selected genes.
        for g in selected:

            all_feature_counts[g] = (
                all_feature_counts.get(g, 0)
                + 1
            )

        for g in GENES:

            if g in selected:

                feature_counts[g] += 1

        # Fallback if LASSO selects nothing.
        if len(selected) == 0:

            ranked = (
                Xtr_adj
                .var()
                .sort_values(
                    ascending=False
                )
            )

            selected = [
                ranked.index[0]
            ]

        # --------------------------------------------------------
        # FINAL OUTER MODEL
        # --------------------------------------------------------

        model = Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=5000,
                    random_state=SEED
                )
            )
        ])

        model.fit(
            Xtr_adj[selected],
            ytr
        )

        p = model.predict_proba(
            Xte_adj[selected]
        )[:, 1]

        results.append({
            "iteration": iteration,
            "AUC": calc_auc(
                yte,
                p
            ),
            "AP": calc_ap(
                yte,
                p
            ),
            "best_C": best_C,
            "n_selected": len(selected),
            "selected_genes":
                ";".join(selected)
        })

        if iteration % 25 == 0:

            log(
                f"Completed "
                f"{iteration}/"
                f"{N_REPEATS * OUTER_FOLDS}"
            )

    results_df = pd.DataFrame(
        results
    )

    feature_df = pd.DataFrame([
        {
            "gene": g,
            "selection_count": n,
            "selection_frequency":
                n / len(results_df)
        }
        for g, n
        in sorted(
            all_feature_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ])

    summary = pd.DataFrame([{
        "outer_folds": OUTER_FOLDS,
        "repeats": N_REPEATS,
        "iterations": len(results_df),
        "mean_AUC":
            results_df["AUC"].mean(),
        "median_AUC":
            results_df["AUC"].median(),
        "SD_AUC":
            results_df["AUC"].std(),
        "AUC_2.5pct":
            results_df["AUC"].quantile(0.025),
        "AUC_97.5pct":
            results_df["AUC"].quantile(0.975),
        "mean_AP":
            results_df["AP"].mean(),
        "median_AP":
            results_df["AP"].median(),
        "ABCA6_selection_frequency":
            feature_counts["ABCA6"] / len(results_df),
        "CRLF1_selection_frequency":
            feature_counts["CRLF1"] / len(results_df),
        "TNFRSF11B_selection_frequency":
            feature_counts["TNFRSF11B"] / len(results_df)
    }])

    results_df.to_csv(
        OUT / "nested_cv_results.csv",
        index=False
    )

    feature_df.to_csv(
        OUT / "nested_cv_feature_selection_frequency.csv",
        index=False
    )

    summary.to_csv(
        OUT / "nested_cv_summary.csv",
        index=False
    )

    log("")
    log("NESTED CV SUMMARY")
    log(
        summary.to_string(
            index=False
        )
    )

    log("")
    log("TOP FEATURES")
    log(
        feature_df.head(30).to_string(
            index=False
        )
    )


# ============================================================
# RANDOM 3-GENE NULL
# ============================================================

def run_random_null(
    expr,
    meta
):

    log("")
    log("=" * 90)
    log("RANDOM 3-GENE NULL")
    log("=" * 90)

    X = expr.copy()

    y = (
        meta["_y"]
        .astype(int)
        .values
    )

    variances = X.var(
        axis=0
    )

    genes = (
        variances
        .sort_values(
            ascending=False
        )
        .head(
            min(
                5342,
                len(variances)
            )
        )
        .index
        .tolist()
    )

    X = X[
        genes
    ]

    rng = np.random.default_rng(
        SEED
    )

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=SEED
    )

    rows = []

    for panel_id in range(
        1,
        N_RANDOM + 1
    ):

        panel = rng.choice(
            genes,
            size=3,
            replace=False
        ).tolist()

        scores = []

        for tr, te in cv.split(
            X,
            y
        ):

            model = Pipeline([
                (
                    "scale",
                    StandardScaler()
                ),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=SEED
                    )
                )
            ])

            model.fit(
                X.iloc[tr][panel],
                y[tr]
            )

            p = model.predict_proba(
                X.iloc[te][panel]
            )[:, 1]

            scores.append(
                calc_auc(
                    y[te],
                    p
                )
            )

        rows.append({
            "panel_id": panel_id,
            "gene1": panel[0],
            "gene2": panel[1],
            "gene3": panel[2],
            "mean_cv_auc":
                np.nanmean(scores)
        })

        if panel_id % 100 == 0:

            log(
                f"Random panels "
                f"{panel_id}/{N_RANDOM}"
            )

    null = pd.DataFrame(
        rows
    )

    # Observed panel under IDENTICAL 5x10 CV.
    observed_scores = []

    for tr, te in cv.split(
        X,
        y
    ):

        model = Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=5000,
                    random_state=SEED
                )
            )
        ])

        model.fit(
            X.iloc[tr][GENES],
            y[tr]
        )

        p = model.predict_proba(
            X.iloc[te][GENES]
        )[:, 1]

        observed_scores.append(
            calc_auc(
                y[te],
                p
            )
        )

    observed = np.nanmean(
        observed_scores
    )

    vals = null[
        "mean_cv_auc"
    ].values

    summary = pd.DataFrame([{
        "observed_panel":
            ";".join(GENES),
        "observed_mean_cv_auc":
            observed,
        "null_mean":
            vals.mean(),
        "null_sd":
            vals.std(ddof=1),
        "null_median":
            np.median(vals),
        "null_q025":
            np.quantile(vals, 0.025),
        "null_q975":
            np.quantile(vals, 0.975),
        "empirical_percentile":
            np.mean(
                vals <= observed
            ),
        "n_random_panels":
            len(vals)
    }])

    null.to_csv(
        OUT / "random_3gene_null_v3.csv",
        index=False
    )

    summary.to_csv(
        OUT / "random_3gene_null_summary_v3.csv",
        index=False
    )

    log("")
    log(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# EXTERNAL GSE5281
# ============================================================

def load_external_expression():

    df = pd.read_csv(
        EXT_EXPR
    )

    # Detect orientation.
    # If genes are rows and GSMs columns.
    first = df.columns[0]

    if df.shape[0] > df.shape[1]:

        expr = df.set_index(
            first
        )

        expr = expr.apply(
            pd.to_numeric,
            errors="coerce"
        )

        expr = expr.T

    else:

        # samples x genes
        expr = df.copy()

        if first not in [
            "Unnamed: 0",
            "GSM",
            "sample",
            "Sample"
        ]:

            expr = expr.set_index(
                first
            )

        else:

            expr = expr.set_index(
                first
            )

        expr = expr.apply(
            pd.to_numeric,
            errors="coerce"
        )

    expr = expr.replace(
        [np.inf, -np.inf],
        np.nan
    )

    expr = expr.fillna(
        expr.median()
    )

    expr.index = (
        expr.index
        .astype(str)
        .str.strip()
    )

    return expr


def run_external(
    train_expr,
    train_meta
):

    if not EXT_EXPR.exists():

        log(
            "GSE5281 expression file missing."
        )
        return

    if not EXT_META.exists():

        log(
            "GSE5281 metadata file missing."
        )
        return

    log("")
    log("=" * 90)
    log("GSE5281 EXTERNAL VALIDATION")
    log("=" * 90)

    ext_expr = load_external_expression()
    ext_meta = load_external_metadata()

    common = [
        s
        for s in ext_meta["_sample"]
        if s in ext_expr.index
    ]

    log(
        f"GSE5281 common samples: "
        f"{len(common)}"
    )

    ext_meta = (
        ext_meta
        .set_index("_sample")
        .loc[common]
    )

    ext_expr = (
        ext_expr
        .loc[common]
    )

    # Check genes.
    missing = [
        g
        for g in GENES
        if g not in train_expr.columns
        or g not in ext_expr.columns
    ]

    if missing:

        log(
            f"Missing genes: {missing}"
        )
        return

    Xtr = train_expr[
        GENES
    ]

    ytr = (
        train_meta["_y"]
        .astype(int)
        .values
    )

    Xext = ext_expr[
        GENES
    ]

    yext = (
        ext_meta["_y"]
        .astype(int)
        .values
    )

    model = Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "lr",
            LogisticRegression(
                max_iter=5000,
                random_state=SEED
            )
        )
    ])

    model.fit(
        Xtr,
        ytr
    )

    p = model.predict_proba(
        Xext
    )[:, 1]

    result = {
        "cohort": "GSE5281",
        "n": len(yext),
        "AD": int(yext.sum()),
        "Control": int((yext == 0).sum()),
        "AUC": calc_auc(
            yext,
            p
        ),
        "AP": calc_ap(
            yext,
            p
        ),
        "Brier": brier_score_loss(
            yext,
            p
        )
    }

    try:

        import statsmodels.api as sm

        pclip = np.clip(
            p,
            1e-6,
            1 - 1e-6
        )

        logit_p = np.log(
            pclip / (1 - pclip)
        )

        Xcal = sm.add_constant(
            logit_p
        )

        cal = sm.Logit(
            yext,
            Xcal
        ).fit(
            disp=False
        )

        result[
            "Calibration_intercept"
        ] = cal.params[0]

        result[
            "Calibration_slope"
        ] = cal.params[1]

    except Exception as e:

        log(
            f"Calibration model failed: {e}"
        )

        result[
            "Calibration_intercept"
        ] = np.nan

        result[
            "Calibration_slope"
        ] = np.nan

    pd.DataFrame(
        [result]
    ).to_csv(
        OUT /
        "GSE5281_locked_3gene_calibration.csv",
        index=False
    )

    pred = ext_meta.copy()

    pred[
        "predicted_probability"
    ] = p

    pred[
        "outcome"
    ] = yext

    pred.to_csv(
        OUT /
        "GSE5281_locked_3gene_predictions.csv"
    )

    log("")
    log(
        "GSE5281 3-GENE RESULT"
    )

    log(
        pd.DataFrame(
            [result]
        ).to_string(
            index=False
        )
    )

    # Age baseline.
    age = ext_meta[
        "_age"
    ].astype(float).values

    age_result = pd.DataFrame([{
        "cohort": "GSE5281",
        "model": "Age",
        "AUC": calc_auc(
            yext,
            age
        ),
        "AP": calc_ap(
            yext,
            age
        )
    }])

    age_result.to_csv(
        OUT /
        "GSE5281_age_baseline.csv",
        index=False
    )

    log("")
    log(
        "GSE5281 AGE BASELINE"
    )

    log(
        age_result.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    log("=" * 90)
    log("REVIEWER VALIDATION V3")
    log("=" * 90)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    expr = load_discovery_expression()

    train_meta = load_discovery_metadata(
        TRAIN_META
    )

    test_meta = load_discovery_metadata(
        TEST_META
    )

    # --------------------------------------------------------
    # ALIGN FULL 62-SAMPLE MATRIX
    # --------------------------------------------------------

    train_expr, train_meta = align(
        expr,
        train_meta
    )

    test_expr, test_meta = align(
        expr,
        test_meta
    )

    log("")
    log(
        f"Training samples aligned: "
        f"{len(train_meta)}"
    )

    log(
        f"Held-out samples aligned: "
        f"{len(test_meta)}"
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    if len(train_meta) != 49:
        raise RuntimeError(
            f"Expected 49 training samples, got "
            f"{len(train_meta)}"
        )

    if len(test_meta) != 13:
        raise RuntimeError(
            f"Expected 13 test samples, got "
            f"{len(test_meta)}"
        )

    # --------------------------------------------------------
    # DEMOGRAPHICS
    # --------------------------------------------------------

    demo_train = demographic_baseline(
        train_meta,
        "GSE48350_TRAIN"
    )

    demo_test = demographic_baseline(
        test_meta,
        "GSE48350_TEST"
    )

    demographics = pd.concat(
        [
            demo_train,
            demo_test
        ],
        ignore_index=True
    )

    demographics.to_csv(
        OUT /
        "GSE48350_demographic_baselines.csv",
        index=False
    )

    log("")
    log(
        demographics.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # CONTROLS-ONLY AGE CORRECTION — DESCRIPTIVE
    # --------------------------------------------------------

    log("")
    log("=" * 90)
    log("CONTROLS-ONLY AGE CORRECTION")
    log("=" * 90)

    age_models, center = fit_age_models(
        train_expr,
        train_meta
    )

    train_adj = apply_age_models(
        train_expr,
        train_meta,
        age_models,
        center
    )

    available = [
        g
        for g in GENES
        if g in train_adj.columns
    ]

    if len(available) == 3:

        y = (
            train_meta["_y"]
            .astype(int)
            .values
        )

        model = Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=5000,
                    random_state=SEED
                )
            )
        ])

        model.fit(
            train_adj[GENES],
            y
        )

        p = model.predict_proba(
            train_adj[GENES]
        )[:, 1]

        result = pd.DataFrame([{
            "model":
                "3gene_controls_only_age_adjusted",
            "AUC":
                calc_auc(y, p),
            "AP":
                calc_ap(y, p)
        }])

        result.to_csv(
            OUT /
            "three_gene_controls_only_age_adjusted.csv",
            index=False
        )

        log(
            result.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # NESTED CV
    # --------------------------------------------------------

    run_nested_cv(
        train_expr,
        train_meta
    )

    # --------------------------------------------------------
    # RANDOM NULL
    # --------------------------------------------------------

    run_random_null(
        train_expr,
        train_meta
    )

    # --------------------------------------------------------
    # EXTERNAL
    # --------------------------------------------------------

    run_external(
        train_expr,
        train_meta
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    elapsed = (
        time.time() - start
    )

    log("")
    log("=" * 90)
    log("VALIDATION V3 COMPLETE")
    log("=" * 90)

    log(
        f"Runtime: "
        f"{elapsed / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
