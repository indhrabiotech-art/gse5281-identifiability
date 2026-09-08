#!/usr/bin/env python3

# =============================================================================
# REVIEWER ROBUSTNESS MASTER ANALYSIS
# Alzheimer's disease 3-gene candidate signature
#
# Candidate genes:
#   ABCA6
#   CRLF1
#   TNFRSF11B
#
# Main objectives:
#   1. Demographic baselines
#   2. Current 3-gene model baseline
#   3. Calibration
#   4. Controls-only age correction
#   5. Repeated nested CV
#   6. Nested feature-selection stability
#   7. Random 3-gene null panels
#
# Project root:
#   ~/project_ml
#
# Fixed random seed:
#   42
#
# IMPORTANT:
# This script is designed to be conservative. It will not silently invent
# metadata or silently substitute datasets.
# =============================================================================

from pathlib import Path
import sys
import os
import json
import time
import warnings
import platform
import subprocess

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

NESTED_REPEATS = 50
OUTER_FOLDS = 5
INNER_FOLDS = 5

RANDOM_NULL_PANELS = 1000

LASSO_C_GRID = [0.1, 0.3, 1.0]

TOP_VARIANCE_N = 5342

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B",
]

ROOT = Path.home() / "project_ml"

META_TRAIN = ROOT / "04_ML" / "GSE48350_RMA_train_meta.csv"
META_TEST = ROOT / "04_ML" / "GSE48350_RMA_test_meta.csv"

TRAIN_MATRIX_CANDIDATES = [
    ROOT / "04_ML" / "GSE48350_RMA_train_age_adjusted.csv",
    ROOT / "04_ML" / "GSE48350_RMA_train_age_adjusted",
    ROOT / "04_ML" / "GSE48350_RMA_train.csv",
    ROOT / "04_ML" / "GSE48350_RMA_ML_matrix.csv",
    ROOT / "03_Preprocessing" / "GSE48350_RMA_genelevel.csv",
]

TEST_MATRIX_CANDIDATES = [
    ROOT / "04_ML" / "GSE48350_RMA_test_age_adjusted.csv",
    ROOT / "04_ML" / "GSE48350_RMA_test.csv",
    ROOT / "03_Preprocessing" / "GSE48350_RMA_genelevel.csv",
]

EXTERNAL_MATRIX_CANDIDATES = [
    ROOT / "04_ML" / "GSE5281_RMA_genelevel_harmonized.csv",
    ROOT / "04_ML" / "GSE5281_RMA_genelevel_age_adjusted.csv",
    ROOT / "04_ML" / "GSE5281_RMA_ML_matrix.csv",
    ROOT / "03_Preprocessing" / "GSE5281_RMA_genelevel.csv",
]

EXTERNAL_META_CANDIDATES = [
    ROOT / "02_Metadata" / "GSE5281_hippocampus_metadata.csv",
    ROOT / "02_Metadata" / "GSE5281_hippocampus_metadata_age.csv",
    ROOT / "02_Metadata" / "GSE5281_hippocampus_metadata_age_corrected.csv",
    ROOT / "04_ML" / "External_Validation" / "GSE5281_validation_metadata.csv",
]

OUT = ROOT / "04_ML" / "Reviewer_Robustness"
OUT.mkdir(parents=True, exist_ok=True)

LOGFILE = OUT / "reviewer_robustness_master.log"

# =============================================================================
# LOGGING
# =============================================================================

def log(msg=""):
    text = str(msg)
    print(text, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def section(title):
    log("")
    log("=" * 90)
    log(title)
    log("=" * 90)


# =============================================================================
# UTILITIES
# =============================================================================

def find_existing(candidates):
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def normalize_sample_id(x):
    return str(x).strip()


def detect_sample_column(df):
    preferred = [
        "index",
        "sample",
        "sample_id",
        "Sample",
        "GSM",
        "gsm",
        "ID",
    ]

    for c in preferred:
        if c in df.columns:
            return c

    # Try first column if it looks like GSM IDs
    for c in df.columns[:5]:
        vals = df[c].astype(str)
        if vals.str.startswith("GSM").mean() > 0.5:
            return c

    return None


def detect_gene_column(df):
    preferred = [
        "gene",
        "Gene",
        "gene_symbol",
        "GeneSymbol",
        "symbol",
        "SYMBOL",
        "Gene Symbol",
    ]

    for c in preferred:
        if c in df.columns:
            return c

    return None


def load_expression_matrix(path):
    log(f"Loading expression matrix: {path}")

    df = pd.read_csv(path)

    # Case 1: genes in rows, samples in columns
    gene_col = detect_gene_column(df)

    if gene_col is not None:
        genes = df[gene_col].astype(str)
        numeric_cols = [
            c for c in df.columns
            if c != gene_col and pd.api.types.is_numeric_dtype(df[c])
        ]

        if len(numeric_cols) > 10:
            expr = df[numeric_cols].copy()
            expr.index = genes
            expr = expr.groupby(expr.index).mean()
            return expr

    # Case 2: samples in rows, genes in columns
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_cols) > 100:
        sample_col = detect_sample_column(df)

        if sample_col is not None:
            expr = df[numeric_cols].copy()
            expr.index = df[sample_col].astype(str)
            return expr.T

    # Case 3: first column is gene symbol
    if df.shape[1] > 10:
        first = df.columns[0]
        vals = df[first].astype(str)

        if vals.str.len().median() < 40:
            numeric = df.drop(columns=[first]).apply(
                pd.to_numeric, errors="coerce"
            )

            if numeric.shape[1] > 10:
                numeric.index = vals
                numeric = numeric.groupby(numeric.index).mean()
                return numeric

    raise RuntimeError(
        f"Could not determine expression matrix orientation for {path}. "
        f"Shape={df.shape}, columns={list(df.columns[:20])}"
    )


def load_metadata(path):
    log(f"Loading metadata: {path}")
    df = pd.read_csv(path)

    required = ["Age", "Sex", "Diagnosis"]

    missing = [x for x in required if x not in df.columns]

    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    sample_col = detect_sample_column(df)

    if sample_col is None:
        raise RuntimeError(
            f"Could not identify sample ID column in {path}"
        )

    df = df.copy()
    df["_sample_id"] = df[sample_col].astype(str).map(normalize_sample_id)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df["Sex"] = (
        df["Sex"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["Diagnosis"] = (
        df["Diagnosis"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["_y"] = df["Diagnosis"].map(
        {
            "ad": 1,
            "alzheimer's disease": 1,
            "alzheimer disease": 1,
            "control": 0,
            "ct": 0,
            "control/ct": 0,
        }
    )

    if df["_y"].isna().any():
        bad = df.loc[df["_y"].isna(), "Diagnosis"].unique()
        raise RuntimeError(
            f"Unknown diagnosis labels in {path}: {bad}"
        )

    if df["Age"].isna().any():
        raise RuntimeError(f"Missing Age values in {path}")

    if df["Sex"].isna().any():
        raise RuntimeError(f"Missing Sex values in {path}")

    return df


def align_expression_metadata(expr, meta):
    expr = expr.copy()
    expr.columns = [normalize_sample_id(x) for x in expr.columns]

    meta = meta.copy()
    meta["_sample_id"] = meta["_sample_id"].map(normalize_sample_id)

    common = [x for x in meta["_sample_id"] if x in expr.columns]

    if len(common) == 0:
        raise RuntimeError(
            "No common sample IDs between expression matrix and metadata."
        )

    meta2 = meta.set_index("_sample_id").loc[common].copy()
    expr2 = expr[common].copy()

    return expr2, meta2


def get_gene_matrix(expr, genes):
    missing = [g for g in genes if g not in expr.index]

    if missing:
        raise RuntimeError(
            f"Required genes missing from expression matrix: {missing}"
        )

    return expr.loc[genes].T.astype(float)


def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score)


def safe_ap(y, score):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, score)


def fit_logistic_predict(X_train, y_train, X_eval):
    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    random_state=SEED
                )
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1], model


def demographic_models(meta, cohort_name):
    y = meta["_y"].values

    results = []

    # -------------------------------------------------------------------------
    # AGE ONLY
    # -------------------------------------------------------------------------

    age_score = meta["Age"].values.astype(float)

    results.append(
        {
            "cohort": cohort_name,
            "model": "Age",
            "n": len(y),
            "AD": int(y.sum()),
            "Control": int((y == 0).sum()),
            "AUC": safe_auc(y, age_score),
            "AP": safe_ap(y, age_score),
            "Brier": brier_score_loss(y, age_score / np.max(age_score)),
        }
    )

    # -------------------------------------------------------------------------
    # SEX ONLY
    # -------------------------------------------------------------------------

    sex_score = (
        meta["Sex"]
        .map({"male": 1, "female": 0})
        .values
        .astype(float)
    )

    results.append(
        {
            "cohort": cohort_name,
            "model": "Sex",
            "n": len(y),
            "AD": int(y.sum()),
            "Control": int((y == 0).sum()),
            "AUC": safe_auc(y, sex_score),
            "AP": safe_ap(y, sex_score),
            "Brier": np.nan,
        }
    )

    # -------------------------------------------------------------------------
    # AGE + SEX
    # -------------------------------------------------------------------------

    X = np.column_stack(
        [
            meta["Age"].values.astype(float),
            sex_score,
        ]
    )

    pred, model = fit_logistic_predict(X, y, X)

    results.append(
        {
            "cohort": cohort_name,
            "model": "Age+Sex",
            "n": len(y),
            "AD": int(y.sum()),
            "Control": int((y == 0).sum()),
            "AUC": safe_auc(y, pred),
            "AP": safe_ap(y, pred),
            "Brier": brier_score_loss(y, pred),
        }
    )

    return pd.DataFrame(results)


# =============================================================================
# CONTROLS-ONLY AGE CORRECTION
# =============================================================================

def controls_only_age_residuals(expr, meta):
    """
    Fit gene ~ age using controls only.

    For each gene:
        residual = expression - predicted expression based on control-derived
                   age relationship.

    This avoids using AD samples to estimate the age-expression relationship.
    """

    log("Running controls-only age correction...")

    X_age = meta["Age"].values.astype(float)

    control_mask = meta["_y"].values == 0

    ages_control = X_age[control_mask]

    corrected = pd.DataFrame(
        index=expr.index,
        columns=expr.columns,
        dtype=float
    )

    age_center = ages_control.mean()

    for gene in expr.index:

        y = expr.loc[gene].values.astype(float)

        y_control = y[control_mask]

        # Simple linear regression:
        # y = intercept + beta * age

        Xc = np.column_stack(
            [
                np.ones(len(ages_control)),
                ages_control
            ]
        )

        beta, *_ = np.linalg.lstsq(
            Xc,
            y_control,
            rcond=None
        )

        intercept = beta[0]
        slope = beta[1]

        # Residualize around control mean age.
        expected = intercept + slope * (
            X_age - age_center
        )

        corrected.loc[gene] = y - expected

    return corrected


# =============================================================================
# LASSO FEATURE SELECTION
# =============================================================================

def lasso_select(
    X,
    y,
    genes,
    C=0.3
):
    """
    L1 logistic regression.

    Returns selected gene names.

    NOTE:
    X must already have any fold-specific preprocessing applied.
    """

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
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
        ]
    )

    pipe.fit(X, y)

    coef = pipe.named_steps["lasso"].coef_[0]

    selected = [
        genes[i]
        for i, c in enumerate(coef)
        if abs(c) > 1e-12
    ]

    return selected, pipe


# =============================================================================
# TOP VARIANCE FILTER
# =============================================================================

def top_variance_genes(X, n=5342):
    """
    X shape:
        samples x genes

    Returns gene columns with highest variance.
    """

    n = min(n, X.shape[1])

    vars_ = X.var(axis=0, ddof=1)

    return vars_.sort_values(
        ascending=False
    ).head(n).index.tolist()


# =============================================================================
# NESTED CV
# =============================================================================

def nested_cv_analysis(
    expr,
    meta,
    cohort_name="GSE48350"
):
    """
    Repeated nested CV over the discovery cohort.

    Outer:
        5 folds × 50 repeats

    Inner:
        5 folds

    Feature selection is performed INSIDE each outer training fold.

    Age correction:
        controls-only within each outer training fold.

    Pipeline:
        fold training
        -> controls-only age correction
        -> top variance filtering
        -> LASSO
        -> logistic prediction
        -> outer test

    The purpose is to estimate honest internal performance and feature
    selection frequency.
    """

    section("REPEATED NESTED CROSS-VALIDATION")

    y = meta["_y"].values.astype(int)

    # Expression:
    # samples x genes
    Xall = expr.T.copy()

    genes = Xall.columns.tolist()

    repeated_outer = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS,
        n_repeats=NESTED_REPEATS,
        random_state=SEED
    )

    results = []
    feature_counts = {}

    iteration = 0

    for train_idx, test_idx in repeated_outer:

        iteration += 1

        X_train = Xall.iloc[train_idx].copy()
        X_test = Xall.iloc[test_idx].copy()

        y_train = y[train_idx]
        y_test = y[test_idx]

        meta_train = meta.iloc[train_idx].copy()
        meta_test = meta.iloc[test_idx].copy()

        # ---------------------------------------------------------------------
        # CONTROLS-ONLY AGE CORRECTION WITHIN OUTER TRAINING FOLD
        # ---------------------------------------------------------------------

        corrected_train = controls_only_age_residuals(
            X_train.T,
            meta_train
        ).T

        # Apply exactly the same control-derived coefficients to test set.
        # We therefore calculate coefficients again explicitly from the
        # training controls and apply them to both sets.

        corrected_test = pd.DataFrame(
            index=X_test.index,
            columns=X_test.columns,
            dtype=float
        )

        control_mask = (
            meta_train["_y"].values == 0
        )

        ages_control = (
            meta_train.loc[
                meta_train["_y"].values == 0,
                "Age"
            ].values.astype(float)
        )

        age_center = ages_control.mean()

        test_ages = meta_test["Age"].values.astype(float)

        for gene in X_train.columns:

            y_control = X_train.loc[
                meta_train.index[
                    meta_train["_y"].values == 0
                ],
                gene
            ].values.astype(float)

            Xc = np.column_stack(
                [
                    np.ones(len(ages_control)),
                    ages_control
                ]
            )

            beta, *_ = np.linalg.lstsq(
                Xc,
                y_control,
                rcond=None
            )

            intercept = beta[0]
            slope = beta[1]

            corrected_train.loc[:, gene] = (
                corrected_train.loc[:, gene]
            )

            expected_test = (
                intercept
                + slope * (test_ages - age_center)
            )

            corrected_test.loc[:, gene] = (
                X_test.loc[:, gene].values
                - expected_test
            )

        # ---------------------------------------------------------------------
        # TOP VARIANCE FILTER
        # ---------------------------------------------------------------------

        selected_variance_genes = top_variance_genes(
            corrected_train,
            TOP_VARIANCE_N
        )

        Xtr = corrected_train[
            selected_variance_genes
        ].copy()

        Xte = corrected_test[
            selected_variance_genes
        ].copy()

        # ---------------------------------------------------------------------
        # INNER CV FOR C
        # ---------------------------------------------------------------------

        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + iteration
        )

        c_scores = []

        for C in LASSO_C_GRID:

            aucs = []

            for inner_train, inner_valid in inner_cv.split(
                Xtr,
                y_train
            ):

                Xi_train = Xtr.iloc[inner_train]
                Xi_valid = Xtr.iloc[inner_valid]

                yi_train = y_train[inner_train]
                yi_valid = y_train[inner_valid]

                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
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
                    ]
                )

                pipe.fit(Xi_train, yi_train)

                pred = pipe.predict_proba(
                    Xi_valid
                )[:, 1]

                aucs.append(
                    safe_auc(
                        yi_valid,
                        pred
                    )
                )

            c_scores.append(
                (
                    C,
                    np.nanmean(aucs)
                )
            )

        best_C = max(
            c_scores,
            key=lambda z: z[1]
        )[0]

        # ---------------------------------------------------------------------
        # FEATURE SELECTION ON OUTER TRAINING DATA ONLY
        # ---------------------------------------------------------------------

        selected_genes, lasso_model = lasso_select(
            Xtr,
            y_train,
            selected_variance_genes,
            C=best_C
        )

        # Prevent empty feature set.
        if len(selected_genes) == 0:

            # Fall back to strongest single feature by training variance.
            selected_genes = [
                selected_variance_genes[0]
            ]

        for gene in selected_genes:
            feature_counts[gene] = (
                feature_counts.get(gene, 0) + 1
            )

        # ---------------------------------------------------------------------
        # FINAL OUTER-FOLD MODEL
        # ---------------------------------------------------------------------

        final_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        max_iter=5000,
                        random_state=SEED
                    )
                )
            ]
        )

        final_model.fit(
            Xtr[selected_genes],
            y_train
        )

        pred = final_model.predict_proba(
            Xte[selected_genes]
        )[:, 1]

        results.append(
            {
                "iteration": iteration,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "best_C": best_C,
                "n_selected_genes": len(selected_genes),
                "selected_genes": ";".join(selected_genes),
                "AUC": safe_auc(y_test, pred),
                "AP": safe_ap(y_test, pred),
            }
        )

        if iteration % 10 == 0:
            log(
                f"Nested CV iteration {iteration}/"
                f"{NESTED_REPEATS * OUTER_FOLDS}"
            )

    results_df = pd.DataFrame(results)

    total_iterations = len(results_df)

    feature_df = pd.DataFrame(
        [
            {
                "gene": gene,
                "selection_count": count,
                "selection_frequency": count / total_iterations
            }
            for gene, count in feature_counts.items()
        ]
    ).sort_values(
        "selection_frequency",
        ascending=False
    )

    results_df.to_csv(
        OUT / "nested_cv_fold_results.csv",
        index=False
    )

    feature_df.to_csv(
        OUT / "nested_cv_feature_selection_frequency.csv",
        index=False
    )

    summary = {
        "n_outer_folds": OUTER_FOLDS,
        "n_repeats": NESTED_REPEATS,
        "total_outer_iterations": total_iterations,
        "mean_AUC": results_df["AUC"].mean(),
        "median_AUC": results_df["AUC"].median(),
        "sd_AUC": results_df["AUC"].std(),
        "q025_AUC": results_df["AUC"].quantile(0.025),
        "q975_AUC": results_df["AUC"].quantile(0.975),
        "mean_AP": results_df["AP"].mean(),
        "median_AP": results_df["AP"].median(),
    }

    pd.DataFrame([summary]).to_csv(
        OUT / "nested_cv_summary.csv",
        index=False
    )

    log("")
    log("Nested CV summary:")
    for k, v in summary.items():
        log(f"{k}: {v}")

    log("")
    log("Top selected genes:")
    log(feature_df.head(20).to_string(index=False))

    return results_df, feature_df


# =============================================================================
# RANDOM 3-GENE NULL
# =============================================================================

def random_panel_null(
    expr,
    meta,
    observed_genes=GENES,
    n_panels=1000
):
    """
    Random 3-gene panel null.

    For each random panel:
        - select 3 genes from the available filtered genes
        - evaluate using repeated stratified CV
        - use fixed pipeline
        - calculate mean CV AUC

    This tests:
        Is the observed panel unusually predictive relative to arbitrary
        three-gene panels?

    NOTE:
    This is a null benchmark, not a permutation test.
    """

    section("RANDOM 3-GENE PANEL NULL")

    y = meta["_y"].values.astype(int)

    Xall = expr.T.copy()

    available_genes = Xall.columns.tolist()

    if len(available_genes) < 3:
        raise RuntimeError(
            "Fewer than three genes available for random null."
        )

    rng = np.random.default_rng(SEED)

    # Use same repeated CV structure for every panel.
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=SEED
    )

    rows = []

    for panel_idx in range(1, n_panels + 1):

        panel = rng.choice(
            available_genes,
            size=3,
            replace=False
        ).tolist()

        X = Xall[panel]

        fold_aucs = []

        for train_idx, test_idx in cv.split(
            X,
            y
        ):

            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logreg",
                        LogisticRegression(
                            max_iter=5000,
                            random_state=SEED
                        )
                    )
                ]
            )

            model.fit(
                X.iloc[train_idx],
                y[train_idx]
            )

            pred = model.predict_proba(
                X.iloc[test_idx]
            )[:, 1]

            fold_aucs.append(
                safe_auc(
                    y[test_idx],
                    pred
                )
            )

        rows.append(
            {
                "panel_id": panel_idx,
                "gene1": panel[0],
                "gene2": panel[1],
                "gene3": panel[2],
                "mean_cv_auc": np.nanmean(fold_aucs),
                "median_cv_auc": np.nanmedian(fold_aucs),
            }
        )

        if panel_idx % 100 == 0:
            log(
                f"Random panel {panel_idx}/{n_panels}"
            )

    null_df = pd.DataFrame(rows)

    null_df.to_csv(
        OUT / "random_3gene_panel_null_1000.csv",
        index=False
    )

    return null_df


# =============================================================================
# CALIBRATION
# =============================================================================

def calibration_statistics(y, pred):
    """
    Calculate:
      - Brier score
      - calibration intercept
      - calibration slope

    Calibration slope/intercept are estimated from:
        logit(predicted probability)

    This is descriptive for an external validation cohort.
    """

    pred = np.clip(
        np.asarray(pred, dtype=float),
        1e-6,
        1 - 1e-6
    )

    y = np.asarray(y, dtype=int)

    brier = brier_score_loss(
        y,
        pred
    )

    logit = np.log(
        pred / (1 - pred)
    ).reshape(-1, 1)

    # Calibration intercept-only:
    intercept_model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000
    )

    try:
        intercept_model.fit(
            np.zeros((len(y), 1)),
            y
        )
        intercept = float(
            intercept_model.intercept_[0]
        )
    except Exception:
        intercept = np.nan

    # Calibration slope + intercept
    cal_model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000
    )

    try:
        cal_model.fit(
            logit,
            y
        )

        slope = float(
            cal_model.coef_[0, 0]
        )

        intercept_full = float(
            cal_model.intercept_[0]
        )

    except Exception:
        slope = np.nan
        intercept_full = np.nan

    return {
        "Brier": brier,
        "Calibration_intercept": intercept_full,
        "Calibration_slope": slope,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    start = time.time()

    section("START REVIEWER ROBUSTNESS MASTER ANALYSIS")

    log(f"Project root: {ROOT}")
    log(f"Python: {sys.version}")
    log(f"Platform: {platform.platform()}")
    log(f"Seed: {SEED}")

    # -------------------------------------------------------------------------
    # CHECK FILES
    # -------------------------------------------------------------------------

    section("FILE DISCOVERY")

    log(f"Training metadata: {META_TRAIN}")
    log(f"Test metadata:     {META_TEST}")

    if not META_TRAIN.exists():
        raise FileNotFoundError(META_TRAIN)

    if not META_TEST.exists():
        raise FileNotFoundError(META_TEST)

    train_matrix = find_existing(
        TRAIN_MATRIX_CANDIDATES
    )

    test_matrix = find_existing(
        TEST_MATRIX_CANDIDATES
    )

    external_matrix = find_existing(
        EXTERNAL_MATRIX_CANDIDATES
    )

    external_meta = find_existing(
        EXTERNAL_META_CANDIDATES
    )

    log(f"Training matrix: {train_matrix}")
    log(f"Test matrix: {test_matrix}")
    log(f"External matrix: {external_matrix}")
    log(f"External metadata: {external_meta}")

    # -------------------------------------------------------------------------
    # LOAD METADATA
    # -------------------------------------------------------------------------

    section("METADATA VALIDATION")

    train_meta = load_metadata(
        META_TRAIN
    )

    test_meta = load_metadata(
        META_TEST
    )

    log(
        train_meta[
            ["Age", "Sex", "Diagnosis"]
        ].describe(include="all").to_string()
    )

    log("")
    log("Training diagnosis:")
    log(train_meta["Diagnosis"].value_counts().to_string())

    log("")
    log("Test diagnosis:")
    log(test_meta["Diagnosis"].value_counts().to_string())

    log("")
    log("Training sex:")
    log(
        pd.crosstab(
            train_meta["Diagnosis"],
            train_meta["Sex"]
        ).to_string()
    )

    log("")
    log("Test sex:")
    log(
        pd.crosstab(
            test_meta["Diagnosis"],
            test_meta["Sex"]
        ).to_string()
    )

    # -------------------------------------------------------------------------
    # DEMOGRAPHIC BASELINES
    # -------------------------------------------------------------------------

    section("DEMOGRAPHIC BASELINES")

    train_demo = demographic_models(
        train_meta,
        "GSE48350_TRAIN"
    )

    test_demo = demographic_models(
        test_meta,
        "GSE48350_TEST"
    )

    demo = pd.concat(
        [train_demo, test_demo],
        ignore_index=True
    )

    demo.to_csv(
        OUT / "demographic_baselines_GSE48350.csv",
        index=False
    )

    log(demo.to_string(index=False))

    # -------------------------------------------------------------------------
    # EXTERNAL METADATA
    # -------------------------------------------------------------------------

    external_meta_df = None

    if external_meta is not None:

        section("EXTERNAL METADATA")

        try:

            external_meta_df = load_metadata(
                external_meta
            )

            log(
                f"External metadata shape: "
                f"{external_meta_df.shape}"
            )

            log(
                external_meta_df[
                    ["Age", "Sex", "Diagnosis"]
                ].describe(include="all").to_string()
            )

            log("")
            log("External diagnosis:")
            log(
                external_meta_df[
                    "Diagnosis"
                ].value_counts().to_string()
            )

            log("")
            log("External sex:")
            log(
                pd.crosstab(
                    external_meta_df["Diagnosis"],
                    external_meta_df["Sex"]
                ).to_string()
            )

            ext_demo = demographic_models(
                external_meta_df,
                "GSE5281"
            )

            ext_demo.to_csv(
                OUT / "demographic_baselines_GSE5281.csv",
                index=False
            )

            ext_demo.to_csv(
                OUT / "demographic_baselines_all_external.csv",
                index=False
            )

            log("")
            log(ext_demo.to_string(index=False))

        except Exception as e:

            log(
                "WARNING: External metadata could not be processed:"
            )
            log(repr(e))

    # -------------------------------------------------------------------------
    # EXPRESSION MATRIX
    # -------------------------------------------------------------------------

    if train_matrix is None:

        log("")
        log(
            "No dedicated training expression matrix found."
        )

        log(
            "Skipping expression-based analyses."
        )

    else:

        section("TRAINING EXPRESSION MATRIX")

        train_expr = load_expression_matrix(
            train_matrix
        )

        log(
            f"Training expression shape: "
            f"{train_expr.shape}"
        )

        # Align to metadata
        train_expr, train_meta_aligned = (
            align_expression_metadata(
                train_expr,
                train_meta
            )
        )

        log(
            f"Aligned training expression shape: "
            f"{train_expr.shape}"
        )

        # ---------------------------------------------------------------------
        # 3-GENE AVAILABILITY
        # ---------------------------------------------------------------------

        section("THREE-GENE AVAILABILITY")

        available = {
            g: g in train_expr.index
            for g in GENES
        }

        log(json.dumps(
            available,
            indent=2
        ))

        if all(available.values()):

            X3 = get_gene_matrix(
                train_expr,
                GENES
            )

            y3 = train_meta_aligned["_y"].values

            pred3, model3 = fit_logistic_predict(
                X3,
                y3,
                X3
            )

            auc3 = safe_auc(
                y3,
                pred3
            )

            ap3 = safe_ap(
                y3,
                pred3
            )

            cal3 = calibration_statistics(
                y3,
                pred3
            )

            current3 = pd.DataFrame(
                [
                    {
                        "cohort": "GSE48350_TRAIN",
                        "model": "Current_3gene",
                        "genes": ";".join(GENES),
                        "AUC": auc3,
                        "AP": ap3,
                        **cal3
                    }
                ]
            )

            current3.to_csv(
                OUT / "current_3gene_training_performance.csv",
                index=False
            )

            log("")
            log("Current 3-gene training performance:")
            log(current3.to_string(index=False))

        # ---------------------------------------------------------------------
        # CONTROLS-ONLY AGE ADJUSTMENT
        # ---------------------------------------------------------------------

        section("CONTROLS-ONLY AGE ADJUSTMENT")

        corrected_train = controls_only_age_residuals(
            train_expr,
            train_meta_aligned
        )

        corrected_train.to_csv(
            OUT / "GSE48350_controls_only_age_adjusted_expression.csv"
        )

        log(
            "Controls-only age-adjusted expression saved."
        )

        if all(available.values()):

            X3_adj = get_gene_matrix(
                corrected_train,
                GENES
            )

            pred_adj, model_adj = fit_logistic_predict(
                X3_adj,
                y3,
                X3_adj
            )

            adj_auc = safe_auc(
                y3,
                pred_adj
            )

            adj_ap = safe_ap(
                y3,
                pred_adj
            )

            adj_cal = calibration_statistics(
                y3,
                pred_adj
            )

            adjusted_result = pd.DataFrame(
                [
                    {
                        "cohort": "GSE48350_TRAIN",
                        "model": "3gene_controls_only_age_adjusted",
                        "genes": ";".join(GENES),
                        "AUC": adj_auc,
                        "AP": adj_ap,
                        **adj_cal
                    }
                ]
            )

            adjusted_result.to_csv(
                OUT / "three_gene_controls_only_age_adjusted.csv",
                index=False
            )

            log("")
            log(
                "Three-gene controls-only age-adjusted result:"
            )
            log(
                adjusted_result.to_string(index=False)
            )

        # ---------------------------------------------------------------------
        # NESTED CV
        # ---------------------------------------------------------------------

        try:

            nested_results, nested_features = (
                nested_cv_analysis(
                    train_expr,
                    train_meta_aligned,
                    "GSE48350"
                )
            )

        except Exception as e:

            log("")
            log("NESTED CV FAILED:")
            log(repr(e))

        # ---------------------------------------------------------------------
        # RANDOM PANEL NULL
        # ---------------------------------------------------------------------

        try:

            # To prevent accidentally sampling from a massive unfiltered
            # matrix, use the top-variance training genes.
            filtered_genes = top_variance_genes(
                train_expr.T,
                TOP_VARIANCE_N
            )

            filtered_expr = train_expr.loc[
                filtered_genes
            ]

            log(
                f"Random null background: "
                f"{len(filtered_genes)} genes"
            )

            null_df = random_panel_null(
                filtered_expr,
                train_meta_aligned,
                GENES,
                RANDOM_NULL_PANELS
            )

            # Observed 3-gene performance using same CV scheme
            Xobs = filtered_expr.loc[
                GENES
            ].T

            yobs = train_meta_aligned["_y"].values

            cv_obs = RepeatedStratifiedKFold(
                n_splits=5,
                n_repeats=10,
                random_state=SEED
            )

            obs_aucs = []

            for tr, te in cv_obs.split(
                Xobs,
                yobs
            ):

                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "logreg",
                            LogisticRegression(
                                max_iter=5000,
                                random_state=SEED
                            )
                        )
                    ]
                )

                model.fit(
                    Xobs.iloc[tr],
                    yobs[tr]
                )

                pred = model.predict_proba(
                    Xobs.iloc[te]
                )[:, 1]

                obs_aucs.append(
                    safe_auc(
                        yobs[te],
                        pred
                    )
                )

            observed_mean_auc = np.nanmean(
                obs_aucs
            )

            null_mean = null_df[
                "mean_cv_auc"
            ].values

            empirical_percentile = (
                np.sum(
                    null_mean <= observed_mean_auc
                ) / len(null_mean)
            )

            random_null_summary = pd.DataFrame(
                [
                    {
                        "observed_panel":
                            ";".join(GENES),
                        "observed_mean_cv_auc":
                            observed_mean_auc,
                        "null_mean":
                            np.mean(null_mean),
                        "null_sd":
                            np.std(null_mean, ddof=1),
                        "null_median":
                            np.median(null_mean),
                        "null_q025":
                            np.quantile(null_mean, 0.025),
                        "null_q975":
                            np.quantile(null_mean, 0.975),
                        "empirical_percentile":
                            empirical_percentile,
                        "n_random_panels":
                            len(null_mean),
                    }
                ]
            )

            random_null_summary.to_csv(
                OUT / "random_3gene_panel_null_summary.csv",
                index=False
            )

            log("")
            log(
                "Random 3-gene null summary:"
            )
            log(
                random_null_summary.to_string(
                    index=False
                )
            )

        except Exception as e:

            log("")
            log("RANDOM PANEL NULL FAILED:")
            log(repr(e))

    # -------------------------------------------------------------------------
    # EXTERNAL EXPRESSION / 3-GENE ANALYSIS
    # -------------------------------------------------------------------------

    if (
        external_matrix is not None
        and external_meta_df is not None
    ):

        section("EXTERNAL GSE5281 ANALYSIS")

        try:

            ext_expr = load_expression_matrix(
                external_matrix
            )

            ext_expr, ext_meta_aligned = (
                align_expression_metadata(
                    ext_expr,
                    external_meta_df
                )
            )

            log(
                f"External expression shape: "
                f"{ext_expr.shape}"
            )

            if all(
                g in ext_expr.index
                for g in GENES
            ):

                Xext3 = get_gene_matrix(
                    ext_expr,
                    GENES
                )

                yext = (
                    ext_meta_aligned["_y"]
                    .values.astype(int)
                )

                # Train 3-gene model on discovery.
                if train_matrix is not None:

                    Xtrain3 = get_gene_matrix(
                        train_expr,
                        GENES
                    )

                    ytrain3 = (
                        train_meta_aligned["_y"]
                        .values.astype(int)
                    )

                    locked_model = Pipeline(
                        [
                            (
                                "scaler",
                                StandardScaler()
                            ),
                            (
                                "logreg",
                                LogisticRegression(
                                    max_iter=5000,
                                    random_state=SEED
                                )
                            )
                        ]
                    )

                    locked_model.fit(
                        Xtrain3,
                        ytrain3
                    )

                    pext = locked_model.predict_proba(
                        Xext3
                    )[:, 1]

                    ext_auc = safe_auc(
                        yext,
                        pext
                    )

                    ext_ap = safe_ap(
                        yext,
                        pext
                    )

                    ext_cal = calibration_statistics(
                        yext,
                        pext
                    )

                    ext_result = pd.DataFrame(
                        [
                            {
                                "cohort": "GSE5281",
                                "model": "Locked_3gene",
                                "genes": ";".join(GENES),
                                "AUC": ext_auc,
                                "AP": ext_ap,
                                **ext_cal
                            }
                        ]
                    )

                    ext_result.to_csv(
                        OUT /
                        "GSE5281_locked_3gene_calibration.csv",
                        index=False
                    )

                    pred_table = ext_meta_aligned.copy()

                    pred_table[
                        "predicted_probability"
                    ] = pext

                    pred_table[
                        "outcome"
                    ] = yext

                    pred_table.to_csv(
                        OUT /
                        "GSE5281_locked_3gene_predictions.csv"
                    )

                    log("")
                    log(
                        "External locked 3-gene performance:"
                    )
                    log(
                        ext_result.to_string(
                            index=False
                        )
                    )

        except Exception as e:

            log("")
            log(
                "External expression analysis failed:"
            )
            log(repr(e))

    # -------------------------------------------------------------------------
    # MASTER SUMMARY
    # -------------------------------------------------------------------------

    section("MASTER SUMMARY")

    summary_files = [
        OUT / "demographic_baselines_GSE48350.csv",
        OUT / "demographic_baselines_GSE5281.csv",
        OUT / "current_3gene_training_performance.csv",
        OUT / "three_gene_controls_only_age_adjusted.csv",
        OUT / "nested_cv_summary.csv",
        OUT / "random_3gene_panel_null_summary.csv",
        OUT / "GSE5281_locked_3gene_calibration.csv",
    ]

    log("Expected output files:")

    for p in summary_files:
        log(
            f"{'[FOUND]' if p.exists() else '[NOT CREATED]'} "
            f"{p}"
        )

    elapsed = time.time() - start

    log("")
    log(
        f"TOTAL RUNTIME: "
        f"{elapsed / 60:.2f} minutes"
    )

    log("")
    log(
        "REVIEWER ROBUSTNESS MASTER ANALYSIS COMPLETE."
    )


if __name__ == "__main__":
    main()

