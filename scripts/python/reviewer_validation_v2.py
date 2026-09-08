import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42
N_REPEATS = 50
OUTER_FOLDS = 5
INNER_FOLDS = 5
N_RANDOM = 1000
GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

ROOT = Path.home() / "project_ml"
OUT = ROOT / "04_ML" / "Reviewer_Robustness_V2"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_EXPR = ROOT / "04_ML" / "GSE48350_RMA_ML_matrix.csv"
TRAIN_META = ROOT / "04_ML" / "GSE48350_RMA_train_meta.csv"
TEST_META = ROOT / "04_ML" / "GSE48350_RMA_test_meta.csv"

EXT_EXPR = ROOT / "04_ML" / "GSE5281_RMA_genelevel_harmonized.csv"
EXT_META = ROOT / "04_ML" / "External_Validation" / "GSE5281_validation_metadata.csv"


def log(x=""):
    print(x, flush=True)


def load_expr(path):
    df = pd.read_csv(path)

    # Existing project matrix is expected to have genes as rows.
    # Detect gene column.
    gene_col = None
    for c in ["gene", "Gene", "symbol", "Symbol", "GeneSymbol"]:
        if c in df.columns:
            gene_col = c
            break

    if gene_col is not None:
        expr = df.drop(columns=[gene_col]).apply(
            pd.to_numeric, errors="coerce"
        )
        expr.index = df[gene_col].astype(str)
    else:
        # First column is normally gene identifier.
        expr = df.iloc[:, 1:].apply(
            pd.to_numeric, errors="coerce"
        )
        expr.index = df.iloc[:, 0].astype(str)

    expr = expr.groupby(expr.index).mean()
    expr = expr.dropna(axis=0, how="all")
    return expr


def load_meta(path, external=False):
    df = pd.read_csv(path)

    if external:
        sample_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else "GSM"
        age_col = "Age_used_years"
        diagnosis_col = "diagnosis"
    else:
        sample_col = "index"
        age_col = "Age"
        diagnosis_col = "Diagnosis"

    df["_sample"] = df[sample_col].astype(str).str.strip()
    df["_age"] = pd.to_numeric(df[age_col], errors="coerce")
    df["_sex"] = (
        df["Sex"].astype(str).str.lower().str.strip()
        if "Sex" in df.columns
        else np.nan
    )

    df["_y"] = (
        df[diagnosis_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .map({"ad": 1, "control": 0})
    )

    return df


def align(expr, meta):
    expr.columns = expr.columns.astype(str).str.strip()

    common = [
        s for s in meta["_sample"]
        if s in expr.columns
    ]

    if not common:
        raise RuntimeError(
            "No matching sample IDs between expression and metadata."
        )

    meta = meta.set_index("_sample").loc[common].copy()
    expr = expr[common].copy()

    return expr, meta


def auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def ap(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)


def demographic_predictions(meta):
    y = meta["_y"].astype(int).values

    age = meta["_age"].astype(float).values

    sex_map = {"female": 0, "male": 1}
    sex = meta["_sex"].map(sex_map).astype(float).values

    rows = []

    rows.append({
        "model": "Age",
        "AUC": auc(y, age),
        "AP": ap(y, age),
    })

    rows.append({
        "model": "Sex",
        "AUC": auc(y, sex),
        "AP": ap(y, sex),
    })

    valid = np.isfinite(sex)

    X = np.column_stack([
        age,
        sex
    ])

    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000,
            random_state=SEED
        ))
    ])

    model.fit(X[valid], y[valid])
    p = model.predict_proba(X[valid])[:, 1]

    rows.append({
        "model": "Age+Sex",
        "AUC": auc(y[valid], p),
        "AP": ap(y[valid], p),
    })

    return pd.DataFrame(rows)


def controls_age_residualize(X, meta_train, meta_eval):
    """
    Fit gene~age using CONTROLS ONLY in training data.
    Apply training-derived coefficients to evaluation samples.
    """

    Xout = pd.DataFrame(
        index=meta_eval.index,
        columns=X.columns,
        dtype=float
    )

    train_control = meta_train["_y"].values == 0

    age_train = meta_train["_age"].values.astype(float)
    age_eval = meta_eval["_age"].values.astype(float)

    age_center = age_train[train_control].mean()

    for gene in X.columns:

        y_control = X.loc[
            meta_train.index[train_control],
            gene
        ].values.astype(float)

        a_control = age_train[train_control]

        lr = LinearRegression()
        lr.fit(
            a_control.reshape(-1, 1),
            y_control
        )

        expected = lr.predict(
            (age_eval - age_center).reshape(-1, 1)
        )

        # The model is fit around centered age.
        # Re-fit explicitly using centered training age to make
        # transformation identical between train and evaluation.
        lr2 = LinearRegression()
        lr2.fit(
            (a_control - age_center).reshape(-1, 1),
            y_control
        )

        expected = lr2.predict(
            (age_eval - age_center).reshape(-1, 1)
        )

        Xout[gene] = (
            X.loc[meta_eval.index, gene].values
            - expected
        )

    return Xout


def fit_predict(Xtr, ytr, Xte):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000,
            random_state=SEED
        ))
    ])

    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def lasso_select(X, y, genes, C):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("lasso", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            max_iter=5000,
            random_state=SEED
        ))
    ])

    model.fit(X, y)

    coef = model.named_steps["lasso"].coef_[0]

    selected = [
        genes[i]
        for i, c in enumerate(coef)
        if abs(c) > 1e-10
    ]

    return selected


def nested_cv(expr, meta):
    log("")
    log("=" * 90)
    log("50 x 5-FOLD NESTED CV")
    log("=" * 90)

    Xall = expr.T.copy()
    y = meta["_y"].astype(int).values

    # Only use genes with finite values.
    Xall = Xall.replace([np.inf, -np.inf], np.nan)
    Xall = Xall.fillna(Xall.median())

    # Top 5342 variance genes determined from the complete matrix here.
    # This is retained only to match the existing project's filtered
    # background. Feature-selection itself remains inside each outer fold.
    variances = Xall.var(axis=0)
    bg_genes = variances.sort_values(
        ascending=False
    ).head(min(5342, Xall.shape[1])).index.tolist()

    Xall = Xall[bg_genes]

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS,
        n_repeats=N_REPEATS,
        random_state=SEED
    )

    rows = []
    feature_counts = {}
    iteration = 0

    # IMPORTANT:
    # split() is required; the previous script failed because it attempted
    # to iterate directly over the RepeatedStratifiedKFold object.
    for train_idx, test_idx in outer.split(Xall, y):

        iteration += 1

        Xtr = Xall.iloc[train_idx].copy()
        Xte = Xall.iloc[test_idx].copy()

        mtr = meta.iloc[train_idx].copy()
        mte = meta.iloc[test_idx].copy()

        ytr = y[train_idx]
        yte = y[test_idx]

        # -------------------------------------------------------------
        # Controls-only age correction
        # -------------------------------------------------------------

        Xtr_adj = controls_age_residualize(
            Xtr,
            mtr,
            mtr
        )

        Xte_adj = controls_age_residualize(
            Xtr,
            mtr,
            mte
        )

        # -------------------------------------------------------------
        # Inner CV chooses LASSO C
        # -------------------------------------------------------------

        inner = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + iteration
        )

        best_C = None
        best_score = -np.inf

        for C in [0.05, 0.1, 0.2, 0.5, 1.0]:

            scores = []

            for itr, iva in inner.split(Xtr_adj, ytr):

                model = Pipeline([
                    ("scale", StandardScaler()),
                    ("lasso", LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=C,
                        max_iter=5000,
                        random_state=SEED
                    ))
                ])

                model.fit(
                    Xtr_adj.iloc[itr],
                    ytr[itr]
                )

                p = model.predict_proba(
                    Xtr_adj.iloc[iva]
                )[:, 1]

                scores.append(
                    auc(ytr[iva], p)
                )

            score = np.nanmean(scores)

            if score > best_score:
                best_score = score
                best_C = C

        # -------------------------------------------------------------
        # Feature selection ONLY on outer training set
        # -------------------------------------------------------------

        genes_bg = Xtr_adj.columns.tolist()

        selected = lasso_select(
            Xtr_adj,
            ytr,
            genes_bg,
            best_C
        )

        # If LASSO selects nothing, use the three locked genes only
        # if they are available; otherwise use highest variance gene.
        if not selected:

            available_locked = [
                g for g in GENES
                if g in Xtr_adj.columns
            ]

            if available_locked:
                selected = available_locked
            else:
                selected = [
                    Xtr_adj.var().sort_values(
                        ascending=False
                    ).index[0]
                ]

        for g in selected:
            feature_counts[g] = (
                feature_counts.get(g, 0) + 1
            )

        # -------------------------------------------------------------
        # Outer-fold prediction
        # -------------------------------------------------------------

        p = fit_predict(
            Xtr_adj[selected],
            ytr,
            Xte_adj[selected]
        )

        rows.append({
            "iteration": iteration,
            "AUC": auc(yte, p),
            "AP": ap(yte, p),
            "best_C": best_C,
            "n_selected": len(selected),
            "selected_genes": ";".join(selected),
        })

        if iteration % 25 == 0:
            log(
                f"Completed {iteration}/"
                f"{N_REPEATS * OUTER_FOLDS}"
            )

    results = pd.DataFrame(rows)

    features = pd.DataFrame([
        {
            "gene": g,
            "selection_count": n,
            "selection_frequency":
                n / len(results)
        }
        for g, n in feature_counts.items()
    ]).sort_values(
        "selection_frequency",
        ascending=False
    )

    summary = pd.DataFrame([{
        "outer_folds": OUTER_FOLDS,
        "repeats": N_REPEATS,
        "iterations": len(results),
        "mean_AUC": results["AUC"].mean(),
        "median_AUC": results["AUC"].median(),
        "SD_AUC": results["AUC"].std(),
        "AUC_2.5pct": results["AUC"].quantile(0.025),
        "AUC_97.5pct": results["AUC"].quantile(0.975),
        "mean_AP": results["AP"].mean(),
        "median_AP": results["AP"].median(),
    }])

    results.to_csv(
        OUT / "nested_cv_results.csv",
        index=False
    )

    features.to_csv(
        OUT / "nested_cv_feature_selection_frequency.csv",
        index=False
    )

    summary.to_csv(
        OUT / "nested_cv_summary.csv",
        index=False
    )

    log("")
    log("NESTED CV SUMMARY")
    log(summary.to_string(index=False))

    log("")
    log("TOP FEATURE SELECTION FREQUENCIES")
    log(features.head(20).to_string(index=False))

    return results, features


def random_null(expr, meta):
    log("")
    log("=" * 90)
    log("RANDOM 3-GENE NULL")
    log("=" * 90)

    X = expr.T.copy()
    y = meta["_y"].astype(int).values

    variances = X.var(axis=0)
    genes = variances.sort_values(
        ascending=False
    ).head(min(5342, X.shape[1])).index.tolist()

    X = X[genes]

    rng = np.random.default_rng(SEED)

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=SEED
    )

    rows = []

    for i in range(1, N_RANDOM + 1):

        panel = rng.choice(
            genes,
            size=3,
            replace=False
        ).tolist()

        scores = []

        for tr, te in cv.split(X[panel], y):

            p = fit_predict(
                X.iloc[tr][panel],
                y[tr],
                X.iloc[te][panel]
            )

            scores.append(
                auc(y[te], p)
            )

        rows.append({
            "panel_id": i,
            "gene1": panel[0],
            "gene2": panel[1],
            "gene3": panel[2],
            "mean_cv_auc": np.nanmean(scores),
        })

        if i % 100 == 0:
            log(f"Random panels: {i}/{N_RANDOM}")

    null = pd.DataFrame(rows)

    # Same 5x10 CV for observed locked panel.
    obs_scores = []

    for tr, te in cv.split(X[GENES], y):

        p = fit_predict(
            X.iloc[tr][GENES],
            y[tr],
            X.iloc[te][GENES]
        )

        obs_scores.append(
            auc(y[te], p)
        )

    observed = np.nanmean(obs_scores)

    vals = null["mean_cv_auc"].values

    summary = pd.DataFrame([{
        "observed_panel": ";".join(GENES),
        "observed_mean_cv_auc": observed,
        "null_mean": vals.mean(),
        "null_sd": vals.std(ddof=1),
        "null_median": np.median(vals),
        "null_q025": np.quantile(vals, 0.025),
        "null_q975": np.quantile(vals, 0.975),
        "empirical_percentile":
            np.mean(vals <= observed),
        "n_random_panels": len(vals)
    }])

    null.to_csv(
        OUT / "random_3gene_null_v2.csv",
        index=False
    )

    summary.to_csv(
        OUT / "random_3gene_null_summary_v2.csv",
        index=False
    )

    log("")
    log(summary.to_string(index=False))


def external_analysis(train_expr, train_meta, ext_expr, ext_meta):

    log("")
    log("=" * 90)
    log("GSE5281 EXTERNAL VALIDATION")
    log("=" * 90)

    train_expr, train_meta = align(
        train_expr,
        train_meta
    )

    ext_expr, ext_meta = align(
        ext_expr,
        ext_meta
    )

    Xtr = train_expr.loc[GENES].T
    Xext = ext_expr.loc[GENES].T

    ytr = train_meta["_y"].astype(int).values
    yext = ext_meta["_y"].astype(int).values

    # Locked model trained only on discovery cohort.
    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000,
            random_state=SEED
        ))
    ])

    model.fit(Xtr, ytr)

    p = model.predict_proba(Xext)[:, 1]

    result = {
        "cohort": "GSE5281",
        "n": len(yext),
        "AD": int(yext.sum()),
        "Control": int((yext == 0).sum()),
        "AUC": auc(yext, p),
        "AP": ap(yext, p),
        "Brier": brier_score_loss(yext, p)
    }

    # Calibration intercept/slope using statsmodels if available.
    try:
        import statsmodels.api as sm

        pclip = np.clip(p, 1e-6, 1 - 1e-6)
        logit_p = np.log(pclip / (1 - pclip))

        Xcal = sm.add_constant(logit_p)

        cal = sm.Logit(
            yext,
            Xcal
        ).fit(disp=False)

        result["Calibration_intercept"] = cal.params[0]
        result["Calibration_slope"] = cal.params[1]

    except Exception:
        result["Calibration_intercept"] = np.nan
        result["Calibration_slope"] = np.nan

    pd.DataFrame([result]).to_csv(
        OUT / "GSE5281_locked_3gene_calibration_v2.csv",
        index=False
    )

    pred = ext_meta.copy()
    pred["predicted_probability"] = p
    pred["outcome"] = yext

    pred.to_csv(
        OUT / "GSE5281_locked_3gene_predictions_v2.csv",
        index=False
    )

    demo = demographic_predictions(
        ext_meta
    )

    demo.to_csv(
        OUT / "GSE5281_demographic_baselines_v2.csv",
        index=False
    )

    log("")
    log("GSE5281 3-GENE")
    log(pd.DataFrame([result]).to_string(index=False))

    log("")
    log("GSE5281 DEMOGRAPHICS")
    log(demo.to_string(index=False))


def main():

    t0 = time.time()

    log("=" * 90)
    log("REVIEWER VALIDATION V2")
    log("=" * 90)

    log(f"Python: {os.sys.version}")
    log(f"Seed: {SEED}")
    log(f"Output: {OUT}")

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    train_expr = load_expr(TRAIN_EXPR)
    train_meta = load_meta(TRAIN_META)

    log(
        f"Training expression: {train_expr.shape}"
    )

    # -------------------------------------------------------------------------
    # BASIC DEMOGRAPHICS
    # -------------------------------------------------------------------------

    demo_train = demographic_predictions(
        train_meta
    )

    test_meta = load_meta(
        TEST_META
    )

    demo_test = demographic_predictions(
        test_meta
    )

    demo_train["cohort"] = "GSE48350_TRAIN"
    demo_test["cohort"] = "GSE48350_TEST"

    pd.concat(
        [demo_train, demo_test],
        ignore_index=True
    ).to_csv(
        OUT / "GSE48350_demographic_baselines_v2.csv",
        index=False
    )

    log("")
    log("GSE48350 DEMOGRAPHICS")
    log(
        pd.concat(
            [demo_train, demo_test],
            ignore_index=True
        ).to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # NESTED CV
    # -------------------------------------------------------------------------

    nested_cv(
        train_expr,
        train_meta
    )

    # -------------------------------------------------------------------------
    # RANDOM NULL
    # -------------------------------------------------------------------------

    random_null(
        train_expr,
        train_meta
    )

    # -------------------------------------------------------------------------
    # EXTERNAL
    # -------------------------------------------------------------------------

    if EXT_EXPR.exists() and EXT_META.exists():

        ext_expr = load_expr(
            EXT_EXPR
        )

        ext_meta = load_meta(
            EXT_META,
            external=True
        )

        external_analysis(
            train_expr,
            train_meta,
            ext_expr,
            ext_meta
        )

    else:

        log(
            "GSE5281 expression or metadata file missing."
        )

    # -------------------------------------------------------------------------
    # FINISH
    # -------------------------------------------------------------------------

    elapsed = time.time() - t0

    log("")
    log("=" * 90)
    log("COMPLETE")
    log("=" * 90)
    log(
        f"Runtime: {elapsed / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
