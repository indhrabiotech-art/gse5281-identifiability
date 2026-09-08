import os
import numpy as np
import pandas as pd
import shap

from xgboost import XGBClassifier


OUTDIR = "04_ML/Final_Model/Validation"
FIGDIR = "06_Manuscript/Figures"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

GENES = [
    "ABCA6",
    "CRLF1",
    "TNFRSF11B"
]

RANDOM_STATE = 42


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train = pd.read_csv(
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    index_col=0
)

train_meta = pd.read_csv(
    "04_ML/GSE48350_RMA_train_meta.csv"
)

# ------------------------------------------------------------
# ALIGN EXPRESSION AND METADATA
# ------------------------------------------------------------

# Expression sample IDs are stored in the expression index.
# Metadata sample IDs are stored in the "index" column.

train = train.copy()
train_meta = train_meta.copy()

train.index = train.index.astype(str)
train_meta["index"] = train_meta["index"].astype(str)

# Align metadata to expression samples
train_meta = train_meta.set_index("index")

common_samples = train.index.intersection(train_meta.index)

print("Expression samples:", len(train))
print("Metadata samples:", len(train_meta))
print("Common samples:", len(common_samples))

if len(common_samples) == 0:
    raise ValueError(
        "No common sample IDs between expression matrix and metadata."
    )

train = train.loc[common_samples]
train_meta = train_meta.loc[common_samples]

X_train = train[GENES].copy()

y_train = (
    train_meta["Diagnosis"]
    .astype(str)
    .str.strip()
    .map({
        "Control": 0,
        "AD": 1
    })
)

valid = y_train.notna()

X_train = X_train.loc[valid].copy()
y_train = y_train.loc[valid].astype(int).copy()

print("Final aligned samples:", len(X_train))
print("AD:", int((y_train == 1).sum()))
print("Control:", int((y_train == 0).sum()))


# ============================================================
# TRAIN THE LOCKED XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=1,
    device="cpu"
)

model.fit(X_train, y_train)


# ============================================================
# SHAP EXPLANATION
# ============================================================

print("=" * 80)
print("FINAL THREE-GENE XGBOOST SHAP ANALYSIS")
print("=" * 80)

print("\nSamples:", len(X_train))
print("Genes:", ", ".join(GENES))

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_train)

shap_values = np.asarray(shap_values)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]


# ============================================================
# SHAP SUMMARY
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

mean_signed_shap = np.mean(
    shap_values,
    axis=0
)

summary = pd.DataFrame({
    "Gene": GENES,
    "Mean_Absolute_SHAP": mean_abs_shap,
    "Mean_Signed_SHAP": mean_signed_shap
})

summary["SHAP_Rank"] = (
    summary["Mean_Absolute_SHAP"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

summary = summary.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print("\n" + "=" * 80)
print("SHAP FEATURE IMPORTANCE")
print("=" * 80)

print(summary.to_string(index=False))


# ============================================================
# SAMPLE-LEVEL SHAP MATRIX
# ============================================================

shap_matrix = pd.DataFrame(
    shap_values,
    columns=GENES,
    index=X_train.index
)

shap_matrix.insert(
    0,
    "Sample",
    X_train.index
)

shap_matrix.to_csv(
    os.path.join(
        OUTDIR,
        "FINAL_THREE_GENE_XGBOOST_SHAP_VALUES.csv"
    ),
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    os.path.join(
        OUTDIR,
        "FINAL_THREE_GENE_XGBOOST_SHAP_IMPORTANCE.csv"
    ),
    index=False
)


# ============================================================
# SHAP DATASET FOR MANUSCRIPT
# ============================================================

publication = X_train.copy()

publication.insert(
    0,
    "Sample",
    X_train.index
)

publication["Diagnosis"] = y_train.values

for gene in GENES:
    publication[gene + "_SHAP"] = shap_matrix[gene].values

publication.to_csv(
    os.path.join(
        OUTDIR,
        "FINAL_THREE_GENE_XGBOOST_SHAP_PUBLICATION_DATA.csv"
    ),
    index=False
)

publication.to_csv(
    os.path.join(
        "06_Manuscript",
        "Tables",
        "Table_final_three_gene_xgboost_shap.csv"
    ),
    index=False
)


# ============================================================
# FIGURES
# ============================================================

print("\nGenerating SHAP figures...")

try:

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=GENES,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGDIR,
            "Figure_XGBoost_SHAP_summary.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    plt.figure()

    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=GENES,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGDIR,
            "Figure_XGBoost_SHAP_importance_bar.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP figures generated successfully.")

except Exception as e:

    print("WARNING: Figure generation failed.")
    print("Error:", e)


print("\n" + "=" * 80)
print("XGBOOST SHAP ANALYSIS COMPLETE")
print("=" * 80)

print("\nSaved:")
print(
    "04_ML/Final_Model/Validation/"
    "FINAL_THREE_GENE_XGBOOST_SHAP_VALUES.csv"
)

print(
    "04_ML/Final_Model/Validation/"
    "FINAL_THREE_GENE_XGBOOST_SHAP_IMPORTANCE.csv"
)

print(
    "04_ML/Final_Model/Validation/"
    "FINAL_THREE_GENE_XGBOOST_SHAP_PUBLICATION_DATA.csv"
)

print(
    "06_Manuscript/Tables/"
    "Table_final_three_gene_xgboost_shap.csv"
)

print(
    "06_Manuscript/Figures/"
    "Figure_XGBoost_SHAP_summary.png"
)

print(
    "06_Manuscript/Figures/"
    "Figure_XGBoost_SHAP_importance_bar.png"
)
