import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    recall_score, classification_report
)

STABLE_GENES = ["ABCA6", "CRLF1", "TNFRSF11B"]

def load_genes_x_samples(path, genes, label):
    df = pd.read_csv(path, index_col=0)
    missing_as_rows = [g for g in genes if g not in df.index]
    if not missing_as_rows:
        print(f"{label}: genes found as ROWS. Shape: {df.shape}")
        return df.loc[genes].T
    missing_as_cols = [g for g in genes if g not in df.columns]
    if not missing_as_cols:
        print(f"{label}: genes found as COLUMNS. Shape: {df.shape}")
        return df[genes]
    print(f"\n{label}: genes NOT found in either rows or columns.")
    print(f"  File shape: {df.shape}")
    raise ValueError(f"{label}: could not locate genes {genes}")

def find_diagnosis_column(meta_df, label):
    candidates = [c for c in meta_df.columns if "diag" in c.lower()]
    if not candidates:
        print(f"\n{label} metadata: no column with 'diag' in its name found.")
        print(f"  Available columns: {list(meta_df.columns)}")
        raise ValueError(f"{label}: could not auto-detect a diagnosis column - see columns printed above.")
    col = candidates[0]
    print(f"{label} metadata: using column '{col}' as diagnosis. Unique values: {meta_df[col].unique()}")
    return col

def to_ad_binary(series):
    return series.astype(str).str.upper().str.contains("AD").astype(int).values

train_meta = pd.read_csv("04_ML/GSE48350_RMA_train_meta.csv")
test_meta  = pd.read_csv("04_ML/GSE48350_RMA_test_meta.csv")
external_meta = pd.read_csv("02_Metadata/GSE5281_hippocampus_metadata_age_corrected.csv")

train_diag_col = find_diagnosis_column(train_meta, "TRAIN")
test_diag_col  = find_diagnosis_column(test_meta, "TEST")
ext_diag_col   = find_diagnosis_column(external_meta, "EXTERNAL")

X_train = load_genes_x_samples("04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv", STABLE_GENES, "TRAIN")
X_test  = load_genes_x_samples("04_ML/GSE48350_RMA_test_top25var_age_adjusted.csv", STABLE_GENES, "TEST")
X_ext   = load_genes_x_samples("04_ML/GSE5281_RMA_genelevel_age_adjusted.csv", STABLE_GENES, "EXTERNAL")

y_train = to_ad_binary(train_meta[train_diag_col])
y_test  = to_ad_binary(test_meta[test_diag_col])
y_ext   = to_ad_binary(external_meta[ext_diag_col])

print("\nTrain:", X_train.shape, " Internal test:", X_test.shape, " External:", X_ext.shape)
print("Train AD/Control:", y_train.sum(), "/", len(y_train)-y_train.sum())
print("Test AD/Control:", y_test.sum(), "/", len(y_test)-y_test.sum())
print("External AD/Control:", y_ext.sum(), "/", len(y_ext)-y_ext.sum())

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
X_ext_s   = scaler.transform(X_ext)

model = LogisticRegression(penalty=None, max_iter=5000)
model.fit(X_train_s, y_train)

print("\nCoefficients:")
for gene, coef in zip(STABLE_GENES, model.coef_[0]):
    print(f"  {gene}: {coef:+.4f}")
print(f"  Intercept: {model.intercept_[0]:+.4f}")

print("\n" + "=" * 60)
print("INTERNAL TEST (GSE48350 held-out, n=%d)" % len(y_test))
print("=" * 60)
y_test_prob = model.predict_proba(X_test_s)[:, 1]
y_test_pred = model.predict(X_test_s)
print("ROC-AUC:", round(roc_auc_score(y_test, y_test_prob), 4))
print("Accuracy:", round(accuracy_score(y_test, y_test_pred), 4))
print("Sensitivity (recall):", round(recall_score(y_test, y_test_pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))

print("\n" + "=" * 60)
print("EXTERNAL VALIDATION (GSE5281, n=%d) - 3-GENE MODEL" % len(y_ext))
print("=" * 60)
y_ext_prob = model.predict_proba(X_ext_s)[:, 1]
y_ext_pred = model.predict(X_ext_s)
print("ROC-AUC:", round(roc_auc_score(y_ext, y_ext_prob), 4))
print("  (compare to 7-gene external AUC: 0.2692)")
print("Accuracy:", round(accuracy_score(y_ext, y_ext_pred), 4))
print("Sensitivity (recall):", round(recall_score(y_ext, y_ext_pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_ext, y_ext_pred))
print("\nFull classification report:")
print(classification_report(y_ext, y_ext_pred, target_names=["Control", "AD"], zero_division=0))

print("\nInterpretation:")
print("  If AUC is meaningfully above 0.5 (e.g. >0.6) and clearly better")
print("  than the 7-gene result (0.2692), this supports the hypothesis")
print("  that SORBS1/PMS2P2/TAB2 were actively hurting external generalization.")
print("  If AUC is still poor, the problem runs deeper than just those three genes.")
