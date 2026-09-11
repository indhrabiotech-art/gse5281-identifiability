#!/usr/bin/env python3
"""
random_signature_null.py

Answers the objection "you have shown one weak signature fails; so what?"

Two null distributions over 1,000 random three-gene sets drawn from the
variance-filtered candidates:

  1. TRANSFER NULL     fit on GSE48350, apply unchanged to GSE5281.
                       Where does the observed 0.662 fall?

  2. WITHIN-GSE5281    leave-one-out cross-validated AUC of the same random
     NULL              sets inside GSE5281 alone. If random genes routinely
                       reach high AUC, GSE5281 "validates" anything, which is
                       the strongest possible statement of the paper's thesis.

Usage:  cd ~/project_ml && python3 Python_scripts/random_signature_null.py
"""

import numpy as np, pandas as pd, json, sys, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import roc_auc_score

os.chdir(os.path.expanduser("~/project_ml"))
OUT = "04_ML/Random_Null"; os.makedirs(OUT, exist_ok=True)

SEED, N_ITER = 2026, 1000
OBSERVED_TRANSFER = 0.661538          # canonical raw transfer ROC-AUC
SIGNATURE = ["ABCA6", "CRLF1", "TNFRSF11B"]

# ----------------------------------------------------------------- load data
def first_existing(paths, what):
    for p in paths:
        if os.path.exists(p): return p
    sys.exit(f"Could not locate {what}. Tried:\n  " + "\n  ".join(paths))

train_p = first_existing([
    "04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
    "04_ML/LASSO/GSE48350_RMA_train_top25var_age_adjusted.csv"], "training matrix")
meta_p  = first_existing([
    "04_ML/GSE48350_RMA_train_meta.csv",
    "04_ML/LASSO/GSE48350_RMA_train_meta.csv"], "training metadata")
ext_p   = first_existing([
    "03_Preprocessing/GSE5281_RMA_genelevel.csv"], "GSE5281 matrix")
ext_m   = first_existing([
    "02_Metadata/GSE5281_hippocampus_metadata.csv"], "GSE5281 metadata")

Xtr = pd.read_csv(train_p, index_col=0)
if Xtr.shape[0] > Xtr.shape[1]: Xtr = Xtr.T      # want samples x genes
mtr = pd.read_csv(meta_p)
gcol = next(c for c in mtr.columns if mtr[c].astype(str).str.startswith("GSM").any())
dcol = next(c for c in mtr.columns if c.lower().startswith(("diagn","group","status")))
mtr[gcol] = mtr[gcol].astype(str).str.split("_").str[0]
idx = [str(i).split("_")[0] for i in Xtr.index]
Xtr.index = idx
mtr = mtr.set_index(gcol).loc[idx]
ytr = mtr[dcol].astype(str).str.contains("ad|alzh|affect", case=False, regex=True).astype(int).values

Xex = pd.read_csv(ext_p, index_col=0).T
Xex.index = [
    str(i).split("_")[0]
    .replace(".CEL", "")
    .replace(".cel", "")
    .replace(".gz", "")
    for i in Xex.index
]
mex = pd.read_csv(ext_m)
mex["GSM"] = mex["GSM"].astype(str)
print("\n===== DEBUG SAMPLE IDs =====")
print("Xex sample IDs:")
print(Xex.index.tolist())

print("\nmex GSM sample IDs:")
print(mex["GSM"].tolist())

print("\nNumber of Xex samples:", len(Xex.index))
print("Number of mex samples:", len(mex))

print("============================\n")
mex = mex.set_index("GSM").loc[Xex.index]
yex = mex["diagnosis"].astype(str).str.contains("ad|alzh|affect", case=False, regex=True).astype(int).values

print(f"training  {Xtr.shape[0]} samples x {Xtr.shape[1]} genes, {ytr.sum()} cases")
print(f"transfer  {Xex.shape[0]} samples x {Xex.shape[1]} genes, {yex.sum()} cases")

candidates = [g for g in Xtr.columns if g in Xex.columns]
print(f"candidate genes present in both: {len(candidates)}")

def model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(penalty="l1", C=0.3, solver="liblinear",
                                            max_iter=10000, random_state=42))

# ---------------------------------------------------------- observed values
m = model(); m.fit(Xtr[SIGNATURE], ytr)
obs_transfer = roc_auc_score(yex, m.predict_proba(Xex[SIGNATURE])[:, 1])
p_obs = cross_val_predict(model(), Xex[SIGNATURE], yex,
                          cv=LeaveOneOut(), method="predict_proba")[:, 1]
obs_within = roc_auc_score(yex, p_obs)
print(f"\nobserved transfer ROC-AUC        {obs_transfer:.4f}  (canonical {OBSERVED_TRANSFER:.4f})")
print(f"observed within-GSE5281 LOO AUC  {obs_within:.4f}")

# ------------------------------------------------------------------ nulls
rng = np.random.default_rng(SEED)
null_transfer, null_within, picked = [], [], []
for k in range(N_ITER):
    g = list(rng.choice(candidates, 3, replace=False))
    try:
        mm = model(); mm.fit(Xtr[g], ytr)
        null_transfer.append(roc_auc_score(yex, mm.predict_proba(Xex[g])[:, 1]))
        pp = cross_val_predict(model(), Xex[g], yex,
                               cv=LeaveOneOut(), method="predict_proba")[:, 1]
        null_within.append(roc_auc_score(yex, pp))
        picked.append(",".join(g))
    except Exception:
        continue
    if (k+1) % 100 == 0:
        print(f"  {k+1}/{N_ITER}", flush=True)

nt = np.array(null_transfer); nw = np.array(null_within)

def describe(v, label, ref, extra=None):
    print(f"\n{label}   (n = {len(v)})")
    print(f"  median {np.median(v):.3f}   mean {v.mean():.3f}   SD {v.std():.3f}")
    print(f"  2.5-97.5 percentile  {np.percentile(v,2.5):.3f} - {np.percentile(v,97.5):.3f}")
    print(f"  proportion >= {ref:.3f}: {(v >= ref).mean():.3f}")
    if extra is not None:
        print(f"  proportion >= {extra:.2f}: {(v >= extra).mean():.3f}")

print("\n=================== RANDOM SIGNATURE NULLS ===================")
describe(nt, "TRANSFER: fit on GSE48350, applied to GSE5281", obs_transfer, 0.70)
describe(nw, "WITHIN GSE5281: leave-one-out AUC", obs_within, 0.90)

pd.DataFrame({"genes": picked, "transfer_auc": nt, "within_auc": nw}) \
  .to_csv(f"{OUT}/random_signature_null_distribution.csv", index=False)

summary = {
  "n_iterations": int(len(nt)), "seed": SEED, "n_candidates": len(candidates),
  "observed_transfer_auc": float(obs_transfer),
  "observed_within_auc": float(obs_within),
  "transfer_null_median": float(np.median(nt)),
  "transfer_null_p_ge_observed": float((nt >= obs_transfer).mean()),
  "within_null_median": float(np.median(nw)),
  "within_null_p_ge_0.90": float((nw >= 0.90).mean()),
  "within_null_p_ge_observed": float((nw >= obs_within).mean()),
}
with open(f"{OUT}/random_signature_null_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n----------------------- INTERPRETATION -----------------------")
if (nt >= obs_transfer).mean() > 0.20:
    print(f"The observed transfer AUC is unremarkable: {(nt>=obs_transfer).mean():.1%} of")
    print("random three-gene sets equal or exceed it. The signature carries no")
    print("transfer advantage over arbitrary genes.")
else:
    print(f"The observed transfer AUC exceeds {1-(nt>=obs_transfer).mean():.1%} of random sets.")
    print("Report this honestly; it is a point in the signature's favour.")
if (nw >= 0.90).mean() > 0.10:
    print(f"\nWithin GSE5281, {(nw>=0.90).mean():.1%} of RANDOM three-gene sets reach")
    print("LOO AUC >= 0.90. The cohort contrast is so large that this dataset")
    print("will validate essentially any signature. This is the strongest")
    print("available statement of the paper's thesis.")
else:
    print(f"\nWithin GSE5281, only {(nw>=0.90).mean():.1%} of random sets reach 0.90.")
    print("The cohort contrast, though present, does not make every signature")
    print("appear to validate. State this rather than overclaiming.")
print(f"\nOutputs in {OUT}")
