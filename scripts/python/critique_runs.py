#!/usr/bin/env python3
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser("~/project_ml"))
OUT="04_ML/Critique_Runs"; os.makedirs(OUT, exist_ok=True)
log=open(f"{OUT}/critique_runs_py_log.txt","w")
def say(*a):
    s=" ".join(str(x) for x in a); print(s, flush=True); log.write(s+"\n")
say("=========== CRITIQUE RUNS (Python) ===========\n")
SIG=["ABCA6","CRLF1","TNFRSF11B"]
def model(): return make_pipeline(StandardScaler(),
    LogisticRegression(penalty="l1",C=0.3,solver="liblinear",
                       max_iter=10000,random_state=42))
def keyf(s): return str(s).split("_")[0].replace(".CEL.gz","").replace(".CEL","")

# ---- R0  leak check ------------------------------------------------------
say("---------- R0  nested-CV leak check ----------")
src="Python_scripts/reviewer_robustness_master.py"
if os.path.exists(src):
    hits=[(i,l.rstrip()) for i,l in enumerate(open(src),1)
          if any(k in l for k in ("top25var","age_adjust","variance","read_csv"))]
    for i,l in hits[:40]: say(f"  {i}: {l[:110]}")
    pre = any("top25var_age_adjusted" in l for _,l in hits)
    say("\n  VERDICT:", "filter/adjustment applied BEFORE the loop -> NOT leak-free"
        if pre else "no pre-filtered matrix read at top level; inspect the loop manually")
else: say("  script not found")

# ---- load canonical data -------------------------------------------------
Xtr=pd.read_csv("04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",index_col=0)
if Xtr.shape[0]>Xtr.shape[1]: Xtr=Xtr.T
Xtr.index=[keyf(i) for i in Xtr.index]
mtr=pd.read_csv("04_ML/GSE48350_RMA_train_meta.csv")
gc=[c for c in mtr.columns if mtr[c].astype(str).str.startswith("GSM").any()][0]
dc=[c for c in mtr.columns if c.lower().startswith(("diagn","group","status"))][0]
mtr[gc]=mtr[gc].map(keyf); mtr=mtr.set_index(gc).loc[Xtr.index]
ytr=mtr[dc].astype(str).str.contains("ad|alzh|affect",case=False).astype(int).values
Xex=pd.read_csv("03_Preprocessing/GSE5281_RMA_genelevel.csv",index_col=0).T
Xex.index=[keyf(i) for i in Xex.index]
mex=pd.read_csv("02_Metadata/GSE5281_hippocampus_metadata.csv")
mex["GSM"]=mex["GSM"].map(keyf); mex=mex.set_index("GSM").loc[Xex.index]
yex=mex["diagnosis"].astype(str).str.contains("ad|alzh|affect",case=False).astype(int).values
say(f"\n  training {Xtr.shape}, cases {ytr.sum()} | transfer {Xex.shape}, cases {yex.sum()}")

# ---- R2  age-restricted transfer ----------------------------------------
say("\n---------- R2  age-restricted transfer ----------")
m4=pd.read_csv("02_Metadata/GSE48350_metadata_grouped.csv"); m4["GSM"]=m4["GSM"].map(keyf)
age=m4.set_index("GSM")["Age"].reindex(Xtr.index).astype(float)
keep=((ytr==1)|(age>=age[ytr==1].min())).values
say(f"  youngest case {age[ytr==1].min():.0f}; retained {keep.sum()} of {len(keep)}"
    f" ({ytr[keep].sum()} cases, {(1-ytr[keep]).sum()} controls)")
mr=model(); mr.fit(Xtr.loc[keep,SIG], ytr[keep])
a_tr=roc_auc_score(ytr[keep], mr.predict_proba(Xtr.loc[keep,SIG])[:,1])
a_ex=roc_auc_score(yex, mr.predict_proba(Xex[SIG])[:,1])
cv=cross_val_predict(model(),Xtr.loc[keep,SIG],ytr[keep],
                     cv=StratifiedKFold(5,shuffle=True,random_state=42),
                     method="predict_proba")[:,1]
say(f"  apparent {a_tr:.4f} | 5-fold {roc_auc_score(ytr[keep],cv):.4f} | TRANSFER {a_ex:.4f}")
pd.DataFrame([{"n":int(keep.sum()),"apparent":a_tr,
               "cv5":roc_auc_score(ytr[keep],cv),"transfer":a_ex}]).to_csv(
    f"{OUT}/R2_age_restricted_transfer.csv",index=False)

# ---- R3  folds below 0.5 -------------------------------------------------
say("\n---------- R3  folds below 0.5 ----------")
cand=[p for p in glob.glob("04_ML/**/*nested*", recursive=True) if p.endswith(".csv")]
done=False
for p in cand:
    try:
        d=pd.read_csv(p); col=[c for c in d.columns if "auc" in c.lower()]
        if col and len(d)>50:
            v=d[col[0]].dropna()
            say(f"  {p}: {(v<0.5).sum()} of {len(v)} folds below 0.5 ({100*(v<0.5).mean():.1f}%)")
            done=True
    except Exception: pass
if not done: say("  per-fold file not found. Files seen:", cand)

# ---- R4  Cohen's d with CIs ---------------------------------------------
say("\n---------- R4  Cohen's d with 95% CI ----------")
def dci(x,y,B=10000,seed=42):
    r=np.random.default_rng(seed)
    f=lambda a,b:(a.mean()-b.mean())/np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
    bs=[f(r.choice(x,len(x),True),r.choice(y,len(y),True)) for _ in range(B)]
    return f(x,y), np.percentile(bs,2.5), np.percentile(bs,97.5)
rows=[]
for g in SIG:
    for lab,X,y in [("discovery",Xtr,ytr),("transfer",Xex,yex)]:
        v=X[g].values; d,lo,hi=dci(v[y==1],v[y==0])
        say(f"  {g:<10} {lab:<10} d={d:6.3f}  95% CI {lo:6.3f} to {hi:6.3f}")
        rows.append({"gene":g,"cohort":lab,"d":d,"lo":lo,"hi":hi})
pd.DataFrame(rows).to_csv(f"{OUT}/R4_cohens_d_ci.csv",index=False)

# ---- R7  discovery null on canonical matrix ------------------------------
say("\n---------- R7  discovery random-panel null (canonical) ----------")
cands=[g for g in Xtr.columns if g in Xex.columns]
cvk=StratifiedKFold(5,shuffle=True,random_state=42)
obs=roc_auc_score(ytr,cross_val_predict(model(),Xtr[SIG],ytr,cv=cvk,
                  method="predict_proba")[:,1])
say(f"  observed 5-fold CV AUC (canonical matrix): {obs:.6f}")
rng=np.random.default_rng(2026); null=[]
for k in range(1000):
    g=list(rng.choice(cands,3,replace=False))
    try: null.append(roc_auc_score(ytr,cross_val_predict(model(),Xtr[g],ytr,
                     cv=cvk,method="predict_proba")[:,1]))
    except Exception: pass
    if (k+1)%200==0: say(f"    {k+1}/1000")
n=np.array(null)
say(f"  null mean {n.mean():.4f} SD {n.std():.4f} median {np.median(n):.4f}")
say(f"  2.5-97.5%  {np.percentile(n,2.5):.4f} - {np.percentile(n,97.5):.4f}")
say(f"  percentile of observed: {(n<obs).mean():.4f}   prop >= observed: {(n>=obs).mean():.4f}")
pd.DataFrame({"null_auc":n}).to_csv(f"{OUT}/R7_discovery_null_canonical.csv",index=False)
json.dump({"observed_cv_auc":float(obs),"null_mean":float(n.mean()),
           "null_median":float(np.median(n)),"percentile":float((n<obs).mean()),
           "n_iter":int(len(n)),"matrix":"train_top25var_age_adjusted"},
          open(f"{OUT}/R7_summary.json","w"),indent=2)
say("\n=========== PY BLOCK DONE ===========")
log.close()
