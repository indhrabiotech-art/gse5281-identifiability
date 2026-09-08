"""Regenerates GSE48350_RMA_ML_matrix.csv with explicit provenance.
Source: 03_Preprocessing/GSE48350_RMA_genelevel.csv
Contents: all 62 samples, gene-level RMA, NO age adjustment.
"""
import pandas as pd
from pathlib import Path
R = Path.home()/"project_ml"
src = pd.read_csv(R/"03_Preprocessing/GSE48350_RMA_genelevel.csv", index_col=0)
if src.shape[0] > src.shape[1]:
    src = src.T
src.index = [str(i).replace(".CEL.gz","").replace(".CEL","").strip() for i in src.index]
old = pd.read_csv(R/"04_ML/GSE48350_RMA_ML_matrix.csv", index_col=0)
print("regenerated:", src.shape, " existing:", old.shape)
common_s = [s for s in old.index if s in src.index]
common_g = [g for g in old.columns if g in src.columns][:200]
if common_s and common_g:
    diff = (src.loc[common_s, common_g].values - old.loc[common_s, common_g].values)
    print(f"max abs diff over {len(common_s)}x{len(common_g)}: {abs(diff).max():.10f}")
    print("IDENTICAL" if abs(diff).max() < 1e-9 else "DIFFERENT - investigate")
src.to_csv(R/"04_ML/GSE48350_RMA_genelevel_all62_UNADJUSTED.csv")
print("written -> 04_ML/GSE48350_RMA_genelevel_all62_UNADJUSTED.csv")
