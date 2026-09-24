"""Benchmark 13 regressors for Surface-Area prediction under three validation schemes.
 - RandomKFold : 5-fold x 2 repeats (interpolation inside the grid; optimistic, for reference)
 - LOTemp      : leave-one-temperature-out (900/1000/1100 C)  -> mimics a *skipped temperature*
 - LOTime      : leave-one-time-out (15/30/45/60/75 min)       -> mimics a *skipped time*
Hyper-parameters are tuned INSIDE each training fold (nested CV) - no leakage.
"""
import sys, time, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import load_raw, metrics, FEATURES
from models import zoo

df = load_raw()
X, y = df[FEATURES].values, df["Sur_Area"].values

def splits():
    for r in range(2):
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=r).split(X)):
            yield "RandomKFold", f"r{r}f{k}", tr, te
    for T in (900, 1000, 1100):
        te = np.where(df.Act_Temp == T)[0]; tr = np.where(df.Act_Temp != T)[0]
        yield "LOTemp", f"T{T}", tr, te
    for t in (15, 30, 45, 60, 75):
        te = np.where(df.Act_Time == t)[0]; tr = np.where(df.Act_Time != t)[0]
        yield "LOTime", f"t{t}", tr, te

names = list(zoo().keys())
oof = []   # out-of-fold predictions
for scheme, fold, tr, te in splits():
    for name, model in zoo().items():
        t0 = time.time(); model.fit(X[tr], y[tr]); p = model.predict(X[te])
        oof.append(pd.DataFrame(dict(scheme=scheme, fold=fold, model=name, idx=te, y=y[te], pred=p)))
    print(scheme, fold, "done", flush=True)
oof = pd.concat(oof); oof.to_csv("results/oof_SA.csv", index=False)

rows = []
for (scheme, name), g in oof.groupby(["scheme", "model"]):
    if scheme == "RandomKFold":   # pooled over the 2 repeats
        m = metrics(g.y, g.pred)
    else:
        m = metrics(g.y, g.pred)
    rows.append(dict(scheme=scheme, model=name, **m))
res = pd.DataFrame(rows)
res.to_csv("results/benchmark_SA_long.csv", index=False)
tab = res.pivot(index="model", columns="scheme", values=["R2", "RMSE", "MAPE"])
print(tab.round(3).to_string())
