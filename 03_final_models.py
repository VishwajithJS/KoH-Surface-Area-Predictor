"""Fit final models, cross-validate them (with uncertainty calibration), and export artefacts."""
import json, joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import load_raw, metrics, FEATURES
from models import gpr, TrendGP

df = load_raw()
X, y = df[FEATURES].values, df.Sur_Area.values
sa_model = lambda: make_pipeline(StandardScaler(), gpr(3, 2.5, restarts=10))

def predict_std(m, X):
    return m.predict(X, return_std=True)

# ---------- surface-area: CV with predictive std ----------
def splits(Xa, dfa, temps=(900, 1000, 1100), times=(15, 30, 45, 60, 75)):
    for T in temps: yield "LOTemp", np.where(dfa.Act_Temp != T)[0], np.where(dfa.Act_Temp == T)[0]
    for t in times: yield "LOTime", np.where(dfa.Act_Time != t)[0], np.where(dfa.Act_Time == t)[0]

rec = []
for s, tr, te in splits(X, df):
    m = sa_model().fit(X[tr], y[tr]); mu, sd = predict_std(m, X[te])
    rec.append(pd.DataFrame(dict(scheme=s, IR=df.IR.values[te], T=df.Act_Temp.values[te], t=df.Act_Time.values[te], y=y[te], mu=mu, sd=sd)))
cv = pd.concat(rec); cv.to_csv("results/cv_SA_gp_with_uncertainty.csv", index=False)
summary = {}
for s, g in cv.groupby("scheme"):
    z = np.abs(g.y - g.mu) / g.sd
    summary[f"SA_{s}"] = {**metrics(g.y, g.mu), "coverage95": float((z <= 1.96).mean()), "coverage68": float((z <= 1.0).mean())}

# ---------- final SA model on all data ----------
final_sa = sa_model().fit(X, y)
gp = final_sa[-1]
ls = gp.kernel_.k1.k2.length_scale        # ARD length-scales in standardised units
summary["SA_kernel"] = str(gp.kernel_)
summary["SA_ARD_lengthscales_std_units(IR,T,t)"] = [float(v) for v in ls]

# Closed-form LOO for a GP with fixed hyper-parameters (Rasmussen & Williams, Eq. 5.12)
Z = final_sa[0].transform(X)
K = gp.kernel_(Z); Ki = np.linalg.inv(K)
y_mean, y_std = y.mean(), y.std()           # same normalisation as normalize_y=True (avoids private sklearn attributes)
yn = (y - y_mean) / y_std
a = Ki @ yn
mu_loo = (yn - a / np.diag(Ki)) * y_std + y_mean
sd_loo = np.sqrt(1.0 / np.diag(Ki)) * y_std
df["SA_loo_pred"] = mu_loo; df["SA_loo_z"] = (y - mu_loo) / sd_loo
summary["SA_LOO_fixed_hyperparams"] = metrics(y, mu_loo)
df.sort_values("SA_loo_z", key=abs, ascending=False).head(12).to_csv("results/data_quality_flags.csv", index=False)
print(df.sort_values("SA_loo_z", key=abs, ascending=False).head(10)[["IR","Act_Temp","Act_Time","Sur_Area","SA_loo_pred","SA_loo_z"]].round(2).to_string())

# ---------- yield: IR=4 grid (35 pts) ----------
d4 = df[df.IR == 4]; X4, y4 = d4[["Act_Temp", "Act_Time"]].values, d4.Yield.values
rec = []
for s, tr, te in splits(X4, d4.reset_index(drop=True)):
    m = TrendGP().fit(X4[tr], y4[tr]); mu, sd = m.predict(X4[te], return_std=True)
    rec.append(pd.DataFrame(dict(scheme=s, y=y4[te], mu=mu, sd=sd)))
cvy = pd.concat(rec); cvy.to_csv("results/cv_Yield_IR4.csv", index=False)
for s, g in cvy.groupby("scheme"):
    z = np.abs(g.y - g.mu) / g.sd
    summary[f"Yield_IR4_{s}"] = {**metrics(g.y, g.mu), "coverage95": float((z <= 1.96).mean())}
final_y = TrendGP().fit(X4, y4)

# conservative interval multipliers calibrated on grouped (leave-one-temp/time-out) CV: 95th pct of |z|
k95 = max(1.96, float(np.quantile(np.abs(cv.y - cv.mu) / cv.sd, 0.95)))
k95y = max(1.96, float(np.quantile(np.abs(cvy.y - cvy.mu) / cvy.sd, 0.95)))
summary["interval_multiplier_SA"], summary["interval_multiplier_Yield"] = k95, k95y

joblib.dump(dict(sa=final_sa, yield_ir4=final_y, k95=k95, k95_yield=k95y, feature_order=FEATURES,
                 sa_domain=dict(IR=[0, 6], Act_Temp=[800, 1200], Act_Time=[0, 90]),
                 yield_note="Yield model trained on IR=4 only (35 points); other IRs have 1 yield point each."),
            "results/koh_models.joblib")
json.dump(summary, open("results/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
