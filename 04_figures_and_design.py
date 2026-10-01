"""Figures for the paper + active-learning (next-experiment) design."""
import os, numpy as np, pandas as pd, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from common import load_raw, FEATURES
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})
os.makedirs("figures", exist_ok=True)   # not tracked by git (empty after clone)
B = joblib.load("results/koh_models.joblib"); sa, yl, k = B["sa"], B["yield_ir4"], B["k95"]
df = load_raw(); X, y = df[FEATURES].values, df.Sur_Area.values
gp, sc = sa[-1], sa[0]
def P(IR, T, t):
    x = np.c_[IR, T, t].astype(float); mu, sd = gp.predict(sc.transform(x), return_std=True); return mu, sd

# ---- Fig 1: benchmark ----
res = pd.read_csv("results/benchmark_SA_long.csv")
order = res[res.scheme == "LOTemp"].sort_values("R2").model.tolist()
fig, ax = plt.subplots(figsize=(8, 4.6)); h = 0.27
for i, (s, c) in enumerate(zip(["LOTemp", "LOTime", "RandomKFold"], ["#c0392b", "#2e86c1", "#95a5a6"])):
    v = res[res.scheme == s].set_index("model").loc[order, "R2"]
    ax.barh(np.arange(len(order)) + (i - 1) * h, v, h, label={"LOTemp": "Leave-one-temperature-out", "LOTime": "Leave-one-time-out", "RandomKFold": "Random 5-fold (x2)"}[s], color=c)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.set_xlabel("R² (out-of-fold, nested tuning)"); ax.set_xlim(0.4, 1.0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False); ax.set_title("Surface-area prediction: 13 models, 3 validation schemes")
fig.savefig("figures/fig1_benchmark.png"); plt.close()

# ---- Fig 2: parity with predictive intervals (grouped CV) ----
cv = pd.read_csv("results/cv_SA_gp_with_uncertainty.csv")
fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, s, ttl in zip(axs, ["LOTemp", "LOTime"], ["Leave-one-temperature-out", "Leave-one-time-out"]):
    g = cv[cv.scheme == s]; ax.errorbar(g.y, g.mu, yerr=k * g.sd, fmt="o", ms=3, alpha=.55, elinewidth=.6, color="#1f4e79")
    lim = [0, 1900]; ax.plot(lim, lim, "k--", lw=.8); ax.set_xlim(lim); ax.set_ylim(lim)
    r2 = 1 - ((g.y - g.mu) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum(); ax.set_title(f"{ttl}  (R²={r2:.3f})")
    ax.set_xlabel("Measured surface area"); ax.set_ylabel("Predicted (GPR, calibrated 95% bars)")
fig.tight_layout(); fig.savefig("figures/fig2_parity_gpr.png"); plt.close()

# ---- Fig 3: response surfaces per IR ----
Tg, tg = np.linspace(800, 1200, 81), np.linspace(0, 90, 91); TT, tt = np.meshgrid(Tg, tg)
fig, axs = plt.subplots(2, 4, figsize=(15, 7), layout="constrained"); vmax = 1900
for IR, ax in zip(range(7), axs.ravel()):
    mu, _ = P(np.full(TT.size, IR), TT.ravel(), tt.ravel()); cs = ax.contourf(TT, tt, mu.reshape(TT.shape), levels=np.linspace(0, vmax, 20), cmap="viridis", extend="max")
    d = df[df.IR == IR]; ax.scatter(d.Act_Temp, d.Act_Time, c="white", s=10, edgecolor="k", lw=.4); ax.set_title(f"IR = {IR}"); ax.set_xlabel("Act. temperature (°C)"); ax.set_ylabel("Act. time (min)")
axs.ravel()[-1].axis("off"); fig.colorbar(cs, ax=axs, label="Predicted surface area", shrink=.8); fig.savefig("figures/fig3_response_surfaces.png"); plt.close()

# ---- Fig 4: curves at skipped conditions with intervals ----
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2)); cols = plt.cm.plasma(np.linspace(0.05, .85, 4))
for IR, c in zip([3, 4, 5, 6], cols):
    Tq = np.linspace(800, 1200, 161); mu, sd = P(np.full_like(Tq, IR), Tq, np.full_like(Tq, 60)); axs[0].plot(Tq, mu, color=c, label=f"IR {IR}"); axs[0].fill_between(Tq, mu - k * sd, mu + k * sd, color=c, alpha=.15)
    d = df[(df.IR == IR) & (df.Act_Time == 60)]; axs[0].scatter(d.Act_Temp, d.Sur_Area, color=c, s=22, edgecolor="k", lw=.5, zorder=3)
    tq = np.linspace(0, 90, 181); mu, sd = P(np.full_like(tq, IR), np.full_like(tq, 1000), tq); axs[1].plot(tq, mu, color=c); axs[1].fill_between(tq, mu - k * sd, mu + k * sd, color=c, alpha=.15)
    d = df[(df.IR == IR) & (df.Act_Temp == 1000)]; axs[1].scatter(d.Act_Time, d.Sur_Area, color=c, s=22, edgecolor="k", lw=.5, zorder=3)
axs[0].set_xlabel("Act. temperature (°C)  [time = 60 min]"); axs[1].set_xlabel("Act. time (min)  [temperature = 1000 °C]")
for a in axs: a.set_ylabel("Surface area")
axs[0].legend(frameon=False, ncol=2); fig.tight_layout(); fig.savefig("figures/fig4_curves_intervals.png"); plt.close()

# ---- Fig 5: yield (IR=4) ----
d4 = df[df.IR == 4]; TT2, tt2 = np.meshgrid(Tg, tg); mu, sd = yl.predict(np.c_[TT2.ravel(), tt2.ravel()], return_std=True)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
cs = axs[0].contourf(TT2, tt2, mu.reshape(TT2.shape), 18, cmap="magma_r"); axs[0].scatter(d4.Act_Temp, d4.Act_Time, c="white", s=12, edgecolor="k", lw=.4); fig.colorbar(cs, ax=axs[0], label="Predicted yield (%)")
cs = axs[1].contourf(TT2, tt2, (k * sd).reshape(TT2.shape), 15, cmap="Blues"); axs[1].scatter(d4.Act_Temp, d4.Act_Time, c="k", s=6); fig.colorbar(cs, ax=axs[1], label="95% half-width")
for a in axs: a.set_xlabel("Act. temperature (°C)"); a.set_ylabel("Act. time (min)")
axs[0].set_title("Yield, IR = 4"); axs[1].set_title("Uncertainty"); fig.tight_layout(); fig.savefig("figures/fig5_yield_IR4.png"); plt.close()

# ---- Active learning: which experiments next? (each costs 2 days) ----
IRs, Ts, ts = np.arange(0, 7), np.arange(800, 1201, 50), np.arange(0, 91, 5)
cand = pd.DataFrame([(a, b, c) for a in IRs for b in Ts for c in ts], columns=FEATURES)
measured = set(map(tuple, df[FEATURES].values)); cand = cand[[tuple(r) not in measured for r in cand.values]].reset_index(drop=True)
Zc, Zt = sc.transform(cand.values.astype(float)), sc.transform(X)
kc = gp.kernel_.k1; noise = gp.kernel_.k2.noise_level
def post_var(Ztrain, Zq):
    K = kc(Ztrain) + noise * np.eye(len(Ztrain)); Ks = kc(Zq, Ztrain); return np.diag(kc(Zq)) - np.einsum("ij,ij->i", Ks @ np.linalg.inv(K), Ks)
sel, Zcur = [], Zt.copy()
for _ in range(10):           # greedy batch: variance does not depend on y -> exact sequential design
    v = post_var(Zcur, Zc); v[sel] = -1; j = int(np.argmax(v)); sel.append(j); Zcur = np.vstack([Zcur, Zc[j]])
explore = cand.loc[sel].copy(); mu, sd = P(explore.IR, explore.Act_Temp, explore.Act_Time); explore["pred_SA"], explore["sd"] = mu, sd
explore.to_csv("results/next_experiments_explore.csv", index=False)
mu, sd = P(cand.IR, cand.Act_Temp, cand.Act_Time); best = y.max(); zz = (mu - best) / sd
cand["pred_SA"], cand["sd"], cand["EI"] = mu, sd, sd * (zz * norm.cdf(zz) + norm.pdf(zz))
cand.sort_values("EI", ascending=False).head(10).to_csv("results/next_experiments_exploit_EI.csv", index=False)
print("EXPLORE (max information):\n", explore.round(1).to_string()); print("\nEXPLOIT (max expected improvement over %.0f):\n" % best, cand.sort_values("EI", ascending=False).head(10).round(1).to_string())

# ---- SHAP (kernel explainer, exact for 3 features) ----
import shap
f = lambda a: gp.predict(sc.transform(a)); ex = shap.KernelExplainer(f, shap.kmeans(X, 15)); sv = ex.shap_values(X, nsamples=200, silent=True)
imp = pd.Series(np.abs(sv).mean(0), index=["IR", "Act. temp", "Act. time"]); imp.to_csv("results/shap_mean_abs_SA.csv")
plt.figure(figsize=(5, 3)); shap.summary_plot(sv, X, feature_names=["IR", "Act. temp", "Act. time"], show=False); plt.savefig("figures/fig6_shap.png"); plt.close()
print(imp)
