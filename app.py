"""Streamlit app:  streamlit run app.py"""
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from predict_api import predict, bundle
from common import load_raw

st.set_page_config(page_title="KOH Activated Carbon Predictor", layout="wide")
st.title("KOH-activated carbon: surface area & yield predictor")
st.caption("Gaussian-process model trained on the lab grid (IR 0-6, 800-1200 °C, 0-90 min). Intervals are conservative 95% bands "
           "calibrated on leave-one-temperature/time-out validation.")

with st.sidebar:
    st.header("Inputs")
    IR = st.number_input("IR (KOH : carbon impregnation ratio)", 0.0, 6.0, 4.0, 1.0)
    T = st.number_input("Activation temperature (°C)", 800.0, 1200.0, 950.0, 10.0)
    t = st.number_input("Activation time (min)", 0.0, 90.0, 50.0, 5.0)

r = predict(IR, T, t)
c1, c2 = st.columns(2)
c1.metric("Surface area", f"{r['surface_area']:.0f}", f"95% interval {r['sa_lo']:.0f} – {r['sa_hi']:.0f}", delta_color="off")
if r["yield_pct"] is not None:
    c2.metric("Yield (%)" + (" – ESTIMATE" if r["yield_is_estimate"] else ""), f"{r['yield_pct']:.2f}", f"interval {r['yield_lo']:.2f} – {r['yield_hi']:.2f}" + (" (excludes IR-scaling assumption)" if r["yield_is_estimate"] else ""), delta_color="off")
else:
    c2.metric("Yield (%)", "n/a", "no yield data for IR 0", delta_color="off")
for w in r["warnings"]: st.warning(w)

b = bundle(); gp, sc, k = b["sa"][-1], b["sa"][0], b["k95"]; df = load_raw()
tq = np.linspace(0, 90, 181); Tq = np.linspace(800, 1200, 161)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, xq, xs, lab, fixed in ((axs[0], tq, "Act_Time", "Activation time (min)", T), (axs[1], Tq, "Act_Temp", "Activation temperature (°C)", t)):
    X = np.c_[np.full_like(xq, IR), np.full_like(xq, T) if xs == "Act_Time" else xq, xq if xs == "Act_Time" else np.full_like(xq, t)]
    mu, sd = gp.predict(sc.transform(X), return_std=True)
    ax.plot(xq, mu, color="#1f4e79"); ax.fill_between(xq, mu - k * sd, mu + k * sd, alpha=.2, color="#1f4e79")
    ax.set_xlabel(lab); ax.set_ylabel("Surface area")
axs[0].set_title(f"vs time  (IR {IR:g}, {T:g} °C)"); axs[1].set_title(f"vs temperature  (IR {IR:g}, {t:g} min)")
# overlay measured points on the matching slice
d = df[(df.IR == round(IR)) & (df.Act_Temp == T)];  axs[0].scatter(d.Act_Time, d.Sur_Area, c="k", s=18, zorder=3)
d = df[(df.IR == round(IR)) & (df.Act_Time == t)];  axs[1].scatter(d.Act_Temp, d.Sur_Area, c="k", s=18, zorder=3)
st.pyplot(fig)
st.subheader("Suggested next experiments")
c3, c4 = st.columns(2)
c3.write("**Max expected improvement** (find a higher surface area)"); c3.dataframe(pd.read_csv("results/next_experiments_exploit_EI.csv").round(1), hide_index=True)
c4.write("**Max information** (reduce model uncertainty)"); c4.dataframe(pd.read_csv("results/next_experiments_explore.csv").round(1), hide_index=True)
