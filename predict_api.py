"""Inference helper used by the CLI and the Streamlit app."""
import joblib, numpy as np, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_B = None
def bundle():
    global _B
    if _B is None: _B = joblib.load(os.path.join(_HERE, "results", "koh_models.joblib"))
    return _B

def predict(IR, act_temp, act_time):
    """Return dict with surface area (+/- calibrated 95% interval), yield (IR=4 only) and warnings."""
    b = bundle(); k = b["k95"]; warn = []
    dom = b["sa_domain"]
    for name, v, key in (("IR", IR, "IR"), ("Act Temp", act_temp, "Act_Temp"), ("Act Time", act_time, "Act_Time")):
        lo, hi = dom[key]
        if not lo <= v <= hi: warn.append(f"{name}={v} is outside the measured range [{lo}, {hi}] - extrapolation, do not trust.")
    x = np.array([[IR, act_temp, act_time]], float)
    mu, sd = b["sa"][-1].predict(b["sa"][0].transform(x), return_std=True)
    out = dict(surface_area=float(mu[0]), sa_lo=float(mu[0] - k * sd[0]), sa_hi=float(mu[0] + k * sd[0]), sa_sd=float(sd[0]),
               yield_pct=None, yield_lo=None, yield_hi=None, yield_is_estimate=False, warnings=warn)
    if int(round(IR)) == 4 and abs(IR - 4) < 1e-9:
        ym, ys = b["yield_ir4"].predict(np.array([[act_temp, act_time]], float), return_std=True)
        ky = b["k95_yield"]
        out.update(yield_pct=float(ym[0]), yield_lo=float(ym[0] - ky * ys[0]), yield_hi=float(ym[0] + ky * ys[0]))
    elif 1 <= IR <= 6:
        # ASSUMPTION-BASED ESTIMATE: IR-4 yield surface x IR scaling factor measured at 900 C / 0 min.
        # Assumes the temperature/time response of yield has the same shape at every IR (not verified by data).
        import pandas as pd
        from common import load_raw
        d = load_raw(); ref = d[(d.Act_Temp == 900) & (d.Act_Time == 0) & d.Yield.notna()].sort_values("IR")
        y4 = float(ref[ref.IR == 4].Yield.iloc[0]); f = float(np.interp(IR, ref.IR.values, ref.Yield.values / y4))
        ym, ys = b["yield_ir4"].predict(np.array([[act_temp, act_time]], float), return_std=True); ky = b["k95_yield"]
        out.update(yield_pct=float(f * ym[0]), yield_lo=float(f * (ym[0] - ky * ys[0])), yield_hi=float(f * (ym[0] + ky * ys[0])), yield_is_estimate=True)
        out["warnings"].append(f"Yield for IR={IR:g} is an ESTIMATE: IR-4 yield surface x scaling factor {f:.2f} taken from a single measurement "
                               "(900 C / 0 min). It assumes the temperature/time trend is the same at every IR; the interval does NOT include that assumption.")
    else:
        out["warnings"].append("No yield measurement exists for IR=0, so yield cannot be estimated.")
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Predict surface area / yield of KOH-activated carbon")
    ap.add_argument("--ir", type=float, required=True); ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--time", type=float, required=True)
    a = ap.parse_args(); r = predict(a.ir, a.temp, a.time)
    print(f"Surface area : {r['surface_area']:.0f}  (95% interval {r['sa_lo']:.0f} - {r['sa_hi']:.0f})")
    if r["yield_pct"] is not None: print(f"Yield        : {r['yield_pct']:.2f}  (95% interval {r['yield_lo']:.2f} - {r['yield_hi']:.2f})")
    for w in r["warnings"]: print("WARNING:", w)
