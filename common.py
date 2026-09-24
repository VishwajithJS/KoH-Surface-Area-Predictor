"""Shared utilities for the KOH-activated-carbon ML project."""
import re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

RAW = "data/AC_Prep_-_KOH.txt"
FEATURES = ["IR", "Act_Temp", "Act_Time"]

def load_raw(path=RAW):
    """Parse the whitespace-separated lab file (Yield is present only on some rows)."""
    rows = []
    for line in open(path):
        p = line.split()
        if not p or not re.fullmatch(r"\d+", p[0]):
            continue
        v = [float(x) for x in p]
        rows.append(v + [np.nan] * (5 - len(v)))
    df = pd.DataFrame(rows, columns=["IR", "Act_Temp", "Act_Time", "Sur_Area", "Yield"])
    df["IR"] = df["IR"].astype(int)
    return df

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(R2=1 - ss_res / ss_tot, RMSE=np.sqrt(np.mean((y - p) ** 2)),
                MAE=np.mean(np.abs(y - p)), MAPE=100 * np.mean(np.abs((y - p) / y)))
