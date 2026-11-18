# Running the research notebook in Google Colab

Everything runs in Google Colab; no local Python installation is needed.

## 1. Open the notebook
Either
* **from GitHub** (after `notebooks/KoH_Surface_Area_Research.ipynb` is pushed to the repository):
  <https://colab.research.google.com/github/VishwajithJS/KoH-Surface-Area-Predictor/blob/main/notebooks/KoH_Surface_Area_Research.ipynb>
* **or by upload:** in Colab choose `File ▸ Upload notebook` and select `notebooks/KoH_Surface_Area_Research.ipynb`.

## 2. (Optional) choose a GPU
`Runtime ▸ Change runtime type ▸ T4 GPU`. A GPU is **not required**: the notebook detects it and uses it for the two
neural networks (ANN, CNN). Tree models and Gaussian processes stay on the CPU because, for ~250 samples, GPU training is
slower. Without a GPU everything runs on the CPU automatically.

## 3. Run everything
`Runtime ▸ Run all`. When asked, allow access to Google Drive (needed so results persist).

Run time: a full first run took 15.5 min on the free Colab CPU runtime (3.4 min on a 24-core PC). Most of it is the leave-one-temperature/time-out benchmark of 15 models.
Re-runs take about half as long because trained models and grouped predictions are cached in Drive.

## 4. Where things go
```
MyDrive/KoH_Surface_Area_Research/
├── data/            original lab files (copied once from GitHub, never overwritten)
├── models/          trained models (.joblib) - re-used on the next run if data and settings are unchanged
├── results/         tables (CSV/Markdown), predictions, run summary (JSON), figure manifest (CSV)
└── figures/
    ├── exploratory/          scatter matrix, measured grid, correlations, outliers
    ├── model_comparison/     benchmarks, Taylor diagrams, R² heat map
    ├── prediction/           actual-vs-predicted, cross plots, relative deviation, histograms
    ├── uncertainty/          GP parity with intervals, calibration, uncertainty maps, response surfaces
    ├── explainability/       Random-Forest importance, SHAP
    ├── experimental_design/  expected improvement, exploration batch, active-learning simulation
    └── publication/          the core set renamed Fig01 ... Fig12 for the paper
```
Every figure is saved as PNG (300 dpi), PDF and SVG (vector). See `figure_manifest.md`.

## 5. Adding new experiments
1. Open `MyDrive/KoH_Surface_Area_Research/data/AC_Prep_-_KOH.txt` (the Drive copy is your master copy).
2. Append a line `IR  Temp  Time  SurfaceArea  [Yield]` for each new sample, in the same format.
3. `Runtime ▸ Run all`. The notebook notices that the data changed and retrains everything.

## 6. Settings (Section 0 of the notebook)
| Setting | Default | Meaning |
|---|---|---|
| `TARGET` | `"Sur_Area"` | `"Yield"` analyses yield (IR = 4 data only) |
| `SEED` | 42 | all random splits and models |
| `SPLIT_FRACTIONS` | 0.70 / 0.15 / 0.15 | train / validation / test |
| `REMOVE_OUTLIERS` | False | drop Monte-Carlo-flagged rows before modelling |
| `RUN_GROUPED_BENCHMARK` | True | leave-one-temperature/time-out for all 15 models (slowest step) |
| `FORCE_RETRAIN` | False | ignore cached models in Drive |
| `DRIVE_MOUNT_TIMEOUT_MIN` | 10 | minutes to wait for the Drive permission dialog |
| `USE_GPU` | `"auto"` | `True` / `False` to force |
| `SA_UNIT` | `"m²/g"` | **assumed** BET unit - confirm and edit |

## 7. Troubleshooting
* **Drive did not mount** - the notebook waits up to 10 minutes for you to approve Drive access
  (`DRIVE_MOUNT_TIMEOUT_MIN`; Colab's own default is only 2). If it still fails, it continues writing to
  `/content/KoH_Surface_Area_Research` (lost when the runtime stops). Stop the run (`Runtime ▸ Interrupt execution`),
  re-run the Drive cell, approve access, then use `Runtime ▸ Run cell and below`.
* **"Could not clone repository"** - Colab is offline or GitHub is unreachable. Upload `AC_Prep_-_KOH.txt` to
  `MyDrive/KoH_Surface_Area_Research/data/` and run again.
* **A section failed** - the error is printed under that section and recorded in the final summary; the rest of the
  notebook still runs. Fix the cause and run again.
* **Results look stale** - set `FORCE_RETRAIN = True` once, or delete the `models/` folder in Drive.
* **Security** - only load `.joblib` files that this notebook created in your own Drive (they use pickle).
