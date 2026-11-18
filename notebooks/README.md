# Notebooks

## `KoH_Surface_Area_Research.ipynb`
The complete research and figure pipeline for the KOH-activated-carbon project, designed for **Google Colab**
(`Runtime ▸ Run all`). Setup instructions: [`../COLAB_SETUP.md`](../COLAB_SETUP.md). Figure list:
[`../figure_manifest.md`](../figure_manifest.md).

The notebook does not import the repository scripts (`common.py`, `models.py`, `0x_*.py`); it re-implements the same
logic so that it runs on its own in Colab. Those scripts and the saved files in `results/` are left unchanged. Where the
saved repository results exist (`results/benchmark_SA_long.csv`, `results/cv_SA_gp_with_uncertainty.csv`,
`results/data_quality_flags.csv`, `results/shap_mean_abs_SA.csv`), the notebook loads them and shows them next to its own
numbers for comparison.

### Sections
| # | Section | What it produces |
|---|---|---|
| 0 | Configuration | target, seed, split, GPU, units, figure formats |
| 1 | Environment setup | installs missing packages; runtime, versions, GPU; seeds; helper functions |
| 2 | Google Drive mounting | `MyDrive/KoH_Surface_Area_Research/` folder tree (created if missing) |
| 3 | Dataset loading | copies the lab files to Drive once (never overwrites); parses `AC_Prep_-_KOH.txt` |
| 4 | Validation & preprocessing | automatic data checks; modelling table without imputation |
| 5 | Exploratory analysis | scatter matrix (paper Fig. 6), measured grid, measured curves |
| 6 | Outlier analysis | Monte Carlo cross-prediction outlier detection + box plots (paper Fig. 8) |
| 7 | Correlation analysis | Pearson / Spearman heat maps with p-values (paper Fig. 7) |
| 8 | Model training / loading | the paper's 15 model families; 70/15/15 split; tuning inside training data; Drive cache |
| 9 | Model benchmarking | Table 1 (tuned structure), Table 2 (metrics); leave-one-temperature/time-out benchmark |
| 10 | Train / validation / test evaluation | R² heat map with generalisation gap; residual diagnostics |
| 11 | Taylor diagrams | paper Fig. 9, for train, validation and test |
| 12 | Actual vs predicted | paper Fig. 10 |
| 13 | Cross-validation plots | paper Fig. 11 + out-of-fold cross plots of the grouped CV |
| 14 | Relative deviation | paper Fig. 12 |
| 15 | Prediction distributions | paper Fig. 13 |
| 16 | Feature importance & SHAP | paper Fig. 14 + SHAP beeswarm, dependence, permutation importance |
| 17 | GP uncertainty | grouped CV with predictive SD, interval calibration (cross-fitted), uncertainty maps, LOO flags |
| 18 | Response surfaces | surfaces per IR, curves with bands, 3-D surface, GP vs tree surfaces |
| 19 | Active learning | expected-improvement maps, max-information batch, retrospective simulation |
| 20 | Publication export | core figures → `figures/publication/`, figure manifest |
| 21 | Final summary | key numbers, section status, unavailable analyses → `results/<TARGET>_run_summary.json` |

### Validation protocols used
* **Random split (Part A, paper protocol):** 70 / 15 / 15 train / validation / test, stratified by IR, seed 42.
  Hyper-parameters by 5-fold CV inside the training set; best model chosen on validation; test reported once.
  Because the experimental grid is regular, this measures *interpolation*.
* **Leave-one-temperature-out / leave-one-time-out (Part B):** each interior temperature (900, 1000, 1100 °C) or time
  (15-75 min) is held out completely; tuning uses inner CV grouped by the same variable. This measures prediction at
  conditions that were never run.
* **Interval calibration:** the 95 % multiplier is cross-fitted (computed without the level it is tested on) so the
  reported coverage is not optimistic.

### Re-running for yield
Set `TARGET = "Yield"` in Section 0 and run all again. Yield is modelled for IR = 4 only (35 points), the only IR with a
full yield grid. Outputs are prefixed `Yield_` and stored next to the surface-area outputs.
