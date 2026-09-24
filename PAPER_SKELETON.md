# Paper skeleton (fill the [brackets] from your lab methods)

**Working title:** Uncertainty-aware Gaussian-process modelling of KOH activation: predicting surface area and yield of
activated carbon from impregnation ratio, temperature and time, with model-guided experiment selection

## Abstract (draft)
KOH-activated carbon preparation is slow (two days per sample), so activation conditions are sampled sparsely. We measured
surface area for a full grid of 7 impregnation ratios (IR 0-6), 5 temperatures (800-1200 °C) and 7 activation times
(0-90 min) (n = 245) and yield for [40] samples. Thirteen regressors were compared under leave-one-temperature-out and
leave-one-time-out validation with nested hyper-parameter tuning. A Gaussian process with an ARD Matern-5/2 kernel gave the
best generalisation to unseen temperatures (R² = 0.908, RMSE = 147) and times (R² = 0.986, RMSE = 53), outperforming
gradient-boosted trees (R² ~ 0.78 on unseen temperatures) and a neural network (0.66). Yield at IR 4 was predicted with
R² = 0.955-0.984. SHAP analysis ranked IR above temperature and time. Predictive uncertainty was calibrated on held-out
conditions and used to propose [n] experiments, of which [n] were performed as prospective validation ([result]).

## 1 Introduction
Chemical activation with KOH; role of IR, temperature, time; cost of experiments; ML in adsorbent/biochar design
(cite Hassan & Kazemi, Sci. Rep. 15:15157, 2025; Zhang et al., Environ. Sci. Technol. 54:7008, 2020); gap: small designed
datasets need uncertainty-aware models and experiment planning, not large-data algorithms.

## 2 Methods
2.1 Materials and activation procedure [precursor, KOH:carbon ratios, furnace, atmosphere, heating rate, washing].
2.2 Characterisation [N2 adsorption, BET, units of surface area, yield definition].
2.3 Dataset (Table 1: grid, counts, yield coverage).
2.4 Models (13 listed in `models.py`) and nested tuning.
2.5 Validation: leave-one-temperature-out, leave-one-time-out, random 5-fold x 2 (reference only).
2.6 Uncertainty calibration (95th percentile of |z|); interpretability (SHAP, ARD length-scales).
2.7 Experiment design: greedy max-variance batch, and expected improvement over current best.

## 3 Results
Table 2 = `results/benchmark_table.md`. Fig. 1 benchmark; Fig. 2 parity with intervals; Fig. 3 response surfaces per IR;
Fig. 4 curves with bands at skipped conditions; Fig. 5 yield (IR 4); Fig. 6 SHAP.
Report honestly: GPR vs ExtraTrees difference is small; the case for GPR is calibrated uncertainty + smoothness.
3.x Optimum region: IR 5, ~1000-1050 °C, 45-60 min. 3.y Prospective validation [new experiments vs predictions].

## 4 Discussion
Why sharp decline beyond 60 min at high temperature (pore collapse/burn-off) [mechanistic literature]; IR 5 vs 6 trade-off;
yield-surface-area trade-off; why tree ensembles extrapolate poorly across temperature; comparison with the reference paper's
data-splitting (random splits over repeated isotherms can leak).

## 5 Limitations and future work
No replicates; single feedstock; yield grid only at IR 4; extend to adsorption uptake (e.g. dye/CO2) and multi-objective optimisation.

## Reproducibility: code + data in this repository (state DOI / repository link).
