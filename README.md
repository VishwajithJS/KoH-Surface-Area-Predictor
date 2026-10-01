# Machine-learning prediction of surface area and yield in KOH-activated carbon

Predicts **surface area** (any IR 0-6, 800-1200 °C, 0-90 min) and **yield** (IR = 4 only) from activation
conditions, with calibrated uncertainty and a built-in "which experiment next?" planner (each real sample costs 2 days).

## Quick start
```bash
pip install -r requirements.txt
python 03_final_models.py                             # once: builds results/koh_models.joblib (not stored in git)
python predict_api.py --ir 4 --temp 950 --time 52      # command-line prediction
streamlit run app.py                                  # interactive app
python -m pytest                                      # smoke tests
```

## Reproduce everything (about 10 min on 1 CPU)
```bash
python 02_benchmark_SA.py         # 13 models x 3 validation schemes, nested tuning  -> results/benchmark_SA_long.csv
python 03_final_models.py         # final GPR (SA) + trend-GP (yield), CV, calibration -> results/koh_models.joblib
python 04_figures_and_design.py   # figures, SHAP, next-experiment recommendations
```
`common.py` parses the raw lab file (`data/AC_Prep_-_KOH.txt`); `models.py` holds the model zoo.

## Design decisions (read before writing the paper)
* **IR is a model input.** It is the strongest driver of surface area (SHAP mean |value|: IR 343, temperature 115, time 102).
* **Validation mimics your real use case.** Hold out a whole temperature (900/1000/1100 °C) or a whole time
  (15-75 min) so the model must predict conditions it has never seen. Random K-fold is reported only for reference
  because neighbouring grid points make it optimistic.
* **Nested tuning**: hyper-parameters are chosen inside each training fold only.
* **Uncertainty**: raw GP 95% intervals covered only ~82-83% of held-out points, so intervals are scaled by a
  multiplier calibrated on grouped CV (3.74 for surface area, 2.19 for yield). They are conservative.
* **Yield** has data only for IR = 4 (35 points) plus one point each at IR 1, 2, 3, 5, 6. For IR 1-6 other than 4 the app gives a clearly-labelled ESTIMATE (IR-4 surface x an IR scaling factor from the single
  900 C / 0 min point); this assumes the T/t trend is the same at every IR and is unvalidated. IR 0 has no yield data. Measure/record yield for the other IRs (a mass ratio) to extend the model.

## Results (surface area, out-of-fold)
See `results/benchmark_table.md`. GPR (Matern-5/2, ARD): R² 0.908 leave-one-temperature-out, 0.986 leave-one-time-out,
RMSE 147 and 53. GPR-RBF ties it; ExtraTrees is close (0.890 / 0.986). Boosted trees ~0.78 on unseen temperatures, MLP 0.66.
Yield (IR 4): R² 0.955 (temperature held out), 0.984 (time held out).

## Data-quality items to verify against lab notebooks (`results/data_quality_flags.csv`)
* IR5 & IR6 both 947 at 900 °C / 0 min; IR5 (800 °C, 15 min) = IR6 (800 °C, 0 min) = 736; IR3 and IR4 at 800 °C / 0 min = 773 / 774.
* IR5 800 °C / 0 min = 504 is a leave-one-out outlier (z = -2.9); IR6 1000 °C / 60 min = 1748 is z = +5.4.
* Yield at 900 °C / 0 min rises with IR (17.24 -> 28.22 for IR 1-4), the opposite of the usual trend. Confirm.

## Next experiments (`results/next_experiments_*.csv`)
* Exploit: IR 5, 1050 °C, 45-60 min (predicted ~1830 ± 50 vs current best 1808).
* Explore: corners/edges such as IR 0 at 850 °C / 0 min, IR 6 at 1150 °C / 90 min.
Run 3-5 of them, add the results to the data file and retrain: this **prospective validation** is the strongest evidence for a paper.

## Limitations
No replicate measurements (noise level unknown), one precursor/furnace/method, surface-area method and units should be stated
in the paper, predictions valid only inside the measured ranges.
