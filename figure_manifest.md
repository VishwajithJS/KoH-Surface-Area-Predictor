# Figure manifest

Figures produced by `notebooks/KoH_Surface_Area_Research.ipynb`. Every figure is written to Google Drive under
`MyDrive/KoH_Surface_Area_Research/figures/<folder>/` as **PNG (300 dpi), PDF and SVG (vector)**, with the file name
`<TARGET>_<ID>.<ext>` where `<TARGET>` is `Sur_Area` (default) or `Yield`. The core set is also copied to
`figures/publication/` under the name in the *Publication* column. Each run additionally writes a dated manifest
(`results/<TARGET>_figure_manifest.csv` and `figures/<TARGET>_figure_manifest.md`) with the exact sample counts.

Sample counts below are for the current surface-area dataset (245 measurements); the notebook recomputes them each run.

| ID | Folder | Section | Title | Reference-paper analogue | n | Validation / data used | Publication |
|---|---|---|---|---|---|---|---|
| F01_scatter_matrix | exploratory | 5. Exploratory data analysis | Scatter matrix | Fig. 6 | 245 | descriptive | Fig01_scatter_matrix |
| F02_measured_grid | exploratory | 5. Exploratory data analysis | Measured grid | — (project) | 245 | descriptive |  |
| F03_measured_curves | exploratory | 5. Exploratory data analysis | Measured response curves | — (project) | 245 | descriptive |  |
| F04_monte_carlo_outliers | exploratory | 6. Outlier analysis | Monte Carlo outlier detection | Fig. 8A | 245 | Monte Carlo cross-prediction, 500 random 75 % subsets | Fig03_outliers |
| F05_boxplots | exploratory | 6. Outlier analysis | Box plots of standardised variables | Fig. 8B | 245 | descriptive |  |
| F06_correlation_heatmap | exploratory | 7. Correlation analysis | Correlation analysis (n = 245; Surface area, all impregnation ratios) | Fig. 7 | 245 | descriptive | Fig02_correlation |
| F07_correlation_IR4_SA_yield | exploratory | 7. Correlation analysis | Surface area and yield at IR = 4 (n = 35) | Fig. 7 | 35 | descriptive |  |
| F08_benchmark_random_split | model_comparison | 9a. Benchmark tables | Benchmark (random split) | Table 2 | 245 | random split, seed 42: train n = 171, validation n = 37, test n = 37 (stratified by IR) |  |
| F09_grouped_benchmark | model_comparison | 9b. Grouped benchmark (leave-one-temperature-out / leave-one-time-out) | Leave-one-temperature/time-out benchmark | — (project) | 245 | leave-one-temperature-out and leave-one-time-out, nested grouped tuning | Fig04_grouped_benchmark |
| F10_r2_train_val_test | model_comparison | 10. Training, validation and testing evaluation | R² heat map train/validation/test | Table 2 | 245 | random split, seed 42: train n = 171, validation n = 37, test n = 37 (stratified by IR) |  |
| F11_residuals_best_model | model_comparison | 10. Training, validation and testing evaluation | Residuals of CNN | — | 245 | random split, seed 42: train n = 171, validation n = 37, test n = 37 (stratified by IR) |  |
| F12_taylor_diagrams | model_comparison | 11. Taylor diagrams | Taylor diagrams | Fig. 9 | 245 | random split, seed 42: train n = 171, validation n = 37, test n = 37 (stratified by IR) | Fig05_taylor |
| F13_actual_vs_predicted_index | prediction | 12. Actual vs predicted plots | Actual vs predicted by index | Fig. 10 | 245 | random split, seed 42: train n = 171, validation n = 37, test n = 37 (stratified by IR) |  |
| F14_cross_plots | prediction | 13. Cross-validation plots | Cross plots: predicted vs actual | Fig. 11 | 245 | random split, seed 42: train n = 171, validation n = 37, test n = 37 (stratified by IR) | Fig06_cross_plots |
| F15_cv_cross_plots_LOTemp | prediction | 13. Cross-validation plots | Leave-one-temperature-out cross-validation | Fig. 11 (grouped CV variant) | 147 | Leave-one-temperature-out, nested grouped tuning |  |
| F15_cv_cross_plots_LOTime | prediction | 13. Cross-validation plots | Leave-one-time-out cross-validation | Fig. 11 (grouped CV variant) | 175 | Leave-one-time-out, nested grouped tuning |  |
| F16_relative_deviation | prediction | 14. Relative deviation plots | Relative deviation | Fig. 12 | 245 | random split, seed 42: train n = 171, validation n = 37, test n = 37 (stratified by IR) |  |
| F17_prediction_distributions | prediction | 15. Prediction distribution histograms | Prediction distributions | Fig. 13 | 245 | random split, seed 42: train n = 171, validation n = 37, test n = 37 (stratified by IR) |  |
| F18_rf_importance_and_shap | explainability | 16. Feature importance and SHAP analysis | Random-Forest importance and mean SHAP | Fig. 14 | 245 | models trained on the training set; SHAP evaluated on all 245 samples | Fig11_importance_shap |
| F19_shap_beeswarm | explainability | 16. Feature importance and SHAP analysis | SHAP summary (beeswarm) | Fig. 14B (extended) | 245 | SHAP on all samples |  |
| F20_shap_dependence | explainability | 16. Feature importance and SHAP analysis | SHAP dependence | — | 245 | SHAP on all samples |  |
| F21_gp_parity_intervals | uncertainty | 17. Gaussian Process uncertainty analysis | GP parity with calibrated 95 % intervals | — (project) | 322 | leave-one-temperature-out and leave-one-time-out | Fig07_gp_parity_intervals |
| F22_interval_calibration | uncertainty | 17. Gaussian Process uncertainty analysis | Prediction-interval calibration | — (project) | 322 | grouped CV; cross-fitted calibration | Fig08_calibration |
| F23_gp_uncertainty_maps | uncertainty | 17. Gaussian Process uncertainty analysis | GP uncertainty maps | — (project) | 245 | final GP on all data; k95 from grouped CV |  |
| F24_response_surfaces | uncertainty | 18. Response surface visualizations | Response surfaces | — (project) | 245 | final GP on all data | Fig09_response_surfaces |
| F25_response_curves | uncertainty | 18. Response surface visualizations | Response curves with intervals | — (project) | 245 | final GP; k95 from grouped CV | Fig10_response_curves |
| F26_response_surface_3d | uncertainty | 18. Response surface visualizations | 3-D response surface | — (project) | 35 | final GP on all data |  |
| F27_surface_model_comparison | uncertainty | 18. Response surface visualizations | Surface comparison GP vs trees | — (project) | 245 | all models refitted on all data |  |
| F28_expected_improvement_maps | experimental_design | 19. Active learning and experimental design | Expected-improvement maps | — (project) | 245 | final GP on all data |  |
| F29_exploration_batch | experimental_design | 19. Active learning and experimental design | Exploration batch | — (project) | 245 | final GP on all data |  |
| F30_active_learning_simulation | experimental_design | 19. Active learning and experimental design | Active-learning simulation | — (project) | 245 | retrospective simulation, 10 repeats | Fig12_active_learning |

## Descriptions

* **F01_scatter_matrix - Scatter matrix.** Pairwise scatter plots with KDE diagonals of the inputs and target.
* **F02_measured_grid - Measured grid.** Annotated heat maps of every measured value (time × temperature) per IR.
* **F03_measured_curves - Measured response curves.** Measured target vs time for each temperature and IR.
* **F04_monte_carlo_outliers - Monte Carlo outlier detection.** Histogram of standardised Monte Carlo cross-prediction errors with ±2 SD bounds; mean vs SD of errors.
* **F05_boxplots - Box plots of standardised variables.** Box plots of z-scored inputs and target.
* **F06_correlation_heatmap - Correlation analysis (n = 245; Surface area, all impregnation ratios).** Pearson and Spearman correlation heat maps with p-values.
* **F07_correlation_IR4_SA_yield - Surface area and yield at IR = 4 (n = 35).** Pearson and Spearman correlation heat maps with p-values.
* **F08_benchmark_random_split - Benchmark (random split).** R² and RMSE of the 15 models on train / validation / test.
* **F09_grouped_benchmark - Leave-one-temperature/time-out benchmark.** R² and RMSE of all models when a whole temperature or time level is unseen.
* **F10_r2_train_val_test - R² heat map train/validation/test.** R² of each model on each set and the train-test gap (over-fitting indicator).
* **F11_residuals_best_model - Residuals of CNN.** Residual vs predicted and residual histograms.
* **F12_taylor_diagrams - Taylor diagrams.** Correlation / normalised SD / centred RMS of each model per set.
* **F13_actual_vs_predicted_index - Actual vs predicted by index.** Measured and predicted values for train, validation and test samples of every model.
* **F14_cross_plots - Cross plots: predicted vs actual.** Predicted vs measured with the y = x bisector.
* **F15_cv_cross_plots_LOTemp - Leave-one-temperature-out cross-validation.** Predicted vs measured with the y = x bisector.
* **F15_cv_cross_plots_LOTime - Leave-one-time-out cross-validation.** Predicted vs measured with the y = x bisector.
* **F16_relative_deviation - Relative deviation.** Absolute relative deviation (%) per sample, per set, per model.
* **F17_prediction_distributions - Prediction distributions.** Histograms of predicted values per set.
* **F18_rf_importance_and_shap - Random-Forest importance and mean SHAP.** A: impurity-based RF importance. B: mean absolute SHAP value of the best model.
* **F19_shap_beeswarm - SHAP summary (beeswarm).** Per-sample SHAP values coloured by feature value.
* **F20_shap_dependence - SHAP dependence.** SHAP value vs feature value, coloured by a second feature.
* **F21_gp_parity_intervals - GP parity with calibrated 95 % intervals.** Out-of-fold GP predictions at unseen temperatures / times with calibrated error bars.
* **F22_interval_calibration - Prediction-interval calibration.** Reliability diagram of raw and cross-fitted calibrated intervals, and distribution of standardised errors.
* **F23_gp_uncertainty_maps - GP uncertainty maps.** Calibrated 95 % interval half-width over temperature × time for each IR.
* **F24_response_surfaces - Response surfaces.** GP mean surface over temperature × time for each IR.
* **F25_response_curves - Response curves with intervals.** GP mean and calibrated band vs temperature and vs time.
* **F26_response_surface_3d - 3-D response surface.** 3-D GP surface with measured points.
* **F27_surface_model_comparison - Surface comparison GP vs trees.** Same-condition surfaces from the GP and tree ensembles (tuned settings, refitted on all data).
* **F28_expected_improvement_maps - Expected-improvement maps.** EI of unmeasured candidate conditions with the top-10 suggestions.
* **F29_exploration_batch - Exploration batch.** Greedy maximum-variance batch of 10 experiments over the uncertainty maps.
* **F30_active_learning_simulation - Active-learning simulation.** Model error and best value found vs number of experiments for random, exploration and EI strategies.

## Reference-paper figures not reproduced

* Figs. 1-5 of Hassan & Kazemi (2025) are schematic diagrams of the workflow and algorithms (no data) and are not recreated.
* The paper's input variables (Abraham descriptors, pore volume, BET area, logCe) do not exist in this dataset; the same
  figure types are produced for this project's inputs (IR, activation temperature, activation time).
