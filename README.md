# Contextual Point-Process Outlier Detection (CPPOD)

This code accompanies the paper:

**Event Outlier Detection in Continuous Time.** Siqi Liu, and Milos Hauskrecht. *International Conference on Machine Learning,* 2021.

## Dependencies

The code runs on Python 3.7. You may need to install (the latest version of) the following libraries:
- numpy
- scipy
- statsmodels
- pytorch
- sklearn
- pandas
- matplotlib

## Reproducing results

Steps:
- Make directories
    - `data/pois`
    - `data/gam`
    - `result/fig`
    - `result/tab`
- Generate data
    - `python simulate_data_pois.py` (Poisson process)
    - `python simulate_data_gam.py` (Gamma process)
- Run baselines
    - `python train_test_baselines.py`
- Train and test CPPOD and PPOD
    - Run the commands in `train_test_cppod_sim.sh`
- Evaluate the performance
    - ROC curves: `python summarize_results.py`
    - AUROC tables: `python summarize_results_std.py`
    - Bounds: `python verify_bounds.py` (some figures are not used and generated only for convenience)
