# Contextual Point-Process Outlier Detection (CPPOD)

This code accompanies the paper:

**Event Outlier Detection in Continuous Time.** Siqi Liu, and Milos Hauskrecht. *International Conference on Machine Learning,* 2021.

## Dependencies

The code runs on Python 3.6 or 3.7. It relies on the following libraries and has been tested on the versions shown, although newer versions may still work:
- python=3.6.10
- pytorch=1.7.0
- numpy=1.18.5
- scipy=1.5.4
- statsmodels=0.11.1
- scikit-learn=0.19.2
- pandas=0.23.4
- matplotlib=2.2.3

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
