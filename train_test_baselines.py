import numpy as np
import scipy.stats as stats
import pickle
import pandas as pd
import os
from baselines import *

datasets = ['gam', 'pois']
outliers = ['commiss', 'omiss']
ps = ["0.1", "0.05", "sin", "pc"]

def detect(name, method, test_set, result_path):
    for outlier in outliers:
        for p in ps:
            np.random.seed(0)
            result = method(test_set[outlier][p])
            result.to_csv(f'{result_path}/{outlier}/{name}_{p}.csv')

def detect_with_param(method, param):
    return lambda x: method(x, param)

for dataset in datasets:
    folder = f'data/{dataset}'
    result_path = f'result/{dataset}'
    with open(f'{folder}/train.pkl', 'rb') as f:
        train_set = pickle.load(f)

    test_set = {}
    for outlier in outliers:
        test_set[outlier] = {}
        os.makedirs(f'{result_path}/{outlier}', exist_ok=True)
        for p in ps:
            with open(f'{folder}/test_{outlier}_{p}.pkl', 'rb') as f:
                test_set[outlier][p] = pickle.load(f)

    detect('rand', detect_rand, test_set, result_path)

    param = fit_len(train_set)
    detect('len', detect_with_param(detect_len, param), test_set, result_path)

    K = 2
    if dataset == 'pois':
        param = fit_model_pois(train_set, K)
        detect('model', detect_with_param(detect_model_pois, param), test_set, result_path)
        with open(f'{folder}/param.pkl', 'rb') as f:
            param = pickle.load(f)
        detect('true', detect_with_param(detect_model_pois, param), test_set, result_path)
    elif dataset == 'gam':
        param = fit_model_gam(train_set, K)
        detect('model', detect_with_param(detect_model_gam, param), test_set, result_path)
        with open(f'{folder}/param.pkl', 'rb') as f:
            param = pickle.load(f)
        detect('true', detect_with_param(detect_model_gam, param), test_set, result_path)
    elif test_set.get('lambda_x') is not None:
        detect('true', detect_model_true, test_set, result_path)
