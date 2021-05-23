import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
from collections import OrderedDict
from util import plot_roc

method_dict = {
    'true': {
        'name':'GT',
        'style':(0, (3, 1, 1, 1, 1, 1)),
    },
    'rand': {
        'name':'RND',
        'style':'dotted',
    },
    'len': {
        'name':'LEN',
        'style':'dashdot',
    },
    'NH': {
        'name':'PPOD',
        'style':'dashed',
    },
    'CNH': {
        'name':'CPPOD',
        'style':'solid',
    },
}
methods = ['rand', 'len', 'NH', 'CNH', 'true']
datasets = ['pois', 'gam']
outliers = ['commiss', 'omiss']
curves = [
    {
        'func': plot_roc,
        'name': 'AUROC',
        'x': 'FPR',
        'y': 'TPR',
        'filename': 'roc',
    },
]
plot = True
matplotlib.rcParams.update({'font.size': 6})

results = []
for dataset in datasets:
    for outlier in outliers:
        folder = f'result/{dataset}/{outlier}'
        for curve in curves:
            if plot:
                fig = plt.figure(figsize=(2.4,1.8))
            for method in methods:
                df = pd.read_csv(f'{folder}/{method}_0.1.csv')
                auc = curve['func'](df['label'], df[f'score_{outlier}'], method_dict[method], plot)
                results.append(OrderedDict({
                    'dataset': dataset,
                    'outlier': outlier,
                    'method': method_dict[method]['name'],
                    'metric': curve['name'],
                    'value': auc,
                }))
            if plot:
                    plt.xlabel(curve['x'])
                    plt.ylabel(curve['y'])
                    plt.legend()
                    filename = curve['filename']
                    fig.savefig(f'result/fig/{filename}_{dataset}_{outlier}.pdf', bbox_inches='tight')
