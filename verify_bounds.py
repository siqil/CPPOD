import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
from collections import OrderedDict
from simulation import *
from baselines import *
from scipy.interpolate import interp1d


def fpr(label, score):
    fpr, tpr, ts = metrics.roc_curve(label, score)
    return ts, fpr

def fdr(label, score):
    p, r, ts = metrics.precision_recall_curve(label, score)
    return ts, 1-p[:-1]

def avg_curve(x, y, n=100):
    def avg(x_g, x, y):
        y_g = []
        for i in range(len(x)):
            f = interp1d(x[i], y[i])
            y_g.append(f(x_g))
        m = np.mean(y_g, axis=0)
        s = np.std(y_g, axis=0)
        return m, s
    left = -np.inf
    right = np.inf
    for i in range(len(x)):
        left = np.maximum(left, np.min(x[i]))
        right = np.minimum(right, np.max(x[i]))
    x_g = np.linspace(left, right, n)
    y_g, y_g_s = avg(x_g, x, y)
    return x_g, y_g, y_g_s

datasets = ['pois','gam']
outliers = ['commiss','omiss']
curves = [
    {
        'func': fpr,
        'name': 'FPR',
        'bound': {
            'commiss': lambda th, p_1: -th * 0.01,
            'omiss': lambda th, p_1: np.exp(-th),
        },
        'x': 'threshold',
        'y': 'FPR',
        'filename': 'fpr',
    },
    {
        'func': fdr,
        'name': 'FDR',
        'bound': {
            'commiss': lambda th, p_1: (-th)/((-th)+p_1),
            'omiss': lambda th, p_1: np.exp(-p_1*th),
        },
        'x': 'threshold',
        'y': 'FDR',
        'filename': 'fdr',
    }
]
matplotlib.rcParams.update({'font.size': 16})
method = 'true'
n_rep = 10
n_test = 20
for dataset in datasets:
    np.random.seed(0)
    folder = f'data/{dataset}'
    with open(f'{folder}/param.pkl', 'rb') as f:
        param = pickle.load(f)
    with open(f'{folder}/train.pkl', 'rb') as f:
        data_train = pickle.load(f)
    with open(f'{folder}/test.pkl', 'rb') as f:
        data_test = pickle.load(f)
    if dataset == 'pois':
        detect = lambda x: detect_model_pois(x, param)
        t_max = 1000
        dt = 0.01
        q = np.array([
            [-0.05, 0.05],
            [0.05, -0.05]
        ])
        sim = PoisMJPSim(q=q, param=param)
    elif dataset == 'gam':
        detect = lambda x: detect_model_gam(x, param)
        t_max = 1000
        dt = 0.01
        q = np.array([
            [-0.05, 0.05],
            [0.05, -0.05]
        ])
        sim = GamMJPSim(q=q, param=param)
    else:
        raise Exception(f'Unexpected dataset {dataset}')
    w = np.inf
    rate = compute_empirical_rate(data_test)
    data_test = []
    p = 0.1
    for i in range(n_rep):
        data_test.append([])
        for j in range(n_test):
            data_test[i].append(sim.sim(t_max, dt))
    for outlier in outliers:
        if outlier == 'omiss':
            outlier_sim = OmissSim(w, p)
            p_1 = p
        elif outlier == 'commiss':
            outlier_sim = CommissSim(p * rate, 1)
            p_1 = p * rate
        else:
            raise Exception(f'Unexpected outlier {outlier}')
        x = {}
        y = {}
        bound = {}
        for name in ['FPR','FDR']:
            x[name] = [None]*n_rep
            y[name] = [None]*n_rep
            bound[name] = [None]*n_rep
        for i in range(n_rep):
            data = copy.deepcopy(data_test[i])
            for j in range(len(data)):
                data[j] = outlier_sim.sim(data[j])
            df = detect(data)
            label = df['label']
            score = df[f'score_{outlier}']
            for curve in curves:
                name = curve['name']
                x[name][i], y[name][i] = curve['func'](label, score,)
        for curve in curves:
            fig = plt.figure(figsize=(4.8,3.6))
            name = curve['name']
            bound = curve['bound'][outlier]
            ax, ay, sy = avg_curve(x[name], y[name], n=100)
            az = bound(ax, p_1)
            plt.errorbar(ax, ay, yerr=sy, label='empirical', capsize=4)
            plt.plot(ax, az, label='bound')
            plt.xlabel(curve['x'])
            plt.ylabel(curve['y'])
            # plt.title(f'{dataset}_{outlier}_{name}')
            plt.legend()
            # plt.show()
            filename = curve['filename']
            fig.savefig(f'result/fig/{filename}_{dataset}_{outlier}.pdf', bbox_inches='tight')

