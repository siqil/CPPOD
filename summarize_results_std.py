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

datasets_title = 'simulated'
methods = ['rand', 'len', 'NH', 'CNH']
datasets = ['pois', 'gam']

outliers = ['commiss','omiss']
curves = [
    {
        'func': plot_roc,
        'name': 'AUROC',
        'x': 'FPR',
        'y': 'TPR',
        'filename': 'roc',
    }
]
ps = ["0.1", "0.05", "sin", "pc"]
matplotlib.rcParams.update({'font.size': 16})
results = []
for dataset in datasets:
    for outlier in outliers:
        folder = f'result/{dataset}/{outlier}'
        for p in ps:
            for method in methods:
                df = pd.read_csv(f'{folder}/{method}_{p}.csv')
                for curve in curves:
                    auc = curve['func'](df['label'], df[f'score_{outlier}'], method, False)
                    results.append(OrderedDict({
                        'dataset': dataset,
                        'outlier': outlier,
                        'method': method,
                        'metric': curve['name'],
                        'p': p,
                        'value': auc,
                    }))

dfr = pd.DataFrame(results)
metric = 'AUROC'
dfr = dfr.loc[dfr['metric'] == metric].drop('metric',axis=1).set_index(['dataset','outlier','p','method'])

if os.path.exists(f'result/bootstrap_{datasets_title}.csv'):
    dfs = pd.read_csv(f'result/bootstrap_{datasets_title}.csv')
else:
    results = []
    for dataset in datasets:
        for outlier in outliers:
            folder = f'result/{dataset}/{outlier}'
            for p in ps:
                for method in methods:
                    dfo = pd.read_csv(f'{folder}/{method}_{p}.csv')
                    n = dfo['seq'].max()
                    print(f'{dataset}_{outlier}_{p}_{method}: {n}')
                    np.random.seed(0)
                    for i in range(20):
                        print(f' {i}', end=".")
                        seq_idx = np.random.randint(n, size=n)
                        dft = []
                        for j in seq_idx:
                            dft.append(dfo.loc[dfo['seq']==j])
                        df = pd.concat(dft)
                        auc = plot_roc(df['label'], df[f'score_{outlier}'], method, False)
                        results.append(OrderedDict({
                            'dataset': dataset,
                            'outlier': outlier,
                            'method': method,
                            'p': p,
                            'rep': i,
                            'value': auc,
                        }))
                    print()
    dfs = pd.DataFrame(results)
    dfs.to_csv(f'result/bootstrap_{datasets_title}.csv')

dfs = dfs.groupby(['dataset','outlier','p','method'])['value'].agg(['std'])

df = dfr.join(dfs)

tb = df.pivot_table(index=['dataset', 'outlier','p'], columns='method')
tb = tb.reindex(index=pd.MultiIndex.from_product([datasets,outliers,ps],names=tb.index.names), columns=pd.MultiIndex.from_product([['value','std'], methods]))

def filter(row):
    rv = row['value']
    rs = row['std']
    max_val = np.max(rv)
    def f(val, s):
        if val == max_val:
            return f'\\textbf{{{val:.3f}}} ($\\pm$ {s:.3f})'
        else:
            return f'{val:.3f} ($\\pm$ {s:.3f})'
    return rv.combine(rs, f)

def compress(values):
    return [f'{index[0].capitalize()[:3] if index[0].islower() else index[0][:3]} ({index[1].capitalize()[0]}) [{index[2]}]' for index in values]

latex_tb = tb.apply(filter, axis=1)
latex_tb.index = compress(latex_tb.index.values)
latex_tb

with open(f'result/tab/{metric}_{datasets_title}_std.tex', 'w') as f:
    latex_tb.to_latex(f, escape=False)
