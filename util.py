import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

def is_sorted(l):
    return all([l[i] <= l[i + 1] for i in range(len(l)-1)])

def plot_method(method, x, y, metric=None):
    name = method['name']
    if metric is None:
        label = name
    else:
        label = f'{name} ({metric:.3f})'
    plt.plot(x, y, label=label, linestyle=method['style'])

def plot_roc(label, score, method, plot=True):
    fpr, tpr, _ = metrics.roc_curve(label, score)
    auc = metrics.roc_auc_score(label, score)
    if plot:
        plot_method(method, fpr, tpr)
    return auc

def plot_pr(label, score, method, plot=True):
    p, r, _ = metrics.precision_recall_curve(label, score)
    auc = metrics.average_precision_score(label, score)
    if plot:
        plot_method(method, r, p)
    return auc

def compute_outlier_ratio(test_set):
    n = len(test_set)
    total = np.zeros(n)
    ones = np.zeros(n)
    for i, seq in enumerate(test_set):
        total[i] = len(seq["label_test"])
        ones[i] = np.sum(seq["label_test"])
    return np.sum(ones) / np.sum(total)


