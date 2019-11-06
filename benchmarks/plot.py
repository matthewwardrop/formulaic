import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'benchmarks.csv')).sort_values('mean')


def grouped_barplot(df, cat, subcat, val, err, subcats=None, **kwargs):
    # based on https://stackoverflow.com/a/42033734
    categories = df[cat].unique()
    x = np.arange(len(categories))
    subcats = subcats or df[subcat].unique()
    offsets = (np.arange(len(subcats)) - np.arange(len(subcats)).mean()) / (len(subcats) + 1.)
    width = np.diff(offsets).mean()
    for i, gr in enumerate(subcats):
        dfg = df[df[subcat] == gr]
        plt.bar(x + offsets[i], dfg[val].values, width=width,
                label="{}".format(gr), yerr=dfg[err].values, capsize=6, **kwargs)
    plt.xlabel(cat)
    plt.ylabel(val)
    plt.xticks(x, categories)
    plt.legend(title=subcat, loc='center left', bbox_to_anchor=(1, 0.5))


def plot_benchmarks(toolings=None):
    plt.figure(dpi=120, figsize=(10, 5))
    grouped_barplot(data, cat='formula', subcat='tooling', val='mean', err='stderr', subcats=toolings, log=True)
    plt.ylim(1e-2, None)
    plt.grid()
    plt.gca().set_axisbelow(True)
    plt.ylabel("Mean Time (s)")
    plt.xlabel("Formula")
    plt.tight_layout()


plot_benchmarks(toolings=['formulaic', 'R', 'patsy', 'formulaic_sparse', 'R_sparse'])
plt.savefig(os.path.join(os.path.dirname(__file__), 'benchmarks.png'))
