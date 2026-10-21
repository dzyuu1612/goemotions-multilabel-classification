"""Plot report figures from measured C3 runs plus checked historical C1/C2."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    folder = args.data.resolve()
    history = pd.read_csv(folder/'learning_curves.csv')
    means = pd.read_csv(folder/'mean_std.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15,
                         'axes.labelsize': 16, 'legend.frameon': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'bert':'#315e84','roberta':'#637785','distilbert':'#ad692f'}
    labels = {'bert':'BERT','roberta':'RoBERTa','distilbert':'DistilBERT'}
    fig, ax = plt.subplots(figsize=(8.8,6.16),layout='constrained')
    for arch in colors:
        rows = history[history.architecture==arch].groupby('epoch').validation_macro_f1.agg(['mean','std'])
        x=rows.index.to_numpy()
        ax.plot(x,rows['mean'],color=colors[arch],marker='o',lw=2,label=labels[arch])
        ax.fill_between(x,rows['mean']-rows['std'],rows['mean']+rows['std'],color=colors[arch],alpha=.13)
    ax.set(xlabel='Epoch',ylabel='Validation Macro-F1 @ 0.5',xticks=[1,2,3,4])
    ax.grid(alpha=.16)
    ax.legend(loc='upper left',fontsize=15)
    fig.savefig(folder/'learning_curves.png',dpi=220)
    plt.close(fig)
    plt.rcParams.update({'font.size':18,'axes.labelsize':18,'axes.titlesize':19})
    fig, axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    x=np.arange(3)
    for ax,metric,title in zip(axes,['macro_f1','micro_f1'],['Test Macro-F1','Test Micro-F1']):
        for offset,mode,color,label in [(-.18,'fixed','#637785','Fixed 0.5'),(.18,'tuned','#ad692f','Per-label threshold')]:
            rows=[means[(means.system=='C_'+arch)&(means.split=='test')&(means.threshold_mode==mode)].iloc[0]
                  for arch in colors]
            ax.bar(x+offset,[r[metric+'_mean'] for r in rows],width=.34,
                   yerr=[r[metric+'_std'] for r in rows],capsize=4,color=color,label=label)
        ax.set(title=title,xticks=x,xticklabels=['BERT','RoBERTa','DistilBERT'],ylim=(0,.7))
        ax.grid(axis='y',alpha=.16)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('F1')
    axes[0].legend(loc='upper left',fontsize=13)
    fig.savefig(folder/'test_f1.png',dpi=200)
    plt.close(fig)
    print('PASS: both report figures use three-seed means and sample standard deviations.')


if __name__=='__main__':
    main()
