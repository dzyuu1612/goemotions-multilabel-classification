"""Figures computed from the same frozen-score tables as Final v7.1.

The functions return Matplotlib figures. They do not train, select thresholds,
or replace saved metrics. Notebooks show each figure directly after its table.
"""
import numpy as np
import matplotlib.pyplot as plt

MODELS = ['C1 BERT', 'C2 RoBERTa', 'C3 DistilBERT']
MODES = ['fixed', 'global', 'tuned']
MODE_NAMES = ['0,5', 'Chung', 'Riêng']
COLORS = ['#99938B', '#2C7C9B', '#B14E34']


def vi(value, digits=4):
    return f'{float(value):.{digits}f}'.replace('.', ',')


def threshold_modes(stats):
    names = ['A TF-IDF balanced', 'B BART-MNLI', *MODELS]
    fig, ax = plt.subplots(figsize=(11, 5.6), layout='constrained')
    for i, (mode, label, color) in enumerate(zip(MODES, MODE_NAMES, COLORS)):
        data = stats[stats['mode'] == mode].set_index('model').loc[names]
        bars = ax.barh(np.arange(5) + (i-1)*.24, data.macro_f1_mean,
                       height=.22, label=label, color=color)
        ax.bar_label(bars, labels=[vi(x) for x in data.macro_f1_mean], padding=4, fontsize=10)
    ax.set_yticks(np.arange(5), names)
    ax.invert_yaxis()
    ax.set(xlim=(0, .62), xlabel='Macro-F1 test')
    ax.legend(ncol=3, title='Ngưỡng đã khóa trên validation', loc='lower left',
              bbox_to_anchor=(0, 1.02), frameon=False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    return fig


def exclusive_errors(taxonomy, seed=2026):
    categories = ['dung_hoan_toan', 'doan_thua', 'dung_mot_phan', 'doan_sai_nhan', 'khong_doan_gi']
    labels = ['Đúng hoàn toàn', 'Đoán thừa', 'Đúng một phần', 'Đoán sai nhãn', 'Không đoán gì']
    colors = ['#3F7D68', '#DAB263', '#7EA8B5', '#BA6654', '#B7B2AC']
    runs = [(model, mode) for model in MODELS for mode in ['fixed', 'tuned']]
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    left = np.zeros(len(runs))
    for category, label, color in zip(categories, labels, colors):
        values = []
        for model, mode in runs:
            row = taxonomy[(taxonomy.model == model) & (taxonomy.seed == seed) &
                           (taxonomy['mode'] == mode) & (taxonomy.category == category)]
            assert len(row) == 1
            values.append(int(row.iloc[0].n))
        ax.barh(np.arange(6), values, left=left, label=label, color=color)
        for y, value, start in zip(range(6), values, left):
            if value > 350:
                ax.text(start+value/2, y, str(value), ha='center', va='center', fontsize=10)
        left += values
    assert np.array_equal(left, np.full(6, 5427))
    ax.set_yticks(np.arange(6), [f'{m} · {"0,5" if mode == "fixed" else "riêng"}' for m, mode in runs])
    ax.invert_yaxis()
    ax.set_xlabel(f'Số câu test · seed {seed} · mỗi hàng = 5.427')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, -.16), fontsize=10)
    return fig


def near_miss(errors, seed=2026):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    definitions = [('FN', 'share_ratio_ge_0_8', 'FN sát ngưỡng: điểm ≥ 0,8×ngưỡng'),
                   ('FP', 'share_margin_le_0_1', 'FP sát ngưỡng: phần vượt ≤ 0,1')]
    for ax, (kind, key, title) in zip(axes, definitions):
        for i, mode in enumerate(['fixed', 'tuned']):
            values = [100*errors[(errors.model == model) & (errors.seed == seed) &
                                (errors['mode'] == mode) & (errors.type == kind)].iloc[0][key]
                      for model in MODELS]
            bars = ax.bar(np.arange(3)+(i-.5)*.34, values, width=.32,
                          color=COLORS[0 if i == 0 else 2], label='0,5' if i == 0 else 'Riêng')
            ax.bar_label(bars, labels=[vi(x, 1)+'%' for x in values], padding=3, fontsize=10)
        ax.set_xticks(np.arange(3), MODELS, rotation=12)
        ax.set(ylim=(0, 40), title=title, ylabel='% trên tổng lỗi cùng loại')
        ax.legend()
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    return fig


def model_f1(actual):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    names = ['bert', 'roberta', 'distilbert']
    for ax, metric in zip(axes, ['macro_f1', 'micro_f1']):
        for i, mode in enumerate(['fixed', 'tuned']):
            grouped = actual[actual['mode'] == mode].groupby('model')[metric]
            means, stds = grouped.mean().loc[names], grouped.std(ddof=1).loc[names]
            ax.bar(np.arange(3)+(i-.5)*.34, means, yerr=stds, width=.32,
                   capsize=3, color=COLORS[0 if i == 0 else 1], label='0,5' if i == 0 else 'Riêng')
        ax.set_xticks(np.arange(3), ['BERT', 'RoBERTa', 'DistilBERT'])
        ax.set(ylim=(0, .7), ylabel=metric.replace('_', '-'))
        ax.legend()
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    return fig


def learning_curves(history, model):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout='constrained')
    for col, seed in enumerate([42, 123, 2026]):
        data = history[(history.model == model) & (history.seed == seed)].sort_values('epoch')
        assert data.epoch.tolist() == list(range(1, 9))
        best = data[data.selected == 1]
        assert len(best) == 1
        for row, metric in enumerate(['macro_f1', 'bce']):
            ax = axes[row, col]
            for split, color in [('train', '#2C7C9B'), ('val', '#B14E34')]:
                ax.plot(data.epoch, data[f'{split}_{metric}'], marker='o', markersize=3,
                        color=color, label='Train (eval)' if split == 'train' else 'Validation')
            ax.axvline(int(best.iloc[0].epoch), linestyle='--', color='#333333', label='Epoch chọn')
            ax.set(xticks=range(1, 9), xlabel='Epoch', ylabel='Macro-F1 @0,5' if row == 0 else 'BCE')
            ax.grid(alpha=.15)
            if row == 0:
                ax.set_title(f'{model} · seed {seed}')
                ax.legend(fontsize=8)
    return fig
