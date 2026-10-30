"""Vietnamese number formatting independent of the operating-system locale."""
from matplotlib.ticker import FuncFormatter

SPLIT_COLORS = ['#7030A0', '#D6A419', '#5C677D']
MODEL_COLORS = ['#167D9A', '#D36A25', '#23855B']

def vi(value, digits=4):
    return f'{value:.{digits}f}'.replace('.', ',')

def decimal_axes(axes):
    for ax in axes:
        for axis in (ax.xaxis, ax.yaxis):
            # Preserve categorical and explicitly labelled axes.
            from matplotlib.ticker import FixedFormatter
            if not isinstance(axis.get_major_formatter(), FixedFormatter):
                axis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'.replace('.', ',')))
