import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def render_ticks(axis, labelsize):
    """
    Style plots for better representation
    :param axis: axes class of plot
    """
    plt.rc('font', weight='bold')
    axis.get_xaxis().set_tick_params(
        which='both', direction='in', width=2.5, labelrotation=0, labelsize=labelsize)
    axis.get_yaxis().set_tick_params(
        which='both', direction='in', width=2.5, labelcolor='k', labelsize=labelsize)
    axis.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.5, b=None, which='both', axis='both')
    # axis.get_xaxis().set_visible(False)
    # axis.get_yaxis().set_visible(False)


fig, axes = plt.subplots(nrows=6, ncols=3, sharex=True, figsize=(12, 12))

for i in range(6):
    for j in range(3):
        render_ticks(axes[i, j], 'x-large')
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticklabels([])

plt.subplots_adjust(hspace=0.05, wspace=0.05)

plt.show()