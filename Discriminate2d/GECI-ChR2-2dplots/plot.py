import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage


def render_ticks(axis, labelsize):
    axis.get_xaxis().set_tick_params(direction='in', width=2, labelsize=labelsize, top='on', bottom='on')
    axis.get_yaxis().set_tick_params(direction='in', width=2, labelsize=labelsize, right='on', left='on')


with open('discrimination_GECI_ChR2.p', 'rb') as f:
    data = pickle.load(f)

with open('discrimination_ChR2_GECI.p', 'rb') as f:
    data_inv = pickle.load(f)

data = ndimage.zoom(data, 1)
data_inv = ndimage.zoom(data_inv, 12)

x = y = np.linspace(0.985, 1.0151, 12)*1600
X, Y = np.meshgrid(x, y)


def plot2d(cmap):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(9., 5))
    levels = np.linspace(0., 8., 51, endpoint=True)
    levels_line = np.linspace(0., 8., 11, endpoint=True)
    im = axes[0].contour(X, Y, data, levels_line, origin='lower', colors='k', linewidths=1., linestyles='dashed', alpha=0.5)
    im = axes[0].contourf(X, Y, data, levels, cmap=cmap, origin='lower')
    # im = axes[1].contour(X, Y, data_inv, levels_line, origin='lower', colors='k', linewidths=1., linestyles='dashed', alpha=0.5)
    # im = axes[1].contourf(X, Y, data_inv, levels, cmap=cmap, origin='lower')
    # axes.add_patch(Circle((1580, 1620), .25, facecolor='k', edgecolor='k'))
    # axes.add_patch(Circle((1580, 1580), .25, facecolor='w', edgecolor='w'))
    # axes.add_patch(Circle((1620, 1620), .25, facecolor='w', edgecolor='w'))

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='7%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax)

    axes[0].set_xlabel('Highest ground $\\omega_v$ (cm$^{-1}$) \n for ChR2', fontsize='x-large', fontweight='bold')
    axes[1].set_xlabel('Highest ground $\\omega_v$ (cm$^{-1}$) \n for ChR2', fontsize='x-large', fontweight='bold')
    axes[0].set_ylabel('Highest ground $\\omega_v$ (cm$^{-1}$) \n for GECI', fontsize='x-large', fontweight='bold')

    render_ticks(axes[0], 'x-large')
    render_ticks(axes[1], 'x-large')
    axes[0].grid(color='k', linestyle='-.', linewidth=.1, alpha=1., which='both', axis='both')
    axes[1].grid(color='k', linestyle='-.', linewidth=.1, alpha=1., which='both', axis='both')
    plt.subplots_adjust(left=0.14, bottom=0.22, hspace=0.0, wspace=0.05)
    plt.savefig(str(cmap) + '.png', format='png')


for cmap in [
            'viridis', 'plasma', 'inferno', 'magma',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
            'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']:
    plot2d('gist_rainbow')