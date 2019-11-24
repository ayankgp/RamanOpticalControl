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


with open('discrimination_GECI_GECI_vib.p', 'rb') as f:
    data = pickle.load(f)

with open('discrimination_ChR2_GECI.p', 'rb') as f:
    data_inv = pickle.load(f)

# data = ndimage.zoom(data, 1)
# data_inv = ndimage.zoom(data_inv, 12)

N_points = 15
x = 1600 - np.linspace(0.985, 1.0, N_points, endpoint=True) * 1600
y = 1 / np.linspace(.1, 1, N_points, endpoint=True)
X, Y = np.meshgrid(x, y)


def plot2d(cmap):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6., 5))
    levels = np.linspace(data.min(), data.max(), 51, endpoint=True)
    levels_line = np.linspace(data.min(), data.max(), 15, endpoint=True)
    im = axes.contour(33.35641/Y, X, data, levels_line, origin='lower', colors='k', linewidths=1., linestyles='dashed', alpha=0.5)
    im = axes.contourf(33.35641/Y, X, data, levels, cmap=cmap, origin='lower')

    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax)

    axes.set_ylabel('$\\omega_{vib}^{GCaMP} - \\omega_{vib}^{ChR2}$ (cm$^{-1}$)', fontsize='x-large', fontweight='bold')
    axes.set_xlabel('Vibrational linewidth $\\tau_{vib}$ (cm$^{-1}$)', fontsize='x-large', fontweight='bold')

    render_ticks(axes, 'x-large')
    axes.grid(color='k', linestyle='-.', linewidth=.1, alpha=1., which='both', axis='both')
    plt.subplots_adjust(left=0.2, bottom=0.2, hspace=0.0, wspace=0.05)
    plt.savefig(str(cmap) + '.png', format='png')


plot2d('RdYlBu')
plt.show()