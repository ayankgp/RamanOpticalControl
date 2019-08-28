import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def render_ticks(axis, labelsize):
    axis.get_xaxis().set_tick_params(direction='in', width=2, labelsize=labelsize, top='on', bottom='on')
    axis.get_yaxis().set_tick_params(direction='in', width=2, labelsize=labelsize, right='on', left='on')


with open('discrimination_GECI_GECI_vib2.p', 'rb') as f:
    data = pickle.load(f)

with open('discrimination_GEVI_GECI.p', 'rb') as f:
    data_inv = pickle.load(f)


data = ndimage.zoom(data, 10) * 18.6 / 8.8
data_inv = ndimage.zoom(data_inv, 10) * 18.6 / 8.8

x = np.linspace(0.985, 1.015, 120)*1600 - 1600
y = np.linspace(10**1.31, 10**(-1), 120, endpoint=False)
print(y)
X, Y = np.meshgrid(y, x)


def plot2d(cmap):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5.5, 5))
    levels = np.linspace(0., 22., 101, endpoint=True)
    levels_line = np.linspace(0., 22., 12, endpoint=True)
    im = axes.contour(33.35*X, Y, data, levels_line, origin='lower', colors='k', linewidths=1., linestyles='dashed', alpha=1)
    im = axes.contourf(33.35*X, Y, data, levels, cmap=cmap, origin='lower')
    # im = axes[1].contour(X, Y, data_inv, levels_line, origin='lower', colors='k', linewidths=1., linestyles='dashed', alpha=0.5)
    # im = axes[1].contourf(X, Y, data_inv, levels, cmap=cmap, origin='lower')
    # axes.add_patch(Circle((1580, 1620), .25, facecolor='k', edgecolor='k'))
    # axes.add_patch(Circle((1580, 1580), .25, facecolor='w', edgecolor='w'))
    # axes.add_patch(Circle((1620, 1620), .25, facecolor='w', edgecolor='w'))

    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='4%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax)

    axes.set_xlabel('$\\tau_{vib}$ (cm$^{-1}$)', fontsize='large', fontweight='bold')
    axes.set_ylabel('$\\omega_v$ (cm$^{-1}$) difference', fontsize='large', fontweight='bold')

    render_ticks(axes, 'x-large')
    render_ticks(axes, 'x-large')
    axes.grid(color='k', linestyle='-.', linewidth=.1, alpha=1., which='both', axis='both')
    axes.grid(color='k', linestyle='-.', linewidth=.1, alpha=1., which='both', axis='both')
    plt.subplots_adjust(left=0.14, bottom=0.22, hspace=0.0, wspace=0.05)
    plt.savefig('GCaMP_ASAP_2d.png', format='png')


def plot3d(cmap):
    fig = plt.figure(figsize=(5., 5))
    axes = fig.add_subplot(1, 1, 1, projection='3d')

    cset = axes.contourf(10**X, Y, data, zdir='z', offset=-1, cmap=cmap, alpha=0.5)
    cset = axes.contour(10**X, Y, data, zdir='z', offset=-1, colors='k', alpha=0.5)
    cset = axes.contourf(10**X, Y, data, zdir='x', offset=1.5, cmap=cmap, alpha=0.5)
    cset = axes.contour(10**X, Y, data, zdir='x', offset=1.5, colors='k', alpha=0.5)
    cset = axes.contourf(10**X, Y, data, zdir='y', offset=-40, cmap=cmap, alpha=0.5)
    cset = axes.contour(10**X, Y, data, zdir='y', offset=-40, colors='k', alpha=0.5)

    axes.plot_surface(10**X, Y, data, rstride=3, cstride=3, cmap=cmap, linewidth=1., alpha=1, antialiased=False)

    axes.plot_wireframe(10**X, Y, data, rstride=10, cstride=10, alpha=0.5, color='k')
    # axes.set_xlim(-1, 1.5)
    axes.set_ylim(-40, 30)
    axes.set_zlim(-1., 22.)
    axes.view_init(30, 127.5)
    axes.tick_params(axis='x', direction='out', length=1, width=2, colors='r')
    axes.tick_params(axis='y', direction='out', length=1, width=2, colors='b')
    axes.set_xlabel('$\\tau_{vib}$ (in ps)', fontsize='medium', fontweight='bold')
    axes.set_ylabel('$\\omega_v$ (cm$^{-1}$) difference', fontsize='medium', fontweight='bold')
    axes.set_zlabel('Discrimination', fontsize='medium', fontweight='bold')
    axes.xaxis.labelpad = 1
    axes.yaxis.labelpad = 2
    axes.zaxis.labelpad = 1
    # axes.set_xticklabels([31, 10, 3.1, 1, 0.31, 0.1][::-1], rotation=0, horizontalalignment='center', va='bottom')
    axes.set_yticklabels([30, 20, 10, 0, -10, -20, -30, -40], rotation=0, horizontalalignment='center', va='bottom')

    # axes = fig.add_subplot(1, 2, 2, projection='3d')
    #
    # cset = axes.contourf(X, Y, data_inv, zdir='z', offset=-1, cmap=cmap, alpha=0.25)
    # cset = axes.contour(X, Y, data_inv, zdir='z', offset=-1, colors='k', alpha=0.5)
    # cset = axes.contourf(X, Y, data_inv, zdir='x', offset=1640, cmap=cmap, alpha=0.25)
    # cset = axes.contour(X, Y, data_inv, zdir='x', offset=1640, colors='k', alpha=0.25)
    # cset = axes.contourf(X, Y, data_inv, zdir='y', offset=1640, cmap=cmap, alpha=0.25)
    # cset = axes.contour(X, Y, data_inv, zdir='y', offset=1640, colors='k', alpha=0.25)
    #
    # axes.plot_surface(X, Y, data_inv, rstride=3, cstride=3, cmap=cmap,
    #                   linewidth=1.5, alpha=1, antialiased=False)
    #
    # axes.plot_wireframe(X, Y, data_inv, rstride=10, cstride=10, alpha=0.5, color='k')
    # axes.set_xlim(1570, 1640)
    # axes.set_ylim(1570, 1640)
    # axes.set_zlim(-1., 5.5)
    #
    # axes.view_init(45, -135)
    # axes.tick_params(axis='x', direction='out', length=6, width=2, colors='r')
    # axes.tick_params(axis='y', direction='out', length=6, width=2, colors='b')
    # axes.set_xlabel('Highest ground $\\omega_v$ (cm$^{-1}$) \n for ChR2', fontsize='medium', fontweight='bold')
    # axes.set_ylabel('Highest ground $\\omega_v$ (cm$^{-1}$) \n for GECI', fontsize='medium', fontweight='bold')
    # axes.set_zlabel('Discrimination Ratio', fontsize='medium', fontweight='bold')
    # axes.xaxis.labelpad = 10
    # axes.yaxis.labelpad = 10
    # axes.zaxis.labelpad = 1
    #
    plt.subplots_adjust(left=0.01, right=0.95, bottom=0.1, top=0.99, wspace=0.05)

    plt.axis('on')
    plt.savefig('plot3d.png', format='png')


for cmap in [
            # 'viridis', 'plasma', 'inferno', 'magma',
            # 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            # 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            # 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
            # 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
            # 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            # 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            # 'Pastel1', 'Pastel2', 'Paired', 'Accent',
            # 'Dark2', 'Set1', 'Set2', 'Set3',
            # 'tab10', 'tab20', 'tab20b', 'tab20c',
            # 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            # 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            # 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'

            'Spectral']:
    plot2d(cmap)
    # plot3d(cmap)

plt.show()
