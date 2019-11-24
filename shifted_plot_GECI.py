import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d
from scipy import arange, array, exp
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.cm as cm
import matplotlib


class SymHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        xx = 0.6*height
        return super(SymHandler, self).create_artists(legend, orig_handle, xdescent, xx, width, height, fontsize, trans)


def render_ticks(axis, labelsize):
    """
    Style plots for better representation
    :param axis: axes class of plot
    """
    axis.get_xaxis().set_tick_params(
        which='both', direction='in', width=1, labelrotation=0, labelsize=labelsize)
    axis.get_yaxis().set_tick_params(
        which='both', direction='in', width=1, labelcolor='k', labelsize=labelsize)


with open('GECI_shift.p', 'rb') as f:
    data = pickle.load(f)

with open('GECI_orig.p', 'rb') as f:
    data_orig = pickle.load(f)

freq_GCaMP = data_orig['orig_freq_GECI']
spectra_GCaMP = data_orig['orig_GECI']

freq_ChR2 = data_orig['orig_freq_ChR2']
spectra_ChR2 = data_orig['orig_ChR2']

exp_spectra_GCaMP = data_orig['exp_spectra_GECI']
exp_spectra_ChR2 = data_orig['exp_spectra_ChR2']

freq_shift = data['shift_freq']
GCaMP_shift = data['shift_GECI']


def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(list(map(pointwise, array(xs))))

    return ufunclike


new_freq = np.linspace(360, 623, 623 - 380 + 1)
f_interp_orig_GCaMP = interp1d(freq_GCaMP, spectra_GCaMP)
f_interp_orig_ChR2 = interp1d(freq_ChR2, spectra_ChR2)
f_interp_shift = interp1d(freq_shift, GCaMP_shift)
f_extrap_orig_GCaMP = extrap1d(f_interp_orig_GCaMP)
f_extrap_orig_ChR2 = extrap1d(f_interp_orig_ChR2)
f_extrap_shift = extrap1d(f_interp_shift)

colors = cm.hot(np.linspace(0.0, 1.0, 10))
set = [1, 2]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9.5, 7))
# axes.plot(new_freq, f_extrap_orig_GCaMP(new_freq), 'r--', label='Fitted Spectra GCaMP')
# axes.plot(new_freq, f_extrap_orig_ChR2(new_freq), 'b--', label='Fitted Spectra ChR2')
axes.plot(new_freq, f_extrap_shift(new_freq), 'k', linewidth=2.0, label='Shifted Spectra')
axes.plot(freq_GCaMP, exp_spectra_GCaMP, 'r', label='Experimental Spectra GCaMP')
axes.plot(freq_ChR2, exp_spectra_ChR2, 'b', label='Experimental Spectra ChR2')
for i in range(1, 10):
    if i in set:
        axes.plot(new_freq, f_extrap_orig_GCaMP(new_freq) * (1 - 0.1 * i) + f_extrap_shift(new_freq) * 0.1 * i, color=['g', 'y'][i-1], label='Mixture ' + str(i*10) + '-' + str(100 - i*10))
axes.legend(handler_map={matplotlib.lines.Line2D: SymHandler()}, loc=9, fontsize='small', prop={'weight': 'bold'}, ncol=2, handleheight=2.4, labelspacing=0.05)
axes.set_xlim(378, 655)
axes.set_ylim(0, 125)
left, bottom, width, height = [0.685, 0.4, 0.2, 0.2]
axes_inset = fig.add_axes([left, bottom, width, height])

axes_inset.semilogy(new_freq, f_extrap_orig_GCaMP(new_freq), 'r', label='Fitted Spectra GCaMP')
axes_inset.semilogy(new_freq, f_extrap_orig_ChR2(new_freq), 'b', label='Fitted Spectra ChR2')
for i in range(1, 10):
    if i in set:
        axes_inset.semilogy(new_freq, f_extrap_orig_GCaMP(new_freq) * (1 - 0.1 * i) + f_extrap_shift(new_freq) * 0.1 * i, color=['g', 'y'][i-1], label='Mixture ' + str(i*10) + '-' + str(100 - i*10))
axes_inset.set_xlim(515, 575)
axes_inset.set_ylim(.5, 100)

render_ticks(axes, 'x-large')
render_ticks(axes_inset, 'x-large')


axes.set_ylabel('Normalised absorption', fontweight='bold', fontsize='large')
axes.set_xlabel('Wavelength (in nm)', fontweight='bold', fontsize='large')

plt.savefig('FinalPaperPlots/GCaMP_shift' + '.png', format='png')
plt.savefig('FinalPaperPlots/GCaMP_shift' + '.eps', format='eps')
plt.show()
