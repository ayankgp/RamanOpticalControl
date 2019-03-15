import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib import cm
from scipy import ndimage


def get_experimental_spectra(mol):
    """
    Calculates interpolated linear spectra data from spectra file.
    :param mol: Spectra file of molecule mol.
    :return: Wavelength (bandwidth for specific molecule),
             Interpolated linear spectra
    """

    data = pd.read_csv(mol, sep=',')
    wavelength = data.values[:, 0]

    absorption = data.values[:, 1]

    func = interp1d(wavelength, absorption, kind='quadratic')
    wavelength_new = 1. / np.linspace(1. / wavelength.max(), 1. / wavelength.min(), 100)
    absorption_new = func(wavelength_new)
    absorption_new *= 100. / absorption_new.max()

    return wavelength_new, absorption_new


def render_ticks(axis, labelsize):
    """
    Style plots for better representation
    :param axis: axes class of plot
    """
    axis.get_xaxis().set_tick_params(
        which='both', direction='in', width=1.5, labelrotation=0, labelsize=labelsize)
    axis.get_yaxis().set_tick_params(
        which='both', direction='in', width=1.5, labelcolor='k', labelsize=labelsize)
    # axis.grid(color='r', linestyle='-.', linewidth=0.5, alpha=0.3, which='both', axis='both')


wl_GECI, abs_GECI = get_experimental_spectra('Data/GCaMP.csv')
wl_ChR2, abs_ChR2 = get_experimental_spectra("Data/ChR2_2.csv")
wl_GEVI, abs_GEVI = get_experimental_spectra("Data/EGFP.csv")
wl_BLUF, abs_BLUF = get_experimental_spectra("Data/BLUF.csv")
wl_LOV, abs_LOV = get_experimental_spectra("Data/LOV.csv")
wl_C1V1, abs_C1V1 = get_experimental_spectra("Data/C1V1.csv")
wl_Chronos, abs_Chronos = get_experimental_spectra("Data/Chronos.csv")
wl_TsChR, abs_TsChR = get_experimental_spectra("Data/TsChR.csv")

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

axes.plot(wl_BLUF, savgol_filter(abs_BLUF, 3, 1), 'b--', linewidth=2., label='BLUF')
axes.plot(wl_ChR2, savgol_filter(abs_ChR2, 11, 2), 'b', linewidth=2., label='ChR2')
axes.plot(wl_GEVI, savgol_filter(abs_GEVI, 3, 1), 'g', linewidth=2., label='ASAP')
axes.plot(wl_GECI, savgol_filter(abs_GECI, 3, 1), 'g--', linewidth=2., label='GCaMP')

# axes.plot(wavelength_TsChR, savgol_filter(absorption_TsChR, 21, 2), linewidth=1., label='TsChR')
# axes.plot(wavelength_C1V1, savgol_filter(absorption_C1V1, 5, 2), linewidth=1., label='C1V1')
# axes.plot(wavelength_Chronos, savgol_filter(absorption_Chronos, 5, 2), linewidth=1., label='Chronos')
# axes.plot(wl_LOV, abs_LOV, linewidth=1.5, label='LOV')

render_ticks(axes, 'large')
render_ticks(axes, 'large')

axes.set_xlim(375, 560)
axes.set_ylim(0, 105)
axes.set_ylabel('Normalised absorption', fontsize='x-large', fontweight='bold')
axes.set_xlabel('Wavelength (in nm)', fontsize='x-large', fontweight='bold')

plt.rc('font', weight='bold')
axes.legend(loc=1, fontsize='x-large')
axes.legend(loc=1, fontsize='x-large')
plt.savefig('SpectralOverlap.eps', format="eps")


# fig, axes = plt.subplots(nrows=1, ncols=1)
# t = np.linspace(-1, 1, 1000)
#
# axes.plot(t, np.cos(np.pi * t / abs(2*t[0]))**2 * (np.cos(270*t) + np.cos(300*t)))
# axes.axis('off')
# # img = plt.imread('blue.png')
# # axes.imshow(img)
# plt.axis('off')
# plt.savefig('field.png', format='png', transparent=True)
plt.show()