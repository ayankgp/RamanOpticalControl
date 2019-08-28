#!/usr/bin/env python

"""
Raman.py:

Class containing C calls for spectra calculation and Raman control.
Plots results obtained from C calls.
"""

__author__ = "Ayan Chattopadhyay"
__affiliation__ = "Princeton University"


# ---------------------------------------------------------------------------- #
#                           LOADING LIBRARY HEADERS                            #
# ---------------------------------------------------------------------------- #

import numpy as np
from types import MethodType, FunctionType
from wrapper import *
import pandas as pd
from scipy.interpolate import interp1d
from multiprocessing import cpu_count
from ctypes import c_int, c_double, c_char_p, POINTER, Structure
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.lines
import matplotlib.pyplot as plt


class SymHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        xx = 0.6*height
        return super(SymHandler, self).create_artists(legend, orig_handle, xdescent, xx, width, height, fontsize, trans)


class ADict(dict):
    """
    Dictionary where you can access keys as attributes
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class RamanOpticalControl:
    """
    Main class initializing molecule and spectra calculation, Raman control
    optimization routines on the molecule.
    """

    def __init__(self, params, **kwargs):
        """
        __init__ function call to initialize variables from the
        parameters for the class instance provided in __main__ and
        add new variables for use in other functions in this class.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.time_A = np.linspace(-params.timeAMP_A, params.timeAMP_A, params.timeDIM_A)
        self.time_R = np.linspace(-params.timeAMP_R, params.timeAMP_R, params.timeDIM_R)

        self.field_A = np.zeros(params.timeDIM_A, dtype=np.complex)
        self.field_R = np.zeros(params.timeDIM_R, dtype=np.complex)

        self.matrix_gamma_pd = np.ascontiguousarray(self.matrix_gamma_pd)
        self.matrix_gamma_dep_GECI = np.ascontiguousarray(self.matrix_gamma_dep)
        self.matrix_gamma_dep_ChR2 = np.ascontiguousarray(self.matrix_gamma_dep)
        self.matrix_gamma_dep_GEVI = np.ascontiguousarray(self.matrix_gamma_dep)
        self.matrix_gamma_dep_BLUF = np.ascontiguousarray(self.matrix_gamma_dep)

        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(params.rho_0)
        self.rho_GECI = np.ascontiguousarray(params.rho_0.copy())
        self.rho_ChR2 = np.ascontiguousarray(params.rho_0.copy())
        self.rho_GEVI = np.ascontiguousarray(params.rho_0.copy())
        self.rho_BLUF = np.ascontiguousarray(params.rho_0.copy())
        self.energies_GECI = np.ascontiguousarray(self.energies_GECI)
        self.energies_ChR2 = np.ascontiguousarray(self.energies_ChR2)
        self.energies_GEVI = np.ascontiguousarray(self.energies_GEVI)
        self.energies_BLUF = np.ascontiguousarray(self.energies_BLUF)

        self.N = len(self.energies_GECI)

        self.abs_spectra_GECI = np.ascontiguousarray(np.zeros(len(self.frequency_A_GECI)))
        self.abs_spectra_ChR2 = np.ascontiguousarray(np.zeros(len(self.frequency_A_ChR2)))
        self.abs_spectra_GEVI = np.ascontiguousarray(np.zeros(len(self.frequency_A_GEVI)))
        self.abs_spectra_BLUF = np.ascontiguousarray(np.zeros(len(self.frequency_A_BLUF)))

        self.abs_dist_GECI = np.ascontiguousarray(np.empty((len(self.prob_GECI), len(self.frequency_A_GECI))))
        self.abs_dist_ChR2 = np.ascontiguousarray(np.empty((len(self.prob_ChR2), len(self.frequency_A_ChR2))))
        self.abs_dist_GEVI = np.ascontiguousarray(np.empty((len(self.prob_GEVI), len(self.frequency_A_GEVI))))
        self.abs_dist_BLUF = np.ascontiguousarray(np.empty((len(self.prob_GEVI), len(self.frequency_A_BLUF))))

        self.dyn_rho_A_GECI = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)
        self.dyn_rho_A_ChR2 = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)
        self.dyn_rho_A_GEVI = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)
        self.dyn_rho_A_BLUF = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)

        self.dyn_rho_R_GECI = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)
        self.dyn_rho_R_ChR2 = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)
        self.dyn_rho_R_GEVI = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)
        self.dyn_rho_R_BLUF = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)

    def create_molecules(self, GECI, ChR2, GEVI, BLUF):
        """
        Creates molecules from class parameters
        """
        #  ----------------------------- CREATING GECI ------------------------  #

        GECI.nDIM = len(self.energies_GECI)
        GECI.energies = self.energies_GECI.ctypes.data_as(POINTER(c_double))
        GECI.matrix_gamma_pd = self.matrix_gamma_pd.ctypes.data_as(POINTER(c_double))
        GECI.matrix_gamma_dep = self.matrix_gamma_dep_GECI.ctypes.data_as(POINTER(c_double))
        GECI.gamma_dep = self.gamma_dep_GECI
        GECI.frequency_A = self.frequency_A_GECI.ctypes.data_as(POINTER(c_double))
        GECI.freqDIM_A = len(self.frequency_A_GECI)
        GECI.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        GECI.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        GECI.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        GECI.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))

        GECI.rho = self.rho_GECI.ctypes.data_as(POINTER(c_complex))
        GECI.abs_spectra = self.abs_spectra_GECI.ctypes.data_as(POINTER(c_double))
        GECI.abs_dist = self.abs_dist_GECI.ctypes.data_as(POINTER(c_double))
        GECI.ref_spectra = self.ref_spectra_GECI.ctypes.data_as(POINTER(c_double))
        GECI.Raman_levels = self.Raman_levels_GECI.ctypes.data_as(POINTER(c_double))
        GECI.levels = self.levels_GECI.ctypes.data_as(POINTER(c_double))
        GECI.dyn_rho_A = self.dyn_rho_A_GECI.ctypes.data_as(POINTER(c_complex))
        GECI.dyn_rho_R = self.dyn_rho_R_GECI.ctypes.data_as(POINTER(c_complex))
        GECI.prob = self.prob_GECI.ctypes.data_as(POINTER(c_double))

        #  ----------------------------- CREATING ChR2 ------------------------  #

        ChR2.nDIM = len(self.energies_ChR2)
        ChR2.energies = self.energies_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.matrix_gamma_pd = self.matrix_gamma_pd.ctypes.data_as(POINTER(c_double))
        ChR2.matrix_gamma_dep = self.matrix_gamma_dep_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.gamma_dep = self.gamma_dep_ChR2
        ChR2.frequency_A = self.frequency_A_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.freqDIM_A = len(self.frequency_A_ChR2)
        ChR2.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        ChR2.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        ChR2.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        ChR2.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))

        ChR2.rho = self.rho_ChR2.ctypes.data_as(POINTER(c_complex))
        ChR2.abs_spectra = self.abs_spectra_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.abs_dist = self.abs_dist_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.ref_spectra = self.ref_spectra_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.Raman_levels = self.Raman_levels_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.levels = self.levels_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.dyn_rho_A = self.dyn_rho_A_ChR2.ctypes.data_as(POINTER(c_complex))
        ChR2.dyn_rho_R = self.dyn_rho_R_ChR2.ctypes.data_as(POINTER(c_complex))
        ChR2.prob = self.prob_ChR2.ctypes.data_as(POINTER(c_double))

        #  ----------------------------- CREATING GEVI ------------------------  #

        GEVI.nDIM = len(self.energies_GEVI)
        GEVI.energies = self.energies_GEVI.ctypes.data_as(POINTER(c_double))
        GEVI.matrix_gamma_pd = self.matrix_gamma_pd.ctypes.data_as(POINTER(c_double))
        GEVI.matrix_gamma_dep = self.matrix_gamma_dep_GEVI.ctypes.data_as(POINTER(c_double))
        GEVI.gamma_dep = self.gamma_dep_GEVI
        GEVI.frequency_A = self.frequency_A_GEVI.ctypes.data_as(POINTER(c_double))
        GEVI.freqDIM_A = len(self.frequency_A_GEVI)
        GEVI.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        GEVI.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        GEVI.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        GEVI.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))

        GEVI.rho = self.rho_GEVI.ctypes.data_as(POINTER(c_complex))
        GEVI.abs_spectra = self.abs_spectra_GEVI.ctypes.data_as(POINTER(c_double))
        GEVI.abs_dist = self.abs_dist_GEVI.ctypes.data_as(POINTER(c_double))
        GEVI.ref_spectra = self.ref_spectra_GEVI.ctypes.data_as(POINTER(c_double))
        GEVI.Raman_levels = self.Raman_levels_GEVI.ctypes.data_as(POINTER(c_double))
        GEVI.levels = self.levels_GEVI.ctypes.data_as(POINTER(c_double))
        GEVI.dyn_rho_A = self.dyn_rho_A_GEVI.ctypes.data_as(POINTER(c_complex))
        GEVI.dyn_rho_R = self.dyn_rho_R_GEVI.ctypes.data_as(POINTER(c_complex))
        GEVI.prob = self.prob_GEVI.ctypes.data_as(POINTER(c_double))

        #  ----------------------------- CREATING BLUF ------------------------  #

        BLUF.nDIM = len(self.energies_BLUF)
        BLUF.energies = self.energies_BLUF.ctypes.data_as(POINTER(c_double))
        BLUF.matrix_gamma_pd = self.matrix_gamma_pd.ctypes.data_as(POINTER(c_double))
        BLUF.matrix_gamma_dep = self.matrix_gamma_dep_BLUF.ctypes.data_as(POINTER(c_double))
        BLUF.gamma_dep = self.gamma_dep_BLUF
        BLUF.frequency_A = self.frequency_A_BLUF.ctypes.data_as(POINTER(c_double))
        BLUF.freqDIM_A = len(self.frequency_A_BLUF)
        BLUF.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        BLUF.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        BLUF.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        BLUF.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))

        BLUF.rho = self.rho_BLUF.ctypes.data_as(POINTER(c_complex))
        BLUF.abs_spectra = self.abs_spectra_BLUF.ctypes.data_as(POINTER(c_double))
        BLUF.abs_dist = self.abs_dist_BLUF.ctypes.data_as(POINTER(c_double))
        BLUF.ref_spectra = self.ref_spectra_BLUF.ctypes.data_as(POINTER(c_double))
        BLUF.Raman_levels = self.Raman_levels_BLUF.ctypes.data_as(POINTER(c_double))
        BLUF.levels = self.levels_BLUF.ctypes.data_as(POINTER(c_double))
        BLUF.dyn_rho_A = self.dyn_rho_A_BLUF.ctypes.data_as(POINTER(c_complex))
        BLUF.dyn_rho_R = self.dyn_rho_R_BLUF.ctypes.data_as(POINTER(c_complex))
        BLUF.prob = self.prob_BLUF.ctypes.data_as(POINTER(c_double))

    def create_parameters_spectra(self, spectra_params, params):
        """
        Creates parameters from class parameters
        """
        spectra_params.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        spectra_params.nDIM = len(self.energies_GECI)
        spectra_params.N_exc = params.N_exc
        spectra_params.time_A = self.time_A.ctypes.data_as(POINTER(c_double))
        spectra_params.time_R = self.time_R.ctypes.data_as(POINTER(c_double))
        spectra_params.timeDIM_A = len(self.time_A)
        spectra_params.timeDIM_R = len(self.time_R)
        spectra_params.field_amp_A = params.field_amp_A
        spectra_params.field_amp_R = params.field_amp_R
        spectra_params.omega_R = params.omega_R
        spectra_params.omega_v = params.omega_v
        spectra_params.omega_e = params.omega_e
        spectra_params.thread_num = params.num_threads
        spectra_params.prob_guess_num = len(self.prob_GECI)
        spectra_params.spectra_lower = params.spectra_lower.ctypes.data_as(POINTER(c_double))
        spectra_params.spectra_upper = params.spectra_upper.ctypes.data_as(POINTER(c_double))
        spectra_params.max_iter = params.max_iter
        spectra_params.control_guess = params.control_guess.ctypes.data_as(POINTER(c_double))
        spectra_params.control_lower = params.control_lower.ctypes.data_as(POINTER(c_double))
        spectra_params.control_upper = params.control_upper.ctypes.data_as(POINTER(c_double))
        spectra_params.guess_num = len(params.control_guess)
        spectra_params.max_iter_control = params.max_iter_control

    def calculate_spectra(self, params):
        GECI = Molecule()
        ChR2 = Molecule()
        GEVI = Molecule()
        BLUF = Molecule()
        self.create_molecules(GECI, ChR2, GEVI, BLUF)
        params_spectra = Parameters()
        self.create_parameters_spectra(params_spectra, params)

        # CalculateSpectra(GECI, params_spectra)
        # CalculateSpectra(ChR2, params_spectra)
        # CalculateSpectra(GEVI, params_spectra)
        CalculateSpectra(BLUF, params_spectra)


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
        which='both', direction='in', width=1, labelrotation=0, labelsize=labelsize)
    axis.get_yaxis().set_tick_params(
        which='both', direction='in', width=1, labelcolor='k', labelsize=labelsize)


def dyn_plot(axes, time_R, time_A, dyn_rho_R, dyn_rho_A, mol_str):

    axes.plot(time_R, dyn_rho_R[0], 'b', label='g1', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[0], 'b', label='g1', linewidth=2.5)
    axes.plot(time_R, dyn_rho_R[3], 'r', label='g4', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[3], 'r', label='g4', linewidth=2.5)
    axes.plot(time_R, dyn_rho_R[4:].sum(axis=0), 'k', label='EXC', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[4:].sum(axis=0), 'k', label='EXC', linewidth=2.5)
    axes.set_ylabel(mol_str, fontweight='bold')


def discrimination_2dplot(N_points, max_iter_control):
    start = time.time()
    params.max_iter_control = max_iter_control

    Z = np.empty((N_points, N_points))
    X = np.linspace(0.985, 1.015, N_points, endpoint=True)
    Y = 10**np.linspace(-1.31, 1, N_points, endpoint=False)
    for x, i in enumerate(X):
        for y, j in enumerate(Y):
            Raman_levels_GECI = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor
            Raman_levels_GEVI = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor * i

            gamma_vib = 2.418884e-5 * j
            matrix_gamma_dep = np.ones_like(matrix_gamma_pd) * gamma_vib
            np.fill_diagonal(matrix_gamma_dep, 0.0)

            params.control_guess = np.asarray([0.000295219, 0.000219229, 0.0140962, 0.0072593, 0.0837432, 600, 7218.48, 64277.3])  # GECI-ChR2-----14.75
            params.control_lower = np.asarray([0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_GECI[3] * 0.990, 1239.84 * energy_factor / 557.5, 600, 5000, 50000])
            params.control_upper = np.asarray([0.001, 0.001, 1.15 * energy_factor, Raman_levels_GECI[3] * 1.010, 1239.84 * energy_factor / 496.5, 600, 7500, 75000])

            # params.control_guess = np.asarray([0.000234753, 7.7756e-05, 0.0187527, Raman_levels_GECI[3] * 0.995, 0.084284])
            # params.control_lower = np.asarray([0.000160, 0.00005, 0.35 * energy_factor, Raman_levels_GECI[3] * 0.99, 1239.84 * energy_factor / 555])
            # params.control_upper = np.asarray([0.000350, 0.00013, 1.15 * energy_factor, Raman_levels_GECI[3] * 1.01, 1239.84 * energy_factor / 515])

            Systems['matrix_gamma_dep'] = matrix_gamma_dep
            Systems['Raman_levels_GECI'] = Raman_levels_GECI
            Systems['Raman_levels_GEVI'] = Raman_levels_GEVI
            # Systems['Raman_levels_ChR2'] = Raman_levels_ChR2

            molecule = RamanOpticalControl(params, **Systems)
            molecule.calculate_spectra(params)

            Z[x][y] = molecule.rho_GECI.diagonal()[4:].sum().real / molecule.rho_GEVI.diagonal()[4:].sum().real
            del molecule

    print(time.time() - start)
    pickle.dump(Z, open("discrimination_GECI_GECI_vib.p", "wb"))
    Z_read = pickle.load(open("discrimination_GECI_GECI_vib.p", "rb"))

    X *= 1600

    plt.figure()
    im = plt.imshow(Z_read, cmap=cm.hot, interpolation='bilinear', extent=[X.min(), X.max(), Y.max(), Y.min()])
    plt.colorbar()


if __name__ == '__main__':

    from scipy.signal import savgol_filter

    # ---------------------------------------------------------------------------- #
    #                             LIST OF CONSTANTS                                #
    # ---------------------------------------------------------------------------- #
    energy_factor = 1. / 27.211385
    time_factor = .02418884 / 1000
    wavelength_freq_factor = 1239.84
    cm_inv2eV_factor = 0.00012398

    # ---------------------------------------------------------------------------- #
    #                  OBTAIN RELEVANT INFORMATION FROM SPECTRA FILES              #
    # ---------------------------------------------------------------------------- #

    #  ----------- READING WAVELENGTH AND LINEAR SPECTRA FROM FILE --------------  #
    wavelength_GECI, absorption_GECI = get_experimental_spectra('Data/GCaMP.csv')
    wavelength_ChR2, absorption_ChR2 = get_experimental_spectra("Data/ChR2_2.csv")
    wavelength_GEVI, absorption_GEVI = get_experimental_spectra("Data/EGFP.csv")
    wavelength_BLUF, absorption_BLUF = get_experimental_spectra("Data/BLUF.csv")

    absorption_GECI = savgol_filter(absorption_GECI, 5, 3)
    absorption_ChR2 = savgol_filter(absorption_ChR2, 15, 3)
    absorption_GEVI = savgol_filter(absorption_GEVI, 5, 3)
    absorption_BLUF = savgol_filter(absorption_BLUF, 5, 3)

    frequency_A_GECI = wavelength_freq_factor * energy_factor / wavelength_GECI
    frequency_A_ChR2 = wavelength_freq_factor * energy_factor / wavelength_ChR2
    frequency_A_GEVI = wavelength_freq_factor * energy_factor / wavelength_GEVI
    frequency_A_BLUF = wavelength_freq_factor * energy_factor / wavelength_BLUF

    # ---------------------------------------------------------------------------- #
    #                      GENERATE MOLECULE PARAMETERS AND MATRICES               #
    # ---------------------------------------------------------------------------- #

    #  ----------------------------- MOLECULAR CONSTANTS ------------------------  #

    N = 8                                   # NUMBER OF ENERGY LEVELS PER SYSTEM
    M = 11                                  # NUMBER OF SYSTEMS PER ENSEMBLE

    N_vib = 4                               # NUMBER OF VIBRATIONAL ENERGY LEVELS IN THE GROUND STATE
    N_exc = N - N_vib                       # NUMBER OF VIBRATIONAL ENERGY LEVELS IN THE EXCITED STATE

    mu_value = 2.                           # VALUE OF TRANSITION DIPOLE MATRIX ELEMENTS IN DEBYE
    gamma_pd = 2.418884e-8                  # POPULATION DECAY GAMMA
    gamma_dep_GECI = 2.00 * 2.418884e-4     # DEPHASING GAMMA FOR GECI
    gamma_dep_ChR2 = 2.50 * 2.418884e-4     # DEPHASING GAMMA FOR ChR2
    gamma_dep_GEVI = 1.75 * 2.418884e-4     # DEPHASING GAMMA FOR GEVI
    gamma_dep_BLUF = 2.25 * 2.418884e-4     # DEPHASING GAMMA FOR GEVI
    gamma_vib = 0.1 * 2.418884e-5           # VIBRATIONAL DEPHASING GAMMA

    #  ------------------------ MOLECULAR MATRICES & VECTORS --------------------  #

    energies_GECI = np.empty(N)
    energies_ChR2 = np.empty(N)
    energies_GEVI = np.empty(N)
    energies_BLUF = np.empty(N)

    levels_GECI = np.asarray(1239.84 * energy_factor / np.linspace(400, 507, 4 * M)[::-1])  # GECI
    levels_ChR2 = np.asarray(1239.84 * energy_factor / np.linspace(370, 540, 4 * M)[::-1])  # ChR2
    levels_GEVI = np.asarray(1239.84 * energy_factor / np.linspace(352, 503, 4 * M)[::-1])  # GEVI
    levels_BLUF = np.asarray(1239.84 * energy_factor / np.linspace(320, 485, 4 * M)[::-1])  # BLUF

    rho_0 = np.zeros((N, N), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j

    mu = mu_value * np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)

    matrix_gamma_pd = np.ones((N, N)) * gamma_pd
    np.fill_diagonal(matrix_gamma_pd, 0.0)
    matrix_gamma_pd = np.tril(matrix_gamma_pd).T

    matrix_gamma_dep = np.ones_like(matrix_gamma_pd) * gamma_vib
    np.fill_diagonal(matrix_gamma_dep, 0.0)

    prob_GECI = np.asarray([0.21236871, 0.21212086, 0.14272493, 0.13512723, 0.11288251, 0.06981559, 0.04798607, 0.03077668, 0.01463422, 0.00558622, 0.01597697])  # GECI-updated
    prob_ChR2 = np.asarray([0.00581433, 0.02331881, 0.0646026, 0.10622365, 0.15318182, 0.15485174, 0.15485035, 0.12426589, 0.0928793,  0.06561301, 0.05439849])  # ChR2-updated
    prob_GEVI = np.asarray([0.19537675, 0.19774475, 0.15882784, 0.1278504,  0.07310059, 0.05344568,  0.04739961, 0.04763354, 0.04330608, 0.03303399, 0.02228078])  # GEVI-updated
    prob_BLUF = np.asarray([0.499283, 0.669991, 0.809212, 0.769626, 0.617798, 0.515183, 0.989902, 0.999971, 0.999369, 0.650253, 0.48709])  # BLUF

    spectra_lower = np.zeros(M)
    spectra_upper = np.ones(M)

    Raman_levels_GECI = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor
    Raman_levels_ChR2 = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor * 0.985
    Raman_levels_GEVI = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor * 0.985
    Raman_levels_BLUF = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor

    frequency_R_GECI = (np.asarray([1000, 1300, 1600])[:, np.newaxis] + 50.*np.linspace(-1, 1, 30)[np.newaxis, :]) * energy_factor * cm_inv2eV_factor
    frequency_R_ChR2 = (np.asarray([1000, 1300, 1600])[:, np.newaxis] + 50.*np.linspace(-1, 1, 30)[np.newaxis, :]) * energy_factor * cm_inv2eV_factor * 0.985
    frequency_R_GEVI = (np.asarray([1000, 1300, 1600])[:, np.newaxis] + 50.*np.linspace(-1, 1, 30)[np.newaxis, :]) * energy_factor * cm_inv2eV_factor * 0.985
    frequency_R_BLUF = (np.asarray([1000, 1300, 1600])[:, np.newaxis] + 50.*np.linspace(-1, 1, 30)[np.newaxis, :]) * energy_factor * cm_inv2eV_factor

    params = ADict(

        N_exc=N_exc,
        num_threads=cpu_count(),

        energy_factor=energy_factor,
        time_factor=time_factor,
        rho_0=rho_0,

        timeDIM_R=1,
        timeAMP_R=.01,
        timeDIM_A=500,
        timeAMP_A=2000,

        field_amp_R=0.0000008,
        field_amp_A=0.000001,

        omega_R=0.75 * energy_factor,
        omega_v=Raman_levels_GECI[3]*0.996,
        omega_e=1239.84*energy_factor/545,

        spectra_lower=spectra_lower,
        spectra_upper=spectra_upper,

        max_iter=100,

        control_guess=np.asarray([0.000226004, 7.11916e-05, 0.0244744, Raman_levels_GECI[3], 1239.84*energy_factor/543.626748]),
        control_lower=np.asarray([0.000135, 0.00005, 0.35 * energy_factor, Raman_levels_GECI[3]*0.990, 1239.84*energy_factor/557.5]),
        control_upper=np.asarray([0.000335, 0.00013, 1.15 * energy_factor, Raman_levels_GECI[3]*1.010, 1239.84*energy_factor/496.5]),

        max_iter_control=1,
    )

    Systems = dict(
        # Constant Parameters
        matrix_gamma_pd=matrix_gamma_pd,
        matrix_gamma_dep=matrix_gamma_dep,
        mu=mu,

        # GECI molecule
        energies_GECI=energies_GECI,
        gamma_dep_GECI=gamma_dep_GECI,
        prob_GECI=prob_GECI,
        # frequency_A_GECI=np.ascontiguousarray(frequency_A_GECI - Raman_levels_GECI[3]),
        frequency_A_GECI=np.ascontiguousarray(frequency_A_GECI),
        ref_spectra_GECI=np.ascontiguousarray(absorption_GECI),
        Raman_levels_GECI=Raman_levels_GECI,
        levels_GECI=levels_GECI,

        # ChR2 molecule
        energies_ChR2=energies_ChR2,
        gamma_dep_ChR2=gamma_dep_ChR2,
        prob_ChR2=prob_ChR2,
        # frequency_A_ChR2=np.ascontiguousarray(frequency_A_ChR2 - Raman_levels_ChR2[3]),
        frequency_A_ChR2=np.ascontiguousarray(frequency_A_ChR2),
        ref_spectra_ChR2=np.ascontiguousarray(absorption_ChR2),
        Raman_levels_ChR2=Raman_levels_ChR2,
        levels_ChR2=levels_ChR2,

        # GEVI molecule
        energies_GEVI=energies_GEVI,
        gamma_dep_GEVI=gamma_dep_GEVI,
        prob_GEVI=prob_GEVI,
        # frequency_A_GEVI=np.ascontiguousarray(frequency_A_GEVI - Raman_levels_GEVI[3]),
        frequency_A_GEVI=np.ascontiguousarray(frequency_A_GEVI),
        ref_spectra_GEVI=np.ascontiguousarray(absorption_GEVI),
        Raman_levels_GEVI=Raman_levels_GEVI,
        levels_GEVI=levels_GEVI,

        # BLUF molecule
        energies_BLUF=energies_BLUF,
        gamma_dep_BLUF=gamma_dep_BLUF,
        prob_BLUF=prob_BLUF,
        # frequency_A_BLUF=np.ascontiguousarray(frequency_A_BLUF - Raman_levels_BLUF[3]),
        frequency_A_BLUF=np.ascontiguousarray(frequency_A_BLUF),
        ref_spectra_BLUF=np.ascontiguousarray(absorption_BLUF),
        Raman_levels_BLUF=Raman_levels_BLUF,
        levels_BLUF=levels_BLUF
    )

    from matplotlib.pylab import cm
    import time
    import pickle

    np.set_printoptions(precision=6)
    molecule = RamanOpticalControl(params, **Systems)
    molecule.calculate_spectra(params)

    colors = [cm.nipy_spectral(x) for x in np.linspace(0., 1., M, endpoint=True)]

    ####################################################################################################################
    #                                                                                                                  #
    #                   PLOTTING ABSORPTION SPECTRA FIT, EXPERIMENTAL SPECTRA, SHIFTS IN SPECTRA                       #
    #                                                                                                                  #
    ####################################################################################################################

    fig_spectra, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))

    # plt.text(0.65, 0.92, 'Raman Shift of GCaMP', horizontalalignment='center', verticalalignment='center', weight='bold')
    # plt.text(0.532, 0.72, '497 nm', horizontalalignment='center', verticalalignment='center', weight='bold')
    # plt.text(0.765, 0.72, '540 nm', horizontalalignment='center', verticalalignment='center', weight='bold')
    #
    # plt.annotate(s='', xy=(541, 103.5), xytext=(495, 103.5), arrowprops=dict(arrowstyle='<|-|>'))
    # plt.annotate(s='', xy=(539.4, 83.5), xytext=(539.4, 97.5), arrowprops=dict(arrowstyle='-|>'))
    # plt.annotate(s='', xy=(497, 83.5), xytext=(497, 97.5), arrowprops=dict(arrowstyle='-|>'))

    axes[0].plot(energy_factor * 1239.84 / molecule.frequency_A_GECI, molecule.abs_spectra_GECI, 'b', label='Fitted GECI', linewidth=2.)
    axes[0].plot(energy_factor * 1239.84 / molecule.frequency_A_GECI, molecule.ref_spectra_GECI, 'b--', label='Actual GECI', linewidth=1.)

    with open('GECI_shift.p', 'rb') as f:
        data = pickle.load(f)

    axes[0].plot(energy_factor * 1239.84 / (molecule.frequency_A_GECI - Raman_levels_GECI[3]), data['shift_GECI'], 'r-', label='Shifted GECI', linewidth=2.)

    # for i in range(M):
    #     axes.fill(energy_factor * 1239.84 / molecule.frequency_A_GECI, molecule.abs_dist_GECI[i], color=colors[i], linewidth=1.5, alpha=0.5)
    #     axes.plot(energy_factor * 1239.84 / molecule.frequency_A_GECI, molecule.abs_dist_GECI[i], 'r--', linewidth=.5, alpha=0.6)

    axes[0].plot(energy_factor * 1239.84 / molecule.frequency_A_ChR2, molecule.abs_spectra_ChR2, 'k--', label='Fitted ChR2', linewidth=1.)
    axes[0].plot(energy_factor * 1239.84 / molecule.frequency_A_ChR2, molecule.ref_spectra_ChR2, 'k-', label='Actual ChR2', linewidth=1.)

    axes[2].plot(energy_factor * 1239.84 / molecule.frequency_A_BLUF, molecule.abs_spectra_BLUF, 'k--', label='Fitted BLUF', linewidth=1.)
    axes[2].plot(energy_factor * 1239.84 / molecule.frequency_A_BLUF, molecule.ref_spectra_BLUF, 'k-', label='Actual BLUF', linewidth=1.)

    # for i in range(M):
    #     axes.fill(energy_factor * 1239.84 / molecule.frequency_A_ChR2, molecule.abs_dist_ChR2[i], color=colors[i], linewidth=1.5, alpha=0.5)
    #     axes.plot(energy_factor * 1239.84 / molecule.frequency_A_ChR2, molecule.abs_dist_ChR2[i], 'r--', linewidth=.5, alpha=0.6)

    axes[0].set_xlabel('Wavelength (in nm)', fontweight='bold', fontsize='large')
    axes[0].set_ylabel('Normalized spectra', fontweight='bold', fontsize='large')
    axes[0].set_xlim(400, 580)
    axes[0].set_ylim(0.5, 155)

    # circle1 = plt.Circle((497, 100), 1.5, color='k')
    # circle2 = plt.Circle((539.4, 100), 1.5, color='k')
    # axes[0].add_artist(circle1).set_zorder(2)
    # axes[0].add_artist(circle2).set_zorder(2)
    axes[0].legend(handler_map={matplotlib.lines.Line2D: SymHandler()}, loc=9, fontsize='xx-small', prop={'weight':'bold'}, ncol=2, handleheight=2.4, labelspacing=0.05)

    for i in range(3):
        render_ticks(axes[i], 'x-large')

    plt.rc('font', weight='bold')
    plt.rc('axes', linewidth=2)

    fig_spectra.subplots_adjust(bottom=0.2, top=0.96, left=0.07, right=0.97, hspace=0.1, wspace=0.04)
    plt.savefig('GCaMP_ChR2_spectra' + '.eps', format='eps')
    plt.savefig('GCaMP_ChR2_spectra' + '.png', format='png')
    plt.show()

    ####################################################################################################################
    # --------------------------------------------- END DOCUMENT ------------------------------------------------------#
    ####################################################################################################################
