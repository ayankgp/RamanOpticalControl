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
from matplotlib import cm

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

        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(params.rho_0)
        self.rho_GECI = np.ascontiguousarray(params.rho_0.copy())
        self.rho_ChR2 = np.ascontiguousarray(params.rho_0.copy())
        self.rho_GEVI = np.ascontiguousarray(params.rho_0.copy())
        self.energies_GECI = np.ascontiguousarray(self.energies_GECI)
        self.energies_ChR2 = np.ascontiguousarray(self.energies_ChR2)
        self.energies_GEVI = np.ascontiguousarray(self.energies_GEVI)

        self.N = len(self.energies_GECI)

        self.abs_spectra_GECI = np.ascontiguousarray(np.zeros(len(self.frequency_A_GECI)))
        self.abs_spectra_ChR2 = np.ascontiguousarray(np.zeros(len(self.frequency_A_ChR2)))
        self.abs_spectra_GEVI = np.ascontiguousarray(np.zeros(len(self.frequency_A_GEVI)))

        self.abs_dist_GECI = np.ascontiguousarray(np.empty((len(self.prob_GECI), len(self.frequency_A_GECI))))
        self.abs_dist_ChR2 = np.ascontiguousarray(np.empty((len(self.prob_ChR2), len(self.frequency_A_ChR2))))
        self.abs_dist_GEVI = np.ascontiguousarray(np.empty((len(self.prob_GEVI), len(self.frequency_A_GEVI))))

        self.dyn_rho_A_GECI = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)
        self.dyn_rho_A_ChR2 = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)
        self.dyn_rho_A_GEVI = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)

        self.dyn_rho_R_GECI = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)
        self.dyn_rho_R_ChR2 = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)
        self.dyn_rho_R_GEVI = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)

    def create_molecules(self, GECI, ChR2, GEVI):
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
        self.create_molecules(GECI, ChR2, GEVI)
        params_spectra = Parameters()
        self.create_parameters_spectra(params_spectra, params)

        # CalculateControl(GECI, ChR2, params_spectra)
        CalculateControl(GECI, GEVI, params_spectra)


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
    plt.rc('font', weight='bold')
    axis.get_xaxis().set_tick_params(
        which='both', direction='in', width=1.25, labelrotation=0, labelsize=labelsize)
    axis.get_yaxis().set_tick_params(
        which='both', direction='in', width=1.25, labelcolor='k', labelsize=labelsize)
    axis.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.5, b=None, which='both', axis='both')
    # axis.get_xaxis().set_visible(False)
    # axis.get_yaxis().set_visible(False)


def dyn_plot(axes, time_R, time_A, dyn_rho_R, dyn_rho_A, mol_str):

    axes.plot(time_R, dyn_rho_R[0], 'b', label='g1', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[0], 'b', linewidth=2.5)
    axes.plot(time_R, dyn_rho_R[3], 'r', label='g4', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[3], 'r', linewidth=2.5)
    axes.plot(time_R, dyn_rho_R[4:].sum(axis=0), 'k', label='EXC', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[4:].sum(axis=0), 'k', linewidth=2.5)
    axes.set_ylabel(mol_str, fontweight='bold', fontsize='large')


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import pickle
    from scipy.signal import savgol_filter
    cm_inv2eV_factor = 0.00012398

    # ---------------------------------------------------------------------------- #
    #                             LIST OF CONSTANTS                                #
    # ---------------------------------------------------------------------------- #
    energy_factor = 1. / 27.211385
    time_factor = .02418884 / 1000
    wavelength_freq_factor = 1239.84

    # ---------------------------------------------------------------------------- #
    #                  OBTAIN RELEVANT INFORMATION FROM SPECTRA FILES              #
    # ---------------------------------------------------------------------------- #

    #  ----------- READING WAVELENGTH AND LINEAR SPECTRA FROM FILE --------------  #
    wavelength_GECI, absorption_GECI = get_experimental_spectra('Data/GCaMP.csv')
    wavelength_ChR2, absorption_ChR2 = get_experimental_spectra("Data/ChR2_2.csv")
    wavelength_GEVI, absorption_GEVI = get_experimental_spectra("Data/EGFP.csv")

    absorption_GECI = savgol_filter(absorption_GECI, 5, 3)
    absorption_ChR2 = savgol_filter(absorption_ChR2, 15, 3)
    absorption_GEVI = savgol_filter(absorption_GEVI, 5, 3)

    frequency_A_GECI = wavelength_freq_factor * energy_factor / wavelength_GECI
    frequency_A_ChR2 = wavelength_freq_factor * energy_factor / wavelength_ChR2
    frequency_A_GEVI = wavelength_freq_factor * energy_factor / wavelength_GEVI

    # ---------------------------------------------------------------------------- #
    #                      GENERATE MOLECULE PARAMETERS AND MATRICES               #
    # ---------------------------------------------------------------------------- #

    #  ----------------------------- MOLECULAR CONSTANTS ------------------------  #

    N = 8  # NUMBER OF ENERGY LEVELS PER SYSTEM
    M = 11  # NUMBER OF SYSTEMS PER ENSEMBLE

    N_vib = 4  # NUMBER OF VIBRATIONAL ENERGY LEVELS IN THE GROUND STATE
    N_exc = N - N_vib  # NUMBER OF VIBRATIONAL ENERGY LEVELS IN THE EXCITED STATE

    mu_value = 5.  # VALUE OF TRANSITION DIPOLE MATRIX ELEMENTS IN DEBYE
    gamma_pd = 2.418884e-8  # POPULATION DECAY GAMMA
    gamma_dep_GECI = 2.00 * 2.418884e-4  # DEPHASING GAMMA FOR GECI
    gamma_dep_ChR2 = 2.50 * 2.418884e-4  # DEPHASING GAMMA FOR ChR2
    gamma_dep_GEVI = 1.75 * 2.418884e-4  # DEPHASING GAMMA FOR GEVI
    gamma_vib = 0.1 * 2.418884e-5  # VIBRATIONAL DEPHASING GAMMA

    #  ------------------------ MOLECULAR MATRICES & VECTORS --------------------  #

    energies_GECI = np.empty(N)
    energies_ChR2 = np.empty(N)
    energies_GEVI = np.empty(N)

    levels_GECI = np.asarray(1239.84 * energy_factor / np.linspace(400, 507, 4 * M)[::-1])  # GECI
    levels_ChR2 = np.asarray(1239.84 * energy_factor / np.linspace(370, 540, 4 * M)[::-1])  # ChR2
    levels_GEVI = np.asarray(1239.84 * energy_factor / np.linspace(352, 503, 4 * M)[::-1])  # GEVI

    rho_0 = np.zeros((N, N), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j

    mu = mu_value * np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)

    matrix_gamma_pd = np.ones((N, N)) * gamma_pd
    np.fill_diagonal(matrix_gamma_pd, 0.0)
    matrix_gamma_pd = np.tril(matrix_gamma_pd).T

    matrix_gamma_dep = np.ones_like(matrix_gamma_pd) * gamma_vib
    np.fill_diagonal(matrix_gamma_dep, 0.0)

    # prob_GECI = np.asarray([0.9999450, 0.998778, 0.672025, 0.636251, 0.531511, 0.328729, 0.225944, 0.144913, 0.0689057, 0.0263029, 0.0752281])  # GECI-updated
    # prob_ChR2 = np.asarray([0.0375477, 0.150588, 0.417190, 0.685970, 0.989216, 1.000000, 0.999991, 0.802483, 0.5997950, 0.4237150, 0.3512940])  # ChR2-updated
    # prob_GEVI = np.asarray([0.988021, 0.999996, 0.803193, 0.64654, 0.36967, 0.270275, 0.2397, 0.240883, 0.218999, 0.167053, 0.112674])  # GEVI-updated

    prob_GECI = np.asarray(
        [0.21236871, 0.21212086, 0.14272493, 0.13512723, 0.11288251, 0.06981559, 0.04798607, 0.03077668, 0.01463422,
         0.00558622, 0.01597697])  # GECI-updated
    prob_ChR2 = np.asarray(
        [0.00581433, 0.02331881, 0.0646026, 0.10622365, 0.15318182, 0.15485174, 0.15485035, 0.12426589, 0.0928793,
         0.06561301, 0.05439849])  # ChR2-updated
    prob_GEVI = np.asarray(
        [0.19537675, 0.19774475, 0.15882784, 0.1278504, 0.07310059, 0.05344568, 0.04739961, 0.04763354, 0.04330608,
         0.03303399, 0.02228078])  # GEVI-updated

    spectra_lower = np.zeros(M)
    spectra_upper = np.ones(M)

    Raman_levels_GECI = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor
    Raman_levels_ChR2 = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor * 0.985
    Raman_levels_GEVI = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor * 0.985

    params = ADict(

        N_exc=N_exc,
        num_threads=cpu_count(),

        energy_factor=energy_factor,
        time_factor=time_factor,
        rho_0=rho_0,

        timeDIM_R=15000,
        timeAMP_R=62500,
        timeDIM_A=4000,
        timeAMP_A=12000,

        field_amp_R=0.000235,
        field_amp_A=0.00008,

        omega_R=0.75 * energy_factor,
        omega_v=Raman_levels_GECI[3] * 0.996,
        omega_e=1239.84 * energy_factor / 545,

        spectra_lower=spectra_lower,
        spectra_upper=spectra_upper,

        max_iter=1,

        control_guess=np.asarray([0.000226004, 7.11916e-05, 0.0244744, 0.00724681, 1239.84 * energy_factor / 543.626748]),  # GECI-ChR2-----7.10967
        # control_guess=np.asarray([0.000223258, 7.50284e-05, 0.0246308, 0.00725043, 1239.84*energy_factor/540.172124]),    # GECI-GEVI-----9.81198
        control_lower=np.asarray([0.000135, 0.00005, 0.35 * energy_factor, Raman_levels_GECI[3] * 0.990, 1239.84 * energy_factor / 557.5]),
        control_upper=np.asarray([0.000335, 0.00013, 1.15 * energy_factor, Raman_levels_GECI[3] * 1.010, 1239.84 * energy_factor / 496.5]),

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
        frequency_A_GECI=np.ascontiguousarray(frequency_A_GECI),
        ref_spectra_GECI=np.ascontiguousarray(absorption_GECI),
        Raman_levels_GECI=Raman_levels_GECI,
        levels_GECI=levels_GECI,

        # ChR2 molecule
        energies_ChR2=energies_ChR2,
        gamma_dep_ChR2=gamma_dep_ChR2,
        prob_ChR2=prob_ChR2,
        frequency_A_ChR2=np.ascontiguousarray(frequency_A_ChR2),
        ref_spectra_ChR2=np.ascontiguousarray(absorption_ChR2),
        Raman_levels_ChR2=Raman_levels_ChR2,
        levels_ChR2=levels_ChR2,

        # GEVI molecule
        energies_GEVI=energies_GEVI,
        gamma_dep_GEVI=gamma_dep_GEVI,
        prob_GEVI=prob_GEVI,
        frequency_A_GEVI=np.ascontiguousarray(frequency_A_GEVI),
        ref_spectra_GEVI=np.ascontiguousarray(absorption_GEVI),
        Raman_levels_GEVI=Raman_levels_GEVI,
        levels_GEVI=levels_GEVI

    )

    colors = [cm.BrBG(x) for x in np.linspace(0., 1., M, endpoint=True)]

    #  ------------------------ SETTING UP PLOT GRIDS --------------------- #

    # gs_top = plt.GridSpec(5, 2, top=0.95, hspace=0.2, wspace=0.05)
    # gs_base = plt.GridSpec(5, 2, hspace=0.025, wspace=0.01)
    # fig = plt.figure(figsize=(14, 11))
    # axbig1 = fig.add_subplot(gs_top[:2, 0])
    # axbig2 = fig.add_subplot(gs_top[:2, 1])
    #
    # axes = [[fig.add_subplot(gs_base[i, j]) for j in range(2)] for i in range(2, 5)]
    #
    # axbig2.set_yticklabels([])
    # for i in range(2):
    #     axes[i][1].set_yticklabels([])
    #     for j in range(2):
    #         axes[i][j].set_xticklabels([])
    # axes[2][1].set_yticklabels([])
    #
    # render_ticks(axbig1, 'x-large')
    # render_ticks(axbig2, 'x-large')
    # for i in range(3):
    #     for j in range(2):
    #         render_ticks(axes[i][j], 'x-large')

    #  ------------------------ READING AND PLOTTING SPECTRAL DATA --------------------- #

    # data_shift = pickle.load(open("all_spectra_shift.p", "rb"))
    # data_fit = pickle.load(open("all_spectra_fit.p", "rb"))
    #
    # GECI_freq = data_fit["GECI_freq"]
    # GECI_spectra_fit = data_fit["GECI_spectra_fit"]
    # GECI_spectra_dist = data_fit["GECI_spectra_dist"]
    # GECI_spectra_ref = data_fit["GECI_spectra_ref"]
    #
    # GEVI_freq = data_fit["GEVI_freq"]
    # GEVI_spectra_fit = data_fit["GEVI_spectra_fit"]
    # GEVI_spectra_dist = data_fit["GEVI_spectra_dist"]
    # GEVI_spectra_ref = data_fit["GEVI_spectra_ref"]
    #
    # GECI_freq_shift = data_shift["GECI_freq_shift"]
    # GECI_spectra_shift = data_shift["GECI_spectra_shift"]
    #
    # GEVI_freq_shift = data_shift["GEVI_freq_shift"]
    # GEVI_spectra_shift = data_shift["GEVI_spectra_shift"]
    #
    # axbig1.plot(GECI_freq, GECI_spectra_ref, 'k', linewidth=1., label='fitted GCaMP \n spectra')
    # axbig1.plot(GECI_freq, GECI_spectra_fit, 'r', linewidth=1., label='experimental \n GCaMP spectra')
    #
    # axbig1.annotate('', xy=(496, 102.5), xytext=(543, 102.5), arrowprops=dict(arrowstyle='<->', color='red'), annotation_clip=False)
    # axbig1.annotate('Raman assisted shift (GCaMP)', xy=(496, 104.5), xytext=(465, 104.5))
    #
    # for i in range(M):
    #     axbig1.fill(GECI_freq, GECI_spectra_dist[i], color=colors[i], linewidth=1.5, alpha=0.75)
    #     axbig1.plot(GECI_freq, GECI_spectra_dist[i], 'r--', linewidth=.5, alpha=0.6)
    #
    # axbig1.plot(GECI_freq_shift, GECI_spectra_shift, 'r--', label='Simulated \n GCaMP shift')
    # axbig1.plot(GEVI_freq, GEVI_spectra_fit, 'b', linewidth=1., label='experimental \n ASAP spectra')
    # axbig1.set_xlim(360., 600.)
    # axbig1.set_ylim(0., 110.)
    # axbig1.set_xlabel('Wavelength (in nm)', weight='bold', fontsize='large')
    # axbig1.set_ylabel('Normalised absorption', weight='bold', fontsize='large')
    # axbig1.legend(prop=dict(weight='normal', size='small'))
    #
    # axbig2.plot(GEVI_freq, GEVI_spectra_ref, 'k', linewidth=1., label='fitted GCaMP \n spectra')
    # axbig2.plot(GEVI_freq, GEVI_spectra_fit, 'b', linewidth=1., label='experimental \n GCaMP spectra')
    #
    # axbig2.annotate('', xy=(495, 102.5), xytext=(535, 102.5), arrowprops=dict(arrowstyle='<->', color='blue'), annotation_clip=False)
    # axbig2.annotate('Raman assisted shift (ASAP)', xy=(496, 104.5), xytext=(465, 104.5))
    #
    # # for i in range(M-1):
    # #     axbig2.fill(GEVI_freq, GEVI_spectra_dist[i], color=colors[i], linewidth=1.5, alpha=0.75)
    # #     axbig2.plot(GEVI_freq, GEVI_spectra_dist[i], 'b--', linewidth=.5, alpha=0.6)
    #
    # axbig2.plot(GEVI_freq_shift, GEVI_spectra_shift, 'b--', label='Simulated \n GCaMP shift')
    # axbig2.plot(GECI_freq, GECI_spectra_fit, 'r', linewidth=1., label='experimental \n ASAP spectra')
    # axbig2.set_xlim(360., 600.)
    # axbig2.set_ylim(0., 110.)
    # axbig2.set_xlabel('Wavelength (in nm)', weight='bold', fontsize='large')
    # axbig2.legend(prop=dict(weight='normal', size='small'))

    ######################################################################################

    #  ------------------------ READING AND PLOTTING SPECTRAL DATA --------------------- #

    ######################################################################################

    np.set_printoptions(precision=6)
    molecule = RamanOpticalControl(params, **Systems)
    molecule.calculate_spectra(params)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6.5, 3.5))
    molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    time_axis = time_factor * (molecule.time_R.max() + np.concatenate((molecule.time_R, molecule.time_A)))
    time_R = time_factor * (molecule.time_R.max() + molecule.time_R)
    time_A = time_factor * (molecule.time_R.max() + molecule.time_A)

    axes[0][0].plot(time_factor * (molecule.time_R.max() + molecule.time_R), 5.142e3 * molecule.field_R.real, 'k', linewidth=1.)
    axes[0][0].plot(time_factor * (molecule.time_R.max() + molecule.time_A), 5.142e3 * molecule.field_A.real, 'darkblue', linewidth=1.)

    dyn_plot(axes[1][0], time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, 'Population \n GECI')
    # dyn_plot(axes[2][0], time_R, time_A, molecule.dyn_rho_R_ChR2.real, molecule.dyn_rho_A_ChR2.real, 'Population \n ChR2')
    dyn_plot(axes[2][0], time_R, time_A, molecule.dyn_rho_R_GEVI.real, molecule.dyn_rho_A_GEVI.real, 'Population \n GEVI')

    axes[2][0].set_xlabel('Time (in ps)', fontweight='bold', fontsize='large')
    axes[0][0].set_ylabel('Electric field \n (in MV/cm)', fontweight='bold', fontsize='large')

    axes[1][0].legend(loc=6, fontsize='x-small')
    axes[2][0].legend(loc=6, fontsize='x-small')

    axes[0][0].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))
    axes[1][0].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))
    axes[2][0].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))

    # ax1 = fig.add_axes([0.42, 0.525, 0.07, 0.0375])
    # ax1.plot(time_factor * (molecule.time_R.max() + molecule.time_R), 5.142e3 * molecule.field_R.real, 'k', linewidth=1.)
    # ax1.set_xlim(0.95, 1.05)
    # ax2 = fig.add_axes([0.42, 0.44, 0.07, 0.0375])
    # ax2.plot(time_factor * (molecule.time_R.max() + molecule.time_A), 5.142e3 * molecule.field_A.real, 'darkblue', linewidth=1.)
    # ax2.set_xlim(3.285, 3.315)
    # ax3 = fig.add_axes([0.38, 0.32, 0.09, 0.0375])
    # dyn_plot(ax3, time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, '')
    # ax3.set_xlim(1.7, 1.755)
    # ax3.set_ylim(0.35, 0.65)
    # ax4 = fig.add_axes([0.38, 0.19, 0.09, 0.0375])
    # dyn_plot(ax4, time_R, time_A, molecule.dyn_rho_R_GEVI.real, molecule.dyn_rho_A_GEVI.real, '')
    # ax4.set_xlim(1.5, 1.55)
    # ax4.get_xaxis().set_visible(False)
    # ax4.set_ylim(0.8, 1.0)
    # ax5 = fig.add_axes([0.38, 0.15, 0.09, 0.0375])
    # dyn_plot(ax5, time_R, time_A, molecule.dyn_rho_R_GEVI.real, molecule.dyn_rho_A_GEVI.real, '')
    # ax5.set_xlim(1.5, 1.55)
    # ax5.set_ylim(0.0, 0.175)
    #
    # for axis in [ax1, ax2, ax3, ax4, ax5]:
    #     render_ticks(axis, 'small')

    del molecule
    params.control_guess=np.asarray([0.000210537, 6.87189e-05, 0.0314729, 0.00714859, 1239.84*energy_factor/533.820184])    # GEVI-GECI-----4.01084
    params.control_lower = np.asarray([0.000160, 0.00005, 0.50 * energy_factor, Raman_levels_GEVI[3] * 0.990, 1239.84 * energy_factor / 555])
    params.control_upper = np.asarray([0.000280, 0.00010, 1.00 * energy_factor, Raman_levels_GEVI[3] * 1.010, 1239.84 * energy_factor / 525])

    molecule = RamanOpticalControl(params, **Systems)
    molecule.calculate_spectra(params)

    molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    time_R = time_factor * (molecule.time_R.max() + molecule.time_R)
    time_A = time_factor * (molecule.time_R.max() + molecule.time_A)

    axes[0][1].plot(time_factor * (molecule.time_R.max() + molecule.time_R), 5.142e3 * molecule.field_R.real, 'k', linewidth=2.)
    axes[0][1].plot(time_factor * (molecule.time_R.max() + molecule.time_A), 5.142e3 * molecule.field_A.real, 'darkblue', linewidth=2.)

    dyn_plot(axes[1][1], time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, '')
    dyn_plot(axes[2][1], time_R, time_A, molecule.dyn_rho_R_GEVI.real, molecule.dyn_rho_A_GEVI.real, '')

    axes[2][1].set_xlabel('Time (in ps)', fontweight='bold', fontsize='large')

    axes[1][1].legend(loc=6, fontsize='x-small')
    axes[2][1].legend(loc=6, fontsize='x-small')

    axes[0][1].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))
    axes[1][1].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))
    axes[2][1].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))

    # ax11 = fig.add_axes([0.81, 0.525, 0.07, 0.0375])
    # ax11.plot(time_factor * (molecule.time_R.max() + molecule.time_R), 5.142e3 * molecule.field_R.real, 'k', linewidth=1.)
    # ax11.set_xlim(0.95, 1.05)
    # ax21 = fig.add_axes([0.81, 0.44, 0.07, 0.0375])
    # ax21.plot(time_factor * (molecule.time_R.max() + molecule.time_A), 5.142e3 * molecule.field_A.real, 'darkblue', linewidth=1.)
    # ax21.set_xlim(3.285, 3.315)
    # ax31 = fig.add_axes([0.76, 0.35, 0.09, 0.0375])
    # dyn_plot(ax31, time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, '')
    # ax31.set_xlim(1.475, 1.525)
    # ax31.get_xaxis().set_visible(False)
    # ax31.set_ylim(0.85, 0.95)
    # ax41 = fig.add_axes([0.76, 0.31, 0.09, 0.0375])
    # dyn_plot(ax41, time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, '')
    # ax41.set_xlim(1.475, 1.525)
    # ax41.set_ylim(0.0, 0.15)
    # ax51 = fig.add_axes([0.76, 0.165, 0.09, 0.0375])
    # dyn_plot(ax51, time_R, time_A, molecule.dyn_rho_R_GEVI.real, molecule.dyn_rho_A_GEVI.real, '')
    # ax51.set_xlim(1.765, 1.795)
    # ax51.set_ylim(0.4, 0.6)
    #
    # for axis in [ax11, ax21, ax31, ax41, ax51]:
    #     render_ticks(axis, 'small')
    #     # axis.grid(b=None)

    plt.savefig('GECI-GEVI.png', format="png")

    print(molecule.rho_GECI.diagonal()[4:].sum().real)
    print(molecule.rho_GEVI.diagonal()[4:].sum().real)
    print(molecule.rho_GECI.diagonal()[4:].sum().real/molecule.rho_GEVI.diagonal()[4:].sum().real)
    print(molecule.rho_GEVI.diagonal()[4:].sum().real/molecule.rho_GECI.diagonal()[4:].sum().real)

    for i in range(3):
        axes[i][1].set_yticklabels([])
        
    fig.subplots_adjust(bottom=0.15, top=0.96, left=0.15, hspace=0.05, wspace=0.05)
    plt.show()