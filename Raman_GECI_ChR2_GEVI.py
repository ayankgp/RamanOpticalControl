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
        spectra_params.timeAMP_A = params.timeAMP_A
        spectra_params.timeAMP_R = params.timeAMP_R
        spectra_params.timeDIM_A = len(self.time_A)
        spectra_params.timeDIM_R = len(self.time_R)
        spectra_params.field_amp_A = params.field_amp_A
        spectra_params.field_amp_R = params.field_amp_R
        spectra_params.omega_R = params.omega_R
        spectra_params.omega_v = params.omega_v
        spectra_params.omega_e = params.omega_e
        spectra_params.d_alpha = params.control_guess[-1]
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

    def control_molA_over_molB(self, params):
        GECI = Molecule()
        ChR2 = Molecule()
        GEVI = Molecule()
        self.create_molecules(GECI, ChR2, GEVI)
        params_spectra = Parameters()
        self.create_parameters_spectra(params_spectra, params)

        # CalculateControl(GECI, ChR2, params_spectra)
        CalculateControl(GECI, GEVI, params_spectra)

    def control_molB_over_molA(self, params):
        GECI = Molecule()
        ChR2 = Molecule()
        GEVI = Molecule()

        self.create_molecules(GECI, ChR2, GEVI)
        params_spectra = Parameters()
        self.create_parameters_spectra(params_spectra, params)

        # CalculateControl(ChR2, GECI, params_spectra)
        CalculateControl(GEVI, GECI, params_spectra)


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

    axes.plot(time_R, dyn_rho_R[0], 'b', label='$\\rho_{g, \\nu=0}$', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[0], 'b', linewidth=2.5)
    # axes.plot(time_R, dyn_rho_R[3], 'r', label='g4', linewidth=1.)
    # axes.plot(time_A, dyn_rho_A[3], 'r', linewidth=2.5)
    axes.plot(time_R, dyn_rho_R[4:].sum(axis=0), 'k', label='$\\rho_{e, total}$', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[4:].sum(axis=0), 'k', linewidth=2.5)
    axes.set_ylabel(mol_str, fontweight='bold', fontsize='medium')


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

    mu_value = 2.  # VALUE OF TRANSITION DIPOLE MATRIX ELEMENTS (TIMES 2.5 DEBYE)
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

        timeDIM_R=10,
        timeAMP_R=1,
        timeDIM_A=3000,
        timeAMP_A=7542,

        field_amp_R=0.000235,
        field_amp_A=0.00008,

        omega_R=0.75 * energy_factor,
        omega_v=Raman_levels_GECI[3] * 0.996,
        omega_e=1239.84 * energy_factor / 545,

        spectra_lower=spectra_lower,
        spectra_upper=spectra_upper,

        max_iter=1,
        control_guess=np.asarray([0, 0.000873791, 0.0241614, 0.00723613, 0.0909597, 600, 7702.36, 1]),  # GECI-GEVI-----EE only
        control_lower=np.asarray([0.0, 0.0001, 0.35 * energy_factor, Raman_levels_GECI[3] * 0.990, 1239.84 * energy_factor / 545, 600, 5000, 1]),
        control_upper=np.asarray([0.0, 0.001, 1.15 * energy_factor, Raman_levels_GECI[3] * 1.010, 1239.84 * energy_factor / 475, 600, 8000, 1]),

        #
        # control_guess=np.asarray([0.00031874, 0.000241155, 0.0144751, 0.0072569, 0.0838227, 600]),  # GECI-ChR2-----14.
        # control_lower=np.asarray([0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_GECI[3] * 0.990, 1239.84 * energy_factor / 557.5, 600]),
        # control_upper=np.asarray([0.001, 0.001, 1.15 * energy_factor, Raman_levels_GECI[3] * 1.010, 1239.84 * energy_factor / 496.5, 600]),

        # control_guess=np.asarray([0.000303476, 0.000299166, 0.0151831, 0.0071445, 0.0858908, 600.]),  # ChR2-GECI-----14.
        # control_lower=np.asarray([0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_ChR2[3] * 0.990, 1239.84 * energy_factor / 550.5, 600]),
        # control_upper=np.asarray([0.001, 0.001, 1.15 * energy_factor, Raman_levels_ChR2[3] * 1.010, 1239.84 * energy_factor / 476.5, 600]),

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

    np.set_printoptions(precision=6)
    molecule = RamanOpticalControl(params, **Systems)
    molecule.control_molA_over_molB(params)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 4.5))
    # axes[0, 1].get_shared_y_axes().join(axes[0, 0], axes[0, 1])

    molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    time_axis = time_factor * (molecule.time_R.max() + np.concatenate((molecule.time_R, molecule.time_A)))
    time_R = time_factor * (molecule.time_R.max() + molecule.time_R)
    time_A = time_factor * (molecule.time_R.max() + molecule.time_A)

    axes[0, 0].plot(time_factor * (molecule.time_R.max() + molecule.time_R), 3.55e7 * molecule.field_R.real.max() * molecule.field_R.real, 'k', linewidth=1.)
    axes[0, 0].plot(time_factor * (molecule.time_R.max() + molecule.time_A), 3.55e7 * molecule.field_A.real.max() * molecule.field_A.real, 'darkblue', linewidth=1.)

    dyn_plot(axes[1, 0], time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, 'Population \n GCaMP')
    # dyn_plot(axes[2, 0], time_R, time_A, molecule.dyn_rho_R_ChR2.real, molecule.dyn_rho_A_ChR2.real, 'Population \n ChR2')
    dyn_plot(axes[2, 0], time_R, time_A, molecule.dyn_rho_R_GEVI.real, molecule.dyn_rho_A_GEVI.real, 'Population \n ASAP')

    # axes[1, 0].plot(time_R, molecule.dyn_rho_R_GECI.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    # axes[1, 0].plot(time_A, molecule.dyn_rho_A_GECI.real[3], 'r', linewidth=2.5)
    # axes[2, 0].plot(time_R, molecule.dyn_rho_R_GEVI.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    # axes[2, 0].plot(time_A, molecule.dyn_rho_A_GEVI.real[3], 'r', linewidth=2.5)
    # axes[2, 0].plot(time_R, molecule.dyn_rho_R_ChR2.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    # axes[2, 0].plot(time_A, molecule.dyn_rho_A_ChR2.real[3], 'r', linewidth=2.5)

    axes[2, 0].set_xlabel('Time (in ps)', fontweight='bold', fontsize='medium')
    axes[0, 0].set_ylabel('Electric field \n (in $GW/cm^2$)', fontweight='bold')
    axes[0, 0].ticklabel_format(style='sci', scilimits=(0, 3))

    axes[1, 0].legend(loc=1, prop={'weight': 'normal', 'size': 'small'})
    axes[2, 0].legend(loc=1, prop={'weight': 'normal', 'size': 'small'})

    axes[0, 0].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))
    axes[1, 0].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))
    axes[2, 0].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))
    render_ticks(axes[0, 0], 'large')
    render_ticks(axes[1, 0], 'large')
    render_ticks(axes[2, 0], 'large')

    print(molecule.rho_GECI.diagonal()[4:].sum().real)
    # print(molecule.rho_ChR2.diagonal()[4:].sum().real)
    print(molecule.rho_GEVI.diagonal()[4:].sum().real)
    # print(molecule.rho_GECI.diagonal()[4:].sum().real/molecule.rho_ChR2.diagonal()[4:].sum().real)
    print(molecule.rho_GECI.diagonal()[4:].sum().real/molecule.rho_GEVI.diagonal()[4:].sum().real)

    # ---------- GECI vs GEVI ----------------- #

    del molecule

    # params.control_guess=np.asarray([0.000316697, 0.000319296, 0.0167604, 0.00714847, 0.0856847, 600])  # ChR2-GECI-----14.
    # params.control_lower=np.asarray([0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_ChR2[3] * 0.990, 1239.84 * energy_factor / 550.5, 600])
    # params.control_upper=np.asarray([0.001, 0.001, 1.15 * energy_factor, Raman_levels_ChR2[3] * 1.010, 1239.84 * energy_factor / 476.5, 600])

    params.timeAMP_R = 62071
    params.timeDIM_R = 15000
    params.timeAMP_A = 7542.48

    params.control_guess = np.asarray([0.00029029, 0.000222322, 0.0246745, 0.00728748, 0.0840499, 600, 7542.48, 62071])  # GECI-GEVI-----9.81198
    params.control_lower = np.asarray([0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_GECI[3] * 0.990, 1239.84 * energy_factor / 557.5, 600, 5000, 50000])
    params.control_upper = np.asarray([0.001, 0.001, 1.15 * energy_factor, Raman_levels_GECI[3] * 1.010, 1239.84 * energy_factor / 496.5, 600, 8000, 80000])

    params.max_iter_control = 1

    molecule = RamanOpticalControl(params, **Systems)
    molecule.control_molA_over_molB(params)

    molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    time_axis = time_factor * (molecule.time_R.max() + np.concatenate((molecule.time_R, molecule.time_A)))
    time_R = time_factor * (molecule.time_R.max() + molecule.time_R)
    time_A = time_factor * (molecule.time_R.max() + molecule.time_A)

    axes[0, 1].plot(time_factor * (molecule.time_R.max() + molecule.time_R), 3.55e7 * molecule.field_R.real.max() * molecule.field_R.real, 'k', linewidth=1.)
    axes[0, 1].plot(time_factor * (molecule.time_R.max() + molecule.time_A), 3.55e7 * molecule.field_A.real.max() * molecule.field_A.real,
                    'darkblue', linewidth=1.)

    dyn_plot(axes[1, 1], time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, '')
    # dyn_plot(axes[2, 1], time_R, time_A, molecule.dyn_rho_R_ChR2.real, molecule.dyn_rho_A_ChR2.real, '')
    dyn_plot(axes[2, 1], time_R, time_A, molecule.dyn_rho_R_GEVI.real, molecule.dyn_rho_A_GEVI.real, '')

    axes[1, 1].plot(time_R, molecule.dyn_rho_R_GECI.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    axes[1, 1].plot(time_A, molecule.dyn_rho_A_GECI.real[3], 'r', linewidth=2.5)
    axes[2, 1].plot(time_R, molecule.dyn_rho_R_GEVI.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    axes[2, 1].plot(time_A, molecule.dyn_rho_A_GEVI.real[3], 'r', linewidth=2.5)
    # axes[2, 1].plot(time_R, molecule.dyn_rho_R_ChR2.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    # axes[2, 1].plot(time_A, molecule.dyn_rho_A_ChR2.real[3], 'r', linewidth=2.5)

    axes[2, 1].set_xlabel('Time (in ps)', fontweight='bold', fontsize='medium')
    axes[0, 1].ticklabel_format(style='sci', scilimits=(0, 3))

    axes[1, 1].legend(loc=6, prop={'weight': 'normal', 'size': 'x-small'})
    axes[2, 1].legend(loc=6, prop={'weight': 'normal', 'size': 'x-small'})

    axes[0, 1].set_xlim(0, 2 * time_factor * (params.timeAMP_R + params.timeAMP_A))
    axes[1, 1].set_xlim(0, 2 * time_factor * (params.timeAMP_R + params.timeAMP_A))
    axes[2, 1].set_xlim(0, 2 * time_factor * (params.timeAMP_R + params.timeAMP_A))
    render_ticks(axes[0, 1], 'large')
    render_ticks(axes[1, 1], 'large')
    render_ticks(axes[2, 1], 'large')

    print(molecule.rho_GECI.diagonal()[4:].sum().real)
    # print(molecule.rho_ChR2.diagonal()[4:].sum().real)
    print(molecule.rho_GEVI.diagonal()[4:].sum().real)
    # print(molecule.rho_ChR2.diagonal()[4:].sum().real / molecule.rho_GECI.diagonal()[4:].sum().real)
    print(molecule.rho_GEVI.diagonal()[4:].sum().real / molecule.rho_GECI.diagonal()[4:].sum().real)

    # for i in range(1, 3):
    #     axes[i][1].set_yticklabels([])
    for i in range(2):
        axes[i][0].set_xticklabels([])
        axes[i][1].set_xticklabels([])

    axes[0, 1].yaxis.set_ticks_position('right')
    axes[1, 1].yaxis.set_ticks_position('right')
    axes[2, 1].yaxis.set_ticks_position('right')

    # ---------- GEVI vs GECI ----------------- #
    del molecule

    # params.control_guess=np.asarray([0.000316697, 0.000319296, 0.0167604, 0.00714847, 0.0856847, 600])  # ChR2-GECI-----14.
    # params.control_lower=np.asarray([0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_ChR2[3] * 0.990, 1239.84 * energy_factor / 550.5, 600])
    # params.control_upper=np.asarray([0.001, 0.001, 1.15 * energy_factor, Raman_levels_ChR2[3] * 1.010, 1239.84 * energy_factor / 476.5, 600])

    params.timeAMP_R = 61120
    params.timeDIM_R = 15000
    params.timeAMP_A = 5960
    params.control_guess = np.asarray([0.000287981, 0.000274194, 0.0358808, 0.00717261, 0.0851248, 600, 5960.78, 61120.7])  # GECI-GEVI-----7.10967
    params.control_lower = np.asarray([0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_GEVI[3] * 0.990, 1239.84 * energy_factor / 557.5, 600, 5000, 50000])
    params.control_upper = np.asarray([0.001, 0.001, 1.15 * energy_factor, Raman_levels_GEVI[3] * 1.010, 1239.84 * energy_factor / 496.5, 600, 8000, 80000])
    params.max_iter_control = 1

    # params.timeDIM_R = 1
    # params.timeAMP_R = 1
    # params.timeDIM_A = 20000
    # params.timeAMP_A = 40000
    #
    # params.control_guess = np.asarray([0, 0.000154152, 0, 0, 1239.84 * energy_factor / 490.678895])  # GECI-GEVI__excitation-----1.39
    # params.control_lower = np.asarray([0, 0.00005, 0, 0, 1239.84 * energy_factor / 557.5])
    # params.control_upper = np.asarray([0, 0.0003, 0, 0, 1239.84 * energy_factor / 476.5])

    molecule = RamanOpticalControl(params, **Systems)
    molecule.control_molB_over_molA(params)

    molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    time_axis = time_factor * (molecule.time_R.max() + np.concatenate((molecule.time_R, molecule.time_A)))
    time_R = time_factor * (molecule.time_R.max() + molecule.time_R)
    time_A = time_factor * (molecule.time_R.max() + molecule.time_A)

    axes[0, 2].plot(time_factor * (molecule.time_R.max() + molecule.time_R), 3.55e7 * molecule.field_R.real.max() * molecule.field_R.real, 'k', linewidth=1.)
    axes[0, 2].plot(time_factor * (molecule.time_R.max() + molecule.time_A), 3.55e7 * molecule.field_A.real.max() * molecule.field_A.real,
                    'darkblue', linewidth=1.)

    dyn_plot(axes[1, 2], time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, '')
    # dyn_plot(axes[2, 2], time_R, time_A, molecule.dyn_rho_R_ChR2.real, molecule.dyn_rho_A_ChR2.real, '')
    dyn_plot(axes[2, 2], time_R, time_A, molecule.dyn_rho_R_GEVI.real, molecule.dyn_rho_A_GEVI.real, '')

    axes[1, 2].plot(time_R, molecule.dyn_rho_R_GECI.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    axes[1, 2].plot(time_A, molecule.dyn_rho_A_GECI.real[3], 'r', linewidth=2.5)
    axes[2, 2].plot(time_R, molecule.dyn_rho_R_GEVI.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    axes[2, 2].plot(time_A, molecule.dyn_rho_A_GEVI.real[3], 'r', linewidth=2.5)
    # axes[2, 2].plot(time_R, molecule.dyn_rho_R_ChR2.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    # axes[2, 2].plot(time_A, molecule.dyn_rho_A_ChR2.real[3], 'r', linewidth=2.5)

    axes[2, 2].set_xlabel('Time (in ps)', fontweight='bold', fontsize='medium')
    axes[0, 2].ticklabel_format(style='sci', scilimits=(0, 3))

    axes[1, 2].legend(loc=6, prop={'weight': 'normal', 'size': 'x-small'})
    axes[2, 2].legend(loc=6, prop={'weight': 'normal', 'size': 'x-small'})

    axes[0, 2].set_xlim(0, 2 * time_factor * (params.timeAMP_R + params.timeAMP_A))
    axes[1, 2].set_xlim(0, 2 * time_factor * (params.timeAMP_R + params.timeAMP_A))
    axes[2, 2].set_xlim(0, 2 * time_factor * (params.timeAMP_R + params.timeAMP_A))
    render_ticks(axes[0, 2], 'large')
    render_ticks(axes[1, 2], 'large')
    render_ticks(axes[2, 2], 'large')

    print(molecule.rho_GECI.diagonal()[4:].sum().real)
    # print(molecule.rho_ChR2.diagonal()[4:].sum().real)
    print(molecule.rho_GEVI.diagonal()[4:].sum().real)
    # print(molecule.rho_ChR2.diagonal()[4:].sum().real / molecule.rho_GECI.diagonal()[4:].sum().real)
    print(molecule.rho_GEVI.diagonal()[4:].sum().real / molecule.rho_GECI.diagonal()[4:].sum().real)

    # for i in range(1, 3):
    #     axes[i][1].set_yticklabels([])
    for i in range(2):
        axes[i][0].set_xticklabels([])
        axes[i][1].set_xticklabels([])
        axes[i][2].set_xticklabels([])

    axes[0, 2].yaxis.set_ticks_position('right')
    axes[1, 2].yaxis.set_ticks_position('right')
    axes[2, 2].yaxis.set_ticks_position('right')

    fig.subplots_adjust(bottom=0.15, top=0.96, left=0.15, hspace=0.1, wspace=0.2)
    plt.savefig('GECI-GEVI.eps', format="eps")
    plt.show()