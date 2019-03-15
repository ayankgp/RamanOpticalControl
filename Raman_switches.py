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

        self.matrix_gamma_dep_ChR2 = np.ascontiguousarray(self.matrix_gamma_dep)
        self.matrix_gamma_dep_BLUF = np.ascontiguousarray(self.matrix_gamma_dep)

        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(params.rho_0)

        self.rho_ChR2 = np.ascontiguousarray(params.rho_0.copy())
        self.rho_BLUF = np.ascontiguousarray(params.rho_0.copy())

        self.energies_ChR2 = np.ascontiguousarray(self.energies_ChR2)
        self.energies_BLUF = np.ascontiguousarray(self.energies_BLUF)

        self.N = len(self.energies_ChR2)

        self.abs_spectra_ChR2 = np.ascontiguousarray(np.zeros(len(self.frequency_A_ChR2)))
        self.abs_spectra_BLUF = np.ascontiguousarray(np.zeros(len(self.frequency_A_BLUF)))

        self.abs_dist_ChR2 = np.ascontiguousarray(np.empty((len(self.prob_ChR2), len(self.frequency_A_ChR2))))
        self.abs_dist_BLUF = np.ascontiguousarray(np.empty((len(self.prob_BLUF), len(self.frequency_A_BLUF))))

        self.dyn_rho_A_ChR2 = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)
        self.dyn_rho_A_BLUF = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)

        self.dyn_rho_R_ChR2 = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)
        self.dyn_rho_R_BLUF = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)

    def create_molecules(self, ChR2, BLUF):
        """
        Creates molecules from class parameters
        """
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
        spectra_params.nDIM = len(self.energies_ChR2)
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
        spectra_params.prob_guess_num = len(self.prob_ChR2)
        spectra_params.spectra_lower = params.spectra_lower.ctypes.data_as(POINTER(c_double))
        spectra_params.spectra_upper = params.spectra_upper.ctypes.data_as(POINTER(c_double))
        spectra_params.max_iter = params.max_iter
        spectra_params.control_guess = params.control_guess.ctypes.data_as(POINTER(c_double))
        spectra_params.control_lower = params.control_lower.ctypes.data_as(POINTER(c_double))
        spectra_params.control_upper = params.control_upper.ctypes.data_as(POINTER(c_double))
        spectra_params.guess_num = len(params.control_guess)
        spectra_params.max_iter_control = params.max_iter_control

    def calculate_spectra(self, params):
        ChR2 = Molecule()
        BLUF = Molecule()
        self.create_molecules(ChR2, BLUF)
        params_spectra = Parameters()
        self.create_parameters_spectra(params_spectra, params)

        # CalculateSpectra(ChR2, params_spectra)
        # CalculateSpectra(BLUF, params_spectra)

        CalculateControl(ChR2, BLUF, params_spectra)
        # CalculateControl(BLUF, ChR2, params_spectra)


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
        which='both', direction='in', width=1.25, labelrotation=0, labelsize=labelsize)
    axis.get_yaxis().set_tick_params(
        which='both', direction='in', width=1.25, labelcolor='k', labelsize=labelsize)
    axis.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.5, b=None, which='both', axis='both')


def dyn_plot(axes, time_R, time_A, dyn_rho_R, dyn_rho_A, mol_str):

    axes.plot(time_R, dyn_rho_R[0], 'b', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[0], 'b', label='g1', linewidth=2.5)
    axes.plot(time_R, dyn_rho_R[3], 'r', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[3], 'r', label='g4', linewidth=2.5)
    axes.plot(time_R, dyn_rho_R[4:].sum(axis=0), 'k', label='EXC', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[4:].sum(axis=0), 'k', linewidth=2.5)
    axes.set_ylabel(mol_str, fontweight='bold', fontsize='medium')


if __name__ == '__main__':

    import matplotlib.pyplot as plt
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
    wavelength_ChR2, absorption_ChR2 = get_experimental_spectra('Data/ChR2_2.csv')
    wavelength_BLUF, absorption_BLUF = get_experimental_spectra("Data/BLUF.csv")

    absorption_ChR2 = savgol_filter(absorption_ChR2, 15, 3)
    absorption_BLUF = savgol_filter(absorption_BLUF, 5, 3)

    frequency_A_ChR2 = wavelength_freq_factor * energy_factor / wavelength_ChR2
    frequency_A_BLUF = wavelength_freq_factor * energy_factor / wavelength_BLUF

    # ---------------------------------------------------------------------------- #
    #                      GENERATE MOLECULE PARAMETERS AND MATRICES               #
    # ---------------------------------------------------------------------------- #

    #  ----------------------------- MOLECULAR CONSTANTS ------------------------  #

    N = 8                                   # NUMBER OF ENERGY LEVELS PER SYSTEM
    M = 11                                  # NUMBER OF SYSTEMS PER ENSEMBLE

    N_vib = 4                               # NUMBER OF VIBRATIONAL ENERGY LEVELS IN THE GROUND STATE
    N_exc = N - N_vib                       # NUMBER OF VIBRATIONAL ENERGY LEVELS IN THE EXCITED STATE

    mu_value = 5.                           # VALUE OF TRANSITION DIPOLE MATRIX ELEMENTS IN DEBYE
    gamma_pd = 2.418884e-8                  # POPULATION DECAY GAMMA
    gamma_dep_ChR2 = 2. * 2.418884e-4     # DEPHASING GAMMA FOR ChR2
    gamma_dep_BLUF = 2.25 * 2.418884e-4     # DEPHASING GAMMA FOR BLUF
    gamma_vib = 0.1 * 2.418884e-5           # VIBRATIONAL DEPHASING GAMMA

    #  ------------------------ MOLECULAR MATRICES & VECTORS --------------------  #

    energies_ChR2 = np.empty(N)
    energies_BLUF = np.empty(N)

    levels_ChR2 = np.asarray(1239.84 * energy_factor / np.linspace(370, 540, 4 * M)[::-1])  # ChR2
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

    prob_ChR2 = np.asarray([0.00581433, 0.02331881, 0.0646026, 0.10622365, 0.15318182, 0.15485174, 0.15485035, 0.12426589, 0.0928793, 0.06561301, 0.05439849])  # ChR2
    prob_BLUF = np.asarray([0.48651200, 0.65047700, 0.7934770, 0.74490600, 0.61047700, 0.49550200, 0.99057400, 1.00000000, 0.9994110, 0.66417900, 0.50762000])  # BLUF

    spectra_lower = np.zeros(M)
    spectra_upper = np.ones(M)

    Raman_levels_BLUF = np.asarray([0, 1000, 1200, 1600]) * energy_factor * cm_inv2eV_factor
    Raman_levels_ChR2 = np.asarray([0, 1000, 1200, 1600]) * energy_factor * cm_inv2eV_factor * 0.985

    params = ADict(

        N_exc=N_exc,
        num_threads=cpu_count(),

        energy_factor=energy_factor,
        time_factor=time_factor,
        rho_0=rho_0,

        timeDIM_R=15000,
        timeAMP_R=75000,
        timeDIM_A=3000,
        timeAMP_A=6000,

        # timeDIM_R=15000,
        # timeAMP_R=62500,
        # timeDIM_A=4000,
        # timeAMP_A=12000,

        # timeDIM_A=250,
        # timeAMP_A=1500,

        field_amp_R=0.000235,
        field_amp_A=0.00008,
        # field_amp_A=0.0000018,

        omega_R=0.75 * energy_factor,
        omega_v=Raman_levels_ChR2[3]*0.996,
        omega_e=1239.84*energy_factor/545,

        spectra_lower=spectra_lower,
        spectra_upper=spectra_upper,

        max_iter=1,

        control_guess=np.asarray([0.000212709, 0.000169697, 0.0311948, 0.00715384, 1239.84*energy_factor/511.849007]),  # ChR2-BLUF
        control_lower=np.asarray([0.000160, 0.00005, 0.35 * energy_factor, Raman_levels_ChR2[3]*0.990, 1239.84*energy_factor/545]),
        control_upper=np.asarray([0.000400, 0.00018, 1.15 * energy_factor, Raman_levels_ChR2[3]*1.010, 1239.84*energy_factor/475]),

        max_iter_control=100,
    )

    Systems = dict(
        # Constant Parameters
        matrix_gamma_pd=matrix_gamma_pd,
        matrix_gamma_dep=matrix_gamma_dep,
        mu=mu,

        # ChR2 molecule
        energies_ChR2=energies_ChR2,
        gamma_dep_ChR2=gamma_dep_ChR2,
        prob_ChR2=prob_ChR2,
        frequency_A_ChR2=np.ascontiguousarray(frequency_A_ChR2),
        ref_spectra_ChR2=np.ascontiguousarray(absorption_ChR2),
        Raman_levels_ChR2=Raman_levels_ChR2,
        levels_ChR2=levels_ChR2,

        # BLUF molecule
        energies_BLUF=energies_BLUF,
        gamma_dep_BLUF=gamma_dep_BLUF,
        prob_BLUF=prob_BLUF,
        frequency_A_BLUF=np.ascontiguousarray(frequency_A_BLUF),
        ref_spectra_BLUF=np.ascontiguousarray(absorption_BLUF),
        Raman_levels_BLUF=Raman_levels_BLUF,
        levels_BLUF=levels_BLUF
    )

    np.set_printoptions(precision=6)
    molecule = RamanOpticalControl(params, **Systems)
    molecule.calculate_spectra(params)

    # fig_spectra, axes = plt.subplots(nrows=1, ncols=1)
    #
    # axes.plot(energy_factor * 1239.84 / molecule.frequency_A_ChR2, molecule.abs_spectra_ChR2, 'r*-', label='Fitted ChR2', linewidth=1.)
    # axes.plot(energy_factor * 1239.84 / molecule.frequency_A_ChR2, molecule.ref_spectra_ChR2, 'k-', label='Actual ChR2', linewidth=2.)
    #
    # axes.plot(energy_factor * 1239.84 / molecule.frequency_A_BLUF, molecule.abs_spectra_BLUF, 'b*-', label='Fitted BLUF', linewidth=1.)
    # axes.plot(energy_factor * 1239.84 / molecule.frequency_A_BLUF, molecule.ref_spectra_BLUF, 'g-', label='Actual BLUF', linewidth=2.)
    #
    # axes.set_xlabel('Wavelength (in nm)', fontweight='bold')
    # axes.set_ylabel('Normalized spectra', fontweight='bold')
    # plt.legend()
    # render_ticks(axes, 'large')

    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(8, 4.5))

    molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    time_axis = time_factor * (molecule.time_R.max() + np.concatenate((molecule.time_R, molecule.time_A)))
    time_R = time_factor * (molecule.time_R.max() + molecule.time_R)
    time_A = time_factor * (molecule.time_R.max() + molecule.time_A)

    axes[0, 0].plot(time_factor * (molecule.time_R.max() + molecule.time_R), 5.142e3 * molecule.field_R.real, 'k', linewidth=1.)
    axes[0, 0].plot(time_factor * (molecule.time_R.max() + molecule.time_A), 5.142e3 * molecule.field_A.real, 'darkblue', linewidth=1.)

    dyn_plot(axes[1, 0], time_R, time_A, molecule.dyn_rho_R_ChR2.real, molecule.dyn_rho_A_ChR2.real, 'Population \n ChR2')
    dyn_plot(axes[2, 0], time_R, time_A, molecule.dyn_rho_R_BLUF.real, molecule.dyn_rho_A_BLUF.real, 'Population \n BLUF')

    axes[2, 0].set_xlabel('Time (in ps)', fontweight='bold')
    axes[0, 0].set_ylabel('Electric field \n (in MV/cm)', fontweight='bold')
    axes[0, 0].ticklabel_format(style='sci', scilimits=(0, 3))

    axes[1, 0].legend(loc=6, fontsize='x-small')
    axes[2, 0].legend(loc=6, fontsize='x-small')

    axes[0, 0].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))
    axes[1, 0].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))
    axes[2, 0].set_xlim(0, 2*time_factor*(params.timeAMP_R + params.timeAMP_A))
    render_ticks(axes[0, 0], 'large')
    render_ticks(axes[1, 0], 'large')
    render_ticks(axes[2, 0], 'large')

    print(molecule.rho_ChR2.diagonal()[4:].sum().real)
    print(molecule.rho_BLUF.diagonal()[4:].sum().real)
    print(molecule.rho_ChR2.diagonal()[4:].sum().real/molecule.rho_BLUF.diagonal()[4:].sum().real)

    # del molecule
    #
    # params.control_guess = np.asarray([0.000201628, 0.000131625, 0.0422529, 0.00727176, 1239.84*energy_factor/516.654698])  # BLUF-ChR2
    # params.control_lower = np.asarray([0.000160, 0.00005, 0.35 * energy_factor, Raman_levels_BLUF[3]*0.990, 1239.84*energy_factor/545])
    # params.control_upper = np.asarray([0.000400, 0.00018, 1.15 * energy_factor, Raman_levels_BLUF[3]*1.010, 1239.84*energy_factor/475])
    #
    # molecule = RamanOpticalControl(params, **Systems)
    # molecule.calculate_spectra(params)
    #
    # molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    # time_axis = time_factor * (molecule.time_R.max() + np.concatenate((molecule.time_R, molecule.time_A)))
    # time_R = time_factor * (molecule.time_R.max() + molecule.time_R)
    # time_A = time_factor * (molecule.time_R.max() + molecule.time_A)
    #
    # axes[0, 1].plot(time_factor * (molecule.time_R.max() + molecule.time_R), 5.142e3 * molecule.field_R.real, 'k',
    #                 linewidth=1.)
    # axes[0, 1].plot(time_factor * (molecule.time_R.max() + molecule.time_A), 5.142e3 * molecule.field_A.real,
    #                 'darkblue', linewidth=1.)
    #
    # dyn_plot(axes[1, 1], time_R, time_A, molecule.dyn_rho_R_ChR2.real, molecule.dyn_rho_A_ChR2.real, '')
    # dyn_plot(axes[2, 1], time_R, time_A, molecule.dyn_rho_R_BLUF.real, molecule.dyn_rho_A_BLUF.real, '')
    #
    # axes[2, 1].set_xlabel('Time (in ps)', fontweight='bold')
    # axes[0, 1].ticklabel_format(style='sci', scilimits=(0, 3))
    #
    # axes[1, 1].legend(loc=6, fontsize='x-small')
    # axes[2, 1].legend(loc=6, fontsize='x-small')
    #
    # axes[0, 1].set_xlim(0, 2 * time_factor * (params.timeAMP_R + params.timeAMP_A))
    # axes[1, 1].set_xlim(0, 2 * time_factor * (params.timeAMP_R + params.timeAMP_A))
    # axes[2, 1].set_xlim(0, 2 * time_factor * (params.timeAMP_R + params.timeAMP_A))
    # render_ticks(axes[0, 1], 'large')
    # render_ticks(axes[1, 1], 'large')
    # render_ticks(axes[2, 1], 'large')
    #
    # print(molecule.rho_ChR2.diagonal()[4:].sum().real)
    # print(molecule.rho_BLUF.diagonal()[4:].sum().real)
    # print(molecule.rho_BLUF.diagonal()[4:].sum().real/molecule.rho_ChR2.diagonal()[4:].sum().real)
    #
    # for i in range(3):
    #     axes[i][1].set_yticklabels([])
    #
    # fig.subplots_adjust(bottom=0.15, top=0.96, left=0.15, hspace=0.05, wspace=0.05)
    # plt.savefig('ChR2-BLUF', format="jpg")
    plt.show()
