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
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        self.matrix_gamma_dep = np.ascontiguousarray(self.matrix_gamma_dep)

        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(params.rho_0)
        self.rho = np.ascontiguousarray(params.rho_0.copy())
        self.energies = np.ascontiguousarray(self.energies)

        self.N = len(self.energies)

        self.dyn_rho_A = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)
        self.dyn_rho_R = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)

    def create_molecules(self, TwoLevel):
        """
        Creates molecules from class parameters
        """
        #  ----------------------------- CREATING GECI ------------------------  #

        TwoLevel.nDIM = len(self.energies)
        TwoLevel.energies = self.energies.ctypes.data_as(POINTER(c_double))
        TwoLevel.matrix_gamma_pd = self.matrix_gamma_pd.ctypes.data_as(POINTER(c_double))
        TwoLevel.matrix_gamma_dep = self.matrix_gamma_dep.ctypes.data_as(POINTER(c_double))
        TwoLevel.gamma_dep = self.gamma_dep
        TwoLevel.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        TwoLevel.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        TwoLevel.d_mu_dx = self.d_mu_dx
        TwoLevel.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        TwoLevel.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))

        TwoLevel.rho = self.rho.ctypes.data_as(POINTER(c_complex))
        TwoLevel.dyn_rho_A = self.dyn_rho_A.ctypes.data_as(POINTER(c_complex))
        TwoLevel.dyn_rho_R = self.dyn_rho_R.ctypes.data_as(POINTER(c_complex))

    def create_parameters_spectra(self, spectra_params, params):
        """
        Creates parameters from class parameters
        """
        spectra_params.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        spectra_params.nDIM = len(self.energies)
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

    def calculate_spectra(self, params):
        TwoLevel = Molecule()
        params_spectra = Parameters()
        self.create_molecules(TwoLevel)
        self.create_parameters_spectra(params_spectra, params)

        RamanTransfer(TwoLevel, params_spectra)


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


def dyn_plot(axes, time_R, dyn_rho_R, mol_str):

    axes.plot(time_R, dyn_rho_R[0], 'b', label='$\\rho_{g, \\nu=0}$', linewidth=1.)
    axes.plot(time_R, dyn_rho_R[1], 'k', label='$\\rho_{e, total}$', linewidth=1.)
    axes.set_ylabel(mol_str, fontweight='bold', fontsize='medium')


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import pickle
    cm_inv2eV_factor = 0.00012398

    # ---------------------------------------------------------------------------- #
    #                             LIST OF CONSTANTS                                #
    # ---------------------------------------------------------------------------- #
    energy_factor = 1. / 27.211385
    time_factor = .02418884 / 1000
    wavelength_freq_factor = 1239.84

    # ---------------------------------------------------------------------------- #
    #                      GENERATE MOLECULE PARAMETERS AND MATRICES               #
    # ---------------------------------------------------------------------------- #

    #  ----------------------------- MOLECULAR CONSTANTS ------------------------  #

    N = 2  # NUMBER OF ENERGY LEVELS PER SYSTEM
    M = 1  # NUMBER OF SYSTEMS PER ENSEMBLE

    mu_value = 5.  # VALUE OF TRANSITION DIPOLE MATRIX ELEMENTS IN DEBYE
    d_mu_dx = 2.14
    gamma_pd = 2.418884e-8  # POPULATION DECAY GAMMA
    gamma_dep = 2.00 * 2.418884e-4  # DEPHASING GAMMA FOR GECI
    gamma_vib = 6.67 * 2.418884e-5  # VIBRATIONAL DEPHASING GAMMA

    #  ------------------------ MOLECULAR MATRICES & VECTORS --------------------  #

    energies = np.asarray([0, 1440]) * energy_factor * cm_inv2eV_factor

    rho_0 = np.zeros((N, N), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j

    mu = mu_value * np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)

    matrix_gamma_pd = np.ones((N, N)) * gamma_pd
    np.fill_diagonal(matrix_gamma_pd, 0.0)
    matrix_gamma_pd = np.tril(matrix_gamma_pd).T

    matrix_gamma_dep = np.ones_like(matrix_gamma_pd) * gamma_vib
    np.fill_diagonal(matrix_gamma_dep, 0.0)

    spectra_lower = np.zeros(M)
    spectra_upper = np.ones(M)

    params = ADict(

        num_threads=cpu_count(),

        energy_factor=energy_factor,
        time_factor=time_factor,
        rho_0=rho_0,

        timeDIM_R=20000,
        timeAMP_R=20000,
        timeDIM_A=4000,
        timeAMP_A=12000,

        field_amp_R=0.0015,
        field_amp_A=0.00008,

        omega_R=0.5 * energy_factor,
        omega_v=energies[1] * 0.996,
        omega_e=1239.84 * energy_factor / 545,
    )

    Systems = dict(
        # Constant Parameters
        matrix_gamma_pd=matrix_gamma_pd,
        matrix_gamma_dep=matrix_gamma_dep,
        mu=mu,
        d_mu_dx=d_mu_dx,

        # GECI molecule
        energies=energies,
        gamma_dep=gamma_dep
    )

    np.set_printoptions(precision=6)
    molecule = RamanOpticalControl(params, **Systems)
    print(molecule.energies)
    molecule.calculate_spectra(params)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 4.5))

    time_R = time_factor * (molecule.time_R.max() + molecule.time_R)

    axes[0].plot(time_R, 3.55e7 * molecule.field_R.real.max() * molecule.field_R.real, 'k', linewidth=1.)

    dyn_plot(axes[1], time_R, molecule.dyn_rho_R.real, 'Population \n TwoLevel')

    axes[1].plot(time_R, molecule.dyn_rho_R.real[1], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)

    axes[1].set_xlabel('Time (in ps)', fontweight='bold', fontsize='medium')
    axes[0].set_ylabel('Electric field \n (in $GW/cm^2$)', fontweight='bold')
    axes[0].ticklabel_format(style='sci', scilimits=(0, 3))

    axes[1].legend(loc=6, prop={'weight': 'normal', 'size': 'small'})

    axes[0].set_xlim(0, 2*time_factor*params.timeAMP_R)
    axes[1].set_xlim(0, 2*time_factor*params.timeAMP_R)
    render_ticks(axes[0], 'large')
    render_ticks(axes[1], 'large')

    print(molecule.rho.diagonal().real)

    axes[0].set_xticklabels([])

    axes[0].yaxis.set_ticks_position('right')
    axes[1].yaxis.set_ticks_position('right')

    fig.subplots_adjust(bottom=0.15, top=0.96, left=0.15, hspace=0.05, wspace=0.05)
    del molecule

    T_size = 20
    Amp_size = 20

    data = np.empty([T_size, Amp_size])
    data1 = np.empty([T_size, Amp_size])
    time_x = np.linspace(2000, 400000, T_size)
    field_y = np.linspace(0.0011, 0.0045, Amp_size)
    for n_i, i in enumerate(time_x):
        for n_j, j in enumerate(field_y):

            params.timeDIM_R = int(i)
            params.timeAMP_R = int(i)
            params.field_amp_R = j

            molecule = RamanOpticalControl(params, **Systems)
            molecule.calculate_spectra(params)
            data[n_i, n_j] = molecule.rho.diagonal()[1]
            data1 = time_x[:, np.newaxis]*time_factor*3.55e7*field_y[np.newaxis, :]**2

            if (time_x[n_i]*time_factor*3.55e7*field_y[n_j]**2 > 1000):
                data[n_i, n_j] = 0.99

            del molecule

    print(data)

    X, Y = np.meshgrid(time_x*time_factor, 3.55e7*field_y**2)

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(9., 9.))
    levels = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0])
    levels_line = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    im = axes.contour(X, Y, data, levels_line, origin='lower', cmap='RdBu', linewidths=2., linestyles='solid')
    im = axes.contourf(X, Y, data, levels, cmap='RdYlBu', origin='lower')


    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='7%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax)

    axes.set_xlabel('Time (ps)', fontsize='x-large', fontweight='bold')
    axes.set_ylabel('Electric field (GW/cm2)', fontsize='x-large', fontweight='bold')

    render_ticks(axes, 'x-large')
    axes.grid(color='k', linestyle='-.', linewidth=.1, alpha=1., which='both', axis='both')
    plt.subplots_adjust(left=0.14, bottom=0.22, hspace=0.0, wspace=0.05)
    plt.savefig('RamanTransfer.png', format='png')
    plt.show()
