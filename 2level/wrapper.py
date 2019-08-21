import os
import ctypes
from ctypes import c_int, c_double, POINTER, Structure


__doc__ = """
Python wrapper for RamanOptogenetics.c
Compile with:
gcc -O3 -shared -o RamanOptogenetics.so RamanOptogenetics.c -lm -fopenmp -fPIC
"""


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [
        ('real', c_double),
        ('imag', c_double)
    ]


class Parameters(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [

        ('rho_0', POINTER(c_complex)),
        ('nDIM', c_int),
        ('time_A', POINTER(c_double)),
        ('time_R', POINTER(c_double)),
        ('timeDIM_A', c_int),
        ('timeDIM_R', c_int),
        ('field_amp_A', c_double),
        ('field_amp_R', c_double),
        ('omega_R', c_double),
        ('omega_v', c_double),
        ('omega_e', c_double),
        ('thread_num', c_int)
    ]


class Molecule(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('nDIM', c_int),
        ('energies', POINTER(c_double)),
        ('matrix_gamma_pd', POINTER(c_double)),
        ('matrix_gamma_dep', POINTER(c_double)),
        ('gamma_dep', c_double),
        ('rho_0', POINTER(c_complex)),
        ('mu', POINTER(c_complex)),
        ('d_mu_dx', c_double),
        ('field_A', POINTER(c_complex)),
        ('field_R', POINTER(c_complex)),
        ('rho', POINTER(c_complex)),
        ('dyn_rho_A', POINTER(c_complex)),
        ('dyn_rho_R', POINTER(c_complex)),
    ]


try:
    # Load the shared library assuming that it is in the same directory
    lib1 = ctypes.cdll.LoadLibrary(os.getcwd() + "/RamanOpticalControl.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o RamanOpticalControl.so RamanOpticalControl.c -lm -lnlopt -fopenmp -fPIC
        """
    )


lib1.RamanTransfer.argtypes = (
    POINTER(Molecule),          # molecule molA
    POINTER(Parameters),        # parameter field_params
)
lib1.RamanTransfer.restype = POINTER(c_complex)


def RamanTransfer(mol, params):
    return lib1.RamanTransfer(
        mol,
        params
    )
