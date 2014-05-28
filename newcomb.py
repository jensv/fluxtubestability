# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:03:00 2014

@author: Jens von der Linden
"""

#Python 3.x compatability
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins.disabled import (apply, cmp, coerce, execfile,
                                      file, long, raw_input, reduce, reload,
                                      unicode, xrange, StandardError)
from future.types import bytes, dict, int, range, str
from future.builtins.misc import (ascii, chr, hex, input, next,
                                  oct, open, round, super)
from future.builtins.iterators import filter, map, zip

#regular imports
import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import dict_convenience as dc
sys.path.append("C:\Users\jensv_000\Dropbox\You_Lab\SpyderProjects\Scripts")
import dict_convenience
import equil_solver


def f_eq(r, k, m, b_z, b_theta):
    r"""
    Return f from Newcomb's paper.

    Parameters
    ----------
    r: ndarray of floats
       radius

    k: float
       axial periodicity number

    m: float
       azimuthal periodicity number

    b_z: ndarray of floats
         axial magnetic field

    b_theta: ndarray of floats
             azimuthal mangetic field

    Returns
    -------
    f: ndarray of floats
       f from Newcomb's paper

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (16)
    """
    return r*f_num_wo_r(r, k, m, b_z, b_theta)/f_denom(r, k, m)


def f_denom(r, k, m):
    r"""
    Return denominator of f from Newcomb's paper.

    Parameters
    ----------
    r: ndarray of floats
       radius

    k: float
       axial periodicity number

    m: float
       azimuthal periodicity number

    Returns
    -------
    f_denom: ndarray of floats
       denominator of f from Newcomb's paper

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (16)
    """
    return k**2*r**2 + m**2


def f_num_wo_r(r, k, m, b_z, b_theta):
    r"""
    Return numerator of f without r from Newcomb's paper.

    Parameters
    ----------
    r: ndarray of floats
       radius

    k: float
       axial periodicity number

    m: float
       azimuthal periodicity number

    b_z: ndarray of floats
         axial magnetic field

    b_theta: ndarray of floats
             azimuthal mangetic field

    Returns
    -------
    f_num_wo_r: ndarray of floats
       numerator of f without r from Newcomb's paper

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (16)
    """
    return (k*r*b_z + m*b_theta)**2


def g_eq_18(r, k, m, b_z, b_theta, p_prime):
    r"""
    Return g from Newcomb's paper.

    Parameters
    ----------
    r: ndarray of floats
       radius

    k: float
       axial periodicity number

    m: float
       azimuthal periodicity number

    b_z: ndarray of floats
         axial magnetic field

    b_theta: ndarray of floats
             azimuthal mangetic field

    p_prime: ndarray of floats


    Returns
    -------
    g: ndarray of floats
       g from Newcomb's paper

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (18)

    Notes
    -----
    Equation (17) is harder to implement due to derivative of Newcomb f.
    """
    term1 = 2*k**2*r**2/(f_denom(r, k, m)) * p_prime
    term2 = (1/r*f_num_wo_r(r, k, m, b_z, b_theta)*(k**2*r**2+m**2-1) /
             f_denom(r, k, m))
    term3 = 2*k**2*r/f_denom(r, k, m)**2*(k**2*r**2*b_z**2-m**2*b_theta**2)
    return term1 + term2 + term3


def splines(a=1, points=500, q0=1.0, k=1, b_z0=1):
    r"""

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    Reference
    ---------

    Example
    -------

    """
    equilibrium = equil_solver.Parabolic_nu2(a, points, q0, k, b_z0)
    return equilibrium.get_splines()


def newcomb_h():
    pass


def newcomb_der(t, y, equil_dict):
    r"""

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    Reference
    ---------

    Example
    -------

    """
    y_prime[0] = y[1]
    y_prime[1] = y[0]*g_eq_18(t, k, m, b_z, b_theta, p_prime)
    return y_prime


def newcomb_der_divide_f(t, y, equil_dict):
    r"""

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    Reference
    ---------

    Example
    -------

    """

    y_prime[0] = y[1]/f_eq(t, k, m, b_z, b_theta)
    y_prime[1] = y[0]*(g_eq_18(t, k, m, b_z, b_theta, p_prime)
                       / f_eq(t, k, m, b_z, b_theta))
    return y_prime


def newcomb_int(divide_f, atol, rtol, rmax, dr, splines):
    r"""
    Integrate Newcomb's Euler Lagrange equation as two odes.

    Parameters
    ----------
    divide_f: bool
              determines which newcomb_der is used
    atol: float
          absolute tolerance
    rtol: float
          relative tolerance
    rmax: float
          maxium radius at which to integrate
    dr: float
        radial step-size

    Returns
    -------
    xi: ndarray of floats (2,M)
        xi and derivative of xi.

    Notes
    -----
    The seperation of the Euler-lagrange equation is based on Alan Glasser's
    cyl code.
    Newcomb's condition states that at each singularity f=0 the integration
    should start from 0.

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (23)
    Alan Glasser (unknown) Cyl code
    """
    xi = []
    if divide_f:
        xi_int = integrate.ode(newcomb_der_divide_f)
    else:
        xi_int = integrate.ode(newcomb_der)

    xi_int.set_integrator('lsoda', atol, rtol)
    xi_int.set_initial_value()
    xi_int.set_f_params(f, g)

    while xi_int.successful() and xi_int.t < r_max-dr:
        xi_int.integrate(xi_int.t + dr)
        xi.append = xi
        crossing_condition(xi)
    return np.array(xi)


def suydam():
    r"""
    Parameters
    ----------
    Returns
    -------
    Notes
    -----
    Reference
    ---------
    Example
    -------
    """
    return r/8*Bz*(qprime/q)**2+pprime


def check_suydam():
    r"""
    Parameters
    ----------
    Returns
    -------
    Notes
    -----
    Reference
    ---------
    Example
    -------
    """
    pass


def check_sing():
    r"""
    Parameters
    ----------
    Returns
    -------
    Notes
    -----
    Reference
    ---------
    Example
    -------
    """
    return k*r*Bz + m*B_theta


def crossing_condition(xi):
    """
    """
    return xi[len(xi)-2]*xi[len(xi)-1] < 0
