# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:03:00 2014

@author: Jens von der Linden
"""


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatability"""


import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import dict_convenience as dc
import sys
sys.path.append("../Scripts")
import dict_convenience
import equil_solver


def f_eq(r, k, m, b_z, b_theta):
    r"""
    Return f from Newcomb's paper.

    Parameters
    ----------
    r : ndarray of floats
        radius

    k : float
        axial periodicity number

    m : float
        azimuthal periodicity number

    b_z : ndarray of floats
        axial magnetic field

    b_theta : ndarray of floats
        azimuthal mangetic field

    Returns
    -------
    f : ndarray of floats
        f from Newcomb's paper

    Notes
    -----
    Implements equation:
    :math:`f = \frac{r (k r B_{z}+ m B_{\theta})^2}{k^{2} r^{2} + m^{2}}`


    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (16)
    """
    f_eq()
    return r*f_num_wo_r(r, k, m, b_z, b_theta)/f_denom(r, k, m)


def f_denom(r, k, m):
    r"""
    Return denominator of f from Newcomb's paper.

    Parameters
    ----------
    r : ndarray of floats
        radius

    k : float
        axial periodicity number

    m : float
        azimuthal periodicity number

    Returns
    -------
    f_denom : ndarray of floats
       denominator of f from Newcomb's paper

    Notes
    -----
    f denominator :math:`k^{2}r^{2}+m^{2}`

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
    r : ndarray of floats
        radius

    k : float
        axial periodicity number

    m : float
        azimuthal periodicity number

    b_z : ndarray of floats
        axial magnetic field

    b_theta : ndarray of floats
        azimuthal mangetic field

    Returns
    -------
    f_num_wo_r : ndarray of floats
        numerator of f without r from Newcomb's paper

    Notes
    -----
    f numerator without r: :math:`(k r B_{z}+m B_{\theta})^{2}`

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (16)
    """
    return (k*r*b_z(r) + m*b_theta(r))**2


def g_eq_18(r, k, m, b_z, b_theta, p_prime):
    r"""
    Return g from Newcomb's paper.

    Parameters
    ----------
    r : ndarray of floats
        radius

    k : float
        axial periodicity number

    m : float
        azimuthal periodicity number

    b_z : ndarray of floats
        axial magnetic field

    b_theta : ndarray of floats
        azimuthal mangetic field

    p_prime : ndarray of floats
        pressure prime

    Returns
    -------
    g : ndarray of floats
        g from Newcomb's paper

    Notes
    -----
    Implements equation
    .. math::
        \frac{2 k^{2} r^{2}}{k^{2} r^{2} + m^{2}} \ frac{dP}{dr} +
        \frac{1}{r}(k r B_{z}+m B_{\theta})^{2}
        \frac{k^{2}r^{2}+m^{2}-1}{k^{2}r^{2}+m^{2}}+
        \frac{2k^{2}r}{(k^{2}r^{2}+m^{2})^{2}}
        (k^{2}r^{2}B_{z}^{2}-m^{2}B_{theta}^{2})

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (18)

    Notes
    -----
    Equation (17) is harder to implement due to derivative of Newcomb f.
    """
    term1 = 2*k**2*r**2/(f_denom(r, k, m)) * p_prime(r)
    if r == 0.:
        term2 = k*b_z(r)*(k**2*r**2+m**2-1) / f_denom(r, k, m)
    else:
        term2 = (1/r*f_num_wo_r(r, k, m, b_z, b_theta)*(k**2*r**2+m**2-1) /
                 f_denom(r, k, m))
    term3 = (2*k**2*r/f_denom(r, k, m)**2*
             (k**2*r**2*b_z(r)**2-m**2*b_theta(r)**2))
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


def newcomb_der(t, y, k, m, b_z, b_theta, p_prime):
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
    y_prime = np.zeros(2)
    y_prime[0] = y[1]
    y_prime[1] = y[0]*g_eq_18(t, k, m, b_z, b_theta, p_prime)
    return y_prime


def newcomb_der_divide_f(t, y, k, m, b_z, b_theta, p_prime):
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
    y_prime = np.zeros(2)
    y_prime[0] = y[1]/f_eq(t, k, m, b_z, b_theta)
    y_prime[1] = y[0]*(g_eq_18(t, k, m, b_z, b_theta, p_prime)
                       / f_eq(t, k, m, b_z, b_theta))
    return y_prime


def newcomb_int(divide_f, r_max, dr, params, atol=None, rtol=None):
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
    (k, m, b_z, b_theta, p_prime) = dc.retrieve(params, ['k', 'm', 'b_z',
                                                         'b_theta', 'pprime'])
    xi = []
    if divide_f:
        xi_int = integrate.ode(newcomb_der_divide_f)
    else:
        xi_int = integrate.ode(newcomb_der)

    if not (atol and rtol):
        xi_int.set_integrator('lsoda')
    else:
        xi_int.set_integrator('lsoda', atol, rtol)
    xi_int.set_initial_value([0, 0], 0)
    xi_int.set_f_params(k, m, b_z, b_theta, p_prime)

    while xi_int.successful() and xi_int.t < r_max-dr:
        xi_int.integrate(xi_int.t + dr)
        xi.append(xi_int.y)
        #crossing_condition(xi)
    return np.array(xi)


def suydam(r, b_z, q_prime, q, p_prime):
    r"""
    Returns suydam condition.

    Parameters
    ----------
    r : ndarray
        radial test points

    b_z : spline
        axial magnetic field

    qprime : spline
        derivative of safety factor

    q : spline
        safety factor

    pprime : spline
        derivative of pressure

    Returns
    -------
    suydam : ndarray


    Notes
    -----
    Returned expression can be checked for elements <=0.

    .. math:: \frac{r}{8}\frac{B_{z}}{\mu_{0}}\frac{q'}{q}^2

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    eq (5).
    Jardin (2010) Computational Mehtods in Plasma Physics. eq (8.84)
    """
    return r/8*b_z*(q_prime/q)**2+p_prime


def check_suydam(r, q_prime, q, p_prime):
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
    if ((suydam(r, q_prime, q, p_prime) <= 0).sum() == 0):
        return (False, np.array([]))
    else:
        return (True, r[(suydam(r, q_prime, q, pprime) <= 0)])


def check_sing(r, k, b_z, m, b_theta, tol):
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
    if ((np.abs(k*r*b_z + m*b_theta) <= tol).sum() == 0):
        return (False, np.array([]))
    else:
        return (True, r[np.abs(k*r*b_z + m*b_theta) <= tol])


def crossing_condition(xi):
    """
    """
    return xi[len(xi)-2]*xi[len(xi)-1] < 0
