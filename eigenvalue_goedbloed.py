# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 10:06:45 2014

@author: Jens von der Linden

Implements the Ideal MHD eigenvalue problem
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatability above"""

import scipy.integrate as inte
import scipy.special as spec
import numpy as np
import sys
sys.path.append("../Scripts")
import dict_convenience as dc

def xi_der():


def f(r, k, m, b_z, b_theta):
    r"""
    Returns F(r).

    Paramters
    ---------
    r: float
        radius at which to evaluate r
    k: float
        axial periodicity number
    m: float
        azimuthal periodicity number
    b_z: float
        axial magnetic field evaluated at r
    b_theta: float
        axial magnetic field evaluated at r

    Returns
    -------
    f : float
        F(r)

    Notes
    -----
    :math:`F(r) = k B_{z}+ \frac{m B_{\theta}}{r}`

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.25)
    """
    return k*b_z + m*b_theta/r


def g(r, k, m, b_z, b_theta):
    r"""
    Returns G(r).

    Paramters
    ---------
    r: float
        radius at which to evaluate r
    k: float
        axial periodicity number
    m: float
        azimuthal periodicity number
    b_z: float
        axial magnetic field evaluated at r
    b_theta: float
        azimuthal magnetic field evaluated at r

    Returns
    -------
    g: float
        G(r)

    Notes
    -----
    :math:`G(r) = \frac{m B_{z}}{r}+ k B_{\theta}`

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.25)
    """
    return m*b_z / r - k*b_theta


def omega_a_sq(f, rho):
    r"""
    Returns Alfven frequency squared.

    Paramters
    ---------
    f: float
        F(r) evaluated at a specific r
    rho: float
         rho evaluated at a specific r

    Returns
    -------
    omega_a_sq: float
        Alfven frequency squared

    Notes
    -----
    :math:`\omega_{alfven}^{2}(r) = \frac{F(r)^{2}}{\rho(r)}`

    Goedbloed has mu0 set to 1.

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.37)
    """
    return f**2 / rho


def omega_f0_sq(r, m, k, gamma, b_theta, b_z, rho, pressure):
    r"""
    Returns f0 frequency squared.

    Paramters
    ---------
    r: float
        radius at which to evaluate r
    k: float
        axial periodicity number
    m: float
        azimuthal periodicity number
    gamma: float
           gamma from equation of state
    b_z: float
        axial magnetic field evaluated at r
    b_theta: float
        azimuthal magnetic field evaluated at r
    rho: float
        density evaluated at r
    pressure: float
        pressure evaluated at r

    Returns
    -------
    omega_f0_sq: float
        f0 frequency squared

    Notes
    -----

    Goedbloed has mu0 set to 1.

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.37)

    """
    k_0_sq = (m**2/r**2+k**2)
    b_sq = b_theta**2 + b_z**2
    v_sound_sq = gamma*pressure / rho
    v_alfven_sq = b_sq / rho
    alpha = (4.*gamma*pressure*f**2 /
             ((m**2/r**2+k**2)*(gamma*pressure + b_sq)**2)
    if alpha > 1.:
        print "Warning: omega_f0 is complex"
    return 0.5*k_0_sq*(v_sound_sq + v_alfven_sq)*(1 + (1 - alpha)**0.5)


def omega_s0_sq(r, m, k, gamma, b_theta, b_z, rho, pressure):
    r"""
    Returns s0 frequency squared.

    Paramters
    ---------
    r: float
        radius at which to evaluate r
    k: float
        axial periodicity number
    m: float
        azimuthal periodicity number
    gamma: float
           gamma from equation of state
    b_z: float
        axial magnetic field evaluated at r
    b_theta: float
        azimuthal magnetic field evaluated at r
    rho: float
        density evaluated at r
    pressure: float
        pressure evaluated at r

    Returns
    -------
    omega_s0_sq: float
        s0 frequency squared

    Notes
    -----

    Goedbloed has mu0 set to 1.

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.37)
    """
    k_0_sq = (m**2/r**2+k**2)
    b_sq = b_theta**2 + b_z**2
    v_sound_sq = gamma*pressure / rho
    v_alfven_sq = b_sq / rho
    alpha = (4.*gamma*pressure*f**2 /
             ((m**2/r**2+k**2)*(gamma*pressure + b_sq)**2)
    if alpha > 1.:
        print "Warning: omega_s0 is complex"
    return 0.5*k_0_sq*(v_sound_sq + v_alfven_sq)*(1 - (1 - alpha)**0.5)


def omega_sound_sq(gamma, f, b_theta, b_z, rho, pressure):
    r"""
    Returns the sonic frequency.

    Paramters
    ---------
    gamma: float
           gamma from equation of state
    f: float
        F(r) evaluated at r
    b_z: float
        axial magnetic field evaluated at r
    b_theta: float
        azimuthal magnetic field evaluated at r
    rho: float
        density evaluated at r
    pressure: float
        pressure evaluated at r

    Returns
    -------
    omega_s0_sq: float
        s0 frequency squared

    Notes
    -----

    Goedbloed has mu0 set to 1.

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.37)

    """
    b_sq = b_theta**2 + b_z**2
    return gamma*pressure/(gamma*pressure+b_sq) * f**2/rho


def n_freq(gamma, b_theta, b_z, pressure, rho, omega_sq, omega_alfven_sq,
           omega_sound_sq):
    r"""
    """
    b_sq = b_theta**2 + b_z**2
    return (rho**2*(gamma*pressure + b_sq)*(omega_sq - omega_alfven_sq)
            *(omega_sq - omega_sound_sq))


def d_freq(rho, omega_sq, omega_s0_sq, omega_f0_sq):
    r"""
    """
    return rho**2*(omega_sq - omega_s0_sq)*(omega_sq - omega_f0_sq)


def n_fb(gamma, f, b_theta, b_z, pressure, rho, omega_sq):
    r"""
    """
    return (rho*omega_sq - f**2)*((gamma*rho + b_sq)
                                  *rho*omega_sq - gamma*pressure*f**2)

def d_fb(r, m, k, gamma, f, b_theta, b_z, rho, pressure, omega_sq ):
    r"""
    """
    b_sq = b_theta**2 + b_z**2
    term1 = rho**2*omega_sq**2
    term2 = -(m**2/r**2 + k**2)*(gamma*pressure + b_sq)*rho*omega_sq
    term3 = (m**2/r**2 + k**2)*gamma*rho*f**2
    return term1 + term2 + term3


def c(r, m, k, gamma, f, b_theta, b_z, rho, pressure, omega_sq):
    r"""
    """
    b_sq = b_theta**2 + b_z**2
    term1 = 2.*b_theta**2/r**2*rho**4*omega_sq**2
    term21 = 2.*m*b_theta*f/r**3
    term22 = ((gamma*pressure + b_sq)*rho*omega_sq - gamma*rho*f**2)
    return term1 + term21*term22


def e(r, m, k, gamma, f, n, b_theta, b_theta_prime, b_z, rho, pressure,
      omega_sq):
    r"""
    """
    term1 = -n/r*((rho*omega_sq-f**2)/r + b_theta_prime/r**2 - 2*b_theta/r**3)
    term2 = -4.*b_theta**4/r**4*rho**2*omega_sq**2
    term31 = 4.*b_theta**2*f**2/r**4
    term32 = ((gamma*pressure + b_sq)*rho*omega_sq - gamma*rho*f**2)
    return term1 + term2 + term31*term32


def chi_der(r, y, c, d, e, n):
    r"""
    """
    chi = y[0]
    Pi = y[1]
    chi_prime = -r/n*(c*chi + d*Pi)
    Pi_prime = -r/n*(e*chi - c*Pi)
    chi = np.array([chi_prime, Pi_prime])
    return chi

def chi_init(r_init, m, k, gamma, d, f, g, b_theta, b_z, rho, pressure,
             omega_sq):
    r"""
    """
    r = r_init
    if m == 0:
        chi_init = r**2
        chi_prime = 2*r
    else:
        chi_init = r**abs(m)
        if abs(m) == 1:
            chi_prime = 0
            chi_prime = abs(m)*r**(abs(m)-1)
    b_sq = b_theta**2 + b_z**2
    Pi_term1 = -n/(r*d)*chi_prime
    Pi_term2 = 2*b_theta**2/r**2
    Pi_term31 = -2*k*b_theta*g/(r**2*d)
    Pi_term32 = (gamma*pressure + b_sq)*rho*omega_sq - gamma*pressure*f**2
    Pi_init = Pi_term1 + (Pi_term2 + Pi_term31*Pi_term32)*chi
    xi_init = np.array([chi_init, Pi_init])
    return xi_init


def chi_boundary_wall():
    r"""
    """
    xi_boundary = 0
    return xi_boundary

def chi_boundary_vacuum():
    r"""
    """
    pass

