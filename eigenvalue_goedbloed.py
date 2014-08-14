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
"""Python 3.x compatibility above"""

import scipy.integrate as inte
import scipy.special as spec
import numpy as np
import sys
import copy


def f(r, k, m, b_theta, b_z):
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


def g(r, k, m, b_theta, b_z):
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


def omega_f0_sq(r, k, m, gamma, f, b_theta, b_z, rho, pressure):
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


    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.37)

    """
    k_0_sq = (m**2/r**2+k**2)
    b_sq = b_theta**2 + b_z**2
    v_sound_sq = gamma*pressure / rho
    v_alfven_sq = b_sq / rho
    alpha = (4.*gamma*pressure*f**2 /
             ((m**2/r**2+k**2)*(gamma*pressure + b_sq)**2))
    if type(alpha) == np.ndarray:
        if alpha.any() > 1:
            print("Warning: omega_f0 is complex")
    elif alpha > 1:
        print("Warning: omega_f0 is complex")
    return 0.5*k_0_sq*(v_sound_sq + v_alfven_sq)*(1 + (1 - alpha)**0.5)


def omega_s0_sq(r, k, m, gamma, f, b_theta, b_z, rho, pressure):
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

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.37)
    """
    k_0_sq = (m**2/r**2+k**2)
    b_sq = b_theta**2 + b_z**2
    v_sound_sq = gamma*pressure / rho
    v_alfven_sq = b_sq / rho
    alpha = (4.*gamma*pressure*f**2 /
             ((m**2/r**2+k**2)*(gamma*pressure + b_sq)**2))
    if type(alpha) == np.ndarray:
        if alpha.any() > 1:
            print("Warning: omega_s0 is complex")
    elif alpha > 1:
        print("Warning: omega_f0 is complex")
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
    omega_sound_sq: float
        sonic frequency squared

    Notes
    -----


    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.37)
    """
    b_sq = b_theta**2 + b_z**2
    return gamma*pressure/(gamma*pressure+b_sq) * f**2/rho


def n_freq(gamma, b_theta, b_z, rho, pressure, omega_sq, omega_alfven_sq,
           omega_sound_sq):
    r"""
    Returns n in the Chi equation set.

    Paramters
    ---------
    gamma: float
           gamma from equation of state
    b_z: float
        axial magnetic field evaluated at r
    b_theta: float
        azimuthal magnetic field evaluated at r
    rho: float
        density evaluated at r
    omega_sq: float
        eigenvalue
    omega_alfven_sq: float
        Alfven frequency evaluated at r
    omega_sound_sq: float
        Sonic frequency evaluated at r
    Returns
    -------
    n_freq: float
        n calculated from frequencies

    Notes
    -----
    Goedbloed gives two equivalent expressions for n. Both are implemented for
    testing purposes.


    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.36)
    """
    b_sq = b_theta**2 + b_z**2
    return (rho**2*(gamma*pressure + b_sq)*(omega_sq - omega_alfven_sq)
            * (omega_sq - omega_sound_sq))


def d_freq(rho, omega_sq, omega_s0_sq, omega_f0_sq):
    r"""
    Returns d in the Chi equation set.

    Paramters
    ---------
    gamma: float
           gamma from equation of state
    b_z: float
        axial magnetic field evaluated at r
    b_theta: float
        azimuthal magnetic field evaluated at r
    rho: float
        density evaluated at r
    omega_sq: float
        eigenvalue
    omega_s0_sq: float
        s0 frequency evaluated at r
    omega_d0_sq: float
        f0 frequency evaluated at r
    Returns
    -------
    d_freq: float
        d calculated from frequencies

    Notes
    -----
    Goedbloed gives two equivalent expressions for d. Both are implemented for
    testing purposes.


    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.36)
    """
    return rho**2*(omega_sq - omega_s0_sq)*(omega_sq - omega_f0_sq)


def n_fb(gamma, f, b_theta, b_z, rho, pressure, omega_sq):
    r"""
    Returns n in the Chi equation set.

    Paramters
    ---------
    gamma: float
           gamma from equation of state
    f: float
        F(r) evaluated at r
    b_theta: float
        axial magnetic field evaluated at r
    b_z: float
        azimuthal magnetic field evaluated at r
    rho: float
        density evaluated at r
    pressure: float
        pressure evaluated at r
    omega_sq: float
        eigenvalue
    Returns
    -------
    n_fb: float
        n calculated from f and B

    Notes
    -----
    Goedbloed gives two equivalent expressions for d. Both are implemented for
    testing purposes.


    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.32)
    """
    b_sq = b_theta**2 + b_z**2
    return (rho*omega_sq - f**2)*((gamma*pressure + b_sq)
                                  * rho*omega_sq - gamma*pressure*f**2)


def d_fb(r, k, m, gamma, f, b_theta, b_z, rho, pressure, omega_sq):
    r"""
    Returns d in the Chi equation set.

    Paramters
    ---------
    r: float
       radius
    k: float
       radial periodicity number
    m: float
       azimnuthal periodicity number
    gamma: float
        gamma from equation of state
    f: float
        F(r) evaluated at r
    b_theta: float
        azimuthal magnetic field evaluated at r
    b_z: float
        axial magnetic field evaluated at r
    rho: float
        density evaluated at r
    pressure: float
        pressure evaluated at r
    omega_sq: float
        eigenvalue
    Returns
    -------
    d_fb: float
        d calculated from f and B

    Notes
    -----
    Goedbloed gives two equivalent expressions for d. Both are implemented for
    testing purposes.


    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.30)
    """
    b_sq = b_theta**2 + b_z**2
    term1 = rho**2*omega_sq**2
    term2 = -(m**2/r**2 + k**2)*(gamma*pressure + b_sq)*rho*omega_sq
    term3 = (m**2/r**2 + k**2)*gamma*pressure*f**2
    return term1 + term2 + term3


def c(r, k, m, gamma, f, b_theta, b_z, rho, pressure, omega_sq):
    r"""
    Returns c from the Chi equation set.

    Paramters
    ---------
    r: float
       radius
    k: float
       radial periodicity number
    m: float
       azimnuthal periodicity number
    gamma: float
        gamma from equation of state
    f: float
        F(r) evaluated at r
    b_theta: float
        azimuthal magnetic field evaluated at r
    b_z: float
        axial magnetic field evaluated at r
    rho: float
        density evaluated at r
    pressure: float
        pressure evaluated at r
    omega_sq: float
        eigenvalue
    Returns
    -------
    c: float
        c

    Notes
    -----

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.42)
    """
    b_sq = b_theta**2 + b_z**2
    term1 = -2.*b_theta**2/r**2*rho**2*omega_sq**2
    term21 = 2.*m*b_theta*f/r**3
    term22 = ((gamma*pressure + b_sq)*rho*omega_sq - gamma*pressure*f**2)
    return term1 + term21*term22


def e(r, k, m, gamma, f, n, b_theta, b_theta_prime, b_z, rho, pressure,
      omega_sq):
    r"""
    Returns e from the Chi equation set.

    Paramters
    ---------
    r: float
       radius
    k: float
       radial periodicity number
    m: float
       azimnuthal periodicity number
    gamma: float
        gamma from equation of state
    f: float
        F(r) evaluated at r
    b_theta: float
        azimuthal magnetic field evaluated at r
    b_theta_prime: float
        derivative of azimuthal field evaluated at r
    b_z: float
        axial magnetic field evaluated at r
    rho: float
        density evaluated at r
    pressure: float
        pressure evaluated at r
    omega_sq: float
        eigenvalue
    Returns
    -------
    e: float
        e

    Notes
    -----

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation (9.42)
    """
    b_sq = b_theta**2 + b_z**2
    derivative = (2.*b_theta*b_theta_prime)/r**2 - (2.*b_theta**2)/r**3
    term1 = -n/r*((rho*omega_sq-f**2)/r + derivative)
    term2 = -4.*b_theta**4/r**4*rho**2*omega_sq**2 - 4.*rho*omega_sq*f**2*b_theta**4/r**4
    term31 = 4.*b_theta**2*f**2/r**4
    term32 = ((gamma*pressure + b_z)*rho*omega_sq - gamma*pressure*f**2)
    return term1 + term2 + term31*term32


def chi_der(r, y, k, m, gamma, b_theta_spl, b_z_spl, rho_spl,
            pressure_spl, omega_sq):
    r"""
    Returns derivatives calculated from Chi equation set for the scipy
    integrators

    Paramters
    ---------
    r: float
       radius (must be here for integrator to work)
    y: float
       integrands (must be here for integrator to work)
    c: float
       c from chi equation
    d: float
       d from chi equation
    e: float
       e from chi equation
    n: float
       n from chi equation
    Returns
    -------
    chi_der: float


    Notes
    -----
    This function is meant to be used by the scipy integrators.

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation
    """
    # print('r requested {0:.16f}',format(r))
    params = {'r': r, 'k': k, 'm': m, 'gamma': gamma, 'omega_sq': omega_sq}
    params['b_theta'] = b_theta_spl(r)
    params['b_theta_prime'] = b_theta_spl.derivative(n=1)(r)
    parmas['b_z'] = b_z_spl(r)
    params['pressure'] = pressure_spl(r)
    params['rho'] = rho_spl(r)

    params['f_ev'] = f(**params)
    params['omega_a_sq_ev'] = omega_a_sq(**params)
    params['omega_s_sq_ev'] = omega_sound_sq(**params)
    params['omega_s0_sq_ev'] = omega_s0_sq(**params)
    params['omega_f0_sq_ev'] = omega_f0_sq(**params)
    params['n_ev'] = n_fb(**params)
    # d_ev = d_freq(**params)
    params['d_ev'] = d_freq(**params)
    params['c_ev'] = c(**params)
    params['e_ev'] = e(**params)
    params['chi'] = y[0]
    params['pi'] = y[1]

    chi_prime, pi_prime = chi_matrix(**params)

    chi = np.array([chi_prime, pi_prime])
    return chi


def chi_matrix(r, n_ev, c_ev, d_ev, e_ev, chi, pi):
    r"""
    """
    chi_prime = -r/n_ev*(c_ev*chi + d_ev*pi)
    pi_prime = -r/n_ev*(e_ev*chi - c_ev*pi)
    return chi_prime, pi_prime


def chi_init(r_init, k, m, gamma, b_theta_spl, b_z_spl, rho_spl,
             pressure_spl, omega_sq):
    r"""
    Returns initial condition for the Chi vector. Derived with a frobenius
    expansion.

    Paramters
    ---------
    r: float
       radius
    k: float
       radial periodicity number
    m: float
       azimnuthal periodicity number
    gamma: float
        gamma from equation of state
    f: float
        F(r) evaluated at r
    g: float
       G(r) evaluated at rsublim
    n: float
       n from chi equation
    Returns
    b_theta: float
        azimuthal magnetic field evaluated at r
    b_theta_prime: float
        derivative of azimuthal field evaluated at r
    b_z: float
        axial magnetic field evaluated at r
    rho: float
        density evaluated at r
    pressure: float
        pressure evaluated at r
    omega_sq: float
        eigenvalue
    Returns
    -------
    e: float
        e

    Notes
    -----

    References
    ----------
    Goedbloed (2010) Principles of MHD Equation
    """
    r = r_init
    if m == 0:
        chi_init = r**2
        chi_prime = 2.*r
    else:
        chi_init = r**abs(m)
        if abs(m) == 1:
            chi_prime = 1.0
        else:
            chi_prime = abs(m)*r**(abs(m)-1)

    init_params = {}
    init_params['r'] = r
    init_params['chi_init'] = chi_init
    init_params['chi_prime'] = chi_prime
    init_params['b_theta'] = b_theta_spl(r)
    init_params['b_z'] = b_z_spl(r)
    init_params['pressure'] = pressure_spl(r)
    init_params['rho'] = rho_spl(r)

    init_params['f_ev'] = f(**init_params)
    init_params['g_ev'] = g(**init_params)
    init_params['omega_a_sq_ev'] = omega_a_sq(**init_params)
    init_params['omega_s_sq_ev'] = omega_sound_sq(**init_params)
    init_params['omega_s0_sq_ev'] = omega_s0_sq(**init_params)
    init_params['omega_f0_sq_ev'] = omega_f0_sq(**init_params)
    init_params['n_ev'] = n_fb(**init_params)

    # d_ev = d_fb(r, k, m, gamma, f_ev, b_theta, b_z, rho, pressure, omega_sq)
    chi_pi_init = chi_pi(**init_params)
    # Pi_init = 1./m*(2.*b_theta/r*f_ev - rho*omega_sq - f_ev**2)*chi_init
    xi_init = np.array([chi_init, chi_pi_init])
    return xi_init


def chi_pi(r, k, b_theta, b_z, n_ev, d_ev, g_ev, f_ev,  gamma, pressure, rho,
           omega_sq, chi_prime, chi_init):
    r"""
    """
    b_sq = b_theta**2 + b_z**2
    Pi_term1 = -n_ev/(r*d_ev)*chi_prime
    Pi_term2 = 2*b_theta**2/r**2
    Pi_term31 = -2*k*b_theta*g_ev/(r**2*d_ev)
    Pi_term32 = (gamma*pressure + b_sq)*rho*omega_sq - gamma*pressure*f_ev**2
    chi_pi_ev = Pi_term1 + (Pi_term2 + Pi_term31*Pi_term32)*chi_init
    return chi_pi_ev


def chi_boundary_wall():
    r"""
    Returns xi at wall boundary.
    """
    xi_boundary = 0
    return xi_boundary


def xi_boundary(params, r):
    r"""
    """
    (mu0, k, m, a, b, pressure) = dc.retrieve(params, ('mu0', 'k', 'm', 'a',
                                                       'b', 'pressure'))
    factor = mu0*k/jardin_f(params, r)**2
    num = spec.ivp(m, k*a)*spec.kvp(m, k*b) - spec.kvp(m, k*a)*spec.ivp(m, k*b)
    den = spec.kv(m, k*a)*spec.ivp(m, k*b) - spec.iv(k*a)*spec.kvp(k*b)
    return factor*num / den*pressure(a)


def chi_integrate(params, r_init, dr, r_max, atol=None, rtol=None, max_step=None):
    r"""
    """
    chi = []

    int_params = copy.deepcopy(params)

    int_params['b_theta_spl'] = params['b_theta']
    int_params['b_z_spl'] = params['b_z']
    int_params['rho_spl'] = params['rho']
    int_params['pressure_spl'] = params['pressure']

    chi_int = inte.ode(chi_der)
    if atol is None or rtol is None:
        chi_int.set_integrator('lsoda', max_step=max_step)
    else:
        chi_int.set_integrator('lsoda', atol, rtol)
    chi_int.set_initial_value(chi_init(r_init, **int_params), r_init)
    chi_int.set_f_params(k, m, gamma, b_theta_spl, b_z_spl, rho_spl,
                         pressure_spl, omega_sq)
    while chi_int.successful() and chi_int.t < r_max-dr:
        #print('point'+str(chi_int.t))
        chi_int.integrate(chi_int.t + dr)
        chi.append(chi_int.y)
        #print((chi_int.successful(), chi_int.t))
    return np.array(chi)
