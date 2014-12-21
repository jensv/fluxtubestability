# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:23:31 2014

@author: Jens von der Linden

Implements Frobenius expansion around a singularity to determine the "small"
solution and check the Suydam condition.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatibility"""

import numpy as np


def alpha_func(r, b_z, b_z_prime, b_theta, b_theta_prime, **kwargs):
    r"""
    Return alpha for Frobenius solution.

    Parameters
    ----------
    r : ndarray of floats
        radial position of singularity
    b_z : ndarray of floats
        axial magnetic field
    b_z_prime : ndarray of floats
        derivative of axial magnetic field
    b_theta : ndarry of floats
        azimuthal magnetic field
    b_theta_prime : ndarry of floats
        derivative of azimuthal magnetic field

    Returns
    -------
    alpha : ndarray of floats
        beta of quadratic equation for the exponents of the Forbenius solutions
    """
    mu = b_theta/(r*b_z)
    mu_prime = (r*b_z*b_theta_prime - b_theta*(b_z + r*b_z_prime)) / (r*b_z)**2
    return r*b_theta**2*b_z**2/(b_theta**2 + b_z**2)*(mu_prime / mu)**2


def beta_func(b_z, b_theta, p_prime, mu_0, **kwargs):
    r"""
    Return beta for Frobenius solution.

    Parameters
    ----------
    b_z : ndarray of floats
        axial magnetic field
    b_theta: ndarray of floats
        azimuhtal magnetic field
    p_prime : ndarry of floats
        derivative of pressure
    Returns
    -------
    beta : ndarray of floats
        beta of quadratic equation for the exponents of the Forbenius solutions
    """
    return 2.*b_theta**2*mu_0/(b_theta + b_z)**2 * p_prime


def nu_1_2(alpha, beta, imaginary=False, **kwargs):
    r"""
    Return exponents of Frobenius solution.

    Paramters
    ---------
    alpha : ndarray of floats
        alpha of quadratic equation for the exponents of the Frobenius
        solutions
    beta : ndarray of floats
        beta of quadratic equation for the exponents of the Forbenius solutions
    Returns
    -------
    nu_1 : ndarray of floats
         exponent 1 of Frobenius solution
    nu_2 : ndarray of floats
         exponent 1 of Frobenius solution
    """
    nu_1 = -0.5 + np.emath.sqrt(0.25 + beta/alpha)
    nu_2 = -0.5 - np.emath.sqrt(0.25 + beta/alpha)
    return nu_1, nu_2


def sings_alpha_beta(r, b_z_spl, b_theta_spl, p_prime_spl, mu_0):
    r"""
    Returns alpha and beta at the singularties.

    Parameters
    ----------
    r : ndarray of floats
        positions of singularities
    b_z_spl : scipy spline object
        axial magnetic field
    b_theta_spl : scipy spline object
        azimuthal magnetic field
    p_prime_spl : scipy spline object
        derivative of pressure

    Returns
    -------
    alpha : ndarry of floats
        alpha of quadratic equation for the exponents of the Frobenius
        solutions
    beta : ndarray of floats
        beta of quadratic equation for the exponents of the Forbenius solutions
    """
    b_z = b_z_spl(r)
    b_z_prime = b_z_spl.derivative()(r)
    b_theta = b_theta_spl(r)
    b_theta_prime = b_theta_spl.derivative()(r)
    p_prime = p_prime_spl(r)
    params = {'r': r, 'b_z': b_z, 'b_z_prime': b_z_prime, 'b_theta': b_theta,
              'b_theta_prime': b_theta_prime, 'p_prime': p_prime, 'mu_0': mu_0}
    alpha = alpha_func(**params)
    beta = beta_func(**params)
    return alpha, beta


def sings_suydam_stable(r, b_z_spl, b_theta_spl, p_prime_spl, mu_0):
    r"""
    Returns bool array. True for singularities that are Suydam stable and False
    for unstable singularities.

    Parameters
    ----------
    r : ndarray of floats
        singularity positions
    b_z_spl : scipy spline object
        axial magnetic field
    b_theta_spl : scipy spline object
        azimuthal magnetic field
    p_prime_spl :scipy spline object
        derivative of pressure

    Returns
    -------
    stable_mask : ndarray of bool
        bool array of Suydam stable singularities can be used as a mask for
        indexing.
    """
    alpha, beta = sings_alpha_beta(r, b_z_spl, b_theta_spl, p_prime_spl, mu_0)
    return suydam_stable(alpha, beta)


def sing_small_solution(r_sing, offset, k, m, b_z_spl, b_theta_spl,
                        p_prime_spl, q_spl, f_func, mu_0, r_request=None):
    r"""
    Returns small solution of Frobenius expansion near singularity in y form of
    ODE set.

    Parameters
    ----------
    r_sing : ndarray of floats
        position of singularities
    offset : float
        offset from singularity for integration
    b_z_spl : scipy spline object
        axial magnetic field
    b_theta_spl : scipy spline object
        azimuthal magnetic field
    p_prime_spl : scipy spline object
        derivative of pressure
    q_spl : scipy spline object
        safety factor
    f_func : function
        function

    Returns
    -------
    small_sol : ndarry of floats (2)
        small solution inf y form of ODE equation set.

    References
    ----------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    """
    alpha, beta = sings_alpha_beta(r_sing, b_z_spl, b_theta_spl, p_prime_spl,
                                   mu_0)

    nu_1, nu_2 = nu_1_2(alpha, beta)
    if not r_request:
        r = r_sing + offset
    else:
        r = r_request
    f_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z_spl(r),
                'b_theta': b_theta_spl(r), 'q': q_spl(r)}
    small_sol = small_solution(r_sing + offset, r_sing, nu_1, nu_2)
    small_sol[1] = f_func(**f_params)*small_sol[1]
    return small_sol


def suydam_stable(alpha, beta):
    r"""
    Returns suydam condition.

    Parameters
    ----------
    alpha : ndarray of floats
        alpha of quadratic equation for the exponents of the Frobenius
        solutions
    beta : scipy spline
        beta of quadratic equation for the exponents of the Forbenius solutions

    Returns
    -------
    suydam : ndarray of bool
        suydam stable

    Notes
    -----
    This ithe criterion for oscillatory solutions near singularity.

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    eq (5).
    Jardin (2010) Computational Mehtods in Plasma Physics. eq (8.84)
    """
    return alpha + 4.*beta > 0.


def small_solution(r, r_sing, nu_1, nu_2):
    r"""
    Returns xi and f*xi_der of the small solution close to a singularity.

    Parameters
    ----------
    r : float
        position at which small solution is desired
    r_sing : float
        position of singularity
    nu_1 : float
        exponent 1 of Frobenius solution
    nu_2 : float
        exponent 2 of Frobenius solution

    Returns
    -------
    small_sol : ndarry of floats (2)
        small solution near singularity

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch (p. 243)
    """
    x = np.abs(r - r_sing)
    if nu_1 > nu_2:
        return np.array([np.real(x**-nu_2), np.real(-nu_2*x**(-nu_2 - 1.))])
    else:
        return np.array([np.real(x**-nu_1), np.real(-nu_1*x**(-nu_1 - 1.))])
