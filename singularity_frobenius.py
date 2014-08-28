# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:23:31 2014

@author: Jens von der Linden

Implments Frobneius expansion around a singularity to determine the "small"
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


def alpha_func(r, b_z, b_z_prime, b_theta, b_theta_prime):
    r"""
    Return alpha for Frobenius solution.
    """
    mu = b_theta/(r*b_z)
    mu_prime = (r*b_z*b_theta_prime - b_theta*(b_z + r*b_z_prime)) / (r*b_z)**2
    return r*b_theta**2*b_z**2/(b_theta**2 + b_z**2)*(mu_prime / mu)**2


def beta_func(b_z, b_theta, p_prime, **kwargs):
    r"""
    Return beta for Frobenius solution.
    """
    return 2*b_theta/(b_theta + b_z)**2 * p_prime


def nu_1_2(alpha, beta, **kwargs):
    r"""
    Return exponents of Frobenius solution.
    """
    nu_1 = 0.5 + np.sqrt(0.25 + beta/alpha)
    nu_2 = 0.5 - np.sqrt(0.25 + beta/alpha)
    return nu_1, nu_2


def sings_alpha_beta(r, b_z_spl, b_theta_spl, p_prime_spl):
    r"""
    """
    b_z = b_z_spl(r)
    b_z_prime = b_z_spl.derivative()(r)
    b_theta = b_theta_spl(r)
    b_theta_prime = b_theta_spl.derivative()(r)
    p_prime = p_prime_spl(r)
    params = {'r': r, 'b_z': b_z, 'b_z_prime': b_z_prime, 'b_theta': b_theta,
              'b_theta_prime': b_theta_prime, 'p_prime': p_prime}
    alpha = alpha_func(**params)
    beta = beta_func(**params)
    return alpha, beta


def sings_suydam_stable(r, b_z_spl, b_theta_spl, p_prime_spl):
    r"""
    """
    alpha, beta = sings_alpha_beta(r, b_z_spl, b_theta_spl, p_prime_spl)
    return suydam_stable(alpha, beta)


def sing_small_solution(r_sing, offset, b_z_spl, b_theta_spl, p_prime_spl):
    r"""
    """
    alpha, beta = sings_alpha_beta(r_sing, b_z_spl, b_theta_spl, p_prime_spl)
    nu_1, nu_2 = nu_1_2(alpha, beta)
    return small_solution(r_sing + offset, r_sing, nu_1, nu_2)


def suydam_stable(alpha, beta):
    r"""
    Returns suydam condition.

    Parameters
    ----------
    alpha : ndarray
        radial test points

    beta : scipy spline
        axial magnetic field

    Returns
    -------
    suydam : ndarray


    Notes
    -----
    Returned expression can be checked for elements <=0.

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    eq (5).
    Jardin (2010) Computational Mehtods in Plasma Physics. eq (8.84)
    """
    return alpha + 4.*beta > 0.


def small_solution(r, r_sing, nu_1, nu_2):
    r"""
    Returns xi and xi_der of the small solution close to a singularity.
    """
    x = np.abs(r - r_sing)
    if nu_1 > nu_2:
        return (x**nu_2, nu_2*x**(nu_2 - 1.))
    else:
        return (x**nu_1, nu_1*x**(nu_1 - 1.))