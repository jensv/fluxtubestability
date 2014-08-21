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


def beta_func(b_z, b_theta, p_prime):
    r"""
    Return beta for Frobenius solution.
    """
    return 2*b_theta/(b_theta + b_z)**2 * p_prime


def nu_1_2(alpha, beta):
    r"""
    Return exponents of Frobenius solution.
    """
    nu_1 = 0.5 + 0.5*np.sqrt(1. + 4.*beta/alpha)
    nu_2 = 0.5 - 0.5*np.sqrt(1. + 4.*beta/alpha)
    return nu_1, nu_2


def suydam_stable(alpha, beta):
    r"""
    Return Ture or False for suydam_stability.
    """
    return alpha() + 4.*beta() > 0.


def small_solution(r, r_sing, nu_1, nu_2):
    r"""
    Returns xi and xi_der of the small solution close to a singularity.
    """
    if nu_1 > nu_2:
        return ((r-r_sing)**nu_2, nu_2*(r-r_sing)**(nu_2 - 1.))
    else:
        return ((r-r_sing)**nu_1, nu_1*(r-r_sing)**(nu_1 - 1.))