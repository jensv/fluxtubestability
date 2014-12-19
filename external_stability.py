# -*- coding: utf-8 -*-
"""
Created on Mon Sep 08 20:42:30 2014

@author: Jens von der Linden
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatibility"""

import scipy.special as spec


def external_stability(params, xi, xi_der, dim_less=False):
    r"""
    Returns external external stability and dW.
    """
    a = params['a']
    b_z = params['b_z'](a)
    b_theta = params['b_theta'](a)
    m = params['m']
    k = params['k']
    magnetic_potential_energy_ratio = params['magnetic_potential_energy_ratio']

    if params['b'] == 'infinity':
        lambda_term = lambda_infinity(**{'a': a, 'k': k, 'm': m})
    else:
        assert isinstance(params['b'], float), "b must be 'infinity' or of \
                                                type float."
        lambda_term = lambda_boundary(**{'a': a, 'b': b, 'k': k, 'm': m})
    f_term = capital_f(**{'a': a, 'k': k, 'm': m, 'b_theta': b_theta,
                          'b_z': b_z})
    f_adjoint_term = f_adjoint(**{'a': a, 'k': k, 'm': m, 'b_theta': b_theta,
                                  'b_z': b_z})
    k_0_sq_term = k_0(**{'k': k, 'm': m, 'a': a})

    term1 = f_term**2*a*xi_der/(k_0_sq_term*xi)
    term2 = f_term*f_adjoint_term/(k_0_sq_term)
    term3 = a**2*f_term**2*lambda_term
    delta_w = (term1+term2*term3)*xi**2
    if dim_less:
        delta_w = magnetic_potential_energy_ratio * delta_w
    stable = delta_w > 0
    return stable, delta_w

def lambda_infinity(a, k, m):
    r"""
    Return lambda term for wall at infinity.
    """
    k_a = spec.kv(m, abs(k)*a)
    k_a_prime = spec.kvp(m, abs(k)*a)

    return -k_a/(abs(k*a)*k_a_prime)


def lambda_boundary(a, b, k, m):
    r"""
    Return lambda term for wall at radius b.
    """
    k_a = spec.kv(m, abs(k)*a)
    k_a_prime = spec.kvp(m, abs(k)*a)
    k_b_prime = spec.kvp(m, abs(k)*b)
    i_a = spec.iv(m, abs(k)*a)
    i_a_prime = spec.ivp(m, abs(k)*a)
    i_b_prime = spec.ivp(m, abs(k)*b)

    factor1 = -k_a/(abs(k*a)*k_a_prime)
    factor2_num = 1. - k_b_prime*i_a/(i_b_prime*k_a)
    factor2_denom = 1. - k_b_prime*i_a_prime/(i_b_prime*k_a_prime)
    return factor1*factor2_num/factor2_denom


def k_0(k, m, a):
    r"""
    Return k_0 term.
    """
    return k**2 + m**2 / a**2


def f_adjoint(a, k, m, b_theta, b_z):
    r"""
    Return adjoint F.
    """
    return k*b_z + m*b_theta/a


def capital_f(a, k, m, b_theta, b_z):
    r"""
    Return F.
    """
    return k*b_z - m*b_theta/a
