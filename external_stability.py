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
from scipy.interpolate import splev
import analytic_condition as ac

def external_stability(params, xi, xi_der, dim_less=False):
    r"""
    Returns external external stability and dW.
    """
    a = params['a']
    b_z = splev(a, params['b_z'])
    b_theta = splev(a, params['b_theta'])
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
    return k*b_z - m*b_theta/a


def capital_f(a, k, m, b_theta, b_z):
    r"""
    Return F.
    """
    return k*b_z + m*b_theta/a


def external_stability_from_notes(params, xi, xi_der, dim_less=False):
    a = params['a']
    b_z = splev(a, params['b_z'])
    b_theta = splev(a, params['b_theta'])
    m = params['m']
    k_bar = params['k']
    magnetic_potential_energy_ratio = params['magnetic_potential_energy_ratio']

    term_params = {'a': a, 'k_bar': k_bar, 'm': m, 'b_z': b_z,
                   'b_theta': b_theta, 'xi': xi, 'xi_der': xi_der}
    delta_w = (plasma_term_from_notes(**term_params) -
               vacuum_term_from_notes(**term_params))
    if dim_less:
        delta_w = magnetic_potential_energy_ratio * delta_w
    stable = delta_w > 0

    return stable, delta_w


def plasma_term_from_notes(a, k_bar, m, b_z, b_theta, xi, xi_der):
    r"""
    Returns plasma energy term as in my derivation.
    """
    f_term = a*(k_bar*b_z + m*b_theta)**2/(k_bar**2 + m**2)
    h_term = (k_bar**2*b_z**2 - m**2*b_theta**2)/(k_bar**2 + m**2)
    return xi**2 * (f_term*xi_der / xi + h_term)


def vacuum_term_from_notes(a, k_bar, m, b_z, b_theta, xi, xi_der):
    r"""
    Returns vacuum energy term as in my derivation.
    """
    k_a = spec.kv(m, abs(k_bar))
    k_a_prime = spec.kvp(m, abs(k_bar))
    term1 = (k_bar*b_z + m*b_theta)**2
    term2 = xi**2/k_bar*k_a/k_a_prime
    return term1*term2

def external_stability_from_analytic_condition(params, xi, xi_der,
                                               without_sing=False,
                                               dim_less=False):
    r"""
    Returns delta_w as given in the analytic condition.

    Optionally remove sing.

    Note
    ----
    When xi goes to zero, this condition is singular due to the delta=xi'/xi
    term.
    """
    a = params['a']
    b_z = splev(a, params['b_z'])
    b_theta_vacuum = splev(a, params['b_theta'])
    m = -params['m']
    k_bar = params['k']
    lambda_bar = 2*b_theta_vacuum / (b_z*a)
    delta = xi_der*a / xi

    if without_sing:
        dW = ac.conditions_without_interface_wo_sing(k_bar, lambda_bar, m, xi,
                                                     xi_der, a)
    else:
        dW = ac.conditions_without_interface(k_bar, lambda_bar, m, delta)
    stable = dW > 0
    return stable, dW
