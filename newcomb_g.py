# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 14:39:57 2014

@author: Jens von der Linden
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatibility"""

import eigenvalue_goedbloed as eg
from newcomb_f import f_denom, f_num_wo_r


def jardin_g_8_80(r, k, m, b_z, b_z_prime, b_theta, b_theta_prime, p_prime, q,
                  q_prime):
    r"""
    """
    term1 = (2.*k**2*r**2)/(k**2*r**2+m**2)*p_prime
    term2 = b_theta**2/r*(m-k*q)**2*(k**2*r**2+m**2-1.)/(k**2*r**2+m**2)
    term3 = 2*k**2*r*b_theta**2/(k**2*r**2+m**2)**2*(k**2*q**2-m**2)
    return term1 + term2 + term3


def jardin_g_8_79(r, k, m, b_z, b_z_prime, b_theta, b_theta_prime, p_prime, q,
                 q_prime):
    r"""
    """
    term1 = 1./r*b_theta**2/(k**2*r**2+m**2)*(k*q + m)**2
    term2 = b_theta**2/r*(m - k*q)
    term3 = 2.*b_theta/r*(r*b_theta_prime + b_theta)
    der_term1 = -2.*k**2*r*b_theta**2/(k**2*r**2 + m**2)**2*(k**2*q**2 -
                                                             m**2)
    der_term2 = 2.*k**2*b_theta**2/(k**2*r**2 + m**2)*q*q_prime
    der_term3 = 2.*b_theta_prime/(k**2*r**2 + m**2)*((k**2*q**2 - m**2) *
                                                     b_theta)
    return term1 + term2 - term3 - der_term1 - der_term2 - der_term3


def newcomb_g_17(r, k, m, b_z, b_z_prime, b_theta, b_theta_prime, p_prime, q,
                 q_prime):
    r"""
    """
    term1 = 1./r*(k*r*b_z - m*b_theta)**2/(k**2*r**2 + m**2)
    term2 = 1./r*(k*r*b_z - m*b_theta)**2
    term3 = 2.*b_theta/r*(r*b_theta_prime + b_theta)
    der_term1 = -2.*k**2*r/(k**2*r**2 + m**2)**2*(k**2*r**2*b_z**2 -
                                                  m**2*b_theta**2)
    der_term2 = 1./(k**2*r**2 + m**2)*(2.*k**2*r**2*b_z*b_z_prime +
                                       2.*k**2*r*b_z**2 -
                                       2.*m**2*b_theta*b_theta_prime)
    return term1 + term2 - term3 - der_term1 - der_term2


def goedbloed_g_0(r, k, m, b_z, b_z_prime, b_theta, b_theta_prime, p_prime, q,
                  q_prime):
    r"""
    """
    params = {'r': r, 'k': k, 'm': m, 'b_z': b_z, 'b_theta': b_theta}
    f = eg.f(**params)
    term1 = 2.*k**2*r**2/(m**2 + k**2*r**2)*p_prime
    term2 = (m**2 + k**2*r**2 - 1)/(m**2 + k**2*r**2)*r*f**2
    term3 = 2.*k**2*r**3*(m*b_theta/r - k*b_z)/(m**2 + k**2*r**2)**2*f
    return term1 + term2 - term3


def newcomb_g_18(r, k, m, b_z, b_z_prime, b_theta, b_theta_prime, p_prime, q,
                 q_prime):
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
    term1 = 2*k**2*r**2/(f_denom(r, k, m)) * p_prime

    term2 = (1/r*f_num_wo_r(r, k, m, b_z, b_theta)*(k**2*r**2+m**2-1) /
             f_denom(r, k, m))
    term3 = (2*k**2*r/f_denom(r, k, m)**2 *
             (k**2*r**2*b_z**2-m**2*b_theta**2))
    return term1 + term2 + term3