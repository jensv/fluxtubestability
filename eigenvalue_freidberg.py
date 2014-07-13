# -*- coding: utf-8 -*-
"""
Created on Wed Jul 02 15:56:41 2014

@author: Jens von der Linden
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


def xi_boundary(params, r):
    r"""
    """
    (mu0, k, m, a, b, pressure) = dc.retrieve(params, ('mu0', 'k', 'm', 'a',
                                                       'b', 'pressure'))
    factor = mu0*k/jardin_f(params, r)**2
    num = spec.ivp(m, k*a)*spec.kvp(m, k*b) - spec.kvp(m, k*a)*spec.ivp(m, k*b)
    den = spec.kv(m, k*a)*spec.ivp(m, k*b) - spec.iv(k*a)*spec.kvp(k*b)
    return factor*num / den*pressure(a)

def evaluate_all(r, k, m, b_z, b_theta, pressure, rho, mu0, gamma):
    r"""
    """
    b_theta_e = b_theta(r)
    b_z_e = b_z(r)
    rho_e = rho(r)
    pressure_e = pressure(r)

    k_0_sq_e = k_0(k, m, r)

    v_alfven_sq_e = v_alfven(b_z_e, b_theta_e, rho_e, mu0)
    v_sound_sq_e = v_sound(gamma, pressure_e, rho_e)

    f_e = f(r, k, m, b_z_e, b_theta_e)
    g_e = g(r, k, m, b_z_e, b_theta_e)

    omega_a_sq_e = omega_a_sq(f_e, rho_e, mu0)
    omega_h_sq_e = omega_h_sq(v_sound_sq_e, v_alfven_sq_e, omega_a_sq_e)
    omega_g_sq_e = omega_g_sq(v_sound_sq_e, v_alfven_sq_e, omega_a_sq_e)
    alpha_sq_e = alpha_fs_sq(k_0_sq_e, v_sound_sq_e, v_alfven_sq_e,
                             omega_a_sq_e)
    omega_f_sq_e = omega_fs_sq(k_0_sq_e, v_sound_sq_e, v_alfven_sq_e,
                               alpha_sq_e)
    omega_s_sq_e = omega_fs_sq(k_0_sq_e, v_sound_sq_e, v_alfven_sq_e,
                               alpha_sq_e)
    a_e = a(r, v_sound_sq_e, v_alfven_sq_e, omega_sq, omega_a_sq_e,
            omega_h_sq_e, omega_f_sq_e, omega_s_sq_e, rho_e)
    c_e = c(r, k, g_e, v_alfven_sq_e, v_sound_sq_e, omega_sq, omega_a_sq_e,
            omega_h_sq_e, omega_g_sq_e, omega_f_sq_e, omega_s_sq_e,
            b_theta_e, rho_e, mu0)

def evaluate_all_der():
    r"""
    """


def xi_der():
    r"""
    """


def f(r, k, m, b_z, b_theta):
    r"""
    """
    return k*b_z + m*b_theta/r


def g(r, k, m, b_z, b_theta):
    r"""
    """
    return m*b_z / r - k*b_theta


def omega_a_sq(f, rho, mu0):
    r"""
    """
    return f**2 / mu_0*rho


def omega_h_sq(v_sound_sq, v_alfven_sq, omega_a_sq):
    r"""
    """
    return ((v_sound_sq / (v_sound_sq + v_alfven_sq)) * omega_a_sq


def omega_g_sq(v_sound_sq, v_alfven_sq, omega_a_sq):
    r"""
    """
    return (v_sound_sq / v_alfven_sq)*omega_a_sq


def omega_f_sq(k_0_sq, v_sound_sq, v_alfven_sq, alpha_sq):
    r"""
    """
    return 0.5*k_0_sq*(v_sound_sq + v_alfven_sq*(1 + (1 - alpha**2)**0.5)


def omega_s_sq(k_0_sq_e, v_sound_sq_e, v_alfven_sq_e, alpha_sq_e):
    r"""
    """
    return 0.5*k_0_sq*(v_sound_sq + v_alfven_sq*(1 - (1 - alpha**2)**0.5)


def alpha_sq(k_0_sq, v_sound_sq, v_alfven_sq, omega_a_sq):
    r"""
    """
    return 4*v_sound_sq*omega_a_sq / (k_0_sq*(v_sound_sq + v_alfven_sq)**2)


def v_sound_sq(gamma, pressure, rho):
    r"""
    """
    return gamma*pressure/rho


def v_alfven_sq(k_0_sq, v_sound_sq, v_alfven_sq, omega_a_sq):
    r"""
    """
    return (b_z(r)**2 + b_theta**2) / (mu0*rho(r))


def k_0_sq(k, m, r):
    r"""
    """
    return k**2 + m**2 / r**2


def a(r, v_sound_sq, v_alfven_sq, omega_sq, omega_a_sq, omega_h_sq,
      omega_f_sq, omega_s_sq, rho):
    r"""
    """
    rho_e = rho(r)
    vsound
    term1 = rho(r)*(v_sound_sq()+v_alfven_sq()) / r
    term2_num = (omega_sq() - omega_a_sq())*(omega_sq() - omega_h_sq())
    term2_denom = (omega_sq() - omega_f_sq())*(omega_sq() - omega_s_sq())
    return term1*term2_num / term2_denom


def c(r, k, m, v_alfven_sq, v_omega_sq, omega_a_sq, omega_g_sq, omega_f_sq, omega_s_sq, b_z,
      b_theta, rho):
    r"""
    """
    term1 = -rho/r*(omega_sq - omega_a_sq)
    term2 = 4*k**2*b_theta**2*v_alfven_sq**2/mu0*r**3
    term3 = (omega_sq - omega_g_sq) / ((omega_sq - omega_f_sq)*
                                       (omega_sq-omega_s_sq))
    term4 =
    return
