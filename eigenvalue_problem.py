# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:52:26 2014

@author: Jens von der Linden
"""

#Python 3.x compatability
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins.disabled import (apply, cmp, coerce, execfile,
                                      file, long, raw_input, reduce, reload,
                                      unicode, xrange, StandardError)
from future.types import bytes, dict, int, range, str
from future.builtins.misc import (ascii, chr, hex, input, next,
                                  oct, open, round, super)
from future.builtins.iterators import filter, map, zip

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


def xi_integrate(params, atol, rtol, r_init, dr, r_max):
    r"""
    """
    xi = []

    xi_int = inte.ode(xi_der)

    xi_int.set_integrator('dopri5', atol, rtol)
    xi_int.set_initial_value(xi_initial_condition(params, r_init))
    xi_int.set_f_params(params)

    while xi_int.successful() and xi_int.t < r_max-dr:
        xi_int.integrate(xi_int.t + dr)
        xi.append(xi_int.y)
    return np.array(xi)


def xi_initial_condition(params, r_init):
    r"""
    """
    (mu0, n, m, constant, omega_sq, b_theta, q,
     rho) = dc.retrieve(params, ('mu0', 'n', 'm', 'constant', 'omega_sq',
                                 'b_theta', 'q', 'rho'))
    r = r_init
    xi = np.zeros(2)
    xi[0] = constant*r
    xi[1] = (1 / m*(b_theta(r)**2 / (mu0*r)*(2*(m - n*q(r)) - (m - n*q(r))**2)
             + rho(r)*omega_sq)*xi[0])
    return xi


def xi_integrate_boundary():
    pass


def xi_der(r, xi_vec, params):
    r"""
    """
    xi_d = np.zeros(2)
    xi_d[0] = (c1(params, r)/jardin_d(params, r)*xi_vec[0] - r*c2(params, r)
               * xi_vec[1])
    xi_d[1] = 1/r*c3(params, r)*xi_vec[0] - c1(params, r)*xi_vec[1]
    return xi_d


def jardin_f(params, r):
    r"""
    """
    k, m, b_theta, b_z = dc.retrieve(params, ('k', 'm', 'b_theta', 'b_z'))
    return m/r*b_theta(r)-k*b_z(r)


def c1(params, r, p_term, f_difference, f_term):
    r"""

    """
    (m, mu0, omega_sq,
     rho, b_theta) = dc.retrieve(params, ('m', 'mu0', 'omega_sq',
                                          'pressure', 'rho', 'b_theta'))
    factor = 2*b_theta(r)/(mu0*r)
    term1 = rho(r)**4*omega_sq**2*b_theta(r)
    term2_factor = -m/r*jardin_f(params, r)
    return factor*(term1 + term2_factor * (p_term - f_term))


def c2(params, r, p_term, f_difference, f_term):
    r"""

    """
    (m, k, omega_sq, rho) = dc.retrieve(params, ('m', 'k', 'omega_sq', 'rho'))
    term1 = rho(r)**2*omega_sq**2
    term2_factor = -(k**2+m**2/r**2)
    return term1 + term2_factor*(p_term - f_term)


def c3(params, r, p_term, f_difference, f_term):
    r"""

    """
    (mu0, omega_sq, state_gamma, pressure, rho, b_theta,
     b_z, d_b_theta_over_r) = dc.retrieve(params, ('mu0', 'omega_sq',
                                                   'state_gamma', 'pressure',
                                                   'rho', 'b_theta', 'b_z',
                                                   'd_b_theta_over_r'))
    term1 = jardin_d(params, r)*(rho(r)*omega_sq - jardin_f(params, r)/mu0 +
                                 2*b_theta(r)/mu0*d_b_theta_over_r(r))
    term2 = rho*omega_sq*f_difference*((2*b_theta(r)**2)/(mu0*r))**2
    term3 = (-(state_gamma*pressure(r)*f_difference + rho(r)*omega_sq*b_z(r)**2
               / mu0)*(2*b_theta(r)*jardin_f(params, r)/(mu0*r)))
    return term1 + term2 + term3


def repeating_terms(params, r):
    r"""
    """
    (mu0, omega_sq, state_gamma, pressure, rho, b_theta,
     b_z) = dc.retrieve(params, ('mu0', 'omega_sq', 'state_gamma', 'pressure',
                                 'rho', 'b_theta', 'b_z'))
    b_sq = b_theta**2 + b_z**2
    p_term = rho(r)*omega_sq*(state_gamma*pressure(r) + b_sq / (2*mu0))
    f_difference = rho(r)*omega_sq - jardin_f(params, r)**2/mu0
    f_term = state_gamma*pressure(r)*jardin_f(params, r)**2/mu0
    return p_term, f_difference, f_term


def jardin_d(params, r, p_term, f_difference, f_term):
    r"""
    """
    (mu0, omega_sq, state_gamma,
     rho, b_theta, b_z) = dc.retrieve(params, ('mu0', 'omega_sq',
                                               'state_gamma', 'rho', 'b_theta',
                                               'b_z'))
    factor1 = (rho(r)*omega_sq-jardin_f(params, r)**2/mu0)
    factor2_term1 = (rho(r)*omega_sq*(state_gamma*rho(r) +
                                      (b_theta(r)**2+b_z**2)/(2*mu0)))
    factor2_term2 = -state_gamma*rho(r)*jardin_f**2/mu0
    return factor1*(factor2_term1 + factor2_term2)
