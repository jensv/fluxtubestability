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


def xi_boundary(params, r):
    r"""
    """
    factor = mu0*k/jardin_f(params, r)**2
    num = spec.ivp(m, k*a)*spec.kvp(m, k*b) - spec.kvp(m, k*a)*spec.ivp(m, k*b)
    den = spec.kv(m, k*a)*spec.ivp(m, k*b) - spec.iv(k*a)*spec.kvp(k*b)
    return factor*num / den*pressure(a)


def xi_integrate(params, r):
    r"""
    """
    xi = []

    xi_int = inte.ode(xi_der)

    xi_int.set_integrator('dopri5', atol, rtol)
    xi_int.set_initial_value()
    xi_int.set_f_params(params)

    while xi_int.successful() and xi_int.t < r_max-dr:
        xi_int.integrate(xi_int.t + dr)
        xi.append = xi
    return np.array(xi)


def xi_integrate_boundary():
    pass


def xi_der(r, xi_vec, params):
    r"""
    """
    xi_d = np.zeros(2)
    xi_d[0] = c1/d*xi_vec[0] - r*c2*xi_vec[1]
    xi_d[1] = 1/r*c3*xi_vec[0] - c1*xi_vec[1]
    return xi_d


def jardin_f(params, r):
    r"""
    """
    return m/r*b_theta(r)-k*b_z(r)


def c1(params, r):
    r"""

    """
    factor = 2*b_theta(r)/(mu0*r)
    term1 = rho(r)**4*omega_sq**2*B_theta(r)
    term2_factor = -m/r*jardin_f(params, r)
    term2 = (rho(r)*omega_sq*(state_gamma*pressure(r) +
                              (b_z(r)**2 + b_theta(r)**2)/mu0)
             - state_gamma*pressure(r)*jardin_f(params, r)**2/mu0)
    return factor*(term1 + term2_factor * term2)


def c2(params, r):
    r"""

    """
    term1 = rho(r)**2*omega_sq**2
    term2 = -(k**2+m**2/r**2)*(rho(r)*omega_sq*(state_gamma*pressure(r) +
                                                (b_theta(r)**2+b_z(r)**2)/mu0)
                               -state_gamma*pressure(r)*jardin_f(params, r)**2/mu0)
    return term1 + term2


def c3(params, r):
    r"""

    """
    term1 = jardin_d(params, r)*(rho(r)*omega_sq - jardin_f(params, r)/mu0 +
                                 2*b_theta(r)/mu0*d_b_theta_over_r(r))
    term2 = (rho*omega_sq*(rho*omega_sq - jardin_f(params, r)**2 / mu0)*
             ((2*b_theta(r)**2)/(mu0*r))**2)
    term3 = (-(state_gamma*pressure(r)*(rho(r)*omega_sq - jardin_f**2/mu0) +
               rho(r)*omega_sq*b_z(r)**2/mu0)*(2*b_theta(r)*jardin_f(params, r)
                                               / (mu_0*r)))
    return term1 + term2 + term3


def jardin_d(params, r):
    r"""
    """
    factor1 = (rho(r)*omega_sq-jardin_f(params, r)**2/mu0)
    factor2_term1 = (rho(r)*omega_sq*(state_gamma*rho(r) +
                                      (b_theta(r)**2+b_z**2)/(2*mu0)))
    factor2_term2 = -state_gamma*rho(r)*jardin_f**2/mu0
    return factor1*(factor2_term1 + factor_term2)
