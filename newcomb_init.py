# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:31:46 2014

@author: Jens von der Linden

Collection of init functions to set initial xi and xi prime for newcomb
integrations.
"""

import numpy as np


def init_geometric_sing(r, k, m, b_z, b_z_prime, b_theta, b_theta_prime,
                        p_prime, q, q_prime, g_func):
    r"""
    Return xi found from Frobenius method at a geometric singularity (i.e. r=0)
    """
    y = np.zeros(2)

    g_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z, 'b_z_prime': b_z_prime,
                'b_theta': b_theta, 'b_theta_prime': b_theta_prime,
                'p_prime': p_prime, 'q': q, 'q_prime': q_prime}

    if m == 0:
        y[0] = r
    else:
        y[0] = r**(abs(m) - 1)

    y[1] = g_func(**g_params)*y[0]

    return y


def init_f_sing():
    r"""
    Return xi found from Frobenius method at an f=0 singularity.
    """
    pass


def init_xi_zero(*args):
    r"""
    Return zero xi and xi_prime.
    """
    y = np.zeros(2)

    y[0] = 0.
    y[1] = 0.

    return y

def init_r_sing_glasser(r, k, m, b_z, b_theta, *args):
    r"""
    Returns initial condition for r close to zero.

    Parameters
    ----------
    r : float
        radial position

    k : float
        axial periodicity number

    m : float
        azimuthal periodicity number

    b_z : scipy spline
        axial magnetic field

    b_theta : scipy spline
        azimuthal magnetic field

    Returns
    -------
    xi : ndarray
         newcomb's xi

    Notes
    -----
    Implements initial condition Alan used in his code.

    .. math ::
        u(0) &= r^{m - 1} \\
        u(1) &= u(0) \frac{(k B_{z} r + m B_{\theta})^{2}}{m^{2}} (m-1)

    Reference
    ---------
    Alan Glasser's newcomb.f code.
    """
    xi = np.zeros(2)
    xi[0] = r**(m - 1)
    xi[1] = xi[0]*((k*b_z(r)*r + m*b_theta(r))/m)**2*(m - 1)
    return xi