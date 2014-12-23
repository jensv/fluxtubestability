# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:31:46 2014

@author: Jens von der Linden

Collection of init functions to set initial xi and xi prime for newcomb
integrations.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatibility"""

import numpy as np

def init_geometric_sing(r, k, m, b_z, b_theta, q, f_func, xi_factor, *args, **kwargs):
    r"""
    Return xi found from Frobenius method at a geometric singularity (i.e. r=0)

    Parameters
    ----------
    r : float
        radius
    k : float
        axial periodicity number
    m : float
        azimuthal periodicity number
    b_z : float
        axial magnetic field
    b_z_prime : float
        derivative of axial magnetic field
    b_theta : float
        azimuthal magnetic field
    b_theta_prime : float
        derivative of azimuthal magnetic field
    p_prime : float
        derivative of pressure
    q : float
        safety factor
    q_prime : float
        derivative of safety factor
    f_func : function
        function used to return Newcomb's g

    Returns
    -------
    y : ndarray of flaots (2)
        intial values for linear set of ODEs representing Euler-Lagrange
        equation. :math:`y(0)=\xi'` and :math:`y(1)=f\xi'`

    Notes
    -----
    The expressions for xi and xi_prime are from power series expansion as
    described in the Newcomb's paper, Goedbloed and Freidberg's MHD books.
    """
    y = np.zeros(2)

    f_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z, 'b_theta': b_theta, 'q': q}

    if m == 0:
        y[0] = r
        y[1] = 1
    else:
        y[0] = r**(abs(m) - 1)
        y[1] = (abs(m) - 1)*r**(abs(m) - 2)

    y[1] = f_func(**f_params)*y[1]

    return y*xi_factor


def init_xi_given(xi, r, k, m, b_z, b_theta, q, f_func, xi_factor, *args, **kwargs):
    r"""
    Return y intizlized with given xi and xi_prime.

    Parameters
    ----------
    r : float
        radius
    k : float
        axial periodicity number
    m : float
        azimuthal periodicity number
    b_z : float
        axial magnetic field
    b_z_prime : float
        derivative of axial magnetic field
    b_theta : float
        azimuthal magnetic field
    b_theta_prime : float
        derivative of azimuthal magnetic field
    p_prime : float
        derivative of pressure
    q : float
        safety factor
    q_prime : float
        derivative of safety factor
    f_func : function
        function used to return Newcomb's f

    Returns
    -------
    y : ndarray of flaots (2)
        intial values for linear set of ODEs representing Euler-Lagrange
        equation. :math:`y(0)=\xi'` and :math:`y(1)=f\xi'`

    Notes
    -----
    The expressions for xi and xi_prime are from power series expansion as
    described in the Newcomb's paper, Goedbloed and Freidberg's MHD books.
    """
    y = np.zeros(2)

    f_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z, 'b_theta': b_theta, 'q': q}

    y[0] = xi[0]
    y[1] = xi[1] * f_func(**f_params)

    return y*xi_factor


def init_r_sing_glasser(r, k, m, b_z, b_theta, xi_factor, *args, **kwargs):
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
    return xi*xi_factor
