# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 20:27:04 2014

@author: Jens von der Linden
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatibility"""

import numpy as np
import scipy.optimize as opt
from scipy.integrate import splev


def identify_singularties(a, b, points, k, m, b_z_spl, b_theta_spl, offset,
                          tol, **kwargs):
    """
    Return list of singular points.

    Parameters
    ----------
    a : float
        radial start of pinch
    b : float
        radial end of pinch
    points : int
        number of points through which to divide f
    k : float
        axial periodicity number
    m : float
        azimuthal periodicity number
    b_z_spl : scipy spline object
        axial magnetic field
    b_theta_spl : scipy spline object
        azimuthal magnetic field

    Returns
    -------
    zero_positions: ndarray of floats (M)
        radial positions at which r equals zero.

    Notes
    -----
    Singular points are found by dividing f into intervals checking for sign
    changes and then running a zero funding method from the scipy optimize
    module.
    """
    params = (k, m, b_z_spl, b_theta_spl)
    r = np.linspace(a, b, points)
    zero_positions = []

    sign = np.sign(f_relevant_part(r, k, m, b_z_spl, b_theta_spl))
    for i in range(points-1):
        if np.allclose(sign[i] + sign[i+1], 0.):
            zero_pos = opt.brentq(f_relevant_part, r[i], r[i+1], args=params)
            zero = f_relevant_part(r[i], k, m, b_z_spl, b_theta_spl)
            if np.isnan(zero) or abs(zero) > tol:
                continue
            else:
                zero_positions.append(zero_pos)

    sings = np.array(zero_positions)

    if not sings.size == 0 and np.allclose(sings[0], 0.):
        sings_wo_0 = np.delete(sings, 0)
    else:
        sings_wo_0 = sings

    integration_points = np.insert((float(a), float(b)), 1, sings_wo_0)
    intervals = [[integration_points[i],
                  integration_points[i+1]]
                 for i in range(integration_points.size-1)]

    return sings, sings_wo_0, intervals


def f_relevant_part(r, k, m, b_z_spl, b_theta_spl):
    """
    Return relevant part of f for singularity detection.

    Parameters
    ----------
    r : ndarray of floats
        radial points
    k : float
        axial periodicity number
    m : float
        azimuthal periodicity number
    b_z_spl : scipy spline object
        axial magnetic field
    b_theta_spl : scipy spline object
        azimuthal magnetic field

    Returns
    -------
    f_relvant_part : ndarray of floats
        The relevant part o    eigenfunctions = []
    eigen_ders = []
    rs_list = []f Newcomb's f for determining f=0. The term that can
        make f=0 when :math:`r \neq 0`

    Notes
    -----
    The relevant part of f is:
    .. math::
       k r B_{z} + m B_{\theta}
    """
    b_theta = splev(r, b_theta_spl)
    b_z = splev(r, b_z_spl)
    return f_relevant_part_func(r, k, m, b_z, b_theta)


def f_relevant_part_func(r, k, m, b_z, b_theta):
    """
    Return relevant part of f for singularity detection. Could be complied be
    with numba.

    Parameters
    ----------
    r : ndarray of floats
        radial points
    k : float
        axial periodicity number
    m : float
        azimuthal periodicity number
    b_z : ndarray of floats
        axial magnetic field
    b_theta : ndarray of floats
        azimuthal magnetic field

    Returns
    -------
    f_relvant_part : ndarray of floats
        The relevant part of Newcomb's f for determining f=0. The term that can
        make f=0 when :math:`r \neq 0`

    Notes
    -----
    The relevant part of f is:
    .. math::
       k r B_{z} + m B_{\theta}
    """
    return k*r*b_z + m*b_theta


def f_relevant_part_der(r, k, m, b_z_spl, b_theta_prime_spl):
    r"""
    """
    b_z = b_z_spl(r)
    b_theta_prime = splev(r, b_theta_prime_spl)
    return f_relevant_part_der_func(r, k, m, b_z, b_theta_prime)


def f_relevant_part_der_func(r, k, m, b_z, b_theta_prime):
    r"""
    """
    return k*b_z + m*b_theta_prime


def f_relevant_part_der_2(r, m, b_theta_prime_prime_spl):
    r"""
    """
    b_theta_prime_prime = splev(r, b_theta_prime_prime_spl)
    return f_relevant_part_der_2_func(r, m, b_theta_prime_prime)


def f_relevant_part_der_2_func(r, m, b_theta_prime_prime):
    r"""
    """
    return m*b_theta_prime_prime