# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 13:43:55 2014

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


def newcomb_f_16(r, k, m, b_z, b_theta, q):
    r"""
    Return f from Newcomb's paper.

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

    Returns
    -------
    f : ndarray of floats
        f from Newcomb's paper

    Notes
    -----
    Implements equation:
    :math:`f = \frac{r (k r B_{z}+ m B_{\theta})^2}{k^{2} r^{2} + m^{2}}`


    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (16)
    """
    params_num = {'r': r, 'k': k, 'm': m, 'b_z': b_z, 'b_theta': b_theta}
    params_denom = {'r': r, 'k': k, 'm': m}
    return r*f_num_wo_r(**params_num)/f_denom(**params_denom)



def goedbloed_f_9_106(r, k, m, b_z, b_theta, q):
    r"""
    """
    params = {'r': r, 'k': k, 'm': m, 'b_z': b_z, 'b_theta': b_theta}
    f = eg.f(**params)
    return r**3*f**2/(m**2 + k**2*r**2)


def jardin_f_8_78(r, k, m, b_z, b_theta, q):
    r"""
    """
    return r*b_theta**2*(m - k*q)**2/(k**2*r**2 + m**2)


def f_denom(r, k, m):
    r"""
    Return denominator of f from Newcomb's paper.

    Parameters
    ----------
    r: ndarray of floats
        radius

    k: float
        axial periodicity number

    m: float
        azimuthal periodicity number

    Returns
    -------
    f_denom: ndarray of floats
         denominator of f from Newcomb's paper

    Notes
    -----
    f denominator :math:`k^{2}r^{2}+m^{2}`

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (16)
    """
    return k**2*r**2 + m**2

def f_num_wo_r(r, k, m, b_z, b_theta):
    r"""
    Return numerator of f without r from Newcomb's paper.

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

    Returns
    -------
    f_num_wo_r : ndarray of floats
        numerator of f without r from Newcomb's paper

    Notes
    -----
    f numerator without r: :math:`(k r B_{z}+m B_{\theta})^{2}`

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (16)
    """
    return (k*r*b_z(r) + m*b_theta(r))**2