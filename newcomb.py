# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:03:00 2014

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
import scipy.integrate as inte
import scipy.optimize as opt


def internal_stability():
    """
    Checks for internal stability accroding to Newcomb's procedure.
    """

    sings = new.identify_singularties(params['a'], params['r_0i'], points,
                                      params['k'], params['m'], params['b_z'],
                                      params['b_theta'])
    if len(sings) != 0:
        suydam_result = check_suydam()
        if len(suydam_result) == :
            print ""

def newcomb_der(r, y, k, m, b_z_spl, b_theta_spl, p_prime_spl, q_spl,
                f_func, g_func):
    r"""

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    Reference
    ---------

    Example
    -------

    """
    y_prime = np.zeros(2)

    g_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z_spl(r),
                'b_z_prime': b_z_spl.derivative()(r),
                'b_theta': b_theta_spl(r),
                'b_theta_prime': b_theta_spl.derivative()(r),
                'p_prime': p_prime_spl(r), 'q': q_spl(r),
                'q_prime': q_spl.derivative()(r)}

    f_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z_spl(r),
                'b_theta': b_theta_spl(r), 'q': q_spl(r)}

    #if np.allclose(f_func(**f_params), 0., atol=1E-10):
    #    print('singularity at r=' + str(r))
    y_prime[0] = y[1] / f_func(**f_params)
    y_prime[1] = y[0]*g_func(**g_params)
    return y_prime


def newcomb_der_divide_f(r, y, k, m, b_z_spl, b_theta_spl, p_prime_spl, q_spl,
                         f_func, g_func):
    r"""

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    Reference
    ---------

    Example
    -------

    """
    y_prime = np.zeros(2)

    g_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z_spl(r),
                'b_z_prime': b_z_spl.derivative()(r),
                'b_theta': b_theta_spl(r),
                'b_theta_prime': b_theta_spl.derivative()(r),
                'p_prime': p_prime_spl(r), 'q': q_spl(r),
                'q_prime': q_spl.derivtive()(r)}

    f_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z_spl(r),
                'b_theta': b_theta_spl(r), 'q': q_spl(r)}

    if np.allclose(f_func(**f_params), 0., atol=10E-5):
        print('singularity at r=' + str(r))

    y_prime[0] = y[1]/f_func(**f_params)

    y_prime[1] = y[0]*(g_func(**g_params)/f_func(**f_params))
    return y_prime


def newcomb_int(r_init, dr, r_max, params, init_func, f_func, g_func,
                atol=None, rtol=None, reverse=False, divide_f=False, prime=0.,
                xi_init=(None, None)):
    r"""
    Integrate Newcomb's Euler Lagrange equation as two odes.

    Parameters
    ----------
    divide_f: bool
              determines which newcomb_der is used
    atol: float
          absolute tolerance
    rtol: float
          relative tolerance
    rmax: float
          maxium radius at which to integrate
    dr: float
        radial step-size

    Returns
    -------
    xi: ndarray of floats (2,M)
        xi and derivative of xi.

    Notes
    -----
    The seperation of the Euler-lagrange equation is based on Alan Glasser's
    cyl code.
    Newcomb's condition states that at each singularity f=0 the integration
    should start from 0.

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (23)
    Alan Glasser (unknown) Cyl code
    """
    (k, m, b_z_spl, b_theta_spl,
     p_prime_spl, q_spl) = map(params.get, ['k', 'm', 'b_z', 'b_theta',
                                            'p_prime', 'q'])

    init_params = {'r': r_init, 'k': k, 'm': m, 'b_z': b_z_spl(r_init),
                   'b_z_prime': b_z_spl.derivative()(r_init),
                   'b_theta': b_theta_spl(r_init),
                   'b_theta_prime': b_theta_spl.derivative()(r_init),
                   'p_prime': p_prime_spl(r_init), 'q': q_spl(r_init),
                   'q_prime': q_spl.derivative()(r_init), 'f_func': f_func,
                   'g_func': g_func, 'prime': prime, 'xi': xi_init}

    xi = []
    rs = []
    if divide_f:
        xi_int = inte.ode(newcomb_der_divide_f)
    else:
        xi_int = inte.ode(newcomb_der)

    if not (atol and rtol):
        xi_int.set_integrator('lsoda')
    else:
        xi_int.set_integrator('lsoda', atol, rtol)
    xi_int.set_initial_value(init_func(**init_params), t=r_init)
    xi_int.set_f_params(k, m, b_z_spl, b_theta_spl, p_prime_spl, q_spl, f_func,
                        g_func)

    xi.append(init_func(**init_params))
    rs.append(r_init)

    if not reverse:
        while xi_int.successful() and xi_int.t < r_max:
            xi_int.integrate(xi_int.t + dr)
            xi.append(xi_int.y)
            rs.append(xi_int.t)
    else:
        while xi_int.successful() and xi_int.t > r_max+dr:
            xi_int.integrate(xi_int.t + dr)
            xi.append(xi_int.y)
            rs.append(xi_int.t)

    return (np.array(xi), np.array(rs))


def identify_singularties(a, b, points, k, m, b_z_spl, b_theta_spl):
    """
    Return list of singular points.
    """
    params = (k, m, b_z_spl, b_theta_spl)
    r = np.linspace(a, b, points)
    zero_positions = []

    sign = np.sign(f_relevant_part(r, k, m, b_z_spl, b_theta_spl))
    for i in range(points-1):
        if np.allclose(sign[i] + sign[i+1], 0.):
            zero_pos = opt.brentq(f_relevant_part, r[i], r[i+1], args=params)
            zero = f_relevant_part(r[i], k, m, b_z_spl, b_theta_spl)
            if np.isnan(zero) or abs(zero) > 1e-2:
                continue
            else:
                zero_positions.append(zero_pos)
    return zero_positions

def f_relevant_part(r, k, m, b_z_spl, b_theta_spl):
    """
    Return relevant part of f for singularity detection.
    """
    b_theta = b_theta_spl(r)
    b_z = b_z_spl(r)
    return f_relevant_part_func(r, k, m, b_z, b_theta)


def f_relevant_part_func(r, k, m, b_z, b_theta):
    """
    Return relevant part of f for singularity detection.
    """
    return k*r*b_z + m*b_theta


def suydam(r, b_z, q_prime, q, p_prime):
    r"""
    Returns suydam condition.

    Parameters
    ----------
    r : ndarray
        radial test points

    b_z : scipy spline
        axial magnetic field

    qprime : scipy spline
        derivative of safety factor

    q : scipy spline
        safety factor

    pprime : scipy spline
        derivative of pressure

    Returns
    -------
    suydam : ndarray


    Notes
    -----
    Returned expression can be checked for elements <=0.

    .. math:: \frac{r}{8}\frac{B_{z}}{\mu_{0}}\frac{q'}{q}^2

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    eq (5).
    Jardin (2010) Computational Mehtods in Plasma Physics. eq (8.84)
    """
    return r/8.*b_z*(q_prime/q)**2+p_prime


def check_suydam(r, b_z_spl, q_spl, p_prime_spl):
    r"""
    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    Reference
    ---------

    Example
    -------

    """
    b_z = b_z_spl(r)
    q = q_spl(r)
    q_prime = q_spl.derivative()(r)
    p_prime = p_prime_spl(r)
    zeros_mask = (suydam(r, b_z, q_prime, q, p_prime) <= 0)
    return r[zeros_mask]




