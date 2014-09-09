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
import newcomb_f as f
import newcomb_g as g
import newcomb_init as init
import singularity_frobenius as frob
import external_stability as ext


def stability(dr, offset, sing_search_points, params,
              init_value=(0.0, 1.0)):
    (stable_internal, xi,
     xi_der, r_array) = internal_stability(dr, offset, sing_search_points,
                                           params, init_value=(0.0, 1.0))
    stable_external, delta_W = ext.external_stability(params, xi[-1],
                                                      xi_der[-1])

    k = params['k']
    m = params['m']
    if not stable_external:
        print("Profile is unstable to external mode k=", k, "m=", m)
    if not stable_internal:
        print("Profile is unstable to internal mode k=", k, "m=", m)
    return stable_internal, stable_external, xi, xi_der, r_array, delta_W


def internal_stability(dr, offset, sing_search_points, params,
                       init_value=(0.0, 1.0)):
    """
    Checks for internal stability accroding to Newcomb's procedure.

    Parameters
    ----------
    dr : float or ndarray (M)
        desired spacing between integration points.
        If float spacing is uniform. If ndarray spacing varies as given.
    offset : float
        offset from geometric (r=0) and f=0 singularties
    sing_seach_points: int
        number of points used to search for sign reversals in f.
    params :
        dictionary of plasma column parameters, including splines, geometry and
        periodicity numbers.
    init_value : optional tuple (2) of float
        intial values for non singularity inital condition. Optionally if not
        provided (0.0, 1.0) is assumed.

    Notes
    -----
    First, Newcomb's f is ecamined for zeros which correspond to singularities
    of the Euler-Lagrange equation. The singularties are tested for Suydam
    stability.
    A list of intervals with the geometric boundaries and singularities
    are created.
    Next the Euler-Lagrange equation is integrated in each interval  with
    scipy.ode using the lsoda solver. At each integration step a check for
    zeros is conducted by looking for sign changes in the integrated
    pertubation function.

    Returns
    -------
    stable : boolean
            stability of input profile

    eigenfunctions: list of ndarray
                    list of eigenfunctions for each interval

    rs: list of ndarray
        list of radii for each value in the eigenfunctions arrays
    """
    stable = True
    eigenfunctions = []
    eigen_ders = []
    rs_list = []

    sing_params = {'a': params['r_0'], 'b': params['a'],
                   'points': sing_search_points, 'k': params['k'],
                   'm': params['m'], 'b_z_spl': params['b_z'],
                   'b_theta_spl': params['b_theta']}
    sings = identify_singularties(**sing_params)
    sings_set = set(sings)
    if 0. in sings_set:
        sings_set.remove(0.)
    suydam_result = check_suydam(sings, params['b_z'], params['b_theta'],
                                 params['p_prime'], params['mu_0'])
    if len(suydam_result) != 0:
        stable = False
        print("Profile is Suydam unstable at r =", suydam_result)

    else:
        integration_points = np.insert(sings, (0, sings.size),
                                       (params['r_0'], params['a']))
        intervals = [[integration_points[i],
                      integration_points[i+1]]
                     for i in range(integration_points.size-1)]
        int_params = {'f_func': f.newcomb_f_16, 'g_func': g.newcomb_g_18,
                      'params': params, 'mu_0': params['mu_0']}
        frob_params = {'offset': offset, 'k': params['k'], 'm': params['m'],
                       'b_z_spl': params['b_z'],
                       'b_theta_spl': params['b_theta'],
                       'p_prime_spl': params['p_prime'],
                       'q_spl': params['q'], 'f_func': f.newcomb_f_16,
                       'mu_0': params['mu_0']}

        special_case, intervals = offset_intervals(intervals, sings_set,
                                                   offset)
        intervals_dr = process_dr(dr, offset, intervals)
        int_params['dr'] = intervals_dr[0]

        int_params['r_max'] = intervals[0][1]
        int_params['r_init'] = intervals[0][0]

        deal_special_case = {'sing': deal_sing, 'geo': deal_geo,
                             None: deal_norm}
        int_params = deal_special_case[special_case](int_params, frob_params,
                                        intervals[0][0], offset, init_value)

        crossing, eigenfunction, eigen_der, rs = newcomb_int(**int_params)
        eigenfunctions.append(eigenfunction)
        eigen_ders.append(eigen_der)
        rs_list.append(rs)
        stable = False if crossing else stable

        for i, interval in enumerate(intervals[1:]):
            # repeat integration for each interval
            int_params['dr'] = intervals_dr[i]
            int_params['r_init'] = interval[0]
            int_params['init_func'] = init.init_xi_given
            frob_params['r_sing'] = interval[0] - offset
            int_params['xi_init'] = frob.sing_small_solution(**frob_params)
            crossing, eigenfunction, eigen_der, rs = newcomb_int(**int_params)
            eigenfunctions.append(eigenfunction)
            eigen_ders.append(eigen_der)
            rs_list.append(rs)
            stable = False if crossing else stable

    eigenfunctions = np.asarray(eigenfunctions)
    eigen_ders = np.asarray(eigen_ders)
    rs_array = np.asarray(rs_list)
    return stable, eigenfunctions, eigen_ders, rs_array


def deal_geo(int_params, *args):
    int_params['init_func'] = init.init_geometric_sing
    return int_params


def deal_sing(int_params, frob_params, interval, offset, *args):
    int_params['init_func'] = init.init_xi_given
    frob_params['r_sing'] = interval - offset
    int_params['xi_init'] = frob.sing_small_solution(**frob_params)
    return int_params


def deal_norm(int_params, frob_params, interval, offset, init_value):
    int_params['init_func'] = init.init_xi_given
    int_params['xi_init'] = init_value
    return int_params

def newcomb_der(r, y, k, m, b_z_spl, b_theta_spl, p_prime_spl, q_spl,
                f_func, g_func, mu_0):
    r"""
    Returns derivatives of Newcomb's Euler-Lagrange equation expressed as a set
    of 2 first order ODEs.

    Parameters
    ----------
    r : float
        radius for which to find derivative
    y : ndarray (2)
        values of :math:`\xi` and :math:`f \xi'`
    k : float
        axial periodicity number
    m : float
        azimuthal periodicity number
    b_z_spl : scipy spline object
        axial magnetic field
    b_theta_spl : scipy spline object
        azimuthal magnetic field
    p_prime_spl : scipy spline object
        derivative of pressure
    q_spl : scipy spline object
        safety factor
    f_func : function
        function which returns f of Newcomb's Euler-Lagrange equation
    g_func : function
        function which returns f of Newcomb's Euler-Lagrange equation
    mu_0 : float
        magnetic permeability of free space

    Returns
    -------
    y_prime : ndarray of floats (2)
        derivatives of y

    Notes
    -----
    The system of ODEs representing the Euler-Lagrange equations is

    .. math::

       \frac{d \xi}{dr} &= \xi' \\
       \frac{d (f \xi')}{dr} &= g \xi
    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a diffuse linear pinch.
    """
    y_prime = np.zeros(2)

    g_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z_spl(r),
                'b_z_prime': b_z_spl.derivative()(r),
                'b_theta': b_theta_spl(r),
                'b_theta_prime': b_theta_spl.derivative()(r),
                'p_prime': p_prime_spl(r), 'q': q_spl(r),
                'q_prime': q_spl.derivative()(r),
                'mu_0': mu_0}

    f_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z_spl(r),
                'b_theta': b_theta_spl(r), 'q': q_spl(r)}

    # if np.allclose(f_func(**f_params), 0., atol=1E-10):
    #    print('singularity at r=' + str(r))
    y_prime[0] = y[1] / f_func(**f_params)
    y_prime[1] = y[0]*g_func(**g_params)
    return y_prime


def newcomb_der_divide_f(r, y, k, m, b_z_spl, b_theta_spl, p_prime_spl, q_spl,
                         f_func, g_func):
    r"""
    This is another formulation of the Euler-Lagrange equation as a set of 2
    ODEs. Alan Glasser used this formulation.

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a diffuse linear pinch.
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


def newcomb_int(r_init, dr, r_max, params, init_func, f_func, g_func, mu_0,
                atol=None, rtol=None, reverse=False, divide_f=False,
                xi_init=(None, None)):
    r"""
    Integrate Newcomb's Euler Lagrange equation as two ODES.

    Parameters
    ----------
    r_init : float
        intial radius at which to start integrating
    dr : ndarray of floats
        radial stepsize between integration points
    rmax : float
        maxium radius at which to integrate
    init_func : function
        function to calculate inital condition
    f_func : function
        function to calculate Newcomb's f
    g_func : function
        function to calculate Newcomb's g
    atol : float
        absolute tolerance
    rtol : float
        relative tolerance
    reverse : false
        flag to integrate in reverse from max to init
    divide_f : bool
          determines which newcomb_der is used
    xi_init : tuple of floats (2)
        initial values for :math:`\xi` and :math:`\xi'`
    ceck_crossing : bool
        check for crossing of 0 by :math:`\xi`
    Returns
    -------
    crossing : bool
        xi crosses zero
    xi : ndarray of floats (2,M)
        xi and derivative of xi.
    rs : ndarray of floats (M)
        radial positions of xi

    Notes
    -----
    Newcomb's condition states that at each singularity f=0 the integration
    should start from the Frobenius solution.

    Reference
    ---------
    Newcomb (1960) Hydromagnetic Stability of a Diffuse Linear Pinch
    Equation (23)
    """
    (k, m, b_z_spl, b_theta_spl,
     p_prime_spl, q_spl) = map(params.get, ['k', 'm', 'b_z', 'b_theta',
                                            'p_prime', 'q'])

    init_params = {'r': r_init, 'k': k, 'm': m, 'b_z': b_z_spl(r_init),
                   'b_theta': b_theta_spl(r_init), 'q': q_spl(r_init),
                   'f_func': f_func, 'xi': xi_init}

    xi = np.empty(dr.size + 1)
    xi_der_f = np.empty(dr.size + 1)
    rs = np.empty(dr.size + 1)

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
                        g_func, mu_0)

    y_init = init_func(**init_params)
    xi[0] = y_init[0]
    xi_der_f[0] = y_init[1]
    rs[0] = r_init

    for i in range(dr.size):
        if not xi_int.successful():
            break
        xi_int.integrate(xi_int.t + dr[i])
        xi[i+1] = xi_int.y[0]
        xi_der_f[i+1] = xi_int.y[1]
        rs[i+1] = xi_int.t

    crossing = False
    crossings = np.where(np.diff(np.sign(xi)))[0]
    if crossings.size != 0:
        crossing = True
        print('Eigenfunction crosses zero near:', np.cumsum(dr)[crossings])
    rs = np.asarray(rs)
    xi_der_f = np.asarray(xi_der_f)
    xi_der = divide_by_f(rs, xi_der_f, k, m, b_z_spl,
                         b_theta_spl, q_spl, f_func)
    return crossing, np.asarray(xi), xi_der, rs


def divide_by_f(r, xi_der_f, k, m, b_z_spl, b_theta_spl, q_spl, f_func):
    r"""
    Divides :math:`y[1]=f \xi'` by f to recover :math:`\xi`.
    """
    f_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z_spl(r),
                'b_theta': b_theta_spl(r), 'q': q_spl(r)}
    return xi_der_f / f_func(**f_params)


def identify_singularties(a, b, points, k, m, b_z_spl, b_theta_spl):
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
            if np.isnan(zero) or abs(zero) > 1e-2:
                continue
            else:
                zero_positions.append(zero_pos)
    return np.array(zero_positions)


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
    b_theta = b_theta_spl(r)
    b_z = b_z_spl(r)
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


def offset_intervals(intervals, sings_set, offset):
    r"""
    Shift interval bundaries off of singularties by offset.
    """
    special_case = None
    if intervals[0][0] <= offset:
        intervals[0][0] = offset
        special_case = 'geo'
    elif intervals[0][0] in sings_set:
        intervals[0][0] += offset
        special_case = 'sing'
    for i, interval in enumerate(intervals):
        if interval[1] in sings_set:
            interval[1] -= offset
            if i < len(intervals)-1:
                intervals[i+1][0] += offset
    return special_case, intervals


def process_dr(dr, offset, intervals):
    r"""
    Return dr array with only elements used in integration. Singularity
    elements are thrown away.
    """
    if isinstance(dr, float) or isinstance(dr, int):
        dr_cum = np.arange(intervals[0][0], intervals[-1][1], dr)
        dr = np.ones(dr_cum.size)*dr
    else:
        dr_cum = np.cumsum(dr)
    if (intervals[0][0] - offset) <= 1E-10:
        dr_cum += intervals[0][0]

    intervals_dr = []
    for interval in intervals:
        index = np.where(dr_cum < interval[1])
        interval_dr = dr[index][np.where(dr_cum[index] > interval[0])]
        intervals_dr.append(interval_dr)
    return intervals_dr


def check_suydam(r, b_z_spl, b_theta_spl, p_prime_spl, mu_0):
    r"""
    Return radial positions at which the Euler-Lagrange equation is singular
    and Suydam's criterion is violated.

    Parameters
    ----------
    r : ndarray of floats (M)
        positions at which f=0.
    b_z_spl : scipy spline object
        axial magnetic field
    b_theta_spl : scipy spline object
        azimuthal magnetic field
    p_prime_spl : scipy spline object
        derivative of pressure
    Returns
    -------
    unstable_r : ndarray of floats (N)
        positions at which plasma column is suydam unstable
    """
    params = {'r': r, 'b_z_spl': b_z_spl, 'b_theta_spl': b_theta_spl,
              'p_prime_spl': p_prime_spl, 'mu_0': mu_0}
    unstable_mask = np.invert(frob.sings_suydam_stable(**params))
    return r[unstable_mask]
