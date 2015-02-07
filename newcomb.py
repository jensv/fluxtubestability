# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:03:00 2014

@author: Jens von der Linden

Module containing functions to test marginal stability of Equilibrium Profiles,
by integrating Newcomb's Euler-Lagrange equation.

The profiles are assumed to be created with equil_solver.py.
"""


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatibility"""

import numpy as np
import numpy.ma as ma
import scipy.integrate as inte
import scipy.interpolate as interp
import newcomb_f as new_f
import newcomb_g as new_g
import newcomb_init as init
import singularity_frobenius as frob
import external_stability as ext
import find_singularties as find_sing
import all_f_g



def stability(dr, offset, suydam_end_offset, sing_search_points, params,
              init_value=(0.0, 1.0), suppress_output=False,
              external_only=True, atol=None, rtol=None, debug_f_g=False):
    r"""
    Examines the total stability of profile: internal, external, Suydam.

    Parameters
    ----------
    dr : float or ndarray
        spacing of requested integration points, can be varying
    offset : float
        offset after which to start integrating after singularties
    suydam_end_offset : float
        offset after which to start integrating after suydam unstable
        singularties
    sing_search_points : int
        number of points at which to look for sign changes
    params : dict
        dictionary of equilibrium and mode params
    init_value : tuple
        initial value, useful for inner conductor boundaries
    suppress_output : bool
        flag to suppress print statments
    external_only : bool
        flag to suppres internal stability checks, only integrate last interval
    atol : float
        absolute tolerance setting for ODE integrator
    rtol : float
        relative tolerance setting for ODE integrator

    Returns
    -------
    stable_internal : bool
        True if no zero corssings of xi are found
    suydam_stable : bool
        True if all singularties (except r=0) are Suydam stable
    stable_external : bool
        True if delta_w > 0
    xi : ndarray
        xi values
    xi_der : ndarray
        derivative of xi
    r_array : ndarray
        radii
    residual_array : ndarray
        array of residuals
    delta_w : foat
        total perturbed potential energy

    Notes
    -----
    Calls interal_stability function.
    If integration succeds to plasma edge external_stability function is called.
    Optional flag only integrates last interval and uses result for
    external_stability.
    """

    all_f_g.all_f = []
    all_f_g.all_g = []
    all_f_g.all_g_term1 = []
    all_f_g.all_g_term2 = []
    all_f_g.all_g_term3 = []
    all_f_g.all_g_term4 = []
    all_f_g.all_g_term5 = []
    all_f_g.all_g_term6 = []
    all_f_g.all_pressure_prime = []
    all_f_g.all_b_theta = []
    all_f_g.all_b_z = []
    all_f_g.all_beta_0 = []
    all_f_g.all_m = []
    all_f_g.all_k = []

    missing_end_params = None
    (stable_internal, suydam_stable, xi, xi_der, r_array,
     residual_array) = internal_stability(dr, offset, suydam_end_offset,
                                          sing_search_points, params,
                                          init_value=init_value,
                                          suppress_output=suppress_output,
                                          external_only=external_only,
                                          atol=atol, rtol=rtol)

    # Test if integration to plasma edge was successful,
    # Can external stability be determined?
    if (r_array.size != 0 and not np.isnan(r_array[-1][-1]) and
        np.abs(r_array[-1][-1] - params['a']) < 1E-1):

        stable_external, delta_w = ext.external_stability(params, xi[-1][-1],
                                                          xi_der[-1][-1],
                                                          dim_less=True)
    else:
        msg = ("Integration to plasma edge did not succeed." +
               "Can not determine external stability.")
        print(msg)
        missing_end_params = params
        stable_external = True
        delta_w = None

    # Output stability messages
    # Can external stability be determined?
    k = params['k']
    m = params['m']
    if not suppress_output:
        if not stable_external:
            print("Profile is unstable to external mode k =", k, "m =", m)
            print("delta_W =", delta_w)
        if not stable_internal and not external_only:
            print("Profile is unstable to internal mode k =", k, "m =", m)
        if (stable_external and stable_internal) or (stable_external and
                                                     external_only):
            print("Profile is stable to mode k = ", k, "m =", m)
            print("delta_W =", delta_w)

    if external_only:
        stable_internal = None
    all_g_terms = [all_f_g.all_g_term1, all_f_g.all_g_term2, all_f_g.all_g_term3,
                   all_f_g.all_g_term4, all_f_g.all_g_term5, all_f_g.all_g_term6]
    all_checks = {'g_terms': all_g_terms, 'b_theta': all_f_g.all_b_theta,
                  'b_z': all_f_g.all_b_z,
                  'pressure_prime': all_f_g.all_pressure_prime,
                  'beta_0': all_f_g.all_beta_0, 'm': all_f_g.all_m,
                  'k': all_f_g.all_k}
    if debug_f_g:
        return (stable_internal, suydam_stable,
                stable_external, xi, xi_der, r_array, residual_array, delta_w,
                missing_end_params, all_f_g.all_f, all_f_g.all_g, all_checks)
    else:
        return (stable_internal, suydam_stable,
                stable_external, xi, xi_der, r_array, residual_array, delta_w,
                missing_end_params)


def internal_stability(dr, offset, suydam_offset, sing_search_points, params,
                       init_value=(0.0, 1.0), suppress_output=False,
                       external_only=True, atol=None, rtol=None):
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
    params : dict
        dictionary of plasma column parameters, including splines, geometry and
        periodicity numbers.
    init_value : optional tuple (2) of float
        intial values for non singularity inital condition. Optionally if not
        provided (0.0, 1.0) is assumed.
    suppress_output : bool
        flag to suppress print statments
    external_only : bool
        flag to suppres internal stability checks, only integrate last interval
    atol : float
        absolute tolerance setting for ODE integrator
    rtol : float
        relative tolerance setting for ODE integrator

    Returns
    -------
    stable : bool
        stability of input profile
    eigenfunctions: list of ndarray
        list of eigenfunctions for each interval
    rs: list of ndarray
        list of radii for each value in the eigenfunctions arrays

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
    """
    stable = True
    suydam_stable = True
    eigenfunctions = []
    eigen_ders = []
    rs_list = []
    residual_list = []

    # Search for singularties
    sing_params = {'a': params['r_0'], 'b': params['a'],
                   'points': sing_search_points, 'k': params['k'],
                   'm': params['m'], 'b_z_spl': params['b_z'],
                   'b_theta_spl': params['b_theta'], 'offset': offset,
                   'tol': 1E-2}
    sings, sings_wo_0, intervals = find_sing.identify_singularties(**sing_params)
    if not suppress_output:
        if not sings.size == 0:
            print("Non-geometric singularties identified at r =", sings)

    # Check singularties for Suydam stability
    suydam_result = check_suydam(sings, params['b_z'], params['b_theta'],
                                 params['p_prime'], params['beta_0'])
    if suydam_result.size != 0:
        if (not suydam_result.size == 1 or not suydam_result[0] == 0.):
            stable = False
            suydam_stable = False
            if not suppress_output:
                print("Profile is Suydam unstable at r =", suydam_result)

    int_params = {'f_func': new_f.newcomb_f_16, 'g_func': new_g.newcomb_g_18_dimless_wo_q,
                  'params': params, 'atol': atol, 'rtol': rtol}
    frob_params = {'offset': offset, 'b_z_spl': params['b_z'],
                   'b_theta_spl': params['b_theta'],
                   'p_prime_spl': params['p_prime'],
                   'q_spl': params['q'], 'f_func': new_f.newcomb_f_16,
                   'beta_0': params['beta_0']}

    #set up integration intervals
    special_case, intervals = offset_intervals(intervals, sings_wo_0,
                                               offset, suydam_result,
                                               suydam_offset)
    intervals_dr, intervals = process_dr(dr, offset, intervals)

    # if external_only integrate last interval
    if external_only:
        int_params['dr'] = intervals_dr[-1]
        int_params['r_max'] = intervals[-1][1]
        int_params['r_init'] = intervals[-1][0]
        int_params['suppress_output'] = suppress_output

        if len(intervals) == 1:
            deal_special_case = {'sing': deal_sing, 'geo': deal_geo,
                                 None: deal_norm}
            int_params = deal_special_case[special_case](int_params,
                                                         frob_params,
                                                         intervals[0][0],
                                                         offset,
                                                         init_value)
        else:
            int_params['init_func'] = init.init_xi_given
            frob_params['r_sing'] = intervals[-1][0] - offset
            int_params['xi_init'] = frob.sing_small_solution(**frob_params)

        (eigenfunctions, eigen_ders, rs_list, stable,
         residual_list) = integrate_interval(int_params, eigenfunctions,
                                             eigen_ders, rs_list, stable,
                                             residual_list)
    # integrate all intervals
    else:
        int_params['dr'] = intervals_dr[0]
        int_params['r_max'] = intervals[0][1]
        int_params['r_init'] = intervals[0][0]
        int_params['suppress_output'] = suppress_output
        deal_special_case = {'sing': deal_sing, 'geo': deal_geo,
                             None: deal_norm}
        int_params = deal_special_case[special_case](int_params,
                                                     frob_params,
                                                     intervals[0][0],
                                                     offset,
                                                     init_value)
        (eigenfunctions, eigen_ders, rs_list, stable,
         residual_list) = integrate_interval(int_params, eigenfunctions,
                                             eigen_ders, rs_list, stable,
                                             residual_list)
        for i, interval in enumerate(intervals[1:]):
            # repeat integration for each interval
            int_params['dr'] = intervals_dr[i+1]
            int_params['r_init'] = interval[0]
            int_params['r_max'] = interval[1]
            int_params['init_func'] = init.init_xi_given
            frob_params['r_sing'] = interval[0] - offset
            int_params['suppress_output'] = suppress_output
            int_params['xi_init'] = frob.sing_small_solution(**frob_params)

            (eigenfunctions, eigen_ders, rs_list, stable,
             residual_list) = integrate_interval(int_params, eigenfunctions,
                                                 eigen_ders, rs_list, stable,
                                                 residual_list)

    eigenfunctions = np.asarray(eigenfunctions)
    eigen_ders = np.asarray(eigen_ders)
    rs_array = np.asarray(rs_list)
    residual_array = np.asarray(residual_list)
    return (stable, suydam_stable, eigenfunctions, eigen_ders, rs_array,
            residual_array)


def integrate_interval(int_params, eigenfunctions, eigen_ders, rs_list, stable,
                       residual_list):
    r"""
    Returns results of interval integration.

    Parameters
    ----------
    int_params : dict
        dictionary with integration info
    eigenfunctions : list (M)
        list of xi ndarrays for each interval
    eigen_ders : list (M)
        list of xi_der ndarrays for each interval
    rs_list :  list (M)
        list of r ndarrays for each interval
    stable : bool
        cumalative stability of all intervals up till now
    residual_list : list (M)
        list of residual ndarrys for each interval
    Returns
    -------
    eigenfunction : list (M+1)
        list of xi ndarrays for each interval
    eigen_ders : list (M+1)
        list of xi_der ndarrays for each interval
    rs_list : list (M+1)
        list of r ndarrays for each interval
    stable : bool
        cumulative stability of all intervals up till now
    residual_list : list (M+1)
        list of residual ndarrys for each interval
    """
    crossing, eigenfunction, eigen_der, rs, residual = newcomb_int(**int_params)
    eigenfunctions.append(eigenfunction)
    eigen_ders.append(eigen_der)
    rs_list.append(rs)
    residual_list.append(residual)
    stable = False if crossing else stable
    return eigenfunctions, eigen_ders, rs_list, stable, residual_list


def deal_geo(int_params, *args):
    r"""
    Returns inital values for r=0 geometric singularity.
    Part of "switch statement" dictionary handeling intial conditions.
    """
    int_params['init_func'] = init.init_geometric_sing
    return int_params


def deal_sing(int_params, frob_params, interval_start, offset, *args):
    r"""
    Returns inital values for :math:`r \neq 0` geometric singularity.
    Part of "switch statement" dictionary handeling intial conditions.
    """
    int_params['init_func'] = init.init_xi_given
    frob_params['r_sing'] = interval_start - offset
    int_params['xi_init'] = frob.sing_small_solution(**frob_params)
    return int_params


def deal_norm(int_params, frob_params, interval_start, offset, init_value):
    r"""
    Returns gicen initial values. e.g. solid inner conductor boundary.
    Part of "switch statement" dictionary handeling intial conditions.
    """
    int_params['init_func'] = init.init_xi_given
    int_params['xi_init'] = init_value
    return int_params


def newcomb_der(r, y, k, m, b_z_spl, b_theta_spl, p_prime_spl, q_spl,
                f_func, g_func, beta_0):
    r"""
    Returns derivatives of Newcomb's Euler-Lagrange equation expressed as a set
    of 2 first order ODEs.

    Parameters
    ----------
    r : floatfirst_element_correction
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
    beta_0 : float
        pressure ratio on axis

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
                'beta_0': beta_0}

    f_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z_spl(r),
                'b_theta': b_theta_spl(r), 'q': q_spl(r)}

    y_prime[0] = y[1] / f_func(**f_params)
    y_prime[1] = y[0]*g_func(**g_params)
    return y_prime


def newcomb_int(r_init, dr, r_max, params, init_func, f_func, g_func,
                atol=None, rtol=None, reverse=False, divide_f=False,
                xi_init=(None, None), suppress_output=False):
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
    residual : ndarray of floats (M)
        residual of 2nd order euler-lagrange ODE.
        Useful as indicator if the ODE is being solved correctly.

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
     p_prime_spl, q_spl, beta_0,
     xi_factor) = map(params.get, ['k', 'm', 'b_z', 'b_theta', 'p_prime', 'q',
                                   'beta_0', 'xi_factor'])

    init_params = {'r': r_init, 'k': k, 'm': m, 'b_z': b_z_spl(r_init),
                   'b_theta': b_theta_spl(r_init), 'q': q_spl(r_init),
                   'f_func': f_func, 'xi': xi_init, 'xi_factor': xi_factor}

    residual_params = {'k': k, 'm': m, 'b_z': b_z_spl, 'b_theta': b_theta_spl,
                       'p_prime': p_prime_spl, 'q': q_spl, 'f_func': f_func,
                       'g_func': g_func, 'beta_0': beta_0}

    xi = np.empty(dr.size + 1)
    xi_der_f = np.empty(dr.size + 1)
    rs = np.empty(dr.size + 1)
    residual = np.empty(dr.size)

    # setup integrator, inital values
    xi_int = inte.ode(newcomb_der)
    if not (rtol):
        xi_int.set_integrator('lsoda')
    else:
        xi_int.set_integrator('lsoda', rtol=rtol)
    xi_int.set_initial_value(init_func(**init_params), t=r_init)
    xi_int.set_f_params(k, m, b_z_spl, b_theta_spl, p_prime_spl, q_spl, f_func,
                        g_func, beta_0)
    y_init = init_func(**init_params)
    xi[0] = y_init[0]
    xi_der_f[0] = y_init[1]
    rs[0] = r_init

    #integration loop
    for i in range(dr.size):
        if not xi_int.successful():
            rs[i+1:] = np.nan
            xi[i+1:] = np.nan
            xi_der_f[i+1:] = np.nan
            break
        xi_int.integrate(xi_int.t + dr[i])
        xi[i+1] = xi_int.y[0]
        xi_der_f[i+1] = xi_int.y[1]
        rs[i+1] = xi_int.t

    # Check for crossings, prepare outputs
    crossing = False
    crossings = np.nonzero(ma.masked_invalid(np.diff(np.sign(xi))))[0]
    if crossings.size != 0:
        crossing = True
        if not suppress_output:
            print('Eigenfunction crosses zero near:', r_init+np.cumsum(dr)[crossings])
    rs = np.asarray(rs)
    xi_der_f = np.asarray(xi_der_f)
    xi_der = divide_by_f(rs, xi_der_f, k, m, b_z_spl,
                         b_theta_spl, q_spl, f_func)
    try:
        residual = determine_residual(xi, xi_der, rs, residual_params)
    except:
        print(rs)
        print(k)
    #residual=None
    return crossing, np.asarray(xi), xi_der, rs, residual


def determine_residual(xi, xi_der, rs, residual_params):
    r"""
    Returns the residual of the 2nd order ODE form of the Euler-Lagrange
    equation.

    Notes
    -----
    :math:`residual = f' \xi' + f \xi'' - g \xi`
    :math:`\xi''` is approximated by the difference between neighboring
    :math:`\xi` values.
    """
    delta_r = np.diff(rs)
    delta_r = np.insert(delta_r, 0, [delta_r[0]])
    xi_der_der = np.gradient(xi_der) / delta_r

    residual_params.update({'r': rs})
    residual_params['b_theta_prime'] = residual_params['b_theta'].derivative()(rs)
    residual_params['b_z_prime'] = residual_params['b_z'].derivative()(rs)
    residual_params['q_prime'] = residual_params['q'].derivative()(rs)

    residual_params['b_z'] = residual_params['b_z'](rs)
    residual_params['b_theta'] = residual_params['b_theta'](rs)
    residual_params['q'] = residual_params['q'](rs)
    residual_params['p_prime'] = residual_params['p_prime'](rs)

    f = residual_params['f_func'](**residual_params)
    f_prime = new_f.f_prime(**residual_params)
    g = residual_params['g_func'](**residual_params)
    residual = f_prime * xi_der + f * xi_der_der - g * xi
    return residual


def divide_by_f(r, xi_der_f, k, m, b_z_spl, b_theta_spl, q_spl, f_func):
    r"""
    Divides :math:`y[1]=f \xi'` by f to recover :math:`\xi`.
    """
    f_params = {'r': r, 'k': k, 'm': m, 'b_z': b_z_spl(r),
                'b_theta': b_theta_spl(r), 'q': q_spl(r)}
    return xi_der_f / f_func(**f_params)


def offset_intervals(intervals, sings, offset, suydam_result, suydam_offset):
    r"""
    Shift interval bundaries off of singularties by offset.
    """
    special_case = None
    if intervals[0][0] <= offset:
        intervals[0][0] = offset
        special_case = 'geo'
    elif np.sum(np.isclose(intervals[0][0], sings) and not sings.size == 0):
        intervals[0][0] += offset
        special_case = 'sing'
    for i, interval in enumerate(intervals):
        if np.sum(np.isclose(interval[1], sings)) and not sings.size == 0:
            interval[1] -= offset
            if i < len(intervals)-1:
                if (i == len(intervals)-2 and not suydam_result.size == 0 and
                        np.sum(np.isclose(interval[1], suydam_result))):
                    intervals[i+1][0] += suydam_offset
                else:
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

    intervals_dr = []
    first_element_correction = 0
    for i, interval in enumerate(intervals):
        if interval[0] > interval[1]:
            interval_dr = np.array([])
            intervals_dr.append(interval_dr)
            interval[0] = interval[1]
            intervals[i] = interval
        else:
            index = np.where(dr_cum < interval[1])
            interval_dr = dr[index][np.where(dr_cum[index] > interval[0])]
            if interval_dr.size != 0:
                interval_dr[0] = interval_dr[0] - first_element_correction
            last_element = interval[1] - dr_cum[index][-1]
            interval_dr = np.append(interval_dr, last_element)
            first_element_correction = last_element
            intervals_dr.append(interval_dr)
    return intervals_dr, intervals


def check_suydam(r, b_z_spl, b_theta_spl, p_prime_spl, beta_0):
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
    beta_0 : float
        beta on axis
    Returns
    -------
    unstable_r : ndarray of floats (N)
        positions at which plasma column is suydam unstable
    """
    params = {'r': r, 'b_z_spl': b_z_spl, 'b_theta_spl': b_theta_spl,
              'p_prime_spl': p_prime_spl, 'beta_0': beta_0}
    unstable_mask = np.invert(frob.sings_suydam_stable(**params))
    return r[unstable_mask]
