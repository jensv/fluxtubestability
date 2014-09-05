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

from itertools import repeat
import numpy as np
import scipy.integrate as inte
import scipy.optimize as opt
import newcomb_f as f
import newcomb_g as g
import newcomb_init as init
import singularity_frobenius as frob
import scipy.constants as consts


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
    rs_list = []

    sing_params = {'a': params['r_0'], 'b': params['a'],
                   'points': sing_search_points, 'k': params['k'],
                   'm': params['m'], 'b_z_spl': params['b_z'],
                   'b_theta_spl': params['b_theta']}
    sings = identify_singularties(**sing_params)
    sings_set = set(sings)
    suydam_result = check_suydam(sings, params['b_z'], params['b_theta'],
                                 params['p_prime'])
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
                      'params': params, 'dr': dr, 'check_crossing': True}
        frob_params = {'offset': offset, 'k': params['k'], 'm': params['m'],
                       'b_z_spl': params['b_z'],
                       'b_theta_spl': params['b_theta'],
                       'p_prime_spl': params['p_prime'],
                       'q_spl': params['q'], 'f_func': f.newcomb_f_16}

        if intervals[0][1] in sings_set:
            # check if endpoint of first interval is singular
            int_params['r_max'] = intervals[0][1] - offset
        else:
            int_params['r_max'] = intervals[0][1]

        if intervals[0][0] == 0.:
            # integration starts at geometric singularity case r=0
            int_params['r_init'] = 0. + offset
            int_params['init_func'] = init.init_geometric_sing
            crossing, eigenfunction, rs = newcomb_int(**int_params)
            eigenfunctions.append(eigenfunction)
            rs_list.append(rs)
            stable = False if crossing else stable

        elif intervals[0][0] in sings_set:
            # integration starts at f=0 sinularity
            int_params['r_init'] = intervals[0][0] + offset
            int_params['init_func'] = init.init_xi_given
            frob_params['r_sing'] = intervals[0][0]
            int_params['xi_init'] = frob.sing_small_solution(**frob_params)
            crossing, eigenfunction, rs = newcomb_int(**int_params)
            eigenfunctions.append(eigenfunction)
            rs_list.append(rs)
            stable = False if crossing else stable

        else:
            # integration starts at non-singular point
            int_params['r_init'] = intervals[0][0]
            int_params['init_func'] = init.init_xi_given
            int_params['xi_init'] = init_value
            crossing, eigenfunction, rs = newcomb_int(**int_params)
            eigenfunctions.append(eigenfunction)
            rs_list.append(rs)
            stable = False if crossing else stable

        for interval in intervals[1:]:
            # repeat integration for each interval
            int_params['r_init'] = interval[0] + offset
            int_params['init_func'] = init.init_xi_given
            frob_params['r_sing'] = interval[0]
            int_params['xi_init'] = frob.sing_small_solution(**frob_params)
            if interval[1] in sings_set:
                int_params['r_max'] = interval[1] - offset
            else:
                int_params['r_max'] = interval[1]
            crossing, eigenfunction, rs = newcomb_int(**int_params)
            eigenfunctions.append(eigenfunction)
            rs_list.append(rs)
            stable = False if crossing else stable

    return stable, eigenfunctions, rs


def newcomb_der(r, y, k, m, b_z_spl, b_theta_spl, p_prime_spl, q_spl,
                f_func, g_func, mu_0=consts.mu_0):
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


def newcomb_int(r_init, dr, r_max, params, init_func, f_func, g_func,
                atol=None, rtol=None, reverse=False, divide_f=False,
                xi_init=(None, None), check_crossing=True):
    r"""
    Integrate Newcomb's Euler Lagrange equation as two ODES.

    Parameters
    ----------
    r_init : float
        intial radius at which to start integrating
    dr : float or ndarray of floats
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

    if isinstance(dr, float):
        dr_generator = repeat(dr)
    else:
        dr_generator = (delta for delta in dr)

    if not reverse:
        while xi_int.successful() and xi_int.t < r_max:
            xi_int.integrate(xi_int.t + dr_generator.next())
            xi.append(xi_int.y)
            rs.append(xi_int.t)
            if check_crossing:
                if xi[0][-1]*xi[0][-2] < 0:
                    return True, np.array(xi), np.array(rs)

    else:
        while xi_int.successful() and xi_int.t > r_max:
            xi_int.integrate(xi_int.t + dr_generator.next())
            xi.append(xi_int.y)
            rs.append(xi_int.t)
            if check_crossing:
                if xi[0][-1]*xi[0][-2] < 0:
                    return True, np.array(xi), np.array(rs)

    return False, np.array(xi), np.array(rs)


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
        The relevant part of Newcomb's f for determining f=0. The term that can
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


def check_suydam(r, b_z_spl, b_theta_spl, p_prime_spl):
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
              'p_prime_spl': p_prime_spl}
    unstable_mask = np.invert(frob.sings_suydam_stable(**params))
    return r[unstable_mask]
