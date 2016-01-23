# -*- coding: utf-8 -*-
"""
Created on Sun Feb 08 15:36:58 2015

@author: Jens von der Linden
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatibility"""

import sys
sys.path.append('scipy_mod')

import fitpack
reload(fitpack)
from fitpack import splev

import numpy as np
from numpy import atleast_1d
import scipy.integrate

import newcomb_init as init
import singularity_frobenius as frob
import find_singularties as find_sing
import external_stability as ext
import newcomb_f as new_f
import newcomb_g as new_g



def stability(params, offset, suydam_offset, suppress_output=False,
              method='lsoda', rtol=None, max_step=None, nsteps=None,
              xi_given=[0., 1.], diagnose=False, sing_search_points=10000,
              f_func=new_f.newcomb_f_16, g_func=new_g.newcomb_g_18_dimless_wo_q,
              skip_external_stability=False, stiff=False, use_jac=True,
              adapt_step_size=False, adapt_min_steps=500):
    r"""
    Determine external stability.

    Parameters
    ----------
    params: dict
        equilibrium parameters including spline coefficients
    offset : float
        offset after which to start integrating after singularties
    suydam_offset : float
        offset after which to start integrating after suydam unstable
        singularties
    suppress_output: boolean
        flag to suppress diagnostic print statements
    method: string
        integration method to use. Either an integrator in scipy.integate.ode
        or 'odeint' for scipy.integrate.odeint
    rtol : float
        passed to ode solver relative tolerance setting for ODE integrator
    max_step: float
        option passed to ode solver, limit for max step size
    nsteps: int
        option passed to ode solder, maximum number of steps allowed during call
        to solver.
    xi_given: tuple of floats
        Initial condition used for xi if equilibrim does not start ar r=0
    diagnose: bool
        flag to print out more diagnostc statements during integration
    f_func: function
        python function to use to calculate f
    g_func: function
        python function to use to calculate g
    Returns
    -------
    stable_external : bool
        True if delta_w > 0
    suydam_stable : bool
        True if all singularties (except r=0) are Suydam stable
    delta_w : float
        total perturbed potential energy
    missing_end_params:
        diagnostic ouput no longer used.
    xi : ndarray
        xi at interval boundary (only last element is relevant)
    xi_der : ndarray
        derivative of xi at interval boundary (only last element is relevant)
    Notes
    -----
    Examines the equilibrium. If the equilibrium has a singularity, the
    frobenius method is used to determine a small solution at an r > than
    instability. If the singularity is suydam unstable no attempt is made to
    calulate external stability.
    If there is no frobenius instability power series solution
    close to r=0 is chosen or if the integration does not start at r=0 a given
    xi is used as boundary condition.
    Only the last interval is integrated.
    To save time xi and xi_der are only evaluated at r=a (under the hood the
    integrator is evaluating xi and xi_der across the interval).
    Xi and Xi_der are plugged into the potential energy equation to determine
    stability.
    """
    params.update({'f_func': f_func, 'g_func': g_func})
    missing_end_params = None

    if params['m'] == -1:
        sing_params = {'a': params['r_0'], 'b': params['a'],
                       'points': sing_search_points, 'k': params['k'],
                       'm': params['m'], 'b_z_spl': params['b_z'],
                       'b_z_prime_spl': params['b_z_prime'],
                       'b_theta_spl': params['b_theta'],
                       'b_theta_prime_spl': params['b_theta_prime'],
                       'p_prime_spl': params['p_prime'], 'offset': offset,
                       'tol': 1E-2, 'beta_0': params['beta_0']}
        (interval,
         starts_with_sing,
         suydam_stable,
         suydam_unstable_interval) = intervals_with_singularties(suppress_output,
                                                                 **sing_params)
    else:
        suydam_stable = True
        starts_with_sing = False
        suydam_unstable_interval = False
        interval = [params['r_0'], params['a']]

    interval, init_value = setup_initial_conditions(interval, starts_with_sing,
                                                    offset, suydam_offset,
                                                    xi_given=xi_given,
                                                    **params)
    if not suydam_unstable_interval:
        if skip_external_stability:
            (xi, xi_der, r_array) = newcomb_int(params, interval,
                                                init_value, method,
                                                diagnose, max_step,
                                                nsteps, rtol,
                                                skip_external_stability=True,
                                                stiff=stiff,
                                                use_jac=use_jac,
                                                adapt_step_size=adapt_step_size,
                                                adapt_min_steps=adapt_min_steps)
            return xi, xi_der

        (stable_external, delta_w,
         missing_end_params, xi, xi_der,
         r_array) = newcomb_int(params, interval, init_value, method,
                                diagnose, max_step, nsteps, rtol,
                                stiff=stiff, use_jac=use_jac,
                                adapt_step_size=adapt_step_size,
                                adapt_min_steps=adapt_min_steps)
    else:
        msg = ("Last singularity is suydam unstable. " +
               "Unable to deterime external instability at k = %.3f."
               % params['k'])
        print(msg)
        delta_w = None
        stable_external = None
        xi = np.asarray([np.nan])
        xi_der = np.asarray([np.nan])
        r_array = np.asarray([np.nan])
    return (stable_external, suydam_stable, delta_w, missing_end_params, xi,
            xi_der, r_array)


def newcomb_der(r, y, k, m, b_z_spl, b_z_prime_spl, b_theta_spl,
                b_theta_prime_spl, p_prime_spl, q_spl, q_prime_spl,
                f_func, g_func, beta_0):
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
    b_z_spl : scipy spline tck tuple
        axial magnetic field
    b_theta_spl : scipy spline tck tuple
        azimuthal magnetic field
    b_theta_prime_spl: scipy spline tck tuple
        radial derivative of azimuthal magnetic field
    p_prime_spl : scipy spline tck tuple
        derivative of pressure
    q_spl : scipy spline tck tuple
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

    r_arr = np.asarray(r)
    r_arr = atleast_1d(r_arr).ravel()

    g_params = {'r': r, 'k': k, 'm': m, 'b_z': splev(r_arr, b_z_spl),
                'b_theta': splev(r_arr, b_theta_spl),
                'p_prime': splev(r_arr, p_prime_spl),
                'beta_0': beta_0}

    f_params = {'r': r, 'k': k, 'm': m, 'b_z': splev(r_arr, b_z_spl),
                'b_theta': splev(r_arr, b_theta_spl), 'q': splev(r_arr, q_spl)}

    y_prime[0] = y[1] / f_func(**f_params)
    y_prime[1] = y[0]*g_func(**g_params)
    return y_prime


def newcomb_jac(r, y, k, m, b_z_spl, b_z_prime_spl, b_theta_spl,
                b_theta_prime_spl, p_prime_spl, q_spl, q_prime_spl,
                f_func, g_func, beta_0):
    r"""
    Jacobian
    """
    r_arr = np.asarray(r)
    r_arr = atleast_1d(r_arr).ravel()
    g_params = {'r': r, 'k': k, 'm': m, 'b_z': splev(r_arr, b_z_spl),
                'b_theta': splev(r_arr, b_theta_spl),
                'p_prime': splev(r_arr, p_prime_spl),
                'beta_0': beta_0}

    f_params = {'r': r, 'k': k, 'm': m, 'b_z': splev(r_arr, b_z_spl),
                'b_theta': splev(r_arr, b_theta_spl), 'q': splev(r_arr, q_spl)}

    jac = np.zeros((2,2))
    jac[0,1] = 1. / f_func(**f_params)
    jac[1,0] = g_func(**g_params)
    return jac

def newcomb_der_for_odeint(y, r, *args):
    r"""
    odeint uses a derivative function with y and r passed as arguments in
    reverse order.
    """
    return newcomb_der(r, y, *args)


def divide_by_f(r, xi_der_f, k, m, b_z_spl, b_theta_spl, q_spl, f_func):
    r"""
    Divides :math:`y[1]=f \xi'` by f to recover :math:`\xi`.
    """
    r_arr = np.asarray(r)
    r_arr = atleast_1d(r_arr).ravel()
    f_params = {'r': r, 'k': k, 'm': m, 'b_z': splev(r_arr, b_z_spl),
                'b_theta': splev(r_arr, b_theta_spl), 'q': splev(r_arr, q_spl)}
    return xi_der_f / f_func(**f_params)


def intervals_with_singularties(suppress_output, **sing_params):
    r"""
    Determines if an interval starts with a singularity, is Suydam unstable.
    """
    starts_with_sing = False
    suydam_unstable_interval = False
    suydam_stable = False
    interval = [sing_params['a'], sing_params['b']]
    (sings,
     sings_wo_0, intervals) = find_sing.identify_singularties(**sing_params)

    if not sings_wo_0.size == 0:
        if not suppress_output:
            print("Non-geometric singularties identified at r =", sings_wo_0)
        interval = [sings_wo_0[-1], sing_params['b']]
        starts_with_sing = True
    # Check singularties for Suydam stability
    suydam_result = check_suydam(sings_wo_0, **sing_params)
    if suydam_result.size != 0:
        suydam_stable = False
        if sings_wo_0.size > 0 and abs(suydam_result[-1] - sings_wo_0[-1]) < 1e-8:
            if not suppress_output:
                print("Profile is Suydam unstable at r =", suydam_result)
            suydam_unstable_interval = True
    else:
        suydam_stable = True
    return interval, starts_with_sing, suydam_stable, suydam_unstable_interval


def setup_initial_conditions(interval, starts_with_sing, offset,
                             suydam_offset, xi_given=[0., 1.], **params):
    r"""
    Returns the initial condition to use for integrating an interval.
    """

    if interval[0] == 0.:
        interval[0] += offset
        if interval[0] > interval[1]:
            interval[0] = interval[1]
        init_params = dict(params)
        r_arr = np.asarray(interval[0])
        r_arr = atleast_1d(r_arr).ravel()
        init_params.update({'b_z': splev(r_arr, params['b_z']),
                            'b_theta': splev(r_arr, params['b_theta']),
                            'q': splev(r_arr, params['q'])})
        init_value = init.init_geometric_sing(interval[0], **init_params)
    else:
        if starts_with_sing:
            if interval[0]+suydam_offset > interval[1]:
                suydam_offset = interval[1] - interval[0]
            frob_params = {'offset': suydam_offset, 'b_z_spl': params['b_z'],
                           'b_z_prime_spl': params['b_z_prime'],
                           'b_theta_spl': params['b_theta'],
                           'b_theta_prime_spl': params['b_theta_prime'],
                           'q_spl': params['q'], 'f_func': new_f.newcomb_f_16,
                           'p_prime_spl': params['p_prime'],
                           'beta_0': params['beta_0'], 'r_sing': interval[0]}
            xi_given = frob.sing_small_solution(**frob_params)
            interval[0] += suydam_offset
            init_params = dict(params)
            r_arr = np.asarray(interval[0])
            r_arr = atleast_1d(r_arr).ravel()
            init_params.update({'b_z': splev(r_arr, params['b_z']),
                                'b_theta': splev(r_arr, params['b_theta']),
                                'q': splev(r_arr, params['q'])})
            init_value = init.init_xi_given(xi_given, r_arr, **init_params)

        else:
            init_params = dict(params)
            init_params.pop('r')
            init_params.update({'b_z': splev(r_arr, params['b_z']),
                                'b_theta': splev(r_arr, params['b_theta']),
                                'q': splev(r_arr, params['q'])})
            init_value = init.init_xi_given(xi_given, interval[0], **init_params)
    return interval, init_value


def check_suydam(r, b_z_spl, b_z_prime_spl, b_theta_spl, b_theta_prime_spl,
                 p_prime_spl, beta_0, **kwargs):
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
    params = {'r': r, 'b_z_spl': b_z_spl, 'b_z_prime_spl': b_z_prime_spl,
              'b_theta_spl': b_theta_spl,
              'b_theta_prime_spl': b_theta_prime_spl,
              'p_prime_spl': p_prime_spl, 'beta_0': beta_0}
    unstable_mask = np.invert(frob.sings_suydam_stable(**params))
    return r[unstable_mask]


def newcomb_int(params, interval, init_value, method, diagnose, max_step,
                nsteps, rtol, skip_external_stability=False, stiff=False,
                use_jac=True, adapt_step_size=False, adapt_min_steps=500):
    r"""
    Integrates newcomb's euler Lagrange equation in a given interval with lsoda
    either with the scipy.ode object oriented interface or with scipy.odeint.
    """
    missing_end_params = None
    #print('k_bar', params['k'], 'interval:', interval[0], interval[1], init_value)
    args = (params['k'], params['m'], params['b_z'], params['b_z_prime'],
            params['b_theta'], params['b_theta_prime'], params['p_prime'],
            params['q'], params['q_prime'], params['f_func'], params['g_func'],
            params['beta_0'])

    if adapt_step_size:
        interval_list = [interval]
        max_step_list = [10.**(np.floor(np.log10(1 - params['core_radius']))-1)]
        nsteps_for_list = 10.**(np.abs(np.log10(max_step_list[0]))+.5) * (1 - params['core_radius'])
        nsteps_for_list = adapt_min_steps if nsteps_for_list < adapt_min_steps else nsteps_for_list
        nsteps_list = [nsteps_for_list]
        if interval[0] < params['core_radius']:
            interval_list.insert(0, [interval[0], params['core_radius']])
            interval_list[1] = [params['core_radius'], interval[1]]
            max_step_list.insert(0, max_step)
            nsteps_list.insert(0, nsteps)
    else:
        interval_list = [interval]
        max_step_list = [max_step]
        nsteps_list = [nsteps]


    for i, interval in enumerate(interval_list):
        max_step = max_step_list[i]
        nsteps = nsteps_list[i]
        if diagnose:
            r_array = np.linspace(interval[0], interval[1], 250)
        else:
            r_array = np.asarray(interval)


        if method == 'lsoda_odeint':
            if 'core_radius' in params.keys():
                transition_points = np.asarray([params['core_radius'],
                                                params['core_radius'] +
                                                params['transition_width'],
                                                params['core_radius'] +
                                                params['transition_width'] +
                                                params['skin_width']])
                tcrit = np.asarray(transition_points[np.less(interval[0], transition_points)])
            else:
                tcrit = None

            integrator_args = {}
            if rtol is not None:
                integrator_args['rtol'] = rtol
            if nsteps is not None:
                integrator_args['mxstep'] = nsteps
            if max_step is not None:
                integrator_args['hmax'] = max_step
            if use_jac:
                results = scipy.integrate.odeint(newcomb_der_for_odeint,
                                                 np.asarray(init_value),
                                                 np.asarray(r_array),
                                                 Dfun=newcomb_jac,
                                                 tcrit=tcrit,
                                                 args=args,
                                                 **integrator_args)
            else:
                results = scipy.integrate.odeint(newcomb_der_for_odeint,
                                                 np.asarray(init_value),
                                                 np.asarray(r_array),
                                                 tcrit=tcrit,
                                                 args=args,
                                                 **integrator_args)
            xi = np.asarray([results[:, 0]]).ravel()
            xi_der_f = np.asarray([results[:, 1]]).ravel()
        else:
            if use_jac:
                integrator = scipy.integrate.ode(newcomb_der, jac=newcomb_jac)
            else:
                integrator = scipy.integrate.ode(newcomb_der)

            integrator_args = {}
            if rtol is not None:
                integrator_args['rtol'] = rtol
            if nsteps is not None:
                integrator_args['nsteps'] = nsteps
            if max_step is not None:
                integrator_args['max_step'] = max_step
            if stiff:
                integrator_args['method'] = 'bdf'

            integrator.set_integrator(method, **integrator_args)
            integrator.set_f_params(*args)
            integrator.set_jac_params(*args)
            integrator.set_initial_value(init_value, interval[0])
            results = np.empty((r_array.size, 2))
            results[0] = init_value
            for i, r in enumerate(r_array[1:]):
                integrator.integrate(r)
                results[i+1, :] = integrator.y
                if not integrator.successful():
                    results[i+1:, :] = [np.nan, np.nan]
                    break
            xi = results[:, 0]
            xi_der_f = results[:, 1]
        init_value = [xi[-1], xi_der_f[-1]]

    xi_der = divide_by_f(r_array,
                         xi_der_f,
                         params['k'],
                         params['m'],
                         params['b_z'],
                         params['b_theta'],
                         params['q'],
                         params['f_func'])

    if np.all(np.isfinite(results[-1])):
        if skip_external_stability:
            return xi, xi_der, r_array
        (stable_external,
         delta_w) = ext.external_stability_from_analytic_condition(params,
                                                                   xi[-1],
                                                                   xi_der[-1],
                                                                   without_sing=True,
                                                                   dim_less=True)
        #print(delta_w)
    else:
        msg = ("Integration to plasma edge did not succeed. " +
               "Can not determine external stability at k = %.3f."
               % params['k'])
        print(msg)
        missing_end_params = params
        stable_external = None
        delta_w = None
    return (stable_external, delta_w, missing_end_params, xi, xi_der, r_array)
