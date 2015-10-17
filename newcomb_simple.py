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

import numpy as np
import scipy.integrate
from scipy.interpolate import splev

import newcomb_init as init
import singularity_frobenius as frob
import find_singularties as find_sing
import external_stability as ext
import newcomb_f as new_f
import newcomb_g as new_g
from copy import deepcopy



def stability(params, offset, suydam_offset, suppress_output=False,
              method='lsoda', rtol=None, max_step=None, nsteps=None,
              xi_given=[0., 1.], diagnose=False, sing_search_points=10000,
              f_func=new_f.newcomb_f_16, g_func=new_g.newcomb_g_18_dimless_wo_q):
    r"""
    Determine external stability.
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
                                                    **params)

    if not suydam_unstable_interval:
        (stable_external, delta_w,
         missing_end_params, xi, xi_der) = newcomb_int(params, interval,
                                                       init_value, method,
                                                       diagnose, max_step,
                                                       nsteps, rtol)
    else:
        if not suppress_output:
            msg = ("Last singularity is suydam unstable." +
                   "Unable to deterime external instability")
            print(msg)
            print(params['k'])
        delta_w = None
        stable_external = None
        xi = np.asarray([np.nan])
        xi_der = np.asarray([np.nan])
    return (stable_external, suydam_stable, delta_w, missing_end_params, xi,
            xi_der)


def newcomb_der(r, y, k, m, b_z_spl, b_z_prime_spl, b_theta_spl,
                b_theta_prime_spl, p_prime_spl, q_spl, q_prime_spl,
                f_func, g_func, beta_0):
    r"""
    Return the derivative of y
    """
    y_prime = np.zeros(2)

    g_params = {'r': r, 'k': k, 'm': m, 'b_z': splev(r, b_z_spl),
                'b_z_prime': splev(r, b_z_prime_spl),
                'b_theta': splev(r, b_theta_spl),
                'b_theta_prime': splev(r, b_theta_prime_spl),
                'p_prime': splev(r, p_prime_spl), 'q': splev(r, q_spl),
                'q_prime': splev(r, q_prime_spl),
                'beta_0': beta_0}

    f_params = {'r': r, 'k': k, 'm': m, 'b_z': splev(r, b_z_spl),
                'b_theta': splev(r, b_theta_spl), 'q': splev(r, q_spl)}

    y_prime[0] = y[1] / f_func(**f_params)
    y_prime[1] = y[0]*g_func(**g_params)
    return y_prime


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
        if not suppress_output:
            print("Profile is Suydam unstable at r =", suydam_result)
        if sings_wo_0.size > 0 and np.allclose(suydam_result[-1], sings_wo_0[-1]):
            suydam_unstable_interval = True
    else:
        suydam_stable = True
    return interval, starts_with_sing, suydam_stable, suydam_unstable_interval


def setup_initial_conditions(interval, starts_with_sing, offset,
                             suydam_offset, **params):
    r"""
    Returns the initial condition to use for integrating an interval.
    """

    if interval[0] == 0.:
        interval[0] += offset
        if interval[0] > interval[1]:
            interval[0] = interval[1]
        init_params = deepcopy(params)
        init_params.update({'b_z': splev(interval[0], params['b_z']),
                           'b_theta': splev(interval[0], params['b_theta']),
                            'q': splev((interval[0]), params['q'])})
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
            init_params = deepcopy(params)
            init_params.update({'b_z': splev(interval[0], params['b_z']),
                                'b_theta': splev(interval[0], params['b_theta']),
                                'q': splev(interval[0], params['q'])})
            init_value = init.init_xi_given(xi_given, interval[0], **init_params)

        else:
            init_value = init.init_xi_given(xi_given, interval[0], **params)
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
                nsteps, rtol):
    r"""
    Integrates newcomb's euler Lagrange equation in a given interval with lsoda
    either with the scipy.ode object oriented interface or with scipy.odeint.
    """
    missing_end_params = None
    #print('k_bar', params['k'], 'interval:', interval[0], interval[1], init_value)
    if diagnose:
        r_array = np.linspace(interval[0], interval[1], 250)
    else:
        r_array = np.asarray(interval)
    args = (params['k'], params['m'], params['b_z'], params['b_z_prime'],
            params['b_theta'], params['b_theta_prime'], params['p_prime'],
            params['q'], params['q_prime'], params['f_func'], params['g_func'],
            params['beta_0'])

    if method == 'lsoda_odeint':
        transition_points = np.asarray([params['core_radius'],
                                        params['core_radius'] +
                                        params['transition_width'],
                                        params['core_radius'] +
                                        params['transition_width'] +
                                        params['skin_width']])
        tcrit = np.asarray(transition_points[np.less(interval[0], transition_points)])

        integrator_args = {}
        if rtol is not None:
            integrator_args['rtol'] = rtol
        if nsteps is not None:
            integrator_args['mxstep'] = nsteps
        if max_step is not None:
            integrator_args['hmax'] = max_step

        results, output = scipy.integrate.odeint(newcomb_der, np.asarray(init_value),
                                                 np.asarray(r_array), args=args, tcrit=tcrit,
                                                 **integrator_args)

    else:
        integrator = scipy.integrate.ode(newcomb_der)

        integrator_args = {}
        if rtol is not None:
            integrator_args['rtol'] = rtol
        if nsteps is not None:
            integrator_args['nsteps'] = nsteps
        if max_step is not None:
            integrator_args['max_step'] = max_step

        integrator.set_integrator(method, **integrator_args)
        integrator.set_f_params(*args)
        integrator.set_initial_value(init_value, interval[0])
        results = np.empty((r_array.size, 2))
        results[0] = init_value
        for i, r in enumerate(r_array[1:]):
            integrator.integrate(r)
            results[i+1, :] = integrator.y
            if not integrator.successful():
                break
        else:
            results[i+1:-1, :] = [np.nan, np.nan]
    #print(results)
    xi = results[:, 0]
    xi_der = results[:, 1]

    if np.all(np.isfinite(results[-1])):
        (stable_external,
         delta_w) = ext.external_stability_from_notes(params, xi[-1],
                                                      xi_der[-1],
                                                      dim_less=True)
        #print(delta_w)
    else:
        msg = ("Integration to plasma edge did not succeed. " +
               "Can not determine external stability.")
        print(msg)
        missing_end_params = params
        stable_external = None
        delta_w = None
    return (stable_external, delta_w, missing_end_params, xi, xi_der)
