# -*- coding: utf-8 -*-
"""
Created on Sun Feb 08 15:36:58 2015

@author: Jens von der Linden
"""

import scipy.integrate
import newcomb_init as init
import singularity_frobenius as frob


def stability(params, offset, suydam_offset, suppress_output=False,
              method='lsoda', rtol=None, max_step=1E-2, nsteps=1000,
              xi_given=[0., 1.], diagnose=False, suppress_output=True):
    r"""
    Determine external stability.
    """

    missing_end_params = None

    if params['m'] == -1:
        sing_params = {'a': params['r_0'], 'b': params['a'],
                       'points': sing_search_points, 'k': params['k'],
                       'm': params['m'], 'b_z_spl': params['b_z'],
                       'b_theta_spl': params['b_theta'], 'offset': offset,
                       'tol': 1E-2}
        (interval,
         starts_with_sing) = intervals_with_singularties(suydam_stable,
                                                         suppress_output,
                                                         **sing_params)[-1]
    else:
        interval = [params['r_0'], params['a']]

    setup_initial_conditions(interval, starts_with_sing, off_set,
                             suydam_offset, **params)

    if diagnose:
        r_array = np.linspace(interval[0], interval[1], 250)
    else:
        r_array = np.asarray(interval)
    args = (params['k'], params['m'], params['b_z_spl'], params['b_theta_spl'],
            params['p_prime_spl'], params['q_spl'], params['f_func'],
            params['g_func'], params['beta_0'])

    if method == 'lsoda':
        transition_points = np.asarray([params['core_radius'],
                                        parmas['core_radius'] +
                                        params['transition_width'],
                                        params['core_radius'] +
                                        params['transition_width'] +
                                        params['skin_width']])
        tcrit = transition_points[np.less(trasition_points, interval[0])]

        resuts = scipy.integrate.odeint(newcomb_der, init_value, r_array,
                                        args=args, tcrit=tcrit, hmax=max_step,
                                        mxstep=nsteps)

    else:
        integrator = scipy.integrate.ode(newcomb_der)
        integrator.set_integrator(method)
        integrator.set_f_params(args)
        integrator.set_initial_value()
        results = np.empty(r_array.size, 2)
        for i, r in enumerate(r_array[1:]):
            integrator.integrate(r)
            if not integrator.sucessful():
                break
        else:
            results[-1, :] = [np.nan, np.nan]

    xi = results[:, 0]
    xi_der = results[:, 1]

    if np.all(np.isfinite(results[-1])):
        (stable_external,
         delta_w) = ext.external_stability_from_notes(params, xi[-1],
                                                      xi_der[-1],
                                                      dim_less=True)
        stable = True if delta_w >= 0. else False
    else:
        msg = ("Integration to plasma edge did not succeed. " +
               "Can not determine external stability.")
        print(msg)
        missing_end_params = params
        stable = None
        delta_w = None
    return (stable, suydam_stable, delta_w, missing_end_params)


def newcomb_der(r, y, k, m, b_z_spl, b_theta_spl, p_prime_spl, q_spl,
                f_func, g_func, beta_0):
    r"""
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


def intervals_with_singularties(stable, suydam_stable, suppress_output,
                                **sing_params):
    r"""
    """"
    starts_with_sing = False
    (sings,
     sings_wo_0, intervals) = find_sing.identify_singularties(**sing_params)

    if not suppress_output:
        if not sings_wo_0.size == 0:
            print("Non-geometric singularties identified at r =", sings_wo_0)
            starts_with_sing = True
            interval = [sings_wo_0[-1], sing_params['b']]
    # Check singularties for Suydam stability
    suydam_result = check_suydam(sings, params['b_z'], params['b_theta'],
                                 params['p_prime'], params['beta_0'])
    if suydam_result.size != 0:
        if (not suydam_result.size == 1 or not suydam_result[0] == 0.):
            suydam_stable = False
            if not suppress_output:
                print("Profile is Suydam unstable at r =", suydam_result)
    else:
        suydam_stable = True
    return interval, starts_with_sing, suydam_stable


def setup_initial_conditions(interval, starts_with_sing, off_set,
                             suydam_offset, **params):

    if interval[0] == 0.:
        interval[0] += offset
        init_values = init.init_geometric_sing(offset, **params)
    else:
        if starts_with_sing:
            frob_params = {'offset': suydam_offset, 'b_z_spl': params['b_z'],
                           'b_theta_spl': params['b_theta'],
                           'p_prime_spl': params['p_prime'],
                           'q_spl': params['q'], 'f_func': new_f.newcomb_f_16,
                           'beta_0': params['beta_0'], 'r_sing': interval[0]}
            xi_given = frob.small_solution(**frob_params)
            interval[0] += suydam_offset
            init_values = init.init_xi_given(xi_given, interval[0], **params)

        else:
            init_value = init.init_xi_given(xi_given, interval[0], **params)

    return interval, init_value
