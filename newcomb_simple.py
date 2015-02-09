# -*- coding: utf-8 -*-
"""
Created on Sun Feb 08 15:36:58 2015

@author: Jens von der Linden
"""

def stability(dr, offset, params, suppress_output=False,
              external_only=True, atol=None, max_step=1E-2, nsteps=1000):
    r"""
    """

    missing_end_params = None

    # Test if integration to plasma edge was successful,
    # Can external stability be determined?
    if (r_array.size != 0 and not np.isnan(r_array[-1][-1]) and
        np.abs(r_array[-1][-1] - params['a']) < 1E-1):

        (stable_external,
         delta_w) = ext.external_stability_from_notes(params, xi[-1][-1],
                                                      xi_der[-1][-1],
                                                      dim_less=True)
    else:
        msg = ("Integration to plasma edge did not succeed. " +
               "Can not determine external stability.")
        print(msg)
        missing_end_params = params
        stable_external = True
        delta_w = None

    # Output stability messages
    # Can external stability be determined?
    k = params['k']
    m = params['m']


def newcomb_der(r, y, k, m, b_z_spl, b_theta_spl, p_prime_spl, q_spl,
                f_func, g_func, beta_0):
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