# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:43:13 2015

@author: jensv

Do a broad scan of skin geometry, epsilo, and spline knots.
"""

import numpy as np
import skin_core_scanner_simple as scss

lambda_bar_space = [0.01, 6., 40]
k_bar_space = [0.01, 3., 40]
for skin_width in np.logspace(np.log10(0.001), np.log10(0.9), 5):
    for transition_width in np.logspace(np.log10(0.001),
                                        np.log10((1. - skin_width)/2.), 5):
        core_radius = 1. - 2.*transition_width - skin_width
        for epsilon in np.logspace(np.log10(0.01), np.log10(1.), 5):

            assert skin_width > 0, 'Negative skin_width'
            assert transition_width > 0, 'Negative transition_width'
            assert core_radius > 0, 'Negative core_radius'
            assert 1.1 >= epsilon >= 0,  'epsilon outside of [0,1] interval'
            assert (0.9 < skin_width + 2.*transition_width + core_radius <
                    1.1), 'total pinch radius not normalized to 1'

#            for points_skin in np.linspace(20, 200, 5):
#                for points_transition in np.linspace(50, 500, 5):
            scss.scan_lambda_k_space(lambda_bar_space, k_bar_space,
                                     skin_width_norm=skin_width,
                                     transition_width_norm=transition_width,
                                     core_radius_norm=core_radius,
                                     epsilon=epsilon)
