# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:43:13 2015

@author: jensv

Do a broad scan of skin geometry, epsilo, and spline knots.
"""

import numpy as np
import skin_core_scanner_simple as scss

lambda_bar_space = [0.01, 6., 75]
k_bar_space = [0.01, 3., 75]
for skin_width in np.logspace(0.001, 0.9, 25):
    for transition_width in np.logspace(0.001, (1. - skin_width)/2., 25):
        core_radius = 1. - 2.*transition_width - skin_width
        for epsilon in np.logspace(0.01, 1., 25):
            for points_skin in np.linspace(20, 200, 5):
                for points_transition in np.linspace(50, 500, 5):
                    scss.scan_lambda_k_space(lambda_bar_space, k_bar_space,
                                             skin_width=skin_width,
                                             transition_width=transition_width,
                                             core_radius=core_radius,
                                             epsilon=epsilon,
                                             points_skin=points_skin,
                                             points_transition=points_transition)
