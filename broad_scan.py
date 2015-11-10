# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:43:13 2015

@author: jensv

Do a broad scan of skin geometry, epsilo, and spline knots.
"""

import numpy as np
import skin_core_scanner_simple as scss

start_index = 95

lambda_bar_space = [0.01, 3., 50]
k_bar_space = [0.01, 1.5, 50]
#for skin_width in np.logspace(np.log10(0.001), np.log10(0.9), 5):
#    for transition_width in np.logspace(np.log10(0.001),
#                                        np.log10((1. - skin_width)/2.), 5):
#        core_radius = 1. - 2.*transition_width - skin_width
#        for epsilon in np.logspace(np.log10(0.01), np.log10(1.), 5):

#skin_widths = np.repeat(np.logspace(np.log10(0.001), np.log10(0.1), 30), 5)
#transition_widths = []
#for skin_width in np.logspace(np.log10(0.001), np.log10(0.9), 5):
#    transition_width = np.logspace(np.log10(0.001),
#                                   np.log10((1. - skin_width)/2.), 5)
#    transition_widths.append(transition_width)
#transition_widths = np.array(transition_widths)
#transition_widths = transition_widths.flatten()
#transition_widths = np.repeat(transition_widths, 5)



#epsilons = np.tile(np.logspace(np.log10(0.01), np.log10(1.), 5), 25)
transition_widths = np.logspace(np.log10(0.001), np.log10(0.1), 50)
skin_widths = np.ones(50) * 0.01
epsilons = np.tile(np.array([0.1, 0.5]), 25)

skin_widths = skin_widths[start_index:]
transition_widths = transition_widths[start_index:]
epsilons = epsilons[start_index:]

for i in xrange(skin_widths.size):
    (skin_width, transition_width,
     epsilon) = skin_widths[i], transition_widths[i], epsilons[i]
    core_radius = 1. - 2.*transition_width - skin_width

    assert skin_width > 0, 'Negative skin_width'
    assert transition_width > 0, 'Negative transition_width'
    assert core_radius >= 0, 'Negative core_radius'
    assert 1.1 >= epsilon >= 0, 'epsilon outside of [0,1] interval'
    assert (0.9 < skin_width + 2.*transition_width + core_radius <
            1.1), 'total pinch radius not normalized to 1'

#            for points_skin in np.linspace(20, 200, 5):
#                for points_transition in np.linspace(50, 500, 5):
    scss.scan_lambda_k_space(lambda_bar_space, k_bar_space,
                             skin_width_norm=skin_width,
                             transition_width_norm=transition_width,
                             core_radius_norm=core_radius,
                             epsilon=epsilon)
