# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:23:58 2015

@author: jensv
"""

import skin_core_scanner_simple as scss
reload(scss)
import equil_solver as es
reload(es)
import newcomb_simple as new
reload(new)

(lambda_a_mesh, k_a_mesh,
 stability_maps) = scss.scan_lambda_k_space([0.01, 3.0, 25.], [0.01, 1.5, 25],
                                            epsilon=0.11, core_radius_norm=0.9,
								 transition_width_norm=0.033,
								 skin_width_norm=0.034,
								 method='lsoda',
								 max_step=1E-2, nsteps=1000)