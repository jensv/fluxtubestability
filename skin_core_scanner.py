# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 11:32:14 2014

@author: Jens von der Linden
"""

import equil_solver as es
import newcomb as new
import numpy as np
from copy import deepcopy
from datetime import datetime
import os
import json


def scan_lambda_k_space(lambda_a_space, k_a_space, integration_points=250,
                        xi_factor=1., magnetic_potential_energy_ratio=1.,
                        **kwargs):
    r"""
    """
    sing_search_points = 1000
    dr = np.linspace(0, 1, integration_points)[1]
    offset = 1E-3
    suydam_end_offset = 1E-3

    k_a_points = np.linspace(k_a_space[0], k_a_space[1], num=k_a_space[2])
    lambda_a_points = np.linspace(lambda_a_space[0], lambda_a_space[1],
                                  num=lambda_a_space[2])

    lambda_a_mesh, k_a_mesh = np.meshgrid(lambda_a_points, k_a_points)
    mesh_shape = lambda_a_mesh.shape
    sub_stability_maps = {-1: np.ones(mesh_shape),
                          0: np.ones(mesh_shape),
                          1: np.ones(mesh_shape)}

    stability_maps = {'internal': sub_stability_maps,
                      'external': deepcopy(sub_stability_maps),
                      'd_w': deepcopy(sub_stability_maps),
                      'd_w_norm': deepcopy(sub_stability_maps),
                      'internal kink': np.empty(mesh_shape),
                      'external kink': np.empty(mesh_shape)}

    for i, lambda_a in enumerate(lambda_a_points):
        print('lambda_bar:', lambda_a)
        for j, k_a in enumerate(k_a_points):
            for m in [-1, 0, 1]:
                profile = es.UnitlessSmoothedCoreSkin(k_bar=k_a, lambda_bar=lambda_a,
                                                      **kwargs)

                params = {'k': k_a, 'm': float(m), 'r_0': 0., 'a': 1.,
                          'b': 'infinity'}
                params_wo_splines = deepcopy(params)
                params.update(profile.get_splines())

                params.update({'xi_factor': xi_factor,
                               'magnetic_potential_energy_ratio': magnetic_potential_energy_ratio,
                               'beta_0': params['beta'](0)})

                results = new.stability(dr, offset, suydam_end_offset,
                                        sing_search_points, params,
                                        suppress_output=True)
                stable_internal = results[0]
                stable_external = results[1]
                xi = results[2]
                xi_der = results[3]
                delta_w = results[5]

                if delta_w is not None:
                    stability_maps['d_w'][m][j][i] = delta_w
                    stability_maps['d_w_norm'][m][j][i] = delta_w
                else:
                    stability_maps['d_w'][m][j][i] = np.nan
                    stability_maps['d_w_norm'][m][j][i] = np.nan

                if not stable_internal:
                    stability_maps['internal'][m][j][i] = 0.
                if not stable_external:
                    stability_maps['external'][m][j][i] = 0.
                if stable_external and delta_w is None:
                    stability_maps['external'][m][j][i] = -1.

    stability_maps['internal kink'] = (stability_maps['internal'][-1] +
                                       stability_maps['internal'][1] > 1.5).astype(int)
    stability_maps['external kink'] = (stability_maps['external'][-1] +
                                       stability_maps['external'][1] > 1.5).astype(int)

    params_wo_splines.update({'lambda_a_space': lambda_a_space,
                              'k_a_space': k_a_space,
                              'sing_search_points': sing_search_points,
                              'dr': dr,
                              'offset': offset,
                              'suydam_end_offset': suydam_end_offset})
    params_wo_splines.update(kwargs)

    date = datetime.now().strftime('%Y-%m-%d-%H-%M')
    if os.getcwd().endswith('ipython_notebooks'):
        path = '../../output/' + date
    else:
        path = '../output/' + date
    os.mkdir(path)

    with open(path+'/params.txt', 'w') as params_file:
        json.dump(params_wo_splines, params_file)

    np.savez(path+'/meshes.npz', lambda_a_mesh=lambda_a_mesh,
             k_a_mesh=k_a_mesh,
             internal_m_neg_1=stability_maps['internal'][-1],
             internal_m_0=stability_maps['internal'][0],
             internal_m_1=stability_maps['internal'][1],
             external_m_neg_1=stability_maps['external'][-1],
             external_m_0=stability_maps['external'][0],
             external_m_1=stability_maps['external'][1],
             internal_kink=stability_maps['internal kink'],
             external_kink=stability_maps['external kink'],
             d_w_m_neg_1=stability_maps['d_w'][-1],
             d_w_m_0=stability_maps['d_w'][0],
             d_w_m_1=stability_maps['d_w'][1],
             d_w_norm_m_neg_1=stability_maps['d_w_norm'][-1],
             d_w_norm_m_0=stability_maps['d_w_norm'][0],
             d_w_norm_m_1=stability_maps['d_w_norm'][1]
             )

    return lambda_a_mesh, k_a_mesh, stability_maps
