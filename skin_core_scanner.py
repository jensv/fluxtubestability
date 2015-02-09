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
import git
import MDSplus as mds
import json


def scan_lambda_k_space_mds(lambda_a_space, k_a_space, integration_points=250,
                            xi_factor=1., magnetic_potential_energy_ratio=1.,
                            rtol=None, **kwargs):
    r"""
    """
    tree = 'skin_core'
    if os.getcwd().endswith('ipython_notebooks'):
        repo = git.Repo('../')
    else:
        repo = git.Repo('.')
    commit = str(repo.head.reference.commit)

    date = datetime.now().strftime('%Y-%m-%d-%H-%M')

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
                      'external kink': np.empty(mesh_shape),
                      'suydam': deepcopy(sub_stability_maps)}

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
                                        suppress_output=True, rtol=rtol)
                stable_internal = results[0]
                stable_suydam = results[1]
                stable_external = results[2]
                xi = results[3]
                xi_der = results[4]
                r_array = results[5]
                residual_array = results[6]
                delta_w = results[7]

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
                if not stable_suydam:
                    stability_maps['suydam'][m][i][j] = 0.

    stability_maps['internal kink'] = (stability_maps['internal'][-1] +
                                       stability_maps['internal'][1] > 1.5).astype(int)
    stability_maps['external kink'] = (stability_maps['external'][-1] +
                                       stability_maps['external'][1] > 1.5).astype(int)


    epsilon = profile.epsilon
    core_radius = profile.core_radius
    transition_width = profile.transition_width
    skin_width = profile.skin_width
    points_core = profile.points_core
    points_transition = profile.points_transition
    points_skin = profile.points_skin

    params_wo_splines.update({'lambda_a_space': lambda_a_space,
                              'k_a_space': k_a_space,
                              'sing_search_points': sing_search_points,
                              'dr': dr,
                              'offset': offset,
                              'suydam_end_offset': suydam_end_offset,
                              'core_radius': core_radius,
                              'transition_width': transition_width,
                              'skin_width': skin_width,
                              'points_core': points_core,
                              'points_transition': points_transition,
                              'points_skin': points_skin})

    params_wo_splines.update(kwargs)

    shot = mds.TreeGetCurrentShotId('skin_core')
    shot += 1
    mds.TreeSetCurrentShotId('skin_core', shot)
    tree = mds.Tree('skin_core', -1, 'edit')
    #mds.TreeCreatePulseFile('skin_core', -1, 0)
    tree.createPulse(0)
    tree = mds.Tree('skin_core', shot, mode='edit')
    tree.getNode('.params:params').putData(params_wo_splines)

    tree.getNode('code_params:datetime').putData(date)
    tree.getNode('code_params:git_commit').putData(commit)

    tree.getNode('output:k_bar_mesh').putData(k_a_mesh)
    tree.getNode('output:lambda_mesh').putData(lambda_a_mesh)
    tree.getNode('output:dw_m_0').putData(stability_maps['d_w'][0])
    tree.getNode('output:dw_m_1').putData(stability_maps['d_w'][1])
    tree.getNode('output:dw_m_neg_1').putData(stability_maps['d_w'][-1])
    tree.getNode('output:suy_m_0').putData(stability_maps['d_w'][-1])
    tree.getNode('output:suy_m_1').putData(stability_maps['d_w'][-1])
    tree.getNode('output:suy_m_neg_1').putData(stability_maps['d_w'][-1])
    tree.write()
    tree.quit()

    print('Saved in Shot:' + str(shot))

    return lambda_a_mesh, k_a_mesh, stability_maps


def scan_lambda_k_space(lambda_a_space, k_a_space, integration_points=250,
                        xi_factor=1., magnetic_potential_energy_ratio=1.,
                        offset=1E-3, r_0=0., init_value=(0.0, 1.0),
                        external_only=True, rtol=None, **kwargs):
    r"""
    """
    sing_search_points = 1000
    dr = np.linspace(0, 1, integration_points)[1]
    suydam_end_offset = offset

    k_a_points = np.linspace(k_a_space[0], k_a_space[1], num=k_a_space[2])
    lambda_a_points = np.linspace(lambda_a_space[0], lambda_a_space[1],
                                  num=lambda_a_space[2])

    lambda_a_mesh, k_a_mesh = np.meshgrid(lambda_a_points, k_a_points)
    mesh_shape = lambda_a_mesh.shape

    delta_map = {-1: np.zeros(mesh_shape),
                  0: np.zeros(mesh_shape),
                  1: np.zeros(mesh_shape)}

    sub_stability_maps = {-1: np.ones(mesh_shape),
                          0: np.ones(mesh_shape),
                          1: np.ones(mesh_shape)}

    stability_maps = {'internal': sub_stability_maps,
                      'external': deepcopy(sub_stability_maps),
                      'd_w': deepcopy(sub_stability_maps),
                      'd_w_norm': deepcopy(sub_stability_maps),
                      'internal kink': np.empty(mesh_shape),
                      'external kink': np.empty(mesh_shape),
                      'suydam': deepcopy(sub_stability_maps)}

    for i, lambda_a in enumerate(lambda_a_points):
        print('lambda_bar:', lambda_a)
        for j, k_a in enumerate(k_a_points):
            for m in [-1, 0, 1]:
                profile = es.UnitlessSmoothedCoreSkin(k_bar=k_a, lambda_bar=lambda_a,
                                                      **kwargs)

                params = {'k': k_a, 'm': float(m), 'r_0': r_0, 'a': 1.,
                          'b': 'infinity'}
                params_wo_splines = deepcopy(params)
                params.update(profile.get_splines())

                params.update({'xi_factor': xi_factor,
                               'magnetic_potential_energy_ratio': magnetic_potential_energy_ratio,
                               'beta_0': params['beta'](0)})
                #dr = np.insert(np.diff(profile.r[profile.r > offset]), 0, offset)
                results = new.stability(dr, offset, suydam_end_offset,
                                        sing_search_points, params,
                                        suppress_output=True,
                                        init_value=init_value,
                                        external_only=external_only, rtol=rtol)
                stable_internal = results[0]
                stable_suydam = results[1]
                stable_external = results[2]
                xi = results[3]
                xi_der = results[4]
                r_array = results[5]
                residual_array = results[6]
                delta_w = results[7]
                delta = xi_der / xi

                delta_map[m][i][j] = delta

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
                if not stable_suydam:
                    stability_maps['suydam'][m][i][j] = 0.

    #normalize
    for m in [-1, 0, 1]:
        delta_w[m] = delta_w[m] / np.abs(stability_maps['d_w'][m]).max()



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

    params_wo_splines.pop('dr', None)
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
             d_w_norm_m_1=stability_maps['d_w_norm'][1],
             suydam_m_0=stability_maps['suydam'][0],
             suydam_m_1=stability_maps['suydam'][1],
             suydam_m_neg_1=stability_maps['suydam'][-1],
             delta_m_0=delta_map[0]
             delta_m_1=delta_map[1]
             delta_m_neg_1=delta_map[-1]
             )

    print('Saved in Directory:' + str(date))

    return lambda_a_mesh, k_a_mesh, stability_maps