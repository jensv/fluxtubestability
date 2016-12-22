# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 22:05:21 2015

@author: jensv

Script to scan k_bar-lambda_bar stability space of profile type.
"""

import equil_solver as es
import newcomb as new
import numpy as np
from copy import deepcopy
from datetime import datetime
import os
import json
import sys
import call_provenance as cp
import sqlite3
import argparse


def scan_lambda_k_space(lambda_bar_space, k_bar_space,
                        xi_factor=1., magnetic_potential_energy_ratio=1.,
                        offset=1E-3, r_0=0., init_value=(0.0, 1.0),
                        rtol=None, max_step=1E-2,
                        nsteps=1000, method='lsoda', suppress_output=True,
                        diagnose=False, stiff=False, use_jac=True,
                        adapt_step_size=True, sing_search_points=1000,
                        profile_type='default',
                        **kwargs):
    r"""
    Scans space given by lambda_a_space and k_a_space for m=0, 1 stability and
    saves several 2D numpy arrays (stability maps) to an npz file.
    """
    lambda_a_space = lambda_bar_space
    k_a_space = k_bar_space
    call_parameters = locals()
    func_name = 'scan_lambda_k_space'

    suydam_end_offset = offset

    k_a_points = np.linspace(k_a_space[0], k_a_space[1], num=k_a_space[2])
    lambda_a_points = np.linspace(lambda_a_space[0], lambda_a_space[1],
                                  num=lambda_a_space[2])

    lambda_a_mesh, k_a_mesh = np.meshgrid(lambda_a_points, k_a_points)

    mesh_shape = lambda_a_mesh.shape

    delta_map = {-1: np.zeros(mesh_shape),
                 0: np.zeros(mesh_shape)}

    sub_stability_maps = {-1: np.ones(mesh_shape),
                          0: np.ones(mesh_shape)}

    stability_maps = {'external': deepcopy(sub_stability_maps),
                      'd_w': deepcopy(sub_stability_maps),
                      'd_w_norm': deepcopy(sub_stability_maps),
                      'suydam': deepcopy(sub_stability_maps)}

    for i, lambda_a in enumerate(lambda_a_points):
        print('lambda_bar = %.3f' % lambda_a)
        for j, k_a in enumerate(k_a_points):
            for m in [-1, 0]:
                results = None
                if profile_type == 'default':
                    profile = es.UnitlessSmoothedCoreSkin(k_bar=k_a,
                                                          lambda_bar=lambda_a,
                                                          **kwargs)


                    params = {'k': k_a, 'm': float(m), 'r_0': r_0, 'a': 1.,
                              'b': 'infinity'}
                    params_wo_splines = deepcopy(params)
                    params.update(profile.get_tck_splines())
                    params.update({'xi_factor': xi_factor,
                                   'magnetic_potential_energy_ratio': magnetic_potential_energy_ratio,
                                   'beta_0': profile.beta_0(),
                                   'core_radius': profile.core_radius,
                                   'transition_width': profile.transition_width,
                                   'skin_width': profile.skin_width,
                                   'points_core': profile.points_core,
                                   'points_transition': profile.points_transition,
                                   'points_skin': profile.points_skin,
                                   'epsilon': profile.epsilon})

                elif profile_type == 'diffuse_core_skin':
                    profile = es.UnitlessExponentialDecaySkin(k_bar=k_a,
                                                              lambda_bar=lambda_a,
                                                              **kwargs)
                    params = {'k': k_a, 'm': float(m), 'r_0': r_0, 'a': 1.,
                               'b': 'infinity'}
                    params_wo_splines = deepcopy(params)
                    params.update(profile.get_tck_splines())
                    params.update({'xi_factor': xi_factor,
                                   'magnetic_potential_energy_ratio': magnetic_potential_energy_ratio,
                                   'beta_0': profile.beta_0(),
                                   'core_radius': profile.core_radius,
                                   'transition_width': None,
                                   'skin_width': profile.skin_width,
                                   'points_core': profile.points_core,
                                   'points_transition': None,
                                   'points_skin': profile.points_skin,
                                   'epsilon': profile.epsilon})
                results = new.stability(params, offset, suydam_end_offset,
                                        sing_search_points=sing_search_points,
                                        suppress_output=suppress_output,
                                        xi_given=init_value,
                                        rtol=rtol, max_step=max_step,
                                        nsteps=nsteps, method=method,
                                        diagnose=diagnose, stiff=stiff,
                                        use_jac=use_jac,
                                        adapt_step_size=adapt_step_size)
                stable_external = results[0]
                stable_suydam = results[1]
                delta_w = results[2]
                xi = results[4]
                xi_der = results[5]

                delta = xi_der[-1] / xi[-1]

                delta_map[m][j][i] = delta

                if delta_w is not None:
                    stability_maps['d_w'][m][j][i] = delta_w
                    stability_maps['d_w_norm'][m][j][i] = delta_w
                else:
                    stability_maps['d_w'][m][j][i] = np.nan
                    stability_maps['d_w_norm'][m][j][i] = np.nan

                if not stable_external:
                    stability_maps['external'][m][j][i] = 0.
                if stable_external and delta_w is None:
                    stability_maps['external'][m][j][i] = -1.
                if not stable_suydam:
                    stability_maps['suydam'][m][j][i] = 0.

    #normalize
    for m in [-1, 0]:
        stability_maps['d_w'][m] = (stability_maps['d_w'][m] /
                                    np.nanmax(np.abs(stability_maps['d_w'][m])))

    params_wo_splines.update({'lambda_a_space': lambda_a_space,
                              'k_a_space': k_a_space,
                              'sing_search_points': sing_search_points,
                              'offset': offset,
                              'suydam_end_offset': suydam_end_offset})
    params_wo_splines.update(kwargs)

    date = datetime.now().strftime('%Y-%m-%d-%H-%M')
    if os.getcwd().endswith('ipython_notebooks'):
        path = '../../output/' + date
        sql_db = '../../output/output.db'
    else:
        path = '../output/' + date
        sql_db = '../output/output.db'
    os.mkdir(path)

    track_provenance(sql_db, func_name, call_parameters, date, params,
                     lambda_a_space, k_a_space)

    params_wo_splines.pop('dr', None)
    with open(path+'/params.txt', 'w') as params_file:
        json.dump(params_wo_splines, params_file)
    np.savez(path+'/meshes.npz', lambda_a_mesh=lambda_a_mesh,
             k_a_mesh=k_a_mesh,
             external_m_neg_1=stability_maps['external'][-1],
             external_m_0=stability_maps['external'][0],
             d_w_m_neg_1=stability_maps['d_w'][-1],
             d_w_m_0=stability_maps['d_w'][0],
             d_w_norm_m_neg_1=stability_maps['d_w_norm'][-1],
             d_w_norm_m_0=stability_maps['d_w_norm'][0],
             suydam_m_0=stability_maps['suydam'][0],
             suydam_m_neg_1=stability_maps['suydam'][-1],
             delta_m_0=delta_map[0],
             delta_m_neg_1=delta_map[-1]
             )

    print('Saved in Directory:' + str(date))

    return lambda_a_mesh, k_a_mesh, stability_maps

def track_provenance(sql_db, func_name, call_parameters, date, params,
                     lambda_a_space, k_a_space):
    r"""
    Save parameters, call, and git commit to make results reproducible and allow
    easy scanning.
    """
    call = func_name + '('
    for key in call_parameters.keys():
        call += key + '=' + str(call_parameters[key]) + ', '
    call = call[:-2] + ')'
    call, git_commit = cp.call_and_git_commit(call=call, call_path=os.getcwd())
    assert os.path.exists(sql_db), ("Run database does not exist."
                                    "Try running init_database.py")
    connection = sqlite3.connect(sql_db)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO Runs(datetime, points_core, points_transition"+
                   ", points_skin, core_radius, transition_width, skin_width,"+
                   "k_bar_start, k_bar_end, k_bar_num, lambda_bar_start," +
                   "lambda_bar_end, lambda_bar_num, epsilon, git_commit," +
                   "python_call) " +
                   "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"+
                   ")", (
                         date,
                         params['points_core'],
                         params['points_transition'],
                         params['points_skin'],
                         params['core_radius'],
                         params['transition_width'],
                         params['skin_width'],
                         k_a_space[0],
                         k_a_space[1],
                         int(k_a_space[2]),
                         lambda_a_space[0],
                         lambda_a_space[1],
                         int(lambda_a_space[2]),
                         params['epsilon'],
                         git_commit,
                         call))
    connection.commit()
    cursor.close()
    connection.close()

def parse_args():
    parser = argparse.ArgumentParser(description="scan specified"
                                     "k_bar-lambda_bar space for specified"
                                     "profiles")
    parser.add_argument('--k_bar_space', help='specify k_bar space',
                        nargs=3, type=float, metavar=('START', 'END', 'POINTS'),
                        default=[0.01, 1.5, 50])
    parser.add_argument('--lambda_bar_space',
                        help="specify lambda_bar space:",
                        metavar=('START', 'END', 'POINTS'),
                        nargs=3, type=float, default=[0.01, 5.0, 50])
    parser.add_argument('--epsilon', help='core current to total current ratio',
                        type=float, default=0.5)
    parser.add_argument('--core_radius_norm', help='normalized core radius',
                        type=float, default=0.6)
    parser.add_argument('--transition_width_norm',
                        help='normalized transition width',
                        type=float, default=0.175)
    parser.add_argument('--skin_width_norm', help='normalized skin width',
                        type=float, default=0.05)
    parser.add_argument('--points_core', help="number of points from which"
                                              "to build core region splines",
                        type=int, default=50)
    parser.add_argument('--points_transition',
                        help="number of points from which to build"
                        "transition splines", type=int, default=50)
    parser.add_argument('--points_skin',
                        help="number of points from which to build"
                        "skin splines", type=int, default=50)
    parser.add_argument('--offset', help="offset after singularities at which"
                                         "to continue integrating from"
                                         "power series solutions",
                        type=float, default=1e-3)
    parser.add_argument('--rtol', help="relative tolerance setting for"
                                       "integrator",
                        type=float, default=None)
    parser.add_argument('--max_step', help="max step for integrator",
                        type=float, default=1e-2)
    parser.add_argument('--nsteps', help='number of steps for integrator',
                        type=int, default=1000)
    parser.add_argument('--method', help="integration method"
                        "from scipy.integrate.ode", type=str, default='lsoda')
    parser.add_argument('--suppress_output', help="flag to suppress output",
                        default=False, action='store_true')
    parser.add_argument('--diagnose', help="flag to make code try to integrate"
                                           "over whole fluxtube radius.",
                        default=False, action='store_true')
    parser.add_argument('--stiff', help="flag to pass stiff flag to integrator",
                        default=False, action='store_true')
    parser.add_argument('--use_jac', help="flag to use Jacobian"
                                          "with integrator",
                        action="store_true")
    parser.add_argument('--adapt_step_size',
                        help="flag to adapt the stepsize requirements"
                        "for the integrator to the fluxtube region", 
                        action="store_true")
    parser.add_argument('--sing_search_points',
                        help="number of points to divide fluxtube radius by in"
                             "search for singularities",
                        type=int, default=1000)

    args = parser.parse_args()
    args.k_bar_space[2] = int(args.k_bar_space[2])
    args.lambda_bar_space[2] = int(args.lambda_bar_space[2])
    return args

def main(args):
    r"""
    Run skin_core_scanner from command line call. 
    """
    scan_lambda_k_space(**vars(args))

if __name__ == "__main__":
    args = parse_args()
    main(args)

