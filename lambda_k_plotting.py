# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 16:03:34 2015

@author: jensv
"""

from __future__ import print_function, unicode_literals, division
from __future__ import absolute_import
from future import standard_library, utils
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import numpy as np
import MDSplus as mds

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, BoundaryNorm
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')


def plot_lambda_k_space_dw(filename, name, mode_to_plot='m_neg_1',
                           show_points=False, lim=None, levels=None, log=False,
                           linthresh=1E-7, bounds=(1.5, 3.0)):
    r"""
    Plot the delta_w of external instabilities in the lambda-k space.
    """
    epsilon_case = np.load(filename)
    lambda_a_mesh = epsilon_case['lambda_a_mesh']
    k_a_mesh = epsilon_case['k_a_mesh']
    external_m_1 = epsilon_case['d_w_m_1']
    external_m_neg_1 = epsilon_case['d_w_m_neg_1']
    external_sausage = epsilon_case['d_w_m_0']
    external_m_1_norm = epsilon_case['d_w_norm_m_1']
    external_m_neg_1_norm = epsilon_case['d_w_norm_m_neg_1']
    external_sausage_norm = epsilon_case['d_w_norm_m_0']
    epsilon_case.close()

    instability_map = {'m_1': external_m_1_norm, 'm_0': external_sausage_norm,
                       'm_neg_1': external_m_neg_1_norm}

    kink_pal = sns.blend_palette([sns.xkcd_rgb["dandelion"],
                                  sns.xkcd_rgb["white"]], 7, as_cmap=True)
    sausage_pal = sns.blend_palette(['orange', 'white'], 7, as_cmap=True)
    instability_palette = {'m_1': kink_pal, 'm_0': sausage_pal,
                           'm_neg_1': kink_pal}

    values = instability_map[mode_to_plot]
    values = values / -np.nanmin(values)

    if levels:
        if log:
            plot = plt.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot],
                                levels=levels, norm=SymLogNorm(linthresh))
        else:
            norm = BoundaryNorm(levels, 256)
            plot = plt.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot],
                                levels=levels, norm=norm)
            cbar = plt.colorbar(label=r'$\delta W$',
                                format=FormatStrFormatter('%.0e'))
            contourlines = plt.contour(lambda_a_mesh, k_a_mesh, values,
                                       levels=levels[:-1], colors='grey')
            cbar.add_lines(contourlines)
    else:
        if log:
            plot = plt.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot],
                                norm=SymLogNorm(linthresh))
        else:
            plot = plt.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot])

    if lim:
        plot.set_clim(lim)
    plt.plot([0.01, 0.1, 1.0, 2.0, 3.0],
             [0.005, 0.05, 0.5, 1.0, 1.5], color='black')

    if show_points:
        plt.scatter(lambda_a_mesh, k_a_mesh, marker='o', c='b', s=5)
    plt.ylim(0.01, bounds[0])
    plt.xlim(0.01, bounds[1])
    plt.ylabel(r'$\bar{k}$', fontsize=25)
    plt.xlabel(r'$\bar{\lambda}$', fontsize=25)
    plt.savefig('../../output/plots/' + name + '.png')
    plt.show()


def plot_dw_mdsplus(tree, shot, name, mode_to_plot='m_neg_1',
                    show_points=False, lim=None, levels=None, log=False,
                    linthresh=1E-7):
    r"""
    Plot lambda k spcae loaded from mds+ tree.
    """
    tree = mds.Tree(tree, shot, 'normal')
    lambda_a_mesh = np.asarray(tree.getNode('.output:lambda_mesh').getData())
    k_a_mesh = np.asarray(tree.getNode('.output:k_bar_mesh').getData())
    external_m_1 = np.asarray(tree.getNode('.output:dw_m_1').getData())
    external_m_0 = np.asarray(tree.getNode('.output:dw_m_0').getData())
    external_m_neg_1 = np.asarray(tree.getNode('.output:dw_m_neg_1').getData())
    suydam_m_0 = np.asarray(tree.getNode('.output:suy_m_0').getData())
    suydam_m_1 = np.asarray(tree.getNode('.output:suy_m_1').getData())
    suydam_m_neg_1 = np.asarray(tree.getNode('.output:suy_m_neg_1').getData())

    instability_map = {'m_1': external_m_1, 'm_0': external_m_0,
                       'm_neg_1': external_m_neg_1}

    kink_pal = sns.blend_palette([sns.xkcd_rgb["dandelion"],
                                  sns.xkcd_rgb["white"]], 7, as_cmap=True)
    sausage_pal = sns.blend_palette(['orange', 'white'], 7, as_cmap=True)
    instability_palette = {'m_1': kink_pal, 'm_0': sausage_pal,
                           'm_neg_1': kink_pal}

    values = instability_map[mode_to_plot]
    values = values / -values.min()

    if levels:
        if log:
            plot = plt.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot],
                                levels=levels, norm=SymLogNorm(linthresh))
        else:
            norm = BoundaryNorm(levels, 256)
            plot = plt.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot],
                                levels=levels, norm=norm)
            cbar = plt.colorbar(label=r'$\delta W$',
                                format=FormatStrFormatter('%.0e'))
            contourlines = plt.contour(lambda_a_mesh, k_a_mesh, values,
                                       levels=levels[:-1], colors='grey')
            cbar.add_lines(contourlines)
    else:
        if log:
            plot = plt.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot],
                                norm=SymLogNorm(linthresh))
        else:
            plot = plt.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot])

    if show_points:
        plt.scatter(lambda_a_mesh, k_a_mesh, marker='o', c='b', s=5)

    if lim:
        plot.set_clim(lim)
    plt.plot([0.01, 0.1, 1.0, 2.0, 3.0],
             [0.005, 0.05, 0.5, 1.0, 1.5], color='black')
    plt.ylim(0.01, 1.5)
    plt.xlim(0.01, 3.)
    plt.ylabel(r'$\bar{k}$', fontsize=25)
    plt.xlabel(r'$\bar{\lambda}$', fontsize=25)
    plt.savefig('../../output/plots/' + name + '.png')
    plt.show()


def plot_lambda_k_space(filename, kink_to_plot='m_1', sausage=True,
                        sausage_pos=(1.8, 0.6)):
    r"""
    Plot the lambda k space, for a specific mode.
    """
    epsilon_case = np.load(filename)
    lambda_a_mesh = epsilon_case['lambda_a_mesh']
    k_a_mesh = epsilon_case['k_a_mesh']
    external_kink = epsilon_case['external_kink']
    external_m_1 = epsilon_case['external_m_1']
    external_m_neg_1 = epsilon_case['external_m_neg_1']
    external_sausage = epsilon_case['external_m_0']
    epsilon_case.close()

    kink_map = {'m_1': external_m_1, 'm_neg_1': external_m_neg_1,
                'kink': external_kink}

    kink_pal = sns.blend_palette(["black", "white", "yellow"], 3, as_cmap=True)
    plt.contourf(lambda_a_mesh, k_a_mesh, kink_map[kink_to_plot],
                 levels=[-1, -0.5, 0.5, 1.], cmap=kink_pal)

    if sausage:
        sausage_pal = sns.blend_palette(["black", "white", "orange"], 3,
                                        as_cmap=True)
        plt.contourf(lambda_a_mesh, k_a_mesh, external_sausage,
                     levels=[-1, -0.5, 0.5, 1.], alpha=0.5, cmap=sausage_pal)

    plt.plot([0.1, 1.0, 2.0, 3.0], [0.05, 0.5, 1.0, 1.5], color='black')
    plt.ylim(0.2, 1.5)

    plt.ylabel(r'$\bar{k}$', fontsize=25)
    plt.xlabel(r'$\bar{\lambda}$', fontsize=25)

    plt.annotate('stable', xy=(0.4, 0.95), fontsize=30)
    plt.annotate('kink', xy=(1.0, 0.3), fontsize=30)
    plt.annotate('sausage', xy=sausage_pos, fontsize=30)
