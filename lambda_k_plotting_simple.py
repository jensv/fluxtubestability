# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 00:30:53 2015

@author: jensv
"""

from __future__ import print_function, unicode_literals, division
from __future__ import absolute_import
from future import standard_library, utils
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import numpy as np
from scipy.special import kv, kvp
import analytic_condition as ac

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, BoundaryNorm
from matplotlib.ticker import FormatStrFormatter, FixedFormatter
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')


def plot_lambda_k_space_dw(filename, epsilon, name, mode_to_plot='m_neg_1',
                           show_points=False, lim=None, levels=None, log=False,
                           linthresh=1E-7, bounds=(1.5, 3.0), floor_norm=False,
                           analytic_compare=False,
                           label_pos=((0.5, 0.4), (2.1, 0.4), (2.8, 0.2))):
    r"""
    Plot the delta_w of external instabilities in the lambda-k space.
    """
    epsilon_case = np.load(filename)
    lambda_a_mesh = epsilon_case['lambda_a_mesh']
    k_a_mesh = epsilon_case['k_a_mesh']
    external_m_neg_1 = epsilon_case['d_w_m_neg_1']
    external_sausage = epsilon_case['d_w_m_0']
    external_m_neg_1_norm = epsilon_case['d_w_norm_m_neg_1']
    external_sausage_norm = epsilon_case['d_w_norm_m_0']
    epsilon_case.close()

    instability_map = {'m_0': external_sausage_norm,
                       'm_neg_1': external_m_neg_1_norm}


    kink_pal = sns.blend_palette([sns.xkcd_rgb["dandelion"],
                                  sns.xkcd_rgb["white"]], 7, as_cmap=True)
    kink_pal = sns.diverging_palette(73, 182, s=72, l=85, sep=1, n=9, as_cmap=True)
    sausage_pal = sns.blend_palette(['orange', 'white'], 7, as_cmap=True)
    sausage_pal = sns.diverging_palette(49, 181, s=99, l=78, sep=1, n=9, as_cmap=True)
    instability_palette = {'m_0': sausage_pal,
                           'm_neg_1': kink_pal}

    values = instability_map[mode_to_plot]

    if floor_norm:
        values = np.clip(values, -100., 100.)
        values = values / -np.nanmin(values)
        values = np.clip(values, -1., 1.)
    else:
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
            cbar.set_label(label=r'$\delta W$', size=45, rotation=0, labelpad=30)
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

    axes = plt.gca()
    axes.set_axis_bgcolor(sns.xkcd_rgb['grey'])

    lambda_bar = np.linspace(0.01, 3., 750)
    k_bar = np.linspace(0.01, 1.5, 750)
    lambda_bar_mesh, k_bar_mesh = np.meshgrid(lambda_bar, k_bar)

    if analytic_compare:
        analytic_comparison(mode_to_plot, k_bar_mesh, lambda_bar_mesh, epsilon,
                            label_pos)

    if show_points:
        plt.scatter(lambda_a_mesh, k_a_mesh, marker='o', c='b', s=5)
    plt.ylim(0.01, bounds[0])
    plt.xlim(0.01, bounds[1])
    axes = plt.gca()
    plt.setp(axes.get_xticklabels(), fontsize=40)
    plt.setp(axes.get_yticklabels(), fontsize=40)
    plt.ylabel(r'$\bar{k}$', fontsize=45, rotation='horizontal', labelpad=30)
    plt.xlabel(r'$\bar{\lambda}$', fontsize=45)
    cbar.ax.tick_params(labelsize=40)
    sns.despine(ax=axes)
    plt.tight_layout()
    plt.savefig('../../output/plots/' + name + '.png')
    plt.show()


def analytic_comparison(mode_to_plot, k_bar_mesh, lambda_bar_mesh, epsilon,
                        label_pos):
    r"""
    Add red lines indicating stability boundaries from analytical model.
    """
    line_labels = FixedFormatter(['-1', '0', '1'])

    assert (mode_to_plot == 'm_neg_1' or
            mode_to_plot == 'm_0'), ("Please specify mode_to_plot as either" +
                                     "m_neg_1 or m_0")

    if mode_to_plot == 'm_neg_1':
        m = 1
        color = 'red'
    if mode_to_plot == 'm_0':
        m = 0
        color = 'red'


    stability_kink_m_neg_1 = ac.conditions(k_bar_mesh, lambda_bar_mesh,
                                           epsilon, m, -1.)
    stability_kink_m_0 = ac.conditions(k_bar_mesh, lambda_bar_mesh,
                                       epsilon, m, 0.)
    stability_kink_m_1 = ac.conditions(k_bar_mesh, lambda_bar_mesh,
                                       epsilon, m, 1)


    stability_kink = stability_kink_m_neg_1 < 0
    stability_kink = stability_kink.astype(float)
    stability_kink[stability_kink_m_neg_1 >= 0] = -1.5
    stability_kink[stability_kink_m_neg_1 < 0] = -0.5
    stability_kink[stability_kink_m_0 < 0] = 0.5
    stability_kink[stability_kink_m_1 < 0] = 1.5

    cs = plt.contour(lambda_bar_mesh, k_bar_mesh, stability_kink,
                     levels=[-1, 0, 1], colors=color, linewidths=5,
                     linestyles='dotted')

    plt.clabel(cs, fmt={-1: r'$\delta = -1$', 0: r'$\delta = 0$',
                        1: r'$\delta = 1$'}, fontsize=40, manual=label_pos)
    return cs


def plot_lambda_k_space_delta(filename, mode_to_plot,
                              clip=False, delta_min=-1.5,
                              delta_max=1., levels=None):
    r"""
    Plot values of delta in lambda k space.
    """
    data_meshes = np.load(filename)
    lambda_mesh = data_meshes['lambda_a_mesh']
    k_mesh = data_meshes['k_a_mesh']
    delta_mesh = data_meshes['delta_m_0']

    if mode_to_plot == 0:
        color = 'green'
        delta_mesh = data_meshes['delta_m_0']
    else:
        #color = sns.xkcd_rgb["dandelion"]
        color = 'green'
        delta_mesh = data_meshes['delta_m_neg_1']

    if clip:
        delta_mesh = np.clip(delta_mesh, delta_min, delta_max)

    colors = sns.light_palette(color, n_colors=6, reverse=True,
                               as_cmap=True)

    if levels:
        plt.contourf(lambda_mesh, k_mesh, delta_mesh, cmap=colors,
                     levels=levels)
    else:
        plt.contourf(lambda_mesh, k_mesh, delta_mesh, cmap=colors)

    cbar = plt.colorbar(label=r'$\delta$')
    cbar.set_label(label=r'$\delta$', size=45, rotation=0, labelpad=30)

    if levels:
        contourlines = plt.contour(lambda_mesh, k_mesh, delta_mesh,
                                   colors='grey', levels=levels)
    else:
        contourlines = plt.contour(lambda_mesh, k_mesh, delta_mesh,
                                   colors='grey')

    cbar.add_lines(contourlines)

    plt.ylim(0.01, 1.5)
    plt.xlim(0.01, 3.0)
    axes = plt.gca()
    axes.set_axis_bgcolor(sns.xkcd_rgb['grey'])
    plt.setp(axes.get_xticklabels(), fontsize=40)
    plt.setp(axes.get_yticklabels(), fontsize=40)
    plt.ylabel(r'$\bar{k}$', fontsize=45, rotation='horizontal', labelpad=30)
    plt.xlabel(r'$\bar{\lambda}$', fontsize=45)
    cbar.ax.tick_params(labelsize=40)
    sns.despine(ax=axes)
    plt.tight_layout()
