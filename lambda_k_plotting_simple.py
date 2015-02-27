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

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, BoundaryNorm
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')


def plot_lambda_k_space_dw(filename, name, mode_to_plot='m_neg_1',
                           show_points=False, lim=None, levels=None, log=False,
                           linthresh=1E-7, bounds=(1.5, 3.0), floor_norm=False):
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