# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:25:43 2015

@author: jensv
"""

import numpy as np
from scipy.special import kv, kvp

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
sns.set_style('ticks')
sns.set_context('poster')


def conditions(k_bar, lambda_bar, epsilon, m, delta):
    r"""
    Return analytic stability condition.
    """
    term1 = (2*k_bar - m*epsilon*lambda_bar)*((delta + 1)*2*k_bar -
                                              (delta - 1)*m*epsilon*
                                              lambda_bar)/(k_bar**2 + m**2)
    term2 = (epsilon**2 - 1) * lambda_bar**2
    term3 = (m*lambda_bar - 2* k_bar)**2/k_bar * (kv(m, np.abs(k_bar)) /
                                                  kvp(m, np.abs(k_bar)))
    return term1 + term2 - term3


def condition_map(epsilon=0.5, delta=0.):
    r"""
    Draw filled contours of sausage (orange), kink(yellow), and stable (white)
    regions for given epsilon and delta values.
    """
    fig = plt.figure(figsize=(10,10))
    lambda_bar = np.linspace(0., 3., 750)
    k_bar = np.linspace(0, 1.5, 750)
    lambda_bar_mesh, k_bar_mesh = np.meshgrid(lambda_bar, k_bar)

    stability_kink = conditions(k_bar_mesh, lambda_bar_mesh, epsilon, 1., delta)
    stability_kink = stability_kink < 0
    stability_sausage = conditions(k_bar_mesh, lambda_bar_mesh, epsilon, 0., delta)
    stability_sausage = stability_sausage < 0
    stability_kink = stability_kink.astype(float)
    stability_kink[stability_sausage] = 2

    cmap = colors.ListedColormap([sns.xkcd_rgb["white"],
                                  sns.xkcd_rgb["yellow"], sns.xkcd_rgb["orange"]])
    plt.contourf(lambda_bar_mesh, k_bar_mesh, stability_kink,
                 cmap=cmap, levels=[0., 0.5, 1.5, 2.])
    plt.contour(lambda_bar_mesh, k_bar_mesh, stability_kink,
                levels=[0., 0.5, 1.5, 2.], colors='grey')
    plt.plot([0, 3.], [0., 1.5], '--', c='black', lw=5)
    axes = plt.gca()

    plt.setp(axes.get_xticklabels(), fontsize=40)
    plt.setp(axes.get_yticklabels(), fontsize=40)
    plt.ylabel(r'$\bar{k}$', fontsize=45, rotation='horizontal', labelpad=25)
    plt.xlabel(r'$\bar{\lambda}$', fontsize=45)
    sns.despine()


def condition_map_variable_delta(filename, mode=1, epsilon=0.5):
    r"""
    Draw filled contours of sausage (orange), kink(yellow), and stable (white)
    regions for given epsilon and delta values.
    """
    data_meshes = np.load(filename)
    lambda_bar_mesh = data_meshes['lambda_a_mesh']
    k_bar_mesh = data_meshes['k_a_mesh']
    delta_mesh = data_meshes['delta_m_0']

    fig = plt.figure(figsize=(10,10))

    cmap = colors.ListedColormap([sns.xkcd_rgb["white"],
                                  sns.xkcd_rgb["yellow"],
                                  sns.xkcd_rgb["orange"]])

    stability_kink = conditions(k_bar_mesh, lambda_bar_mesh, epsilon, 1.,
                                delta_mesh)
    stability_kink = stability_kink < 0
    stability_sausage = conditions(k_bar_mesh, lambda_bar_mesh, epsilon, 0.,
                                   delta_mesh)
    stability_sausage = stability_sausage < 0

    if mode == 0:
        stability = stability_sausage
        cmap = colors.ListedColormap([sns.xkcd_rgb["white"],
                                      sns.xkcd_rgb["orange"]])
    else:
        stability = stability_kink
        cmap = colors.ListedColormap([sns.xkcd_rgb["white"],
                                      sns.xkcd_rgb["yellow"]])

    plt.contourf(lambda_bar_mesh, k_bar_mesh, stability,
                 cmap=cmap, levels=[0.5, 1.5])
    plt.contour(lambda_bar_mesh, k_bar_mesh, stability,
                levels=[0.5, 1.5], colors='grey')
    plt.plot([0, 3.], [0., 1.5], '--', c='black', lw=5)
    axes = plt.gca()

    plt.setp(axes.get_xticklabels(), fontsize=40)
    plt.setp(axes.get_yticklabels(), fontsize=40)
    plt.ylabel(r'$\bar{k}$', fontsize=45, rotation='horizontal', labelpad=25)
    plt.xlabel(r'$\bar{\lambda}$', fontsize=45)
    sns.despine()
