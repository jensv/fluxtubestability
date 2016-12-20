# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:25:43 2015

@author: jensv

Analytic stability condition derived for lengthening 
current-carrying magnetic flux tube with core and skin
currents.
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

    Parameters
    ----------
    k_bar : float
        normalized inverse aspect ratio
    lambda_bar : float
        normalized current-to-magnetic flux ratio
    epsilon : float
        core to total current ratio
    m : float
        azimuthal periodicity number
    delta : float
        abruptness parameter

    Returns
    -------
    delta_w : float
        perturbed potential energy of marginal stability case
    """
    term1 = conditions_plasma_term(k_bar, lambda_bar, epsilon, m, delta)
    term2 = conditions_interface_term(k_bar, lambda_bar, epsilon, m, delta)
    term3 = conditions_vacuum_term(k_bar, lambda_bar, m, delta)
    return term1 + term2 - term3


def conditions_without_interface(k_bar, lambda_bar, m, delta):
    r"""
    Return analytic stability condition minus interface term (term2).

    Parameters
    ----------
    k_bar : float
        normalized inverse aspect ratio
    lambda_bar : float
        normalized current-to-magnetic flux ratio
    m : float
        azimuthal periodicity number
    delta : float
        abruptness parameter

    Returns
    -------
    delta_w_without_interface : float
        perturbed potential energy without interface term.

    Notes
    -----
    For profiles with current smoothly going to zero at the delta_w term is zero.
    """
    term1 = conditions_smooth_plasma_term(k_bar, lambda_bar, m, delta)
    term3 = conditions_vacuum_term(k_bar, lambda_bar, m, delta)
    return term1 - term3


def conditions_without_interface_wo_sing(k_bar, lambda_bar, m, xi,
                                         xi_der, a):
    r"""
    Multiply analytic expression with xi squared to avoid singularity.

    Parameters
    ----------
    k_bar : float
        normalized inverse aspect ratio
    lambda_bar : float
        normalized current-to-magnetic flux ratio
    m : float
        azimuthal periodicity number
    xi : float
        solution to Euler-Lagrange equation at boundary
    xi_der : float
        derivative of solution to Euler-Lagrange equation at boundary
    a : float
        radius of current-carrying magnetic flux tube
    Returns
    -------
    delta_w_without_interface_wo_sing : float
        perturbed potential energy without interface or singularity

    Notes
    -----
    delta can be singular when xi goes through zero. This form is multiplied
    by xi**2 to avoid singularity.
    """
    term1 = conditions_smooth_plasma_term_wo_sing(k_bar, lambda_bar,
                                                  m, xi, xi_der, a)
    term3 = conditions_vacuum_term_wo_sing(k_bar, lambda_bar, m, xi)
    return term1 - term3


def conditions_plasma_term(k_bar, lambda_bar, epsilon, m, delta):
    r"""
    Returns plasma term of analytic stability condition.

    Parameters
    ----------
    k_bar : float
        normalized inverse aspect ratio
    lambda_bar : float
        normalized current-to-magnetic flux ratio
    epsilon : float
        core to total current ratio
    m : float
        azimuthal periodicity number
    delta : float
        abruptness parameter

    Returns
    -------
    delta_w_plasma_term : float
        perturbed potential energy plasma term due to internal
        currents.
    """
    term1 = (2.*k_bar - m*epsilon*lambda_bar)*((delta + 1)*2.*k_bar -
                                               (delta - 1)*m*epsilon *
                                               lambda_bar)/(k_bar**2 + m**2)
    return term1


def conditions_smooth_plasma_term_wo_sing(k_bar, lambda_bar, m, xi,
                                          xi_der, a):
    r"""
    Multiply analytic expression with xi squared to avoid singularity.

    Parameters
    ----------
    k_bar : float
        normalized inverse aspect ratio
    lambda_bar : float
        normalized current-to-magnetic flux ratio
    m : float
        azimuthal periodicity number
    xi : float
        solution to Euler-Lagrange equation at boundary
    xi_der : float
        derivative of solution to Euler-Lagrange equation at boundary
    a : float
        radius of current-carrying magnetic flux tube

    """
    epsilon = 1.
    term1 = (2.*k_bar - m*epsilon*lambda_bar)*((xi_der*a*xi + xi**2)*2.*k_bar -
                                               (xi_der*a*xi - xi**2)*m*epsilon*
                                               lambda_bar)/(k_bar**2 + m**2)
    return term1


def conditions_smooth_plasma_term(k_bar, lambda_bar, m, delta):
    r"""
    Returns plasma term of analytic condition with epsilon set to 1. This
    should be relvant for a profile that smoothly goes to zero current,
    since b_v(a) = b_p(a) in that case.

    Parameters
    ----------
    k_bar : float
        normalized inverse aspect ratio
    lambda_bar : float
        normalized current-to-magnetic flux ratio
    m : float
        azimuthal periodicity number
    delta : float
        abruptness parameter

    """
    epsilon = 1.
    term1 = conditions_plasma_term(k_bar, lambda_bar, epsilon, m, delta)
    return term1


def conditions_interface_term(k_bar, lambda_bar, epsilon, m, delta):
    r"""
    Returns interface term of analytic stability condition.

    Parameters
    ----------
    k_bar : float
        normalized inverse aspect ratio
    lambda_bar : float
        normalized current-to-magnetic flux ratio
    epsilon : float
        core to total current ratio
    m : float
        azimuthal periodicity number
    delta : float
        abruptness parameter

    """
    term2 = (epsilon**2 - 1) * lambda_bar**2
    return term2


def conditions_vacuum_term(k_bar, lambda_bar, m, delta):
    r"""
    Returns vacuum term of analytic stability condition.

    Parameters
    ----------
    k_bar : float
        normalized inverse aspect ratio
    lambda_bar : float
        normalized current-to-magnetic flux ratio
    m : float
        azimuthal periodicity number
    delta : float
        abruptness parameter
    """
    term3 = (m*lambda_bar - 2.*k_bar)**2/k_bar*(kv(m, np.abs(k_bar)) /
                                                kvp(m, np.abs(k_bar)))
    return term3

def conditions_vacuum_term_wo_sing(k_bar, lambda_bar, m, xi):
    r"""
    Multiply analytic expression with xi squared to avoid singularity.

    Parameters
    ----------
    k_bar : float
        normalized inverse aspect ratio
    lambda_bar : float
        normalized current-to-magnetic flux ratio
    m : float
        azimuthal periodicity number
    xi : float
        Euler-Lagrange solution
    """
    term3 = xi**2 * (m*lambda_bar - 2.*k_bar)**2/k_bar*(kv(m, np.abs(k_bar)) /
                                                     kvp(m, np.abs(k_bar)))
    return term3


def condition_map(epsilon=0.5, delta=0.):
    r"""
    Draw filled contours of sausage (orange), kink(yellow), and stable (white)
    regions for given epsilon and delta values.

    Parameters
    ----------
    epsilon : float
        core to total current ratio
    delta : float
        abruptness parameter
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


def condition_map_variable_delta(filename, mode=1, epsilon=0.5,
                                 conditions_func=conditions_without_interface):
    r"""
    Draw filled contours of sausage (orange), kink(yellow), and stable (white)
    regions for given epsilon and delta values.
    Delta values are loaded from a .npz mesh file.

    Parameters
    ----------
    filename : string
        filename from which to load lambda_bar, k_bar and delta values.
    mode : int
        azimuthal mode number 0 or 1
    epsilon : float
        core current to total current ratio
    conditions_func : function
        conditions function to use
    """
    data_meshes = np.load(filename)
    lambda_bar_mesh = data_meshes['lambda_a_mesh']
    k_bar_mesh = data_meshes['k_a_mesh']
    delta_mesh = data_meshes['delta_m_0']

    fig = plt.figure(figsize=(10,10))

    cmap = colors.ListedColormap([sns.xkcd_rgb["white"],
                                  sns.xkcd_rgb["yellow"],
                                  sns.xkcd_rgb["orange"]])

    stability_kink = conditions_func(k_bar_mesh, lambda_bar_mesh, epsilon, 1.,
                                     delta_mesh)
    stability_kink = stability_kink < 0
    stability_sausage = conditions_func(k_bar_mesh, lambda_bar_mesh, epsilon, 0.,
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
    plt.show()
