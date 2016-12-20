"""
Created on Tue Nov 22 11:12:58 2016

@author: Jens von der Linden
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatibility"""

import sys
sys.path.append('scipy_mod')

import fitpack
reload(fitpack)
from fitpack import splev

import numpy as np
from numpy import atleast_1d
import scipy.integrate

import singularity_frobenius as frob

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

def plot_all_profiles_suydam(profile, normalize=False,
                             mu_0=None, axes=None, title=None):
    r"""
    """
    if not axes:
        axes = plt.gca()
    r = np.linspace(0, 1, 250)
    splines = profile.get_splines()
    j_z = splines['j_z'](r)
    b_z = splines['b_z'](r)
    b_z_prime = splines['b_z'].derivative()(r)
    b_theta = splines['b_theta'](r)
    b_theta_prime = splines['b_theta'].derivative()(r)
    p_prime = splines['p_prime'](r)
    p = splines['pressure'](r)
    safety_factor = splines['q'](r)
    safety_factor_prime = splines['q'].derivative()(r)
    if mu_0:
       beta_0 = mu_0
    else:
        beta_0 = profile.beta_0()
    alpha = frob.alpha_func(r, b_z, b_z_prime, b_theta, b_theta_prime)
    beta = frob.beta_func(b_z, b_theta, p_prime, beta_0)
    suydam_mu = alpha + 4.*beta
    suydam_q = r*b_z**2./(8. * beta_0)*(safety_factor_prime/safety_factor)**2. + p_prime
    if 'j_theta' in splines.keys():
        print('true')
        j_theta = splines['j_theta'](r)
    else:
        print('false')
        j_theta = None
    if normalize:
        j_z = j_z/np.max(np.abs(j_z))
        b_theta = b_theta/np.max(np.abs(b_theta))
        alpha = alpha/np.max(np.abs(alpha))
        beta = beta/np.max(np.abs(beta))
        suydam_q = suydam_q/np.max(np.abs(suydam_q))
        suydam_mu = suydam_mu/np.nanmax(np.abs(suydam_mu))
        p = p/np.max(np.abs(p))
        p_prime =  p_prime/np.max(np.abs(p_prime))
        safety_factor_prime =  safety_factor_prime/np.max(np.abs(safety_factor_prime))
    if 'j_theta' in splines.keys():
            j_theta = j_theta/np.nanmax(np.abs(j_theta))
    axes.plot(r, j_z, c='#087804', label=r'$j_z$')
    if 'j_theta' in splines.keys():
        axes.plot(r, j_theta, c='#6fc276', label=r'$j_\theta$')
    axes.plot(r, b_theta, c='#e50000', label=r'$B_\theta$')
    axes.plot(r, p_prime, c='#7e1e9c', label=r"$p'$")
    axes.plot(r, p, c='#bf77f6', label=r"$p$")
    axes.plot(r, safety_factor, c='#000000', label=r"$q$")
    axes.plot(r, safety_factor_prime, c='#7d7f7c', label=r"$q'$")
    axes.plot(r[1:], suydam_q[1:], c='#acfffc', label=r"suydam_q")
    #axes.plot(r[1:], suydam_mu[1:], c='#82cafc', label=r"suydam_mu")
    if title:
        axes.set_title(title)
    axes.legend(loc='best')
    return axes

def plot_suydam(profile, normalize=False):
    r"""
    """
    axes = plt.gca()
    r = np.linspace(0, 1, 250)
    splines = profile.get_splines()
    j_z = splines['j_z'](r)
    b_z = splines['b_z'](r)
    b_z_prime = splines['b_z'].derivative()(r)
    b_theta = splines['b_theta'](r)
    b_theta_prime = splines['b_theta'](derivative)(r)
    p_prime = splines['p_prime'](r)
    p = splines['pressure'](r)
    safety_factor = splines['q'](r)
    safety_factor_prime = splines['q'].derivative()(r)
    beta_0 = profile.beta_0()
    alpha = frob.alpha_func(r, b_z, b_z_prime, b_theta, b_theta_prime)
    beta = frob.beta_func(b_z, b_theta, p_prime, beta_0)
    suydam_mu = alpha + 4.*beta
    suydam_q = r*b_z**2./(8. * beta_0)*(safety_factor_prime/safety_factor)**2. + p_prime
    if normalize:
        j_z = j_z/np.max(np.abs(j_z))
        b_theta = b_theta/np.max(np.abs(b_theta))
        alpha = alpha/np.max(np.abs(alpha))
        beta = beta/np.max(np.abs(beta))
        suydam_q = suydam_q/np.max(np.abs(suydam_q))
        suydam_mu = suydam_mu/np.nanmax(np.abs(suydam_mu))
        p_prime =  p_prime/np.max(np.abs(p_prime))
        safety_factor_prime = safety_factor_prime/np.max(np.abs(safety_factor_prime)) 
    #axes.plot(r, j_z, c='#087804', label=r'$j_z$')
    #axes.plot(r, b_theta, c='#e50000', label=r'$B_\theta$')
    #axes.plot(r, p_prime, c='#7e1e9c', label=r"$p'$")
    #axes.plot(r, p, c='#bf77f6', label=r"$p$")
    #axes.plot(r, safety_factor, c='#000000', label=r"$q$")
    #axes.plot(r, safety_factor_prime, c='#7d7f7c', label=r"$q'$")
    axes.plot(r, suydam_q, c='#acfffc', label=r"suydam_q")
    axes.plot(r, suydam_mu, c='#82cafc', label=r"suydam_mu")
    axes.legend(loc='best')
    return suydam_q, suydam_mu, axes
