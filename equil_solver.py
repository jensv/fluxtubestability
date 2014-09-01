# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:19:28 2014

@author: Jens von der Linden
"""


from __future__ import print_function, unicode_literals, division
from __future__ import absolute_import
from future import standard_library, utils
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
# Python 3.x compatability

import numpy as np
import sympy as sp
import scipy.interpolate as interp
import scipy.integrate as inte


class EquilSolver(object):

    def ___init__(self, a, points):
        self.r = np.linspace(0, a, points)

    def set_splines(self, param_points):
        splines = {}
        r = self.r
        d_b_theta_r_request = False
        for key, value in param_points.items():
            if key == 'd_b_theta_over_r':
                d_b_theta_r_request = True
            else:
                splines[key] = interp.InterpolatedUnivariateSpline(r, value,
                                                                   k=3)
        if d_b_theta_r_request:
            splines['d_b_theta_over_r'] = self.d_b_theta_over_r(r,
                                                           splines['b_theta'])
        self.splines = splines

    def get_splines(self):
        return self.splines

    def q(self, r):
        if r[0] == 0.:
            q_to_return = np.ones(r.size)*self.q0
            q_to_return[1:] = r[1:]*self.k*self.b_z(r)/self.b_theta(r)
        else:
            q_to_return = r*self.k*self.b_z(r)/self.b_theta(r)
        return q_to_return


class ParabolicNu2(EquilSolver):

    def __init__(self, a=1, points=500, q0=1.0, k=1, b_z0=1, temp=1.0, qa=None):
        self.r = np.linspace(0, a, points)
        self.q0 = q0
        if qa != None:
            q0 = qa/3.
        self.k = k
        self.b_z0 = b_z0
        self.temp = temp
        r = self.r
        self.j0 = self.get_j0()
        param_points = {'j_z': self.j_z(r), 'b_theta': self.b_theta(r),
                        'b_z': self.b_z(r), 'p_prime': self.pprime(r),
                        'pressure': self.pressure(r), 'q': self.q(r),
                        'rho': self.rho(r),
                        'd_b_theta_over_r': None}
        self.set_splines(param_points)

    def get_j0(self):
        self.j0 = 1
        return self.k*2*self.b_z(0)/self.q0

    def j_z(self, r):
        return self.j0*(1 - r**2)**2

    def b_theta(self, r):
        j0 = self.j0
        return j0*r/2 - j0*r**3/2 + j0*r**5/6

    def b_z(self, r):
        b_z0 = self.b_z0
        if isinstance(r, float) or isinstance(r, int):
            return np.ones(1) * b_z0
        else:
            return np.ones(r.size) * b_z0

    def pprime(self, r):
        j0 = self.j0
        return (-j0**2*r/2 + 3*j0**2*r**3/2 - 5*j0**2*r**5/3 + 5*j0**2*r**7/6
                - j0**2*r**9/6)

    def pressure(self, r):
        j0 = self.j0
        return (47*j0**2/720 - j0**2*r**2/4 + 3*j0**2*r**4/8 - 5*j0**2*r**6/18
                + 5*j0**2*r**8/48 - j0**2*r**10/60)

    def rho(self, r):
        return np.ones(r.size)
        #return self.pressure(r)/self.temp

    def d_b_theta_over_r(self, r, b_theta):
        b_over_r = interp.InterpolatedUnivariateSpline(r, b_theta.derivative()(r)
                                                       /r, k=3)
        return b_over_r.derivative()


class NewcombConstantPressure(EquilSolver):
    """
    Creates splines describing the constant pressure profile at the end of
    Newcomb's 1960 paper.
    """

    def __init__(self, a=0.1, r_0i=0.5, k=1, b_z0=0.1, b_thetai=0.1,
                 points=500):
        self.r = np.linspace(a, r_0i, points)
        self.a = a
        self.r_0i = r_0i
        self.k = k
        self.b_z0 = b_z0
        self.b_thetai = b_thetai
        r = self.r
        param_points = {'j_z': self.get_j_z(r), 'b_theta': self.b_theta(r),
                        'b_z': self.b_z(r), 'p_prime': self.pprime(r),
                        'pressure': self.pressure(r), 'q': self.q(r),
                        'rho': self.rho(r)}
        self.set_splines(param_points)

    def get_j_z(self, r):
        return np.zeros(r.size)

    def b_theta(self, r):
        return self.b_thetai*self.r_0i/r

    def b_z(self, r):
        return np.ones(r.size)*self.b_z0

    def pprime(self, r):
        return np.zeros(r.size)

    def pressure(self, r):
        return np.ones(r.size)

    def rho(self, r):
        return np.ones(r.size)


class SmoothedCoreSkin(EquilSolver):
    r"""
    Creates splines describing a smooth skin and core current profile.
    """

    def __init__(self, points_core=20, points_transition=50, points_skin=20,
                 r_core=0.7, r_transition=0.1, r_skin=0.1, k=1., b_z0=0.1,
                 epsilon=None, beta=None):

        self.points_core = points_core
        self.points_transition = points_transition
        self.points_skin = points_skin
        self.r_core = r_core
        self.r_transition = r_transition
        self.r_skin = r_skin

        mask = np.ones(points_transition + 2, dtype=bool)
        mask[[0, -1]] = False
        r1 = np.linspace(0., r_core, points_core)
        r2 = np.linspace(r_core, r_core + r_transition, points_transition + 2)
        r2 = r2[mask]
        r3 = np.linspace(r_core + r_transition, r_core + r_transition + r_skin,
                         points_skin)
        r4 = np.linspace(r_core + r_transition + r_skin, r_core +
                         2*r_transition + r_skin, points_transition + 2)
        r4 = r4[mask]
        self.r = np.concatenate((r1, r2, r3, r4))

        self.k = k
        self.b_z0 = b_z0

    def smooth(self, x1, x2, g1, g2, x):
        """
        Smoothing method by Alan Glasser.
        """
        delta_x = (x2 - x1) / 2.
        x_bar = (x2 + x1) / 2.
        delta_g = (g2 - g1) / 2.
        g_bar = (g1 + g2) / 2.
        z = (x - x_bar) / delta_x
        return g_bar + self.f(z)*delta_g

    def smooth_f(self, z):
        r"""
        Smoothing polynominal by Alan Glasser.
        """
        return z/8.*(3.*z**4 - 10.*z**2 + 15.)

    def get_j_z(self, r):
        r"""
        For now always returns complete j_z.
        """
        total_points = (self.points_core + 2*self.points_transition +
                        self.points_skin)

        points1 = self.points_core
        points2 = self.points_core + self.points_transition
        points3 = self.points_core + self.points_transition + self.points_skin
        points4 = (self.points_core + 2*self.points_transition +
                   self.points_skin)

        boundary1 = self.r_core
        boundary2 = self.r_core + self.r_transition
        boundary3 = self.r_core + self.r_transition + self.r_skin
        boundary4 = self.r_core + 2*self.r_transition + self.r_skin

        j_z = np.zeros(total_points)
        j_z[:points1] = self.j_core
        j_z[points1:points2] = self.smooth(boundary1, boundary2, self.j_core,
                                           self.j_skin,
                                           self.r[points1:points2])
        j_z[points2:points3] = self.j_skin
        j_z[points3:points4] = self.smooth(boundary3, boundary4, self.j_skin,
                                           0., self.r[points3:points4])
        return j_z

    def b_theta(self, r):
        r"""
        """
        b_theta_r_integrator = inte.ode(self.b_theta_r_prime)
        b_theta_r_integrator.set_integrator('lsoda')
        b_theta_r_integrator.set_initial_value(0., t=0.)
        for position in r[1:]:
            if b_theta_r_integrator.successful():
                b_theta_r_integrator.integrate(t=position)
            else:
                break
        return b_theta

    def b_theta_r_prime(r, y):
        r"""
        """
        return self.splines['j_z'](r)*r

    def b_z(self, r):
        r"""
        """
        return np.ones(r.size)*self.b_z0

    def pprime(self, r):
        r"""
        """
        return self.splines['b_theta']*self.splines['j_z']*

    def pressure(self, r):
        r"""
        """
        pressure_integrator = inte.ode(self.pprime)

    def rho(self, r):
        r"""
        """
        return np.ones(r.size)


class HardCoreZPinch(EquilSolver):
    pass


class SharpCoreSkin(EquilSolver):
    pass