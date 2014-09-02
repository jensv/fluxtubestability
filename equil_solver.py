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
from collections import OrderedDict
import scipy.interpolate as interp
import scipy.integrate as inte


class EquilSolver(object):
    r"""
    General Equilibrium Solver parent implements spline generation and safety
    factor calculation.
    """

    def set_splines(self, param_points):
        r"""
        Returns Splines requested with a dictionary of spline names and point
        generating functions.
        """
        splines = {}
        r = self.r
        for key, value in param_points.items():
            splines[key] = interp.InterpolatedUnivariateSpline(r, value,
                                                               k=3)
        self.splines = splines

    def get_splines(self):
        r"""
        Returns splines of instance.
        """
        return self.splines

    def q(self, r):
        r"""
        Returns safety factor evaluated at points.
        """
        if r[0] == 0.:
            q_to_return = np.ones(r.size)*self.q0
            q_to_return[1:] = r[1:]*self.k*self.b_z(r)/self.b_theta(r)
        else:
            q_to_return = r*self.k*self.b_z(r)/self.b_theta(r)
        return q_to_return

    def rho(self, r):
        r"""
        Return density. Set to one evrywhere. Density plays no role for
        Newcomb. However, the general eigenvalue problem has stability issues.
        """
        return np.ones(r.size)


class ParabolicNu2(EquilSolver):
    r"""
    Creates splines for parabolic current profile pinch.
    """

    def __init__(self, a=1, points=500, q0=1.0, k=1, b_z0=1, temp=1.0,
                 qa=None):
        r"""
        Initalize parameters defining parabolic pinch and create splines.
        """
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
                        'b_z': self.b_z(r), 'p_prime': self.p_prime(r),
                        'pressure': self.pressure(r), 'q': self.q(r),
                        'rho': self.rho(r)}
        self.set_splines(param_points)

    def get_j0(self):
        r"""
        Return j0 for a b_z and q0 paramters of pinch.
        """
        self.j0 = 1
        return self.k*2*self.b_z(0)/self.q0

    def j_z(self, r):
        r"""
        Return parabolic axial current profile.
        """
        return self.j0*(1 - r**2)**2

    def b_theta(self, r):
        r"""
        Return azimuthhal magnetic field for a parabolic current profile pinch.
        """
        j0 = self.j0
        return j0*r/2 - j0*r**3/2 + j0*r**5/6

    def b_z(self, r):
        r"""
        Return constant axial magnetic field.
        """
        b_z0 = self.b_z0
        r = np.asarray(r)
        return np.ones(r.size) * b_z0

    def p_prime(self, r):
        r"""
        Returns pressure_prime profile for a arabolic current profile pinch.
        """
        j0 = self.j0
        return (-j0**2*r/2 + 3*j0**2*r**3/2 - 5*j0**2*r**5/3 + 5*j0**2*r**7/6
                - j0**2*r**9/6)

    def pressure(self, r):
        r"""
        Returns pressure profile for a arabolic current profile pinch.
        """
        j0 = self.j0
        return (47*j0**2/720 - j0**2*r**2/4 + 3*j0**2*r**4/8 - 5*j0**2*r**6/18
                + 5*j0**2*r**8/48 - j0**2*r**10/60)


class NewcombConstantPressure(EquilSolver):
    """
    Creates splines describing the constant pressure profile at the end of
    Newcomb's 1960 paper.
    """

    def __init__(self, a=0.1, r_0i=0.5, k=1, b_z0=0.1, b_thetai=0.1,
                 points=500):
        r"""
        Initialize parameters defining parabolic pinch and create splines.
        """
        self.r = np.linspace(a, r_0i, points)
        self.a = a
        self.r_0i = r_0i
        self.k = k
        self.b_z0 = b_z0
        self.b_thetai = b_thetai
        r = self.r
        param_points = {'j_z': self.get_j_z(r), 'b_theta': self.b_theta(r),
                        'b_z': self.b_z(r), 'p_prime': self.p_prime(r),
                        'pressure': self.pressure(r), 'q': self.q(r),
                        'rho': self.rho(r)}
        self.set_splines(param_points)

    def get_j_z(self, r):
        r"""
        Return zero current.
        """
        r = np.asarray(r)
        return np.zeros(r.size)

    def b_theta(self, r):
        r"""
        Return b_theta current profile as given in Newcomb's paper.
        """
        return self.b_thetai*self.r_0i/r

    def b_z(self, r):
        r"""
        Return constant b_z field.
        """
        return np.ones(r.size)*self.b_z0

    def p_prime(self, r):
        r"""
        Return pressure_prime profile.
        """
        r = np.asarray(r)
        return np.zeros(r.size)

    def pressure(self, r):
        r"""
        Return pressure.
        """
        r = np.asarray(r)
        return np.ones(r.size)


class SmoothedCoreSkin(EquilSolver):
    r"""
    Creates splines describing a smooth skin and core current profile.
    """
    def __init__(self, points_core=20, points_transition=50, points_skin=20,
                 r_core=0.7, r_transition=0.1, r_skin=0.1, k=1.,
                 j_core=0.1, epsilon=0.3, lambda_bar=0.5, q0=):
        r"""
        Initialize parameters defining smooth skin and core profile
        and create splines.
        """

        self.points_core = points_core
        self.points_transition = points_transition
        self.points_skin = points_skin
        self.r_core = r_core
        self.r_transition = r_transition
        self.r_skin = r_skin

        mask = np.ones(points_transition + 2, dtype=bool)
        mask[[0, -1]] = False
        self.r1 = np.linspace(0., r_core, points_core)
        r2 = np.linspace(r_core, r_core + r_transition, points_transition + 2)
        self.r2 = r2[mask]
        self.r3 = np.linspace(r_core + r_transition, r_core + r_transition + r_skin,
                         points_skin)
        r4 = np.linspace(r_core + r_transition + r_skin, r_core +
                         2*r_transition + r_skin, points_transition + 2)
        self.r4 = r4[mask]
        self.r = np.concatenate((self.r1, self.r2, self.r3, self.r4))

        self.k = k
        self.j_core = j_core
        self.epsilon = epsilon
        self.lambda_bar = lambda_bar


        param_points = OrderedDict([('j_z', self.j_z(self.r)),
                                    ('b_theta', self.b_theta(self.r)),
                                    ('b_z', self.b_z(self.r)),
                                    ('p_prime', self.pprime(self.r)),
                                    ('pressure', self.pressure(self.r)),
                                    ('q', self.q(self.r)),
                                    ('rho', self.rho(self.r))])

        self.set_splines(param_points)

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

    def get_j_z_skin(self):
        r"""
        Returs j_z_skin based on j_z_core, geometry and epsilon of pinch.
        """

        return

    def get_b_z():
        r"""
        Returns b_z based on j_z, geometry and lambda_bar of pinch.
        """
        return

    def j_z(self, r):
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
                                  r = np.asarray(r)         self.j_skin,
                                           self.r[points1:points2])
        j_z[points2:points3] = self.j_skin
        j_z[points3:points4] = self.smooth(boundary3, boundary4, self.j_skin,
                                           0., self.r[points3:points4])
        return j_z

    def b_theta(self, r):
        r"""
        Return b_theta at given r values.
        """
        b_theta_r_integrator = inte.ode(self.__b_theta_r_prime__)
        b_theta_r_integrator.set_integrator('lsoda')
        b_theta_r_integrator.set_initial_value(0., t=0.)
        b_theta_r_integrator.set_f_params(self)
        b_theta_array = [0.]
        for position in r[1:]:
            if b_theta_r_integrator.successful():
                b_theta_r_integrator.integrate(t=position)
                b_theta_array.append(b_theta_r_integrator.y)
            else:
                break
        return np.array(b_theta_array)

    def __b_theta_r_prime__(r, y, self):
        r"""
        Return b_theta_r_prime at given r values. To be used for integration.
        """
        return self.splines['j_z'](r)*r

    def b_z(self, r):
        r"""
        """
        return np.ones(r.size)*self.b_z0

    def p_prime(self, r):
        r"""
        Return pressure_prime at given r values. To be used for integration.
        """
        return -self.splines['b_theta'](r)*self.splines['j_z'](r)

    def __p_prime_for_integrating__(r, y, self):
        r"""
        Return pressure_prime at given r values. To be used for integration.
        """
        return [self.splines['j_z'](r)*r, -y[0]/r*self.splines['j_z'](r)]

    def pressure(self, r):
        r"""
        Return pressure_prime at given r values. To be used for integration.
        """
        r_reverse = r[::-1]
        pressure_integrator = inte.ode(self.__p_prime_for_integrating__)
        pressure_integrator.set_integrator('lsoda')
        pressure_integrator.set_initial_value([0., 0.], r_reverse[0])
        pressure_integrator.set_f_params(self)
        pressure_array = [0.]
        for position in r_reverse[1:]:
            if pressure_integrator.successful():
                pressure_integrator.integrate(t=position)
                pressure_array.append(pressure_integrator.y[1])
            else:
                break
        return np.array(pressure_array)

    def q(self, r):
        r"""
        Returns safety factor evaluated at points.
        """
        if r[0] == 0.:
            q_to_return = np.ones(r.size)*self.q0
            q_to_return[1:] = (r[1:]*self.k*self.splines['b_z'](r)/
                               self.splines['b_theta'](r))
        else:
            q_to_return = (r*self.k*self.splines['b_z'](r)/
                           self.splines['b_theta'](r))
        return q_to_return


class HardCoreZPinch(EquilSolver):
    r"""
    """
    def
    pass


class SharpCoreSkin(EquilSolver):
    pass