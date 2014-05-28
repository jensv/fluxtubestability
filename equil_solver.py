# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:19:28 2014

@author: Jens von der Linden
"""

#Python 3.x compatability
from __future__ import print_function, unicode_literals, division
from __future__ import absolute_import
from future import standard_library, utils
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import numpy as np
import sympy as sp
import scipy.interpolate as interp


class Equil_solver(object):

    def ___init__(self, a, points):
        self.r = np.linspace(0, a, points)

    def set_splines(self, param_points):
        splines = {}
        r = self.r
        for key, value in param_points.items():
            splines[key] = interp.InterpolatedUnivariateSpline(r, value, k=3)
        self.splines = splines

    def get_splines(self):
        return self.splines


class Parabolic_nu2(Equil_solver):

    def __init__(self, a=1, points=500, q0=1.0, k=1, b_z0=1):
        self.r = np.linspace(0, a, points)
        self.q0 = q0
        self.k = k
        self.b_z0 = b_z0
        r = self.r
        self.j0 = self.get_j0()
        param_points = {'j_z': self.j_z(r), 'b_theta': self.b_theta(r),
                        'b_z': self.b_z(r), 'pprime': self.pprime(r),
                        'pressure': self.pressure(r)}
        self.set_splines(param_points)

    def get_j0(self):
        self.j0 = 1
        return self.k*self.b_z(0)/self.b_theta_over_r(0)

    def j_z(self, r):
        return self.j0*(1 - r**2)**2

    def b_theta(self, r):
        j0 = self.j0
        return j0*r/2 - j0*r**3/2 + j0*r**5/6

    def b_theta_over_r(self, r):
        j0 = self.j0
        return j0*1/2 - j0*r**2/2 + j0*r**4/6

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


class Smoothed_core_skin(Equil_solver):
    pass


class Sharp_core_skin(Equil_solver):
    pass