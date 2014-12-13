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
"""Python 3.x compatability"""

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
            splines[key] = interp.InterpolatedUnivariateSpline(r,
                                                               value(self.r),
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
            q_to_return[1:] = r[1:]*self.k*self.b_z(r[1:])/self.b_theta(r[1:])
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
                 qa=None, mu_0=1.):
        r"""
        Initalize parameters defining parabolic pinch and create splines.
        """
        self.r = np.linspace(0, a, points)
        if qa is not None:
            q0 = qa/3.
        self.q0 = q0
        self.mu_0 = mu_0
        self.k = k
        self.b_z0 = b_z0
        self.temp = temp
        self.j0 = self.get_j0()
        param_points = {'j_z': self.j_z, 'b_theta': self.b_theta,
                        'b_z': self.b_z, 'p_prime': self.p_prime,
                        'pressure': self.pressure, 'q': self.q,
                        'rho': self.rho}
        self.set_splines(param_points)

    def get_j0(self):
        r"""
        Return j0 for a b_z and q0 paramters of pinch.
        """
        self.j0 = 1
        return self.k*2.*self.b_z(0)/(self.q0*self.mu_0)

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
        return self.mu_0*(j0*r/2 - j0*r**3/2. + j0*r**5/6.)

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
        return self.mu_0*(-j0**2*r/2. + 3.*j0**2*r**3/2.
                          - 5.*j0**2*r**5/3. + 5.*j0**2*r**7/6.
                          - j0**2*r**9/6.)

    def pressure(self, r):
        r"""
        Returns pressure profile for a arabolic current profile pinch.
        """
        j0 = self.j0
        return self.mu_0*(47.*j0**2/720. - j0**2*r**2/4. + 3.*j0**2*r**4/8.
                          - 5.*j0**2*r**6/18. + 5.*j0**2*r**8/48.
                          - j0**2*r**10/60.)


class NuCurrentConstructor(object):

    def __init__(self, a=1., nu=2):
        j0, r, mu_0, k, b_z, q0, qa = sp.symbols('j_0 r mu_0 k b_z \
                                                  q_0 q_a')
        current_sym = j0*(1 - r**2)**nu
        b_theta_sym = mu_0*sp.integrate(current_sym*r, r, conds='none')/r
        p_prime_sym = -current_sym*b_theta_sym
        pressure_sym = sp.integrate(p_prime_sym, r, conds='none')
        pressure_norm_sym = pressure_sym - pressure_sym.subs(r, a)
        q_sym = sp.cancel(r*k*b_z/b_theta_sym)
        q0_sym = q_sym.subs(r, 0)
        qa_sym = q_sym.subs(r, a)
        j0_defined_by_q0_sym = sp.solve(q0_sym - q0, j0)[0]
        j0_defined_by_qa_sym = sp.solve(qa_sym - qa, j0)[0]

        # create lambda functions of expressions
        self.current_func = sp.lambdify((r, j0), current_sym,
                                        modules=str('numpy'))
        self.b_theta_func = sp.lambdify((r, j0, mu_0), sp.cancel(b_theta_sym),
                                        modules=str('numpy'))
        self.p_prime_func = sp.lambdify((r, j0, mu_0), sp.cancel(p_prime_sym),
                                        modules=str('numpy'))
        self.pressure_func = sp.lambdify((r, j0, mu_0), pressure_norm_sym,
                                         modules=str('numpy'))
        self.q_func = sp.lambdify((r, k, b_z, j0, mu_0), q_sym,
                                  modules=str('numpy'))
        self.j0_defined_by_q0_func = sp.lambdify((k, b_z, mu_0, q0),
                                                 j0_defined_by_q0_sym,
                                                 modules=str('numpy'))
        self.j0_defined_by_qa_func = sp.lambdify((k, b_z, mu_0, qa),
                                                 j0_defined_by_qa_sym,
                                                 modules=str('numpy'))


class NuCurrentProfile(EquilSolver):
    r"""
    """

    def __init__(self, nu_constructor=NuCurrentConstructor(), nu=2, a=1,
                 points=500, q0=1.0, k=1., b_z0=1., temp=1.0, qa=None, mu_0=1.):
        r"""

        """

        self.get_j_z = nu_constructor.current_func
        self.get_b_theta = nu_constructor.b_theta_func
        self.get_p_prime = nu_constructor.p_prime_func
        self.get_pressure = nu_constructor.pressure_func
        self.get_q = nu_constructor.q_func
        self.get_j0_given_q0 = nu_constructor.j0_defined_by_q0_func
        self.get_j0_given_qa = nu_constructor.j0_defined_by_qa_func

        self.r = np.linspace(0, a, points)
        self.nu = nu
        if qa is not None:
            self.j0 = self.get_j0_given_qa(k, b_z0, mu_0, qa)
        else:
            self.j0 = self.get_j0_given_q0(k, b_z0, mu_0, q0)
        self.q0 = q0
        self.mu_0 = mu_0
        self.k = k
        self.b_z0 = b_z0
        self.temp = temp
        param_points = {'j_z': self.j_z, 'b_theta': self.b_theta,
                        'b_z': self.b_z, 'p_prime': self.p_prime,
                        'pressure': self.pressure, 'q': self.q,
                        'rho': self.rho}
        self.set_splines(param_points)

    def j_z(self, r):
        r"""
        Return nu axial current profile.
        """
        j_z_value = self.get_j_z(r, self.j0)
        if isinstance(j_z_value, int) or isinstance(j_z_value, float):
            j_z_value = np.ones(r.size)*j_z_value
        return j_z_value

    def b_theta(self, r):
        r"""
        Return azimuthhal magnetic field for a nu current profile pinch.
        """
        return self.get_b_theta(r, self.j0, self.mu_0)

    def b_z(self, r):
        r"""

        """
        b_z0 = self.b_z0
        r = np.asarray(r)
        return np.ones(r.size) * self.b_z0

    def p_prime(self, r):
        r"""

        """
        return self.get_p_prime(r, self.j0, self.mu_0)

    def pressure(self, r):
        r"""

        """
        return self.get_pressure(r, self.j0, self.mu_0)

    def q(self, r):
        r"""
        Returns safety factor evaluated at points.
        """
        q_value = self.get_q(r, self.k, self.b_z0, self.j0, self.mu_0)
        if isinstance(q_value, int) or isinstance(q_value, float):
            q_value = np.ones(r.size)*q_value
        return q_value


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
        param_points = {'j_z': self.get_j_z, 'b_theta': self.b_theta,
                        'b_z': self.b_z, 'p_prime': self.p_prime,
                        'pressure': self.pressure, 'q': self.q,
                        'rho': self.rho}
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
                 core_radius=0.7, transition_width=0.1, skin_width=0.1, k=1.,
                 j_core=0.1, epsilon=0.3, lambda_bar=0.5, mu_0=1., q0=1.1,
                 b_z0=0.1, determinator='j_core'):
        r"""
        Initialize parameters defining smooth skin and core profile
        and create splines.
        """
        self.mu_0 = mu_0
        self.q0 = q0

        self.points_core = points_core
        self.points_transition = points_transition
        self.points_skin = points_skin
        self.core_radius = core_radius
        self.transition_width = transition_width
        self.skin_width = skin_width

        mask = np.ones(points_transition + 2, dtype=bool)
        mask[[0, -1]] = False
        self.r1 = np.linspace(0., core_radius, points_core)
        r2 = np.linspace(core_radius, core_radius + transition_width,
                         points_transition + 2)
        self.r2 = r2[mask]
        self.r3 = np.linspace(core_radius + transition_width,
                              core_radius + transition_width + skin_width,
                              points_skin)
        r4 = np.linspace(core_radius + transition_width + skin_width,
                         core_radius + 2*transition_width + skin_width,
                         points_transition + 2)
        self.r4 = r4[mask]
        self.r = np.concatenate((self.r1, self.r2, self.r3, self.r4))

        self.k = k
        self.epsilon = epsilon
        self.lambda_bar = lambda_bar

        if determinator == 'q0':
            self.b_z0 = b_z0
            self.j_core = self.get_j_z_core()
            self.ratio = self.get_ratio()
            self.j_skin = j_core*self.ratio
            self.j_skin = self.get_j_z_skin()

        if determinator == 'j_core':
            self.j_core = j_core
            self.ratio = self.get_ratio()
            self.j_skin = j_core*self.ratio
            self.current = self.get_current()
            self.b_z0 = self.get_b_z()

        param_points = OrderedDict([('j_z', self.j_z),
                                    ('b_theta', self.b_theta),
                                    ('b_z', self.b_z),
                                    ('p_prime', self.p_prime),
                                    ('pressure', self.pressure),
                                    ('q', self.q),
                                    ('rho', self.rho)])

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
        return g_bar + self.smooth_f(z)*delta_g

    def smooth_f(self, z):
        r"""
        Smoothing polynominal by Alan Glasser.
        """
        return z/8.*(3.*z**4 - 10.*z**2 + 15.)

    def get_j_z_core(self):
        r"""
        """
        return (2.*self.b_z0*self.k/(self.mu_0*self.q0))

    def get_ratio(self):
        r"""
        Returs j_z_skin based on j_z_core, geometry and epsilon of pinch.
        """
        a = self.core_radius + 2*self.transition_width + self.skin_width
        term1 = -7.*a**2
        term2 = 7.*a*self.skin_width
        term3 = 14.*a*self.transition_width
        term4 = 7.*a**2*self.epsilon
        term5 = -14.*a*self.skin_width*self.epsilon
        term6 = 7.*self.skin_width**2*self.epsilon
        term7 = -21.*a*self.transition_width*self.epsilon
        term8 = 21.*self.skin_width*self.transition_width*self.epsilon
        term9 = 16.*self.transition_width**2*self.epsilon
        denominator = (7.*(2.*a - self.skin_width -
                       2.*self.transition_width) *
                       (self.skin_width + self.transition_width) *
                       self.epsilon)
        return -(term1 + term2 + term3 + term4 + term5 + term6 +
                 term7 + term8 + term9) / denominator

    def get_current(self):
        r"""
        """
        a = self.core_radius + 2*self.transition_width + self.skin_width
        term1 = 7.*a**2*self.j_core
        term2 = -14.*a*self.skin_width*self.j_core
        term3 = 14.*a*self.skin_width*self.j_skin
        term4 = -21.*a*self.transition_width*self.j_core
        term5 = 14.*a*self.transition_width*self.j_skin
        term6 = 7.*self.skin_width**2*self.j_core
        term7 = -7.*self.skin_width**2*self.j_skin
        term8 = 21.*self.skin_width*self.transition_width*self.j_core
        term9 = -21.*self.skin_width*self.transition_width*self.j_skin
        term10 = 16.*self.transition_width**2*self.j_core
        term11 = -14.*self.transition_width**2*self.j_skin
        return np.pi/7.*(term1 + term2 + term3 + term4 + term5 + term6 +
                         term7 + term8 + term9 + term10 + term11)

    def get_b_z(self):
        r"""
        Returns b_z based on j_z, geometry and lambda_bar of pinch.
        """
        a = self.core_radius + 2.*self.transition_width + self.skin_width
        return self.current*self.mu_0/(np.pi*a**2*self.lambda_bar)

    def j_z(self, dummy_r):
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

        boundary1 = self.core_radius
        boundary2 = self.core_radius + self.transition_width
        boundary3 = self.core_radius + self.transition_width + self.skin_width
        boundary4 = (self.core_radius + 2*self.transition_width +
                     self.skin_width)

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
        Return b_theta at given r values.
        """
        b_theta_r_integrator = inte.ode(b_theta_r_prime_func)
        b_theta_r_integrator.set_integrator('lsoda')
        b_theta_r_integrator.set_f_params(self.splines['j_z'], self.mu_0)
        b_theta_r_integrator.set_initial_value(0., t=0.)
        b_theta_array = np.empty(r.size)
        b_theta_array[0] = 0.
        for i, position in enumerate(r[1:]):
            if b_theta_r_integrator.successful():
                b_theta_r_integrator.integrate(position)
                b_theta_array[i+1] = (b_theta_r_integrator.y/position)
            else:
                break
        return np.array(b_theta_array)

    def b_z(self, r):
        r"""
        Returns constant axial magnetic field.
        """
        return np.ones(r.size)*self.b_z0

    def p_prime(self, r):
        r"""
        Return pressure_prime at given r values. To be used for integration.
        """
        return -self.splines['b_theta'](r)*self.splines['j_z'](r)

    def pressure(self, r):
        r"""
        Return pressure_prime at given r values. To be used for integration.
        """
        pressure_integrator = inte.ode(p_prime_func)
        pressure_integrator.set_integrator('lsoda')
        pressure_integrator.set_initial_value(0., 0.)
        pressure_integrator.set_f_params(self.splines['j_z'],
                                         self.splines['b_theta'])
        pressure_unnorm = np.empty(r.size)
        pressure_unnorm[0] = 0.
        for i, position in enumerate(r[1:]):
            if pressure_integrator.successful():
                pressure_integrator.integrate(t=position)
                pressure_unnorm[i+1] = pressure_integrator.y
            else:
                break
        pressure_norm = pressure_unnorm - pressure_unnorm[-1]
        return pressure_norm

    def get_beta(self, r):
        r"""
        """
        beta_numerator = 2.*self.mu_0*self.splines['pressure'](self.r)
        beta_denominator = ((self.splines['b_z'](self.r))**2 +
                            (self.splines['b_theta'](self.r))**2)
        return interp.InterpolatedUnivariateSpline(self.r, beta_numerator/beta_denominator)

    def get_beta_average(self, r):
        pass

    def q(self, r):
        r"""
        Returns safety factor evaluated at points.
        """
        if r[0] == 0.:
            q0 = self.k*self.b_z0/(0.5*self.mu_0*self.j_core)
            q_to_return = np.ones(r.size)*q0
            q_to_return[1:] = (r[1:]*self.k*self.splines['b_z'](r[1:]) /
                               self.splines['b_theta'](r[1:]))
        else:
            q_to_return = (r*self.k*self.splines['b_z'](r) /
                           self.splines['b_theta'](r))
        return q_to_return


class UnitlessSmoothedCoreSkin(EquilSolver):
    r"""
    Creates splines describing a smooth skin and core current profile.
    """
    def __init__(self, points_core=20, points_transition=50, points_skin=20,
                 core_radius_norm=0.7, transition_width_norm=0.1, 
                 skin_width_norm=0.1, k_bar=1., beta=0.1,
                 j_0=1.0, epsilon=0.3, lambda_bar=0.5):
        r"""
        Initialize parameters defining smooth skin and core profile
        and create splines.
        """
        self.points_core = points_core
        self.points_transition = points_transition
        self.points_skin = points_skin
        self.core_radius = core_radius_norm
        self.transition_width = transition_width_norm
        self.skin_width = skin_width_norm
        self.r_0 = self.core_radius + 2*self.transition_width + self.skin_width
        self.r = self.r_points()
        
        self.k_bar = k_bar
        self.epsilon = epsilon
        self.lambda_bar = lambda_bar 
 
        self.splines = {}
    
        self.j_skin = self.get_j_skin_norm()
        self.make_spline('j_z', self.r, self.j_z(self.r))        
        
        self.b_theta_integrand_array = self.b_theta_integrand()
        self.q_0 = self.get_q_0()
        self.make_spline('b_theta', self.r, self.b_theta(self.r))
        self.make_spline('b_z', self.r, self.b_z(self.r))
        self.make_spline('pressure', self.r, self.pressure(self.r))
        self.make_spline('p_prime', self.r, )
        self.make_spline('q', self.r)
        self.make_spline('rho', self.r)        
        

    def r_points(self):
        r"""
        """
        (points_core, points_transition, 
         points_skin)                    = (self.points_core, 
                                            self.points_transition,
                                            self.points_skin)
        (core_radius, transition_width, 
         skin_width)                     = (self.core_radius, 
                                            self.transition_width,
                                            self.skin_width)
        mask = np.ones(points_transition + 2, dtype=bool)
        mask[[0, -1]] = False
        self.r1 = np.linspace(0., core_radius, points_core)
        r2 = np.linspace(core_radius, core_radius +
                         transition_width, points_transition + 2)
        self.r2 = r2[mask]
        self.r3 = np.linspace(core_radius + transition_width,
                              core_radius + transition_width + 
                              skin_width, points_skin)
        r4 = np.linspace(core_radius + transition_width + 
                         skin_width,
                         core_radius + 2*transition_width + 
                         skin_width, points_transition + 2)
        self.r4 = r4[mask]
        r = np.concatenate((self.r1, self.r2, self.r3, self.r4))
        return r
        
    def make_spline(self, key, r, values):
        r"""
        """
        self.splines[key] = interp.InterpolatedUnivariateSpline(r,
                                                                values,
                                                                k=3)
    def smooth(self, x1, x2, g1, g2, x):
        """
        Smoothing method by Alan Glasser.
        """
        delta_x = (x2 - x1) / 2.
        x_bar = (x2 + x1) / 2.
        delta_g = (g2 - g1) / 2.
        g_bar = (g1 + g2) / 2.
        z = (x - x_bar) / delta_x
        return g_bar + self.smooth_f(z)*delta_g

    def smooth_f(self, z):
        r"""
        Smoothing polynominal by Alan Glasser.
        """
        return z/8.*(3.*z**4 - 10.*z**2 + 15.)

    def get_j_skin_norm(self):
        r"""
        Returs j_z_skin based on j_z_core, geometry and epsilon of pinch.
        """
        (epsilon, skin_width, 
         transition_width, r_0) = (self.epsilon, self.skin_width, 
                                   self.transition_width, self.r_0) 
        
        factor1 = skin_width + epsilon*r_0 - r_0
        term1 = 16.*skin_width**2
        term2 = 21.*skin_width*transition_width
        term3 = -21.*skin_width*r_0
        term4 = 7.*transition_width**2
        term5 = -14.*transition_width*r_0
        term6 = 7.*r_0**2
        numerator = factor1*(term1 + term2 + term3 + term4)

        term1 = 14.*skin_width**3
        term2 = 21.*skin_width**2*transition_width
        term3 = 9.*skin_width**2*epsilon*r_0
        term2 = -28.*skin_width**2*r_0
        term3 = 7*skin_width*transition_width**2
        term4 = 21.*skin_width*transition_width*epsilon*r_0
        term5 = -35.*skin_width*transition_width*r_0
        term6 = -7.*skin_width*epsilon*r_0**2
        term7 = 14.*skin_width*r_0**2
        term8 = 7.*transition_width**2*epsilon*r_0
        term9 = -7.*transition_width**2*r_0
        term10 = -14.*transition_width*epsilon*r_0**2
        term11 = 14.*transition_width*r_0**2
        denominator = (term1 + term2 + term3 + term4 + term5 + term6 + term7 
                       + term8 + term9 + term10 + term11)
                       
        return numerator/denominator
        
    def j_z(self, dummy_r):
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

        boundary1 = self.core_radius
        boundary2 = self.core_radius + self.transition_width
        boundary3 = self.core_radius + self.transition_width + self.skin_width
        boundary4 = (self.core_radius + 2*self.transition_width +
                     self.skin_width)

        j_z = np.zeros(total_points)
        j_z[:points1] = 1.
        j_z[points1:points2] = self.smooth(boundary1, boundary2, 1.,
                                           self.j_skin,
                                           self.r[points1:points2])
        j_z[points2:points3] = self.j_skin
        j_z[points3:points4] = self.smooth(boundary3, boundary4, self.j_skin,
                                           0., self.r[points3:points4])
        return j_z     
        
    def b_theta_integrand():
        r"""
        """
        b_theta_r_integrator = inte.ode(b_theta_r_prime_func)
        b_theta_r_integrator.set_integrator('lsoda')
        b_theta_r_integrator.set_f_params(self.splines['j_z'], 1.0)
        b_theta_r_integrator.set_initial_value(0., t=0.)
        b_theta_integrand_array = np.empty(r.size)
        b_theta_integrand_array[0] = 0.
        for i, position in enumerate(r[1:]):
            if b_theta_r_integrator.successful():
                b_theta_r_integrator.integrate(position)
                b_theta_integrand_array[i+1] = (b_theta_r_integrator.y/position)
            else:
                break
        return b_theta_integrand_array
        
    def get_q_0(self):
        r"""
        """
        return self.b_theta_integrand_array[-1]*4.*self.k_bar/self.lambda_bar

    def b_theta(self, r):
        r"""
        Return b_theta at given r values.
        """
        b_theta_array = self.b_theta_integrand_array*2.*self.k_bar/self.q_0
        return b_theta_array

    def b_z(self, r):
        r"""
        Returns constant axial magnetic field.
        """
        self.b_z0 = 2.*self.b_theta_array[-1]/self.lambda_bar
        return np.ones(r.size)*self.b_z0

    def p_prime(self, r):
        r"""
        Return pressure_prime at given r values. To be used for integration.
        """
        return -self.splines['b_theta'](r)*self.splines['j_z'](r)

    def pressure(self, r):
        r"""
        Return pressure_prime at given r values. To be used for integration.
        """
        pressure_integrator = inte.ode(p_prime_func)
        pressure_integrator.set_integrator('lsoda')
        pressure_integrator.set_initial_value(0., 0.)
        pressure_integrator.set_f_params(self.splines['j_z'],
                                         self.splines['b_theta'])
        pressure_unnorm = np.empty(r.size)
        pressure_unnorm[0] = 0.
        for i, position in enumerate(r[1:]):
            if pressure_integrator.successful():
                pressure_integrator.integrate(t=position)
                pressure_unnorm[i+1] = pressure_integrator.y
            else:
                break
        pressure_norm = pressure_unnorm - pressure_unnorm[-1]
        pressure = pressure_norm*4.*self.k_bar/(self.beta*self.q_0)
        return pressure

    def q(self, r):
        r"""
        Returns safety factor evaluated at points.
        """
        if r[0] == 0.:
            q_to_return = np.ones(r.size)*self.q_0
            q_to_return[1:] = (r[1:]*self.k_bar*self.splines['b_z'](r[1:]) /
                               self.splines['b_theta'](r[1:]))
        else:
            q_to_return = (r*self.k_bar*self.splines['b_z'](r) /
                           self.splines['b_theta'](r))
        return q_to_return


class HardCoreZPinch(EquilSolver):
    r"""
    Create profiles for the Hard-Core Z-pinch as described in Freidberg Ideal
    MHD.
    """
    def __init__(self, i_c=0.1, i_p=0.2, r_c=0.1, r_a=1.0, k=1.0, points=500,
                 mu_0=1.):
        self.k = k
        self.i_c = i_c
        self.i_p = i_p
        self.r_c = r_c
        self.r_a = r_a
        self.current = i_c + i_p
        self.k_p = 3./2.*(5./3.)**(5./2.)
        self.k_i = mu_0*((i_c + i_p)/(2.*np.pi*r_c))**2
        self.mu_0 = mu_0

        self.r = np.linspace(r_c, r_a, points)

        param_points = {'j_z': self.j_z,
                        'b_theta': self.b_theta,
                        'b_z': self.b_z,
                        'p_prime': self.p_prime,
                        'pressure': self.pressure,
                        'q': self.q,
                        'rho': self.rho,
                        'stability': self.stability_criterion}

        self.set_splines(param_points)

    def b_z(self, r):
        r"""
        Returns axial field.
        """
        r = np.asarray(r)
        return np.zeros(r.size)

    def b_theta(self, r):
        r"""
        Returns azimuthal field.
        """
        r = np.asarray(r)
        x = r**2 / self.r_c**2
        b_theta = np.sqrt(self.k_i/x - 2.*self.mu_0*self.k_p*(9.*x-5.) /
                          (3*x*x**(3./2.)))
        return b_theta

    def pressure(self, r):
        r"""
        Returns pressure.
        """
        r = np.asarray(r)
        x = r**2 / self.r_c**2
        return self.k_p*(x - 1.)/x**(5./2.)

    def p_prime(self, r):
        r"""
        Returns derivative of pressure.
        """
        r = np.asarray(r)
        r_sym, r_c, k_p = sp.symbols('r r_c K_p')
        x = r_sym**2 / self.r_c**2
        pressure_sym = k_p*(x - 1)/x**(2.5)
        p_prime_func = sp.lambdify((r_sym, r_c, k_p),
                                   sp.diff(pressure_sym, r_sym), modules=np)
        return p_prime_func(r, self.r_c, self.k_p)

    def j_z(self, r):
        r"""
        Returns axial current.
        """
        r = np.asarray(r)

        r_sym, r_c, k_p, k_i = sp.symbols('r r_c K_p K_i')
        x = r_sym**2 / r_c**2
        b_theta = sp.sqrt(k_i/x - 2.*k_p*(9.*x - 5.)/(3*x*x**(1.5)))
        j_z_sym = sp.diff(r_sym*b_theta, r_sym)/(r_sym*self.mu_0)
        j_z_func = sp.lambdify((r_sym, r_c, k_p, k_i), j_z_sym,
                               modules=np)
        return j_z_func(r, self.r_c, self.k_p, self.k_i)

    def beta(self, r):
        r"""
        Returns Beta
        """
        r = np.asarray(r)
        return 32./2.*np.pi**2*self.r_c**2/(self.mu_0*self.current)*self.k_p

    def stability_criterion(self, r):
        r"""
        Returns stability criterion. If array is  greater than 0 any where the
        profile is unstable.
        """
        r = np.asarray(r)
        x_sym, r_c, current, k_p = sp.symbols('x r_c I K_p')
        beta = 32./3.*sp.pi**2*r_c**2/(self.mu_0*current**2)*k_p
        b_theta_norm_sq = 1. - beta/4. * (9.*x_sym - 5.)/x_sym**(1.5)
        stab_sym = sp.diff(b_theta_norm_sq/x_sym**(0.5), x_sym)
        stab_func = sp.lambdify((x_sym, r_c, k_p, current), stab_sym,
                                modules=np)
        return stab_func(r**2/self.r_c**2, self.r_c, self.k_p, self.current)


class SharpCoreSkin(EquilSolver):
    pass


def b_theta_r_prime_func(r, y, j_z, mu_0):
    r"""
    Return b_theta_r_prime at given r values. To be used for integration.
    """
    return j_z(r)*r*mu_0


def p_prime_func(r, y, j_z, b_theta):
    r"""
    Return pressure_prime at given r values. To be used for integration.
    """
    return -b_theta(r)*j_z(r)
