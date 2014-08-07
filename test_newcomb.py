# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:05:57 2014

@author: Jens von der Linden

Tests based on relations in Goedbloed (2010) Principles of MHD, Jardin &
Newcomb paper.
All equation references are from these books unless ofterwise noted.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
"""Python 3.x compatibility"""

import math
import numpy as np
from numpy import atleast_1d as d1
from numpy.testing import assert_allclose, assert_array_less
import newcomb_g as g
import newcomb_f as f


class TestParabolic(object):
    r"""
    Compare f and g outputs for a parabolic current profile.
    """

    def setUp(self):
        r"""
        Create a parabolic axial current equilibrium profile.
        """
        import equil_solver as es
        parabolic = es.Parabolic_nu2()
        params = {'mu0': 1.0, 'k': (2*math.pi)/1.0, 'n': 1.0, 'm': 1.0,
                  'a': 1.0, 'b': 10.0, 'omega_sq': -5.0, 'gamma': 1.0,
                  'constant': 1.0}
        splines = parabolic.get_splines()
        params.update(splines)
        self.params = params
        self.delta = 10E-5

    def test_jardin_f_negative(self):
        r"""
        Test that f is never negative for all r.
        """
        k, m, b_z, b_theta, q = map(self.params.get, ['k', 'm', 'b_z',
                                                      'b_theta', 'q'])
        r = np.linspace(0, 1, 100)
        f_params = {'r': r, 'k': d1(k), 'm': d1(m),
                    'b_z': b_z(r), 'b_theta': b_theta(r),
                    'q': q(r)}

        assert_array_less(-1E-14, f.jardin_f_8_78(**f_params))

    def test_jardin_f_newcomb_f(self):
        r"""
        Test that the jardin_f and newcomb_f expressions are equivalent.
        """
        k, m, b_z, b_theta, q = map(self.params.get, ['k', 'm', 'b_z',
                                                      'b_theta', 'q'])
        r = np.linspace(0, 1, 100)
        f_params = {'r': r, 'k': d1(k), 'm': d1(m),
                    'b_z': b_z(r), 'b_theta': b_theta(r), 'q': q(r)}

        assert_allclose(f.jardin_f_8_78(**f_params),
                        f.newcomb_f_16(**f_params))

    def test_jardin_f_goedbloed_f(self):
        r"""
        Test that the jardin_f and goedbloed_f expressions are equivalent.
        """
        k, m, b_z, b_theta, q = map(self.params.get, ['k', 'm', 'b_z',
                                                      'b_theta', 'q'])
        r = np.linspace(0, 1, 100)
        f_params = {'r': r, 'k': d1(k), 'm': d1(m),
                    'b_z': b_z(r), 'b_theta': b_theta(r), 'q': q(r)}

        assert_allclose(f.jardin_f_8_78(**f_params),
                        f.goedbloed_f_9_106(**f_params))


    def test_goedbloed_f_newcomb_f(self):
        r"""
        Test that the goedbloed_f and newcomb_f expressions are equivalent.
        """
        k, m, b_z, b_theta, q = map(self.params.get, ['k', 'm', 'b_z',
                                                      'b_theta', 'q'])
        r = np.linspace(0, 1, 100)
        f_params = {'r': r, 'k': d1(k), 'm': d1(m),
                    'b_z': b_z(r), 'b_theta': b_theta(r), 'q': q(r)}
        assert_allclose(f.jardin_f_8_78(**f_params),
                        f.goedbloed_f_9_106(**f_params))

    def test_jardin_g_8_79_and_8_80(self):
        r"""
        Test that the jardin equation (8.79) and jardin (8.80) equation
        are equivalent.
        """
        (k, m, b_z, b_theta,
         q, p_prime) = map(self.params.get, ['k', 'm', 'b_z', 'b_theta', 'q',
                                             'p_prime'])
        r = np.linspace(0, 1, 100)
        g_params = {'r': r, 'k': d1(k), 'm': d1(m),
                    'b_z': b_z(r), 'b_z_prime': b_z.derivative()(r),
                    'b_theta': b_theta(r),
                    'b_theta_prime': b_theta.derivative()(r),
                    'p_prime': p_prime(r), 'q': q(r),
                    'q_prime': q.derivative()(r)}

        assert_allclose(g.jardin_g_8_79(**g_params),
                        g.jardin_g_8_80(**g_params))

    def test_jardin_g_8_80_goedbloed_g(self):
        r"""
        Test that the jardin g (8.80) and goedbloed g expressions are
        equivalent.
        """
        (k, m, b_z, b_theta,
         q, p_prime) = map(self.params.get, ['k', 'm', 'b_z', 'b_theta', 'q',
                                             'p_prime'])
        r = np.linspace(0, 1, 100)
        g_params = {'r': r, 'k': d1(k), 'm': d1(m),
                    'b_z': b_z(r), 'b_z_prime': b_z.derivative()(r),
                    'b_theta': b_theta(r),
                    'b_theta_prime': b_theta.derivative()(r),
                    'p_prime': p_prime(r), 'q': q(r),
                    'q_prime': q.derivative()(r)}

        assert_allclose(g.jardin_g_8_80(**g_params),
                        g.goedbloed_g_0(**g_params))

    def test_jardin_g_8_80_newcomb_g_18(self):
        r"""
        Test that the jardin g (8.80) and newcomb g (18) expressions are
        equivalent.
        """
        (k, m, b_z, b_theta,
         q, p_prime) = map(self.params.get, ['k', 'm', 'b_z', 'b_theta', 'q',
                                             'p_prime'])
        r = np.linspace(0, 1, 100)

        g_params = {'r': r, 'k': d1(k), 'm': d1(m),
                    'b_z': b_z(r), 'b_z_prime': b_z.derivative()(r),
                    'b_theta': b_theta(r),
                    'b_theta_prime': b_theta.derivative()(r),
                    'p_prime': p_prime(r), 'q': q(r),
                    'q_prime': q.derivative()(r)}

        assert_allclose(g.jardin_g_8_80(**g_params),
                        g.newcomb_g_18(**g_params))

    def test_goedbloed_g_newcomb_g_18(self):
        r"""
        Test that the goedbloed g and newcomb g (18) expressions are
        equivalent.
        """
        (k, m, b_z, b_theta,
         q, p_prime) = map(self.params.get, ['k', 'm', 'b_z', 'b_theta', 'q',
                                             'p_prime'])
        r = np.linspace(0, 1, 100)
        g_params = {'r': r, 'k': d1(k), 'm': d1(m),
                    'b_z': b_z(r), 'b_z_prime': b_z.derivative()(r),
                    'b_theta': b_theta(r),
                    'b_theta_prime': b_theta.derivative()(r),
                    'p_prime': p_prime(r), 'q': q(r),
                    'q_prime': q.derivative()(r)}

        assert_allclose(g.goedbloed_g_0(**g_params),
                        g.newcomb_g_18(**g_params))

    def test_jardin_g_8_79_goedbloed_g(self):
        r"""
        Test that the jardin g (8.79) and goedbloed g expressions are
        equivalent.
        """
        (k, m, b_z, b_theta,
         q, p_prime) = map(self.params.get, ['k', 'm', 'b_z', 'b_theta', 'q',
                                             'p_prime'])
        r = np.linspace(0, 1, 100)
        g_params = {'r': r, 'k': d1(k), 'm': d1(m),
                    'b_z': b_z(r), 'b_z_prime': b_z.derivative()(r),
                    'b_theta': b_theta(r),
                    'b_theta_prime': b_theta.derivative()(r),
                    'p_prime': p_prime(r), 'q': q(r),
                    'q_prime': q.derivative()(r)}

        assert_allclose(g.jardin_g_8_79(**g_params),
                        g.goedbloed_g_0(**g_params))

    def test_jardin_g_8_79_newcomb_g_18(self):
        r"""
        Test that the jardin g (8.79) and newcomb g (18) expressions are
        equivalent.
        """
        (k, m, b_z, b_theta,
         q, p_prime) = map(self.params.get, ['k', 'm', 'b_z', 'b_theta', 'q',
                                             'p_prime'])
        r = np.linspace(0, 1, 100)
        g_params = {'r': r, 'k': d1(k), 'm': d1(m),
                    'b_z': b_z(r), 'b_z_prime': b_z.derivative()(r),
                    'b_theta': b_theta(r),
                    'b_theta_prime': b_theta.derivative()(r),
                    'p_prime': p_prime(r), 'q': q(r),
                    'q_prime': q.derivative()(r)}

        assert_allclose(g.jardin_g_8_79(**g_params),
                        g.newcomb_g_18(**g_params))

    def test_suydam_q_suydam_alpha_beta(self):
        r"""
        Not implemented
        """
        pass
        # r = np.linspace(0, 1, 100)
        # assert_almost_equal(suydam_q(), suydam_alpha_beta())

    def test_suydam_q_suydam_mu(self):
        r"""
        Not implemented
        """
        pass
        # r = np.linspace(0, 1, 100)
        # assert_almost_equal(suydam_q(), suydam_mu())

    def test_jardin_small_solution_goedbloed_small_solution(self):
        r"""
        Not implemented
        """
        pass
        # r = np.linspace(0, 1, 100)
        # assert_almost_equal(jardin_small_solution(), newcomb_small_solution())
