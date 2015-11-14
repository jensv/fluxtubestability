# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:44:38 2015

@author: jensv
"""

import numpy as np
import numpy.testing as test
import scipy.integrate as integrate
import equil_solver as es
from scipy.interpolate import splev
from nose.tools import with_setup

test_equil = None

def setup_func():
    r"""
    Generate test equilibrium.
    """
    global test_equil
    test_equil = es.UnitlessSmoothedCoreSkin(core_radius_norm=0.9,
                                             transition_width_norm=0.033,
                                             skin_width_norm=0.034,
                                             epsilon=0.9, lambda_bar=.5)


def teardown_func():
    pass


@with_setup(setup_func, teardown_func)
def test_epsilon():
    r"""
    Test that ratio of b_theta gives epsilon.
    """
    r_core = test_equil.core_radius
    a = (test_equil.core_radius + 2.*test_equil.transition_width +
         test_equil.skin_width)
    b_theta_tck = test_equil.get_tck_splines()['b_theta']
    epsilon_from_b_theta_ratio = (splev(r_core, b_theta_tck) /
                                  splev(a, b_theta_tck))
    test.assert_approx_equal(epsilon_from_b_theta_ratio, test_equil.epsilon,
                             significant=3)


@with_setup(setup_func, teardown_func)
def test_lambda_bar():
    r"""
    Test that lambda bar is given by ratio of total current to magnetic flux.
    """
    a = (test_equil.core_radius + 2.*test_equil.transition_width +
         test_equil.skin_width)
    b_theta_tck = test_equil.get_tck_splines()['b_theta']
    b_z_tck = test_equil.get_tck_splines()['b_z']
    calculated_lambda = (2.*splev(a, b_theta_tck) /
                         (splev(a, b_z_tck)))
    test.assert_approx_equal(calculated_lambda, test_equil.lambda_bar,
                             significant=3)
