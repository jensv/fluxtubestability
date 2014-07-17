# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 10:29:40 2014

@author: Jens von der Linden

Tests based on relations in Goedbloed (2010) Principles of MHD.
All equation references are from the book unless ofterwise noted.
"""

import eigenvalue_goedbloed as eigen

import math
from nose.tools import assert_almost_equal, assert_greater_equal

delta = 10E-5


def equil():
    """
    Create equilibrium for testing.

    Can start with simple constant density, axial current and field.
    """
    import equil_solver as es
    parabolic = es.Parabolic_nu2()
    params = {'mu0': 1.0, 'k': (2*math.pi)/1.0, 'n': 1.0, 'm': 1.0, 'a': 1.0,
              'b': 10.0, 'omega_sq': -5.0, 'gamma': 1.0, 'constant': 1.0}
    splines = parabolic.get_splines()
    params.update(splines)
    return params


def test_freq_ordering():
    """
    Tests the frequency ordering given in equation (9.38).
    """
    params = equil()
    k = params['k']
    m = params['m']
    gamma = params['gamma']
    b_theta_spl = params['b_theta']
    b_z_spl = params['b_z']
    rho_spl = params['rho']
    pressure_spl = params['pressure']

    radii = [.1, .25, .5, .75, .9]
    for radius in radii:

        b_theta = b_theta_spl(radius)
        b_z = b_z_spl(radius)
        pressure = pressure_spl(radius)
        rho = rho_spl(radius)
        f_ev = eigen.f(radius, k, m, b_theta, b_z)

        test_omega_alfven = eigen.omega_a_sq(f_ev, rho)
        test_omega_sound = eigen.omega_sound_sq(gamma, f_ev, b_theta, b_z, rho,
                                                pressure)
        test_omega_s0 = eigen.omega_s0_sq(radius, k, m, gamma, f_ev, b_theta,
                                          b_z, rho, pressure)
        test_omega_f0 = eigen.omega_f0_sq(radius, k, m, gamma, f_ev, b_theta,
                                          b_z, rho, pressure)

        assert_greater_equal(test_omega_s0, test_omega_sound)
        assert_greater_equal(test_omega_alfven, test_omega_s0)
        assert_greater_equal(test_omega_f0, test_omega_alfven)


def test_n():
    """
    Tests different formulations for n equations (9.32) & (9.36).
    """
    params = equil()
    k = params['k']
    m = params['m']
    gamma = params['gamma']
    b_theta_spl = params['b_theta']
    b_z_spl = params['b_z']
    rho_spl = params['rho']
    pressure_spl = params['pressure']
    omega_sq = params['omega_sq']

    radius = 0.5

    b_theta = b_theta_spl(radius)
    b_z = b_z_spl(radius)
    pressure = pressure_spl(radius)
    rho = rho_spl(radius)
    f_ev = eigen.f(radius, k, m, b_theta, b_z)

    omega_a_sq_ev = eigen.omega_a_sq(f_ev, rho)
    omega_s_sq_ev = eigen.omega_sound_sq(gamma, f_ev, b_theta, b_z, rho,
                                         pressure)

    test_n_freq = eigen.n_freq(gamma, b_theta, b_z, pressure, rho, omega_sq,
                               omega_a_sq_ev, omega_s_sq_ev)
    test_n_fb = eigen.n_fb(gamma, f_ev, b_theta, b_z, pressure, rho, omega_sq)

    assert_almost_equal(test_n_freq, test_n_fb, delta=delta)


def test_d():
    """
    Tests different formulations for d equations (9.30) & (9.36).
    """
    params = equil()
    k = params['k']
    m = params['m']
    gamma = params['gamma']
    b_theta_spl = params['b_theta']
    b_z_spl = params['b_z']
    rho_spl = params['rho']
    pressure_spl = params['pressure']
    omega_sq = params['omega_sq']

    radius = 0.5

    b_theta = b_theta_spl(radius)
    b_z = b_z_spl(radius)
    pressure = pressure_spl(radius)
    rho = rho_spl(radius)
    f_ev = eigen.f(radius, k, m, b_theta, b_z)

    omega_s0_sq = eigen.omega_s0_sq(radius, k, m, gamma, f_ev, b_theta, b_z,
                                    rho, pressure)
    omega_f0_sq = eigen.omega_f0_sq(radius, k, m, gamma, f_ev, b_theta, b_z,
                                    rho, pressure)

    test_d_freq = eigen.d_freq(rho, omega_sq, omega_s0_sq, omega_f0_sq)
    test_d_fb = eigen.d_fb(radius, k, m, gamma, f_ev, b_theta, b_z, rho,
                           pressure, omega_sq)

    assert_almost_equal(test_d_freq, test_d_fb, delta=delta)


def test_freq_limits():
    """
    Tests frequency limis given below equation (9.38).

    Notes
    -----
    May not be possible to implement numerically as these limits are for r->0,
    where the functions blow up.
    """
    pass


def test_newcomb_limit():
    """
    Tests limit of omega_sq=0 where the eigenvalue equations should become
    newcomb's equations.

    Notes
    -----
    Have to think about if it will be possible to implement this limit
    numerically.
    """
    pass