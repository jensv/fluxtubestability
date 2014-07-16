# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 10:29:40 2014

@author: Jens von der Linden

Tests based on relations in Goedbloed (2010) Principles of MHD.
All equation references are from the book unless ofterwise noted.
"""

import eigenvalue_goedbloed as eigen
from nose.tools import assert_almost_equal, assert_greater_equal

delta = 10E-5
test_equil()


def test_equil():
    """
    Create equilibrium for testing.

    Can start with simple constant density, axial current and field.
    """

def test_freq_ordering():
    """
    Tests the frequency ordering given in equation (9.38).
    """
    radii = []
    for radius in radii:
        test_omega_alfven = eigen.omega_alfven()
        test_omega_sound = eigen.omega_sound()
        test_omega_s0 = eigen.omega_s0()
        test_omega_f0 = eigen.omega_f0()

        assert_greater_equal(test_omega_s0, test_omega_sound)
        assert_greater_equal(test_omega_alfven, test_omega_s0)
        assert_greater_equal(test_omega_f0, test_omega_alfven)


def test_n():
    """
    Tests different formulations for n equations (9.32) & (9.36).
    """
    test_n_freq = eigen.n_freq()
    test_n_fb = eigen.n_fb()

    assert_almost_equal(test_n_freq, test_n_fb, delta=delta)


def test_d():
    """
    Tests different formulations for d equations (9.30) & (9.36).
    """
    test_d_freq = eigen.d_freq()
    test_d_fb = eigen.d_fb()

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