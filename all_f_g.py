# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 10:28:37 2015

@author: Jens von der Linden

global variables all_f and all_g save value of f,g and r each time an f_func, or
g_func is called.
In the form [r, f].
To call fs  convert to numpy array and all_f[:, 1] for rs all_f[:, 0].
Used for debugging.
"""
all_f = []
all_g = []
all_g_term1 = []
all_g_term2 = []
all_g_term3 = []
all_g_term4 = []
all_g_term5 = []
all_g_term6 = []
all_pressure_prime = []
all_b_theta = []
all_b_z = []
all_beta_0 = []
all_m = []
all_k = []