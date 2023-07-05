import numpy as np

global pi, stefan, kappa_0, G, gamma, Delta_ab, mu, m_H, k_B, alpha, beta, M_sun, L_sun, AU, T_eq_earth, M_earth, R_earth, eta_0, t_sat, a0, b_cutoff, R_sun

pi = np.pi
stefan = 5.67e-8
kappa_0 = 2.294e-8
G = 6.674e-11
gamma = 5/3
Delta_ab = (gamma-1)/gamma
mu = 2.35
m_H = 1.67e-27
k_B = 1.381e-23
alpha = 0.68
beta = 0.45
M_sun = 1.989e30
L_sun = 3.828e26
AU = 1.496e11
S_earth = (L_sun / (4 * pi * AU * AU))
T_eq_earth = (L_sun / (16 * stefan * pi * AU * AU))*0.25
M_earth = 5.927e24
R_earth = 6.378e6
eta_0 = 0.1
eta_0 = 0.17
Myr = 1e6 * 365 * 24 * 60 * 60
day = 24 * 60 * 60
a0 = 0.5
b_cutoff = 0.7
R_sun = 6.957e8