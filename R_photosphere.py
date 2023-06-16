import numpy as np
import math
from scipy.optimize import brentq
from constants import *


# /////////////////////////// IMPORT TABULATED INTEGRALS ///////////////////// #

# dR_Rc_array = np.loadtxt("dR_Rc_array.csv", delimiter=',')
# I2_array = np.loadtxt("I2.csv", delimiter=',')
# I2_I1_array = np.loadtxt("I2_I1.csv", delimiter=',')

# /////////////////////////// INTERPOLATION OF INTEGRALS ///////////////////// #

def I2(dR_Rc):
    log_dR_Rc = np.log10(dR_Rc)
    log_I2 = np.log(1.2598238012325735 + np.exp(-2.4535812359864062*1.0798633578446974*log_dR_Rc)) / -1.0798633578446974 - 4.8814961274562990e-01
    I2 = 10**log_I2
    return I2


def I2_I1(dR_Rc):
    log_dR_Rc = np.log10(dR_Rc)
    log_I2_I1 = -7.6521840215423576e-01 / (1.0 + np.exp(-(log_dR_Rc-5.7429970641208375e-02)/6.4338705851296174e-01))**1.8214254374336605E+00
    I2_I1 = 10**log_I2_I1
    return I2_I1

# //////////////////////////// CALCULATE STATE VARIABLES ///////////////////// #

def calculate_T_eq(M_star, smet, a, t, Lbol_interp):
    """
    Calculates the equilibrium temperature of the planet:
    M_star: Mass of host star (solar masses)
    a = semi-major axis of orbit (AU)
    """

    if Lbol_interp:
        if t > 3000:
            Lbol = 10**Lbol_interp([M_star, smet, np.log10(3000 * 1e6)])[0] * L_sun
        else:
            Lbol = 10**Lbol_interp([M_star, smet, np.log10(t * 1e6)])[0] * L_sun

        a_meters = a * AU
        T_eq = (Lbol / (16 * stefan * pi * a_meters* a_meters))**0.25

    else:

        T_eq = T_eq_earth * (1/a)**0.5 * (M_star)**(0.25 * 4.5)


    return T_eq


def calculate_sound_speed_squared(T_eq):
    """
    Calculates the isothermal sound speed of atmosphere:
    T_eq = Equilibrium temperature of planet (K)
    """

    c_s_squared = (k_B * T_eq) / (mu * m_H)
    return c_s_squared

# ///////////////////// ANALYTIC MODEL EQUATIONS FOR R_rcb /////////////////// #


def R_rcb_equation(R_rcb, T_eq, c_s_squared, KH_timescale_seconds, M_core_kg, R_core_meters, X, use_Lcore):

    if use_Lcore:
        psi = 1.0
    else:
        psi=0.0

    # collecting physical constants and terms which aren't function of R_rcb
    c1 = (4 * pi * mu * m_H / (M_core_kg * k_B))
    c2 = ((Delta_ab * G * M_core_kg / c_s_squared)**(1/(gamma-1)))
    c3 = ((1088 * pi * stefan * (T_eq**(3-alpha-beta)) * KH_timescale_seconds / (3 * kappa_0 * M_core_kg * (17*X+psi)))**(1/(alpha+1)))

    c = c1 * c2 * c3

    # full equation
    equation = c * (R_rcb**3) * I2((R_rcb/R_core_meters)-1) * ((1/R_rcb)**(1/(gamma-1))) * ((I2_I1((R_rcb/R_core_meters)-1) * R_rcb)**(1/(alpha+1))) - X

    return equation



# /////////////////////////////// SOLVE FOR R_rcb //////////////////////////// #

def solve_Rho_rcb_and_R_rcb(T_eq, c_s_squared, KH_timescale_seconds, M_core_kg, R_core_meters, X, R_guess, use_Lcore):
    """
    Calculates solution of R_rcb_equation using Newton-Raphson/secant method. Then
    finds Rho_rcb using the solution R_rcb:

    T_eq: Equilibrium temperature of planet (K)
    c_s_squared: isothermal sound speed (m^2 s^-2)
    a: semi-major axis of orbit (AU)
    KH_timescale_seconds: Kelvin-Helmholtz (cooling) timescale for atmosphere (seconds)
    M_core_kg: Mass of planet core (kg)
    R_core_meters: Radius of planet core (m)
    X: Envelope mass fraction M_env / M_core
    """


    if R_guess == 0.0:
        sign_test1 = np.sign(R_rcb_equation(R_core_meters*1.0001, T_eq, c_s_squared, KH_timescale_seconds, M_core_kg, R_core_meters, X, use_Lcore))
        sign_test2 = np.sign(R_rcb_equation(500*R_core_meters, T_eq, c_s_squared, KH_timescale_seconds,M_core_kg, R_core_meters, X, use_Lcore))
        if sign_test1 == sign_test2:
            return 0.0, 0.0
        R_rcb = brentq(R_rcb_equation, R_core_meters*1.0001, 500*R_earth, args=(T_eq, c_s_squared, KH_timescale_seconds,
                       M_core_kg, R_core_meters, X, use_Lcore), disp=False)

    else:
        sign_test1 = np.sign(R_rcb_equation(R_core_meters*1.0001, T_eq, c_s_squared, KH_timescale_seconds,M_core_kg, R_core_meters, X, use_Lcore))
        sign_test2 = np.sign(R_rcb_equation(R_earth*(1.0+R_guess), T_eq, c_s_squared, KH_timescale_seconds,M_core_kg, R_core_meters, X, use_Lcore))
        if sign_test1 == sign_test2:
            return 0.0, 0.0
        R_rcb = brentq(R_rcb_equation, R_core_meters*1.0001, R_earth*(1.0+R_guess), args=(T_eq, c_s_squared, KH_timescale_seconds,
                        M_core_kg, R_core_meters, X, use_Lcore), disp=False)



    Rho_rcb_1 = (mu * m_H / k_B)
    if use_Lcore:
        psi = 1.0
    else:
        psi = 0.0
    Rho_rcb_2 = ((I2_I1((R_rcb/R_core_meters)-1) * 1088 * pi * stefan * (T_eq**(3-alpha-beta)) * KH_timescale_seconds * R_rcb / (3 * kappa_0 * M_core_kg * (17*X+psi)))**(1/(alpha+1)))
    Rho_rcb = Rho_rcb_1 * Rho_rcb_2


    return R_rcb, Rho_rcb



# ////////////////////////// SOLVE FOR R_photosphere ///////////////////////// #

def calculate_R_photosphere(t, M_star, smet, a, M_core, R_core, X, KH_timescale_cutoff, R_guess, Lbol_interp, use_Lcore, return_rcb=False):
    """
    Returns the photospheric radius of the planet (in meters):

    M_star: Mass of host star (solar masses)
    a: semi-major axis of orbit (AU)
    KH_timescale = Kelvin-Helmholtz (cooling) timescale for atmosphere (years)
    M_core = Mass of planet core (Earth masses)
    R_core = Radius of planet core (Earth Radius)
    X = Envelope mass fraction M_env / M_core

    """

    # calculate temperature and sound speed for system
    T_eq = calculate_T_eq(M_star, smet, a, t, Lbol_interp)
    c_s_squared = calculate_sound_speed_squared(T_eq)

    if t < KH_timescale_cutoff:
        KH_timescale_seconds = KH_timescale_cutoff*Myr
    else:
        KH_timescale_seconds = t*Myr

    # convert to SI
    M_core_kg = M_core * M_earth
    R_core_meters = R_core * R_earth

    # solve simultaneous equations for radiative-convective boundary radius and density
    R_rcb, Rho_rcb = solve_Rho_rcb_and_R_rcb(T_eq, c_s_squared, KH_timescale_seconds, M_core_kg, R_core_meters, X, R_guess, use_Lcore)

    if (R_rcb, Rho_rcb) == (0.0, 0.0):
        if return_rcb:
            return 0.0, 0.0
        else:
            return 0.0

    # calculate gravitational constant (assumed constant)
    g = G * M_core_kg / (R_rcb*R_rcb)

    # calcualte scale height
    H = (k_B * T_eq) / (mu * m_H * g)

    # locate photosphere by finding pressure at which P=2/3 * g/kappa
    P_photosphere = (2 * g / (3 * kappa_0 * T_eq**beta))**(1/(alpha+1))

    # calculate photospheric density
    Rho_photosphere = P_photosphere * mu * m_H / (k_B * T_eq)

    # calculate photospheric radius
    R_photosphere = R_rcb + H * np.log(Rho_rcb / Rho_photosphere)

    # print('Rrcb / Rb = ', R_rcb / (G * M_core_kg / (2*c_s_squared)))

    if return_rcb:
        return R_rcb, Rho_rcb
    else:
        return R_photosphere



