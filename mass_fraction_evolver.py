import numpy as np
from numpy import log10, exp, sqrt
from numpy import random as rand
import joblib
from scipy import stats
from matplotlib import pyplot as plt
import pickle
import pandas as pd


from constants import *
import R_photosphere
import RKF45

transition = False 

efficiency_scaler = np.load("eff_shape_file.npy", allow_pickle=True).item()
Mp_scale = 7.603262769401823e+27
norm = 0.981533205679987
norm_eff = 0.1642377032138817


# ////////////////////////// FORTNEY ET AL 2007 M-R RELATIONS ////////////////////// #

def R_imf(imf, M_core): # ice mass fraction

    _R1 = (0.0912*imf + 0.1603) * (log10(M_core)*log10(M_core))
    _R2 = (0.3330*imf + 0.7387) *  log10(M_core)
    _R3 = (0.4639*imf + 1.1193)

    return _R1 + _R2 + _R3

def R_rmf(rmf, M_core): # rock mass fraction

    _R1 = (0.0592*rmf + 0.0975) * (log10(M_core)*log10(M_core))
    _R2 = (0.2337*rmf + 0.4938) *  log10(M_core)
    _R3 = (0.3102*rmf + 0.7932)

    return _R1 + _R2 + _R3


def calculate_R_core(composition, M_core):

    if composition <= 0.0:
        imf = -1.0 * composition
        R_core = R_imf(imf, M_core)
    if composition > 0.0:
        rmf = 1.0 - composition
        R_core = R_rmf(rmf, M_core)

    return R_core


def calculate_tX(t, X, M_core, M_star, smet, a, R_core, 
                 KH_timescale_cutoff, R_guess, Lbol_interp, 
                 logLx_offset, XUV_transition, use_Lcore, use_CPML, return_LxLbol=False):

    """
    This function calculated the mass-loss timescale of the atmosphere. See Owen
    and Wu (2017) for details.
    """

    global transition # this is for transition between X-ray and EUV driven mass-loss (not important here)

    transition = False

    # convert to SI units
    M_core_kg = M_core * M_earth
    a_meters = a * AU
    t_seconds = t * Myr
    R_core_meters = R_core * R_earth

    # Calculate envelope mass (kg)
    M_env_kg = X * M_core_kg
    t_sat = 100 * Myr * (M_star**(-1.0))
    
    # Calulate photospheric radius
    R_ph = R_photosphere.calculate_R_photosphere(t, M_star, smet, a, M_core, R_core, X, KH_timescale_cutoff, R_guess, Lbol_interp, use_Lcore)

    if R_ph == 0.0:
        return 0.0

    if use_CPML == 0: # XUV Photoevaporation
        

        # use MIST stellar evolution tracks to calclate high-energy flux
        if Lbol_interp:

            LxLbol_sat = 10**(-3.50) * (M_star**(-0.5))
            LeuvLbol_sat = 10**(-4.00) * (M_star**(-0.5))        
            if t_seconds < t_sat:
                LxLbol = LxLbol_sat
                LeuvLbol = LeuvLbol_sat
            else:
                LxLbol = LxLbol_sat * (t_seconds/t_sat)**(-1.3)
                LeuvLbol = LeuvLbol_sat * (t_seconds/t_sat)**(-0.65)

            if t > 3000:
                Lbol = 10**Lbol_interp([M_star, smet, np.log10(3000 * 1e6)])[0]
            else:
                Lbol = 10**Lbol_interp([M_star, smet, np.log10(t * 1e6)])[0]
            Lx = L_sun * LxLbol * Lbol
            Leuv = L_sun * LeuvLbol * Lbol

        # else use simple mass-luminosity relation
        else:
            # calculate saturation luminosity for photoevaporation
            L_sat = 10**(-3.50) * L_sun * (M_star**(0.5))
            if t_seconds < t_sat:
                Lx = L_sat
            else:
                Lx = L_sat * (t_seconds/t_sat)**(-1.5)
            Leuv = None
        
        # incorporate scatter (if required)
        logLx = log10(Lx)
        logLx = logLx + logLx_offset
        Lx = 10**logLx

        if Lbol_interp:
            logLeuv = log10(Leuv)
            logLeuv = logLeuv + logLx_offset
            Leuv = 10**logLeuv

        if return_LxLbol:
            return Lx/(Lbol)

        # simple mass-loss efficiency scaling
        # escape_velocity = np.sqrt(2*G*M_core_kg / R_core_meters) * 0.001
        # eta = eta_0 * (escape_velocity / 23)**(-0.42)
        # eta = eta_0 * (escape_velocity / 15)**(-2.0)

        # use efficiency scaling from Owen & Jackson 2012 (see https://github.com/jo276/EvapMass/tree/eff_mod)
        R_ph_cgs = R_ph * 1e2
        M_p_cgs = M_core * (1.0 + X) * M_earth * 1e3
        scaled_eff = 10.**efficiency_scaler(np.log10(R_ph_cgs*Mp_scale/M_p_cgs))
        mass_scale = (1. + (np.sqrt(M_p_cgs/1e29))**10.)**(1./10.)
        eta = scaled_eff * mass_scale * norm_eff / norm
        # eta = 0.10

        # Calculate mass loss rate due to photoevaporation
        M_env_dot_PE = eta * R_ph**3 * Lx / (4 * a_meters * a_meters * G * M_core_kg)

        if XUV_transition: # not important
            print('warning, have not changed code to calculated transitions!')
            
            if transition == True:
                M_env_dot = eta * R_ph**3 * Leuv / (4 * a_meters * a_meters * G * M_core_kg)
            else:
                Leuv_crit = 2e29 * (a / 0.1)**2 * (M_env_dot / 1e9)**2 * (R_ph / (10*R_earth))**-3
                if Leuv > Leuv_crit:
                    M_env_dot = eta * R_ph**3 * Leuv / (4 * a_meters * a_meters * G * M_core_kg)
                    transition = True


    else: # Core-powered mass-loss (note this requires use_Lcore == True)

        if t < KH_timescale_cutoff:
            KH_timescale_seconds = KH_timescale_cutoff*Myr
        else:
            KH_timescale_seconds = t*Myr

        T_eq = R_photosphere.calculate_T_eq(M_star, smet, a, t, Lbol_interp)
        c_s_squared = R_photosphere.calculate_sound_speed_squared(T_eq)
        R_bondi = G * M_core_kg * (X+1) / (2*c_s_squared)

        R_rcb, rho_rcb = R_photosphere.calculate_R_photosphere(t, M_star, smet, a, M_core, R_core, X, KH_timescale_cutoff, R_guess, Lbol_interp, use_Lcore, return_rcb=True)
        M_env_dot_EL = (((17*X+1) * M_core_kg * R_core_meters) / (17 * KH_timescale_seconds * R_rcb)) * (1 / R_photosphere.I2_I1((R_rcb/R_core_meters)-1))
        # M_env_dot_BL = 4.0 * pi * R_bondi * R_bondi * sqrt(c_s_squared) * rho_rcb * exp( - G * M_core_kg / (c_s_squared * R_rcb) )
        M_env_dot_BL = 4 * pi * R_bondi * R_bondi * sqrt(c_s_squared) * rho_rcb * exp( 1.5 - (2*R_bondi/R_rcb) ) 

        M_env_dot_CPML = min(M_env_dot_EL, M_env_dot_BL)

        # if M_env_dot_BL <= M_env_dot_EL:
        #     M_env_dot_CPML = M_env_dot_CPML * 10
    
    # print(use_CPML, M_env_dot_PE, M_env_dot_CPML)
    if use_CPML == 1:
        M_env_dot = M_env_dot_CPML
    elif use_CPML == 0:
        M_env_dot = M_env_dot_PE
    else:
        raise Exception("use_CPML must be 0,1")
        
    # Calculate mass-loss timescale
    tX = (M_env_kg / M_env_dot) / Myr

    return tX

# /////////////////////////// MASS FRACTION EVOLUTION ODE //////////////////// #

def dXdt_ODE(t,  X, parameters):

    """
    This function presents the ODE for atmospheric mass-loss dX/dt = - X / tX
    where X = M_atmosphere / M_core and tX is the mass-loss timeascale.
    """
    M_core, M_star, smet, a, R_core, KH_timescale_cutoff, R_guess, Lbol_interp, logLx_offset, XUV_transition, use_Lcore, use_CPML = parameters
    tX = calculate_tX(t, X, M_core, M_star, smet, a, R_core, KH_timescale_cutoff, R_guess, Lbol_interp, logLx_offset, XUV_transition, use_Lcore, use_CPML)

    if tX == 0.0:
        return 0.0

    dXdt = - X / tX

    return dXdt


# /////////////////// EVOLVE MASS FRACTION ACCORDING TO ODE ////////////////// #

def RK45_driver(t_start, t_stop, dt_try, accuracy, initial_X, composition, 
                M_core, period, M_star, smet, KH_timescale_cutoff, logLx_scatter=0.0,
                return_history=False, MIST_interp=False, set_offset=False, XUV_transition=False,
                use_Lcore=True, use_CPML=0):

    """
    This function controls the integration of the mass-loss ODE. It calls upon
    the "calculate_R_photosphere" function from the R_photosphere file as well
    as the "step_control" function in the RKF45 file to calculate the evolution
    of the mass fraction X.
    """

    if return_history:
        t_array = []
        X_array = []
        Rph_array = []


    # add scatter to Lx for each star
    if set_offset:
        logLx_offset = set_offset
    else:
        logLx_offset = rand.normal(0,logLx_scatter)


    R_core = calculate_R_core(composition, M_core)
    # orbital period to semi-major axis measured in AU
    a = (((period * 24 * 60 * 60)**2 * G * M_star * M_sun / (4 * pi * pi))**(1/3)) / AU
    #calculate initial photospheric radius
    R_ph_init = R_photosphere.calculate_R_photosphere(t_start, M_star, smet, a, M_core,R_core, initial_X, KH_timescale_cutoff, 0.0, MIST_interp, use_Lcore) / R_earth

    # define initial variables
    if return_history:
        t_array.append(t_start)
        X_array.append(initial_X)
        Rph_array.append(R_ph_init)

    X = initial_X
    t = t_start
    R_ph = R_ph_init
    dt = dt_try

    # setup loop
    while t < t_stop:
        # print(t, R_ph)
        #perform an adaptive RK45 step
        [t_new, X_new, dt_next] = RKF45.step_control(t, X, dt, dXdt_ODE, accuracy, parameters=[M_core, M_star, smet, a, R_core, KH_timescale_cutoff, R_ph, MIST_interp, logLx_offset, XUV_transition, use_Lcore, use_CPML])

        if t_new >= t_stop:
            t_new = t_stop
            X_new, _ = RKF45.RKCashKarp_Step45(t_stop, X, dXdt_ODE, t_stop-t, [M_core, M_star, smet, a, R_core, KH_timescale_cutoff, R_ph, MIST_interp, logLx_offset, XUV_transition, use_Lcore, use_CPML])

        if [t_new, X_new, dt_next] == [0.0, 0.0, 0.0]:
            return 0.0, 0.0

        # calculate new R_ph
        R_ph_new = R_photosphere.calculate_R_photosphere(t_new, M_star, smet, a, M_core, R_core, X_new, KH_timescale_cutoff, 0.0, MIST_interp, use_Lcore) / R_earth
        if R_ph_new == 0.0:
            print('here1')
            return 0.0, 0.0
        R_rcb, Rho_rcb = R_photosphere.calculate_R_photosphere(t_new, M_star, smet, a, M_core, R_core, X_new, KH_timescale_cutoff, 0.0, MIST_interp, use_Lcore, return_rcb=True)
        if [R_rcb, Rho_rcb] == [0.0, 0.0]:
            print('here2')
            return 0.0, 0.0
        
        T_eq = R_photosphere.calculate_T_eq(M_star, smet, a, t, MIST_interp)
        c_s_squared = R_photosphere.calculate_sound_speed_squared(T_eq)
        R_bondi = G * M_core * M_earth * (X_new+1) / (2*c_s_squared)
        mlr = 4 * pi * R_bondi * R_bondi * sqrt(c_s_squared) * Rho_rcb * exp( 1.5 - (2*R_bondi/R_rcb) ) 

        R_rcb = R_rcb / R_earth

        # if X becomes very small, we can assume all atmosphere is removed
        check_strip = False
        if X_new <= 1e-4:
            check_strip = True
        if R_ph_new <= R_core:
            check_strip = True
        if R_ph_new <= R_core or R_ph_new-R_rcb >= R_rcb-R_core:
            check_strip = True
        if check_strip:
            if return_history:
                t_array.append(t_new)
                X_array.append(1e-4)
                Rph_array.append(R_core)

                t_array.append(t_stop)
                X_array.append(1e-4)
                Rph_array.append(R_core)
                if XUV_transition:
                    return t_array, X_array, Rph_array, transition
                else:
                    return t_array, X_array, Rph_array
            else:
                if XUV_transition:
                    R_core, 1e-4, transition
                else:
                    return R_core, 1e-4
            
        # update step size and t according to step-control
        if return_history:
            t_array.append(t_new)
            X_array.append(X_new)
            Rph_array.append(R_ph_new)

            
        t = t_new
        dt = dt_next
        X = X_new
        R_ph = R_ph_new

    if return_history:
        if XUV_transition:
            return t_array, X_array, Rph_array, transition
        else:
            return t_array, X_array, Rph_array
    else:
        if XUV_transition:
            return R_ph, X, transition
        else:
            return R_ph, X



