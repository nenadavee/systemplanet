import time
import numpy as np
from numpy import log10, random, exp
from scipy import interpolate, special, stats
import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.exceptions import DataConversionWarning
# from mpl_toolkits.mplot3d import Axes3D 
# from PyAstronomy import pyasl
# import matplotlib.pyplot as animation
from system_var import planet_vars
from mathy_stuff import period_distribution_CDF, bernstein_CDF

from constants import *
Lbol_interp = joblib.load('PMS_interp.pkl')


class Planet():
    def __init__(self, Nplanets, cmass_disp, Mstar) -> None:
        self.Mcoeffs = [0.00,0.84,0.57,0.00,0.00]
        self.Pcdf = period_distribution_CDF(
            power1=0.4,power2=0.4,cutoff=5.0
        )
        # define cumulative distribution function for core mass
        self.M_cdf = bernstein_CDF(self.Mcoeffs, lims=[0.6,100])
        
        self.first_planet_generated = False
        self.first_planet_Mcore = -1
        self.cmass_disp = cmass_disp

        for planet_var in planet_vars:
            setattr(self, planet_var, -1)
        
        self.generate_planet_variables(Nplanets,Mstar)

    def generate_planet_variables(self, Nplanets, Mstar):
        self.Pc = self.Pcdf(random.uniform())
        
        # Mcore
        if self.first_planet_generated:
            while not 0.6 < self.Mcore < 20.0:
                self.Mcore = random.normal(self.first_planet_Mcore, self.cmass_disp)
        else:
            self.Mcore = 10**self.M_cdf(random.uniform())
            self.first_planet_Mcore = self.Mcore
            self.first_planet_generated = True
        
        # Period
        while not 1.0 < self.period < 100.0:
            self.period = self.Pc * random.lognormal(0, Nplanets*0.21)

        # Ecc
        while not (0.0 <= self.ecc < 1.0):
            self.ecc = np.random.rayleigh(0.020)
        
        # Inclination
        while not 0 <= self.inclination <= 90:
            self.inclination = np.random.rayleigh(1.40)
        
        # periastron
        while not 0 <= self.periastron <= 2 * np.pi:
            self.periastron = np.random.uniform(0, 2 * np.pi)
        
        # sma
        self.sma = (((self.period * 24 * 60 * 60)**2 * G * Mstar * M_sun / (4 * np.pi * np.pi))**(1/3)) / AU
        
        # Xinit stuff
        if self.period == 0:
            self.Xinit = 0
        else:
             # initial atmospheric mass fraction according to scaling relation of Ginzburg et al. 2016
            L = 10**Lbol_interp([Mstar, 0.0, np.log10(3000 * 1e6)])[0]*L_sun
            
            Teq = ( (L / (16*stefan*pi)) * (4*pi*pi / (G * Mstar * M_sun * (self.period * day)**2))**(2/3) )**0.25
            
            self.Xinit = 0.01 * (self.Mcore**0.4) * ((Teq / 1000)**0.25)
            
            if self.Xinit < 0.001:
                self.Xinit = 1e-4
            if self.Xinit > 0.5:
                self.Xinit = 0.5
        
        self.composition = 0.33
        self.KHtimescale = 100
        self.system_age = 5000
    
    def __repr__(self) -> str:
        return(
            f"sma: {self.sma}\n"
            f"inclination: {self.inclination}\n"
            f"perioas: {self.periastron}\n"
            f"ecc: {self.ecc}\n"
        )
    
        
if __name__ == "__main__":
    Mstar = 0
    while not 0.5 < Mstar < 1.5:
        Mstar = random.normal(loc=1.0, scale=0.15)
    planet = Planet(cmass_disp=3, Nplanets=5, Mstar=Mstar)
    planet2 = Planet(Nplanets=3, Mstar=Mstar, cmass_disp=0,)
    print(planet)
    print(planet2)

    # planet.period_distribution_CDF()
    # planet.generate_planet_variables()
    