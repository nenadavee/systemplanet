import numpy as np
from numpy import log10, random, exp
from numpy import random as rand
from scipy import interpolate, special, stats
from math import comb
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sklearn
from sklearn.exceptions import DataConversionWarning
from mpl_toolkits.mplot3d import Axes3D 
from PyAstronomy import pyasl
import matplotlib.pyplot as animation

from system_var import system_vars
import time

import mass_fraction_evolver
from constants import *

Lbol_interp = joblib.load('PMS_interp.pkl')

class System():

    def __init__(self, Nplanets, system_ID, cmass_disp, X_disp, model=0, plot=False):
        self.Nplanets = Nplanets                     # number of planets in system
        self.ID = system_ID                          # system ID
        print(type(self.ID))
        self.Mcoeffs = [0.00,0.84,0.57,0.00,0.00]    # polynomial coefficients for probability distribution

        for system_var in system_vars:
            setattr(self, system_var, np.zeros(self.Nplanets))

        self.Ptype = [0] * self.Nplanets             # asign array for planet type i.e. super-Earth or sub-Neptune
        self.transits = [0] * self.Nplanets          # asign array for whether planet transits or not
        self.detected = [0] * self.Nplanets          # asign array for whether planet is detected or not


        self.hillradius = np.zeros(self.Nplanets - 1)# asign array for planet PAIR Mutual Hill Radius
        self.instabcriterion = np.zeros(self.Nplanets - 1)  # asign array for instability criterion calculated from MHR
        self.sorted_indices = []                     # asign array for planet semi major axis sorted indices 
        self.sorted_sma = []                         # asign array for planet sorted semi major axis
        self.sorted_Mcore = []                       # asign array for planet mass core sorted with new indices
        self.sorted_ecc = []                         # asign array for planet orbit eccentricity sorted with new indices

        self.Pcdf = period_distribution_CDF(power1=0.4,power2=0.4,cutoff=5.0)        # define cumulative distribution function for period
        
        # define cumulative distribution function for core mass
        self.M_cdf = bernstein_CDF(self.Mcoeffs, lims=[0.6,100])

        self.cmass_disp = cmass_disp                 # asign a core mass dispersion
        self.X_disp = X_disp                         # asign a initial atmospheric mass fraction dispersion

        self.model = model                           # choose which evolutionary model to use photoevaporation=0, core-powered mass-loss=1
        self.plot = plot                             # decide whether you want to plot the system architecture, default is False

        self.generate()             # generate planet parameters
        self.evolve(plot_system=self.plot)           # evolve the system through photoevaporation or core-powered mass-loss
        self.plot_orbits()                           # plots the system orbits using generate variables 
        self.probabilities()                         # calculate observing probabilities
        self.tabulate()                              # tabulate results
        self.ratios()                                # calculate period and radii ratios


    def generate(self):
        self.Pc = self.Pcdf(random.uniform())

        self.Mstar = self.generate_stellar_mass()
        
        # for each planet
        for i in range(self.Nplanets):
            self.Mcore[i] = self.generate_core_mass(i)
            self.period[i] = self.generate_period()
            self.sma[i] = self.calculate_sma(self.period[i])

            self.ecc[i] = self.generate_eccentricity()
            self.inclination[i] = self.generate_inclination()
            self.periastron[i] = self.generate_periastron()

            print(f"ecc: {self.ecc[0]}")
            time.sleep(10000)
            self.Xinit[i] = self.generate_Xinit(i)

            self.composition[i] = 0.33
            self.KHtimescale[i] = 100
            self.system_age[i] = 5000

        self.check_stable_orbits()

    def generate_stellar_mass(self):
        while True:
            Mstar = random.normal(loc=1.0, scale=0.15)
            if 0.5 < Mstar < 1.5:
                return Mstar

    def generate_core_mass(self, i):
        while True:
            if i == 0:
                Mcore = 10 ** self.M_cdf(random.uniform())
            else:
                Mcore = random.normal(self.Mcore[0], self.cmass_disp)
            if 0.6 < Mcore < 20.0:
                return Mcore

    def generate_period(self):
        while True:
            period = self.Pc * random.lognormal(0, self.Nplanets * 0.21)
            if 1.0 < period < 100.0:
                return period

    def generate_eccentricity(self):
        while True:
            ecc = np.random.rayleigh(0.020)
            if 0.0 <= ecc < 1.0:
                return ecc
    
    def generate_periastron(self):
        while True:
            periastron = np.random.uniform(0, 2 * np.pi)
            if 0 <= periastron <= 2 * np.pi:
                return periastron
    
    def generate_inclination(self):
        while True:
            inclination = np.random.rayleigh(1.40)
            if 0 <= inclination <= 90:
                return inclination

    def generate_Xinit(self, i):
        Xinit_check = False

        while not Xinit_check:
            if self.period[i] == 0:
                Xinit = 0
                Xinit_check = True
            else:
                L = 10**Lbol_interp([self.Mstar, 0.0, np.log10(3000 * 1e6)])[0] * L_sun
                
                Teq = ((L / (16 * stefan * pi)) * (4 * pi * pi / (G * self.Mstar * M_sun * (self.period[i] * day)**2))**(2/3))**0.25
                
                Xinit = 0.01 * (self.Mcore[i]**0.4) * ((Teq / 1000)**0.25)

                if Xinit < 0.001:
                    Xinit = 1e-4
                if Xinit > 0.5:
                    Xinit = 0.5

                Xinit_check = True

        return Xinit


    def calculate_sma(self, period):
        sma = (((period * 24 * 60 * 60) ** 2 * G * self.Mstar * M_sun / (4 * np.pi * np.pi)) ** (1 / 3)) / AU
        return sma

    def check_stable_orbits(self):
        Stableorb_check = False

        while not Stableorb_check:
            self.sort_planets_by_sma()
            pairs = self.Nplanets - 1

            for p in range(pairs):
                sorted_i = self.sorted_indices[p]
                sorted_k = self.sorted_indices[p + 1]

                self.hillradius[p] = ((((self.sorted_sma[sorted_i] + self.sorted_sma[sorted_k])) * AU / 2)) * ((((self.sorted_Mcore[sorted_i] / self.sorted_Mcore[sorted_k]) * M_earth) / (3 * self.Mstar * M_sun)) ** (1/3))
                self.instabcriterion[p] = (((self.sorted_sma[sorted_k] * AU) * (1 - self.sorted_ecc[sorted_k])) - ((self.sorted_sma[sorted_i] * AU) * (1 + self.sorted_ecc[sorted_i])))/self.hillradius[p]

                if self.instabcriterion[p] >= 8.0:
                    Stableorb_check = True
        return

    def sort_planets_by_sma(self):
        self.sorted_indices = np.argsort(self.sma)
        self.sorted_sma = self.sma[self.sorted_indices]
        self.sorted_Mcore = self.Mcore[self.sorted_indices]
        self.sorted_ecc = self.ecc[self.sorted_indices]


    def plot_orbits(self):
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         
         num_points = 200
        
         for i in range(self.Nplanets):
             
             print(i, "sma", self.sma[i] * AU, "incl", self.inclination[i], "perias", self.periastron[i],"ecc",  self.ecc[i]) # a good check

             sma = self.sma[i]
             ecc = self.ecc[i]
             perias = self.periastron[i]
             incl = self.inclination[i]
             period = self.period[i]

             orbit = pyasl.KeplerEllipse(sma, period, e = ecc, i = incl, w = perias) # in sma calculation sma / AU to make units match well

             # 3D coordinates 
             t = np.linspace(0, 100, num_points)
             pos = orbit.xyzPos(t)

             print("pos is", pos ) # this is a check 

             ax.plot(pos[:,0], pos[:,1], pos[:,2])

         ax.set_xlabel("Orbit X")
         ax.set_ylabel("Orbit Y")
         ax.set_zlabel("Orbit Z")

         plt.show()


    def evolve(self, plot_system=False):
        self.smet = 0
        for i in range(self.Nplanets):

            # if period generator failed, ignore this planet
            if self.period[i] == 0:
                continue

            self.Rplanet[i], self.Xfinal[i] = mass_fraction_evolver.RK45_driver(t_start=10.0, t_stop=self.system_age[i], dt_try=0.001, accuracy=1e-5,
                                                                                initial_X=self.Xinit[i], composition=self.composition[i],
                                                                                M_core=self.Mcore[i], period=self.period[i], M_star=self.Mstar, smet=self.smet,
                                                                                KH_timescale_cutoff=self.KHtimescale[i], return_history=False, MIST_interp=Lbol_interp, 
                                                                                set_offset=False, XUV_transition=False, use_Lcore=True, use_CPML=self.model)

            # planet mass is sum of core and atm. masses
            self.Mplanet[i] = self.Mcore[i] * ( 1.0 + self.Xfinal[i] )

            # is the planet a super-Earth of sub-Neptune
            if self.Xfinal[i] <= 1e-4:
                self.Ptype[i] = 'SE'
            else:
                self.Ptype[i] = 'SN'

        # can choose to plot the system architecture
        if plot_system:
            plt.figure(0)
            for i in range(len(self.period)):

                if self.Ptype[i] == 'SE':
                    plt.scatter(self.period[i], [float(self.ID[-1])], s=10*self.Rplanet[i]**2, color='firebrick')
                else:
                    plt.scatter(self.period[i], [float(self.ID[-1])], s=10*self.Rplanet[i]**2, color='C0')

            plt.xscale('log')
            plt.xlim([1,100])
            plt.xlabel('Orbital Period (days)')
            plt.ylabel('System ID')
            plt.tight_layout()

                
    def probabilities(self):

        Bimpact_check = False

        for i in range(self.Nplanets):

            if self.period[i] == 0:
                continue

            self.sma[i] = (((self.period[i] * 24 * 60 * 60)**2 * G * self.Mstar * M_sun / (4 * np.pi * np.pi))**(1/3)) #done before
            self.P_transit[i] = 0.7 * self.Mstar * R_sun / self.sma[i] #prob of transit, remove, calc Bimpact

        if len([i for i in self.P_transit if i > 0]) == 0:
            return
        else:
            self.P_transit_sys = min(i for i in self.P_transit if i > 0)
            U_rand = random.uniform()

        # max(P_tr) = 0.3 for 1.5Msol, at 1 day orbital period
        if U_rand <= self.P_transit_sys * 3:
            self.transits = 1
        else:
            self.transits = 0

        for i in range(self.Nplanets):

            if self.period[i] == 0:
                continue

            m_i = ((self.Rplanet[i]*R_earth) / (self.Mstar * R_sun))**2 * np.sqrt((4*365)/self.period[i]) * (1/5e-5)
            self.P_detection[i] = stats.gamma.cdf(m_i,17.56,scale=0.49)

            # print(self.P_detection[i], U_rand)
            if U_rand < self.P_detection[i] and self.transits:
                self.detected[i] = 1
            else:
                self.detected[i] = 0
        
        for i in range(self.Nplanets):
            while Bimpact_check == False: 

                self.Rstar = (self.Mstar ** 0.8) # solar units 

                self.bimpactpar[i] = ((((self.sma[i] * AU ) * (np.cos(self.inclination[i])))/ self.Rstar * R_sun) * ((1-(self.ecc[i] ** 2)) / (1 + (self.ecc[i] * np.sin(self.periastron[i])))))

                if self.bimpactpar[i] < (1 + (self.Rplanet[i] / self.Rstar)):
                    Bimpact_check = True

                    print ("impact parameter is:", self.bimpactpar[i])

    def tabulate(self):
        
        d = {
            'ID': self.ID,
            'smass': self.Mstar,
            'period': self.period,
            'prad': self.Rplanet,
            'Xinit': self.Xinit,
            'Xfinal': self.Xfinal,
            'cmass': self.Mcore,
            'pmass': self.Mplanet,
            'composition': self.composition,
            'type': self.Ptype,
            'prob': self.probability,
            'detected': self.detected
        }
        
        self.df = pd.DataFrame(data=d)
        self.df = self.df.query('period > 0')
        self.Nplanets = len(self.df)
        self.df = self.df.sort_values('period')
        self.df = self.df.reset_index()


    def ratios(self):
        self.RjRi = []
        self.PjPi = []
        self.Pmax = []
        self.ratio_type = []
        self.ratio_detect = []

        if self.Nplanets == 1:
            pass
        else:
            for i in range(self.Nplanets-1):

                for j in range(i+1,self.Nplanets):

                    if self.df['detected'][i] == 1 and self.df['detected'][j] == 1:
                        self.ratio_detect.append(1)
                    else:
                        self.ratio_detect.append(0)

                    self.RjRi.append(self.df['prad'][j] / self.df['prad'][i])
                    self.PjPi.append(self.df['period'][j] / self.df['period'][i])
                    self.Pmax.append(self.df['period'][j])

                    if self.df['type'][i] == 'SE':
                        if self.df['type'][j] == 'SE':
                            self.ratio_type.append('SE_SE')
                        else:
                            self.ratio_type.append('SE_SN')
                    else:
                        if self.df['type'][j] == 'SE':
                            self.ratio_type.append('SN_SE')
                        else:
                            self.ratio_type.append('SN_SN')
        
        d = {'RjRi': self.RjRi,
             'PjPi': self.PjPi,
             'Pmax': self.Pmax,
             'ratio_type': self.ratio_type,
             'ratio_detect': self.ratio_detect}

        self.df_ratio = pd.DataFrame(data=d)


def bernstein_poly(x, order, coefficients):
    """
    Bernstein polynomials. See appendix of Rogers & Owen 2021
    """

    coefficients = np.array(coefficients)
    poly_array = np.array([special.binom(order, i)*(x**i)*((1-x)**(order-i)) for i in range(order+1)])
    B = np.dot(coefficients, poly_array)

    return B


def bernstein_PDF(coeffs, loglims=[-4.0,0.0]):

    order_X = len(coeffs) - 1
    X_poly_min, X_poly_max = bernstein_poly(0, order_X, coeffs), bernstein_poly(1, order_X, coeffs)
    X_poly_norm = X_poly_max - X_poly_min
    X_norm = loglims[1] - loglims[0]
    U_X = random.uniform()
    return 10**(((X_norm/X_poly_norm) * ((bernstein_poly(U_X, order_X, coeffs)) - X_poly_min)) + loglims[0])


def bernstein_CDF(coeffs, lims=[0.6,100]):
    """
    Creates CDF for Bernstein polynomial, used for core mass distribution. See appendix of Rogers & Owen 2021
    """

    x = np.linspace(log10(lims[0]), log10(lims[1]), 50)
    bern_x = np.linspace(0.0,1.0,50)
    bern_pdf = [bernstein_poly(i, len(coeffs)-1, coeffs) for i in bern_x]

    # normalise pdf, calculate cdf
    bern_pdf = bern_pdf / np.sum(bern_pdf)
    bern_cdf = np.cumsum(bern_pdf)
    bern_cdf[0] = 0.0
    bern_cdf[-1] = 1.0

    # remove repeats
    bern_cdf, mask = np.unique(bern_cdf, return_index=True)
    x_mask = x[mask]

    # interpolate
    cdf_interp = interpolate.interp1d(bern_cdf, x_mask)

    return cdf_interp


def period_distribution_CDF(power1=2.31, power2=-0.08, cutoff=5.75):

    """
    Creates CDF for underlying period distribution (see Figure 2 Rogers & Owen 2021)
    """

    P_range = np.logspace(0,2,100)
    pdf = []

    # create pdf using double power law
    for i in range(len(P_range)):
        if P_range[i] <= cutoff:
            pdf_i = P_range[i] ** power1
            pdf.append(pdf_i)
        else:
            pdf_i = (cutoff ** (power1-power2)) * (P_range[i] ** power2)
            pdf.append(pdf_i)

    # normalise pdf, calculate cdf
    pdf = pdf / np.sum(pdf)
    cdf = np.cumsum(pdf)
    cdf[0] = 0.0
    cdf[-1] = 1.0

    # remove repeats
    cdf, mask = np.unique(cdf, return_index=True)
    P_mask = P_range[mask]

    # interpolate
    cdf_interp = interpolate.interp1d(cdf, P_range)

    return cdf_interp

def run_systems(Nsystems, cmass_disp, X_disp):

    for i in range(Nsystems):
        print('{0} / {1}'.format(i+1,Nsystems))
        _N = 1
        # asign random number of planets for this system (ensure it's greater than 0!)
        while _N <= 1:
            _N = int(random.poisson(lam=3.5))

        # generate and evolve system
        _system = System( _N, 'system_{0}'.format(i), cmass_disp, X_disp, model=0, plot=True) 

        # assign pandas dataframe for the systems
        if i == 0:
            df_planets = _system.df
            df_ratios = _system.df_ratio

        # add to pandas dataframe for new system
        else:
            df_planets = pd.concat([df_planets, _system.df], ignore_index=True)
            df_ratios = pd.concat([df_ratios, _system.df_ratio], ignore_index=True)

    return df_planets, df_ratios


run_systems(1, 0.0, 0.0)
plt.tight_layout()
plt.show()

# if "__name__" == "__main__":
#     print("HI")

