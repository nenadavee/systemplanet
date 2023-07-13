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


import mass_fraction_evolver
from constants import *

Lbol_interp = joblib.load('PMS_interp.pkl')

class System():

    def __init__(self, Nplanets, system_ID, cmass_disp, X_disp, model=0, plot=False):
        self.Nplanets = Nplanets                     # number of planets in system
        self.ID = system_ID                          # system ID
        self.Mcoeffs = [0.00,0.84,0.57,0.00,0.00]    # polynomial coefficients for probability distribution

        self.Mcore = np.zeros(self.Nplanets)         # asign array for planet core masses
        self.Xinit = np.zeros(self.Nplanets)         # asign array for planet initial atm. mass fractions
        self.composition = np.zeros(self.Nplanets)   # asign array for planet core compositions
        self.period = np.zeros(self.Nplanets)        # asign array for planet periods
        self.sma = np.zeros(self.Nplanets)           # asign array for planet semi-major axis
        self.KHtimescale = np.zeros(self.Nplanets)   # asign array for planet initial kelvin-helmholtz (cooling) timescales 
        self.system_age = np.zeros(self.Nplanets)    # asign array for planet (observed) ages
        self.Rplanet = np.zeros(self.Nplanets)       # asign array for planet radii
        self.Mplanet = np.zeros(self.Nplanets)       # asign array for planet masses (core + atm.)
        self.Xfinal = np.zeros(self.Nplanets)        # asign array for planet final atm. mass fractions
        self.Ptype = [0] * self.Nplanets             # asign array for planet type i.e. super-Earth or sub-Neptune
        self.P_transit = np.zeros(self.Nplanets)     # asign array for planet probability of transiting
        self.P_detection = np.zeros(self.Nplanets)   # asign array for planet probability of being detected 
        self.probability = np.zeros(self.Nplanets)   # asign array for planet probabilities of transiting and being detected
        self.transits = [0] * self.Nplanets          # asign array for whether planet transits or not
        self.detected = [0] * self.Nplanets          # asign array for whether planet is detected or not
        self.ecc = np.zeros(self.Nplanets)           # asign array for planet eccentricity
        self.inclination = np.zeros(self.Nplanets)   # asign array for planet inclination
        self.periastron = np.zeros(self.Nplanets)    # asign array for planet argument of periastron
        self.bimpactpar = np.zeros(self.Nplanets)    # asign array for b impact parameter 
        self.Rstar = np.zeros(self.Nplanets)         # asign array for stellar body Radius
        self.hillradius = np.zeros(self.Nplanets - 1)# asign array for planet PAIR Mutual Hill Radius
        self.sorted_indices = []                     # asign array for planet semi major axis sorted indices 
        self.sorted_sma = []                         # asign array for planet sorted semi major axis
        self.sorted_Mcore = [] # asign array for planet mass core sorted with new indices
        self.sorted_ecc = []
        self.long_asc_node = np.zeros(self.Nplanets) # asign array for longitude of the ascending node for the i-th planet in system
        self.instabcriterion = np.zeros(self.Nplanets - 1)
        self.Pcdf = period_distribution_CDF(power1=0.4,power2=0.4,cutoff=5.0)        # define cumulative distribution function for period 
        self.M_cdf = bernstein_CDF(self.Mcoeffs,                                    # define cumulative distribution function for core mass 
                                   lims=[0.6,100])

        self.cmass_disp = cmass_disp                 # asign a core mass dispersion
        self.X_disp = X_disp                         # asign a initial atmospheric mass fraction dispersion

        self.model = model                           # choose which evolutionary model to use photoevaporation=0, core-powered mass-loss=1
        self.plot = plot                             # decide whether you want to plot the system architecture, default is False

        self.generate()
        self.evolve(plot_system=self.plot)           # evolve the system through photoevaporation or core-powered mass-loss
        self.plot_orbits()                           # plots the system orbits using generate variables 
        # self.probabilities()                         # calculate observing probabilities
        # self.tabulate()                              # tabulate results
        # self.ratios()                                # calculate period and radii ratios


    # generate planet parameters
    def generate(self):
        Mcore_check = False
        Xinit_check = False
        Mstar_check = False

        self.Pc = self.Pcdf(random.uniform())
        # add a check that 1 < Pc < 100
        
        # draw random stellar mass according to Gaussian distributions between [0.5,1.5]Msol
        while Mstar_check == False:
            self.Mstar = random.normal(loc=1.0, scale=0.15)
            self.smet = 0.0
            if 0.5 < self.Mstar < 1.5:
                Mstar_check = True


        # for each planet, asign period, core mass and Xinit
        for i in range(self.Nplanets):
            Xinit_check = False
            Mcore_check = False
            Per_check = False
            Ecc_check = False
            Incl_check = False
            Periastron_check = False
            Stableorb_check = False

            while Mcore_check == False:
                # fist planet core mass from bernstein polynimial (see Rogers and Owen 2021)
                if i == 0:
                    self.Mcore[i] = 10**self.M_cdf(random.uniform()) 
                # all others have the same mass but with some core mass dispersion cmass_disp
                else:
                    self.Mcore[i] = random.normal(self.Mcore[0], self.cmass_disp)
                if 0.6 < self.Mcore[i] < 20.0:
                    Mcore_check = True

            while Per_check == False:
                self.period[i] = self.Pc * random.lognormal(0, self.Nplanets*0.21)
                # sma = distance from star, 
                # [10, 9, 18.3]
                self.sma[i] = (((self.period[i] * 24 * 60 * 60)**2 * G * self.Mstar * M_sun / (4 * np.pi * np.pi))**(1/3)) / AU

               # print(i, self.period[i], self.Pc)

                if 1.0 < self.period[i] < 100.0:
                    Per_check = True

            while Ecc_check == False:
                self.ecc[i] = np.random.rayleigh(0.020)
                # orbital eccentricities drawn from a rayleigh distr. self.ecc is the (e) created for each planet

                print(i, self.ecc[i])

                if 0.0 <= self.ecc[i] < 1.0:
                    Ecc_check = True

            while Incl_check == False:
                self.inclination[i] = np.random.rayleigh(1.40)

                if 0 <= self.inclination[i] <= 90:
                    Incl_check = True

            while Periastron_check == False:
                self.periastron[i] = np.random.uniform(0, 2 * np.pi)
                
                if 0 <= self.periastron[i] <= 2 * np.pi:
                    Periastron_check = True

            while Xinit_check == False:

                # if period generator (above) failed, it returns sets self.period[i]=0. We will ignore this planet.
                if self.period[i] == 0:
                    self.Xinit[i] = 0
                    Xinit_check = True
                else:
                    # initial atmospheric mass fraction according to scaling relation of Ginzburg et al. 2016
                    L = 10**Lbol_interp([self.Mstar, 0.0, np.log10(3000 * 1e6)])[0] * L_sun
                    
                    Teq = ( (L / (16*stefan*pi)) * (4*pi*pi / (G * self.Mstar * M_sun * (self.period[i] * day)**2))**(2/3) )**0.25
                    
                    self.Xinit[i] = 0.01 * (self.Mcore[i]**0.4) * ((Teq / 1000)**0.25)
                    
                    if self.Xinit[i] < 0.001:
                        self.Xinit[i] = 1e-4
                    if self.Xinit[i] > 0.5:
                        self.Xinit[i] = 0.5
                    Xinit_check = True

            # assume all planets have Earth-like core composition, initial KH timescale of 100Myr and are observed at 5 Gyr
            self.composition[i] = 0.33
            self.KHtimescale[i] = 100
            self.system_age[i] = 5000


        while Stableorb_check == False:
                 
            self.sorted_indices = np.argsort(self.sma)
            self.sorted_sma = self.sma[self.sorted_indices]
            self.sorted_Mcore = self.Mcore[self.sorted_indices]
            
            pairs = self.Nplanets - 1 # 5 planets, 4 pairs [ A B C D E ] -> Sort [ C D B A E ] -> Pairs CD DB BA AE hmmmm...
                 
            for p in range(pairs - 5):
                sorted_i = self.sorted_indices[p]  # Index of the first planet in the pair
                sorted_k = self.sorted_indices[p + 1]  # Index of the second planet in the pair
                     
                self.hillradius[p] = (((self.sorted_sma[sorted_i] + self.sorted_sma[sorted_k]) * AU)/ 2) * ((((self.sorted_Mcore[sorted_i] / self.sorted_Mcore[sorted_k]) * M_sun) / (3 * self.Mstar * M_sun)) ** (1/3))

                self.instabcriterion[p] = (((self.sorted_sma[sorted_k] * AU) * (1 - self.sorted_ecc[sorted_k]))-((self.sorted_sma[sorted_i] * AU) * (1 + self.sorted_ecc[sorted_i])))/ self.hillradius[p]

                print("Mutual Hill Radius", sorted_i, "-", sorted_k, ":", self.hillradius)
                

                if self.instabcriterion[p] >= 8.0: 
                    Stableorb_check = True


    def plot_orbits(self):
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         
         num_points = 100
        
         for i in range(self.Nplanets):
             
             print(i, self.sma[i] * AU, self.inclination[i], self.periastron[i], self.ecc[i]) # a good check

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
                    print(type(self.ID))
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
                #check sma, check by force system to have small incl no ecc, near 0 b 
                #sma 1 solar radii if incl 90, no ecc, b= 1

                if self.bimpactpar[i] < (1 + (self.Rplanet[i] / self.Rstar)):
                    Bimpact_check = True

                    print ("impact parameter is:", self.bimpactpar[i])

    def tabulate(self):
        
        d = {'ID': self.ID,
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
             'detected': self.detected}
        
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
        # print(self.df_ratio)

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