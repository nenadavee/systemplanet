import time
import numpy as np
from numpy import log10, random, exp
from numpy import random as rand
from scipy import interpolate, special, stats
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import DataConversionWarning
from mpl_toolkits.mplot3d import Axes3D 
from PyAstronomy import pyasl
import matplotlib.pyplot as animation

from Planet import Planet
from system_var import system_vars

import mass_fraction_evolver
from constants import *

Lbol_interp = joblib.load('PMS_interp.pkl')

class System():
    def __init__(self, Nplanets, system_ID, cmass_disp, X_disp, model=0, plot=False):
        self.Nplanets = Nplanets # num of planets
        self.ID = system_ID
        # self.stable_system = False
        # self.sufficient_transit = False
        self.X_disp = X_disp # initial atmospheric mass fraction dispersion
        self.plot = plot # plot or not

        #* for the partner planets
        self.hillradius = np.zeros(self.Nplanets - 1)
        self.instabcriterion = np.zeros(self.Nplanets - 1)

        # populate system variables
        for system_var in system_vars:
            setattr(self, system_var, 0)

        while not 0.5 < self.Mstar < 1.5:
            self.Mstar = random.normal(loc=1.0, scale=0.15)

        # Create a stable system
        while True: # inf loop
            self.planets = [Planet(Nplanets, cmass_disp, self.Mstar) for _ in range(Nplanets)]
            if not self.check_stable_orbits():
                continue
            self.evolve(plot_system=plot)
            if not self.dectect_transit():
                continue
        
            
        #     print(self)
        #     time.sleep(5)
        print("Successfully created stable system! :D")

        self.model = model                           # choose which evolutionary model to use photoevaporation=0, core-powered mass-loss=1

        # 1. self.generate()             # generate planet parameters
        # 2. self.evolve(plot_system=self.plot)           # evolve the system through photoevaporation or core-powered mass-loss
        # 3. self.plot_orbits()                           # plots the system orbits using generate variables 
        # 4. self.probabilities()                         # calculate observing probabilities
        # 5. self.tabulate()                              # tabulate results
        # 6. self.ratios()                                # calculate period and radii ratios
    def __repr__(self) -> str:
        return(
            f"nplanets: {self.Nplanets}\n"
            f"Id: {self.ID}\n"
            f"X_disp: {self.X_disp}\n"
            f"instab: {self.instabcriterion}\n"
        )

    def check_stable_orbits(self) -> bool:
        # sorted copy of planet's sma
        sorted_planets_sma = sorted(self.planets, key=lambda planet: planet.sma)
        i = 0
        # [ C D B A E ]  original
        # bc there needs to be partners -> splits to partner a & b
        while sorted_planets_sma:
            part_a = sorted_planets_sma.pop(0)
            try:
                part_b = sorted_planets_sma[0]
            except Exception:
                break
            self.hillradius[i] = (((part_a.sma + part_b.sma) * AU) / 2) * ((((part_a.Mcore + part_b.Mcore) * M_earth) / (3 * self.Mstar * M_sun)) ** (1/3))

            self.instabcriterion[i] = (((part_b.sma * AU) * (1 - part_b.ecc)) - ((part_a.sma * AU) * (1 + part_a.ecc)))/self.hillradius[i]

            if self.instabcriterion[i] <= 8.0:
                # regenerate system -> unstable system
                print("UNSTABLE SYSTEM!!!!! Regenerating....")
                return False
            i += 1
        # self.stable_system = True
        return True

    def evolve(self, plot_system=False):
        for planet in self.planets:
            # if period generator failed, ignore this planet
            if planet.period == 0:
                continue

            planet.Rplanet, planet.Xfinal = mass_fraction_evolver.RK45_driver(
                    t_start=10.0, t_stop=planet.system_age, dt_try=0.001, accuracy=1e-5,
                    initial_X=planet.Xinit, composition=planet.composition,
                    M_core=planet.Mcore, period=planet.period, M_star=self.Mstar, smet=self.smet,
                    KH_timescale_cutoff=planet.KHtimescale, return_history=False, MIST_interp=Lbol_interp, 
                    set_offset=False, XUV_transition=False, use_Lcore=True, use_CPML=self.model
                )

            # planet mass is sum of core and atm. masses
            planet.Mplanet = planet.Mcore * (1.0 + planet.Xfinal)

            # is the planet a super-Earth of sub-Neptune
            if planet.Xfinal <= 1e-4:
                planet.Ptype = 'SE'
            else:
                planet.Ptype = 'SN'

        # can choose to plot the system architecture
        if plot_system:
            plt.figure(0)
            for planet in self.planets:
                if planet.Ptype == 'SE':
                    plt.scatter(planet.period, float(str(self.ID)[-1]), s=10*planet.Rplanet**2, color='firebrick')
                else:
                    plt.scatter(planet.period, float(str(self.ID)[-1]), s=10*planet.Rplanet**2, color='C0')

            plt.xscale('log')
            plt.xlim([1,100])
            plt.xlabel('Orbital Period (days)')
            plt.ylabel('System ID')
            plt.tight_layout()
            plt.show()

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

    def dectect_transit(self) -> bool:
        transiting_planets = 0
        self.Rstar = self.Mstar**0.8
        
        for planet in self.planets:
            bimpactpar = ((((planet.sma * AU) * (np.cos(planet.inclination)))/ self.Rstar * R_sun) * ((1-(planet.ecc ** 2)) / (1 + (planet.ecc * np.sin(planet.periastron)))))

            wills_theorem = (1 + (planet.Rplanet * R_earth/ self.Rstar* R_sun))
            if bimpactpar < wills_theorem:
                transiting_planets += 1
            
            if transiting_planets == 2:
                return True
                # regen
        return False
    
    def is_strong_signal():
        Signoise_check = False

        # After this, we assume the planets will transit (+2), we can work out if the signal to noise is large enough.
        while Signoise_check == False:
                for i in range(self.Nplanets): 
                    self.snrmi[i] = ((self.Rplanet[i] * R_earth / (self.Rstar * R_sun)  2) * (((4 * 365) / self.period[i])  (1/2)) * (1/CDPP))
                # also known as [mi] in the Fulton Paper

                print ("signal to noise is", self.snrmi[i]) 

                self.probdetection[i] = stats.gammacdf(self.snrmi[i],17.56, 1.00,scale=0.49)

                print ("probability of detection %", self.probdetection[i])

                self.randdetect[i] = np.random.uniform(0, 1)
                # If random number between 0 and 1 is less than self.probdetection then it is detected !!

                if self.randdetect[i] < self.probdetection[i]: 
                    Signoise_check = True 

                    print("DETECTED")


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

if __name__ == "__main__":
    # run_systems(1, 0.0, 0.0)
    # plt.tight_layout()
    # plt.show()
    system = System(Nplanets=3, system_ID=1, cmass_disp=0, X_disp=0)
    system.evolve(True)