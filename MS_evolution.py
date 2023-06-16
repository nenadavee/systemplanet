import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, LinearNDInterpolator, RegularGridInterpolator
import pickle
import joblib

import read_mist_models


# THIS FILE LOADS AND STORES A SERIES OF STELLAR EVOLUTION MODELS
# THE GOAL IS TO SAVE A .PKL FILE THAT WE CAN USE TO INTERPOLATE STELLAR BOLOMETRIC 
# LUMINOSITY AS FUNCTION OF STELLAR MASS, METALLICITY AND TIME. 
# MODELS COME FROM MIST https://waps.cfa.harvard.edu/MIST/

smass = np.linspace(0.2,1.8,)
smass = [0.20,0.25,0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.45,0.50,
         0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.92,0.94,0.96,0.98,1.00,1.02,1.04,
         1.06,1.08,1.10,1.12,1.14,1.16,1.18,1.20,1.22,1.24,1.26,1.28,1.30,1.32,1.34,
         1.36,1.38,1.40,1.42,1.44,1.46,1.48,1.50,1.52,1.54,1.56,1.58,1.60,1.62,1.64,
         1.66,1.68,1.70,1.72,1.74,1.76,1.78,1.80]

Z = [-1.0,-0.75,-0.50,-0.25,
     0.00, 0.25, 0.50]

Z_str = ['m1.00','m0.75','m0.50','m0.25',
         'p0.00','p0.25','p0.50']

data = []

t_grid = np.linspace(7,10,100)

def interp(interp_obj, t, smass): 
    t_MS = np.log10(0.1 * 10**10 * smass**-2.5)

    if t < t_MS:
        return interp_obj(t)
    else:
        return interp_obj(t_MS)


################### DUMP AND INTERP EACH MIST FILE ################### 
data = np.ndarray((len(smass), len(Z), len(t_grid)))
for i in range(len(smass)):
    for j in range(len(Z)):
        smass_str = "{:05d}".format(round(smass[i]*100))
        eep = read_mist_models.EEP('./MIST_v1.2_feh_{0}_afe_p0.0_vvcrit0.4_EEPS/{1}M.track.eep'.format(Z_str[j], smass_str))

        _interp = interp1d(np.log10(eep.eeps['star_age']), eep.eeps['log_L'])
        _logL = [interp(_interp, t, smass[i]) for t in t_grid]
        for k in range(len(t_grid)):
            data[i,j,k] = _logL[k]

np.savetxt('Lbol.txt', data.flatten(), delimiter=',')
np.savetxt('Lbol_smass.txt', smass, delimiter=',')
np.savetxt('Lbol_smet.txt', Z, delimiter=',')
np.savetxt('Lbol_logt.txt', t_grid, delimiter=',')
joblib.dump(data, 'luminosities.pkl')   


################# INTERP LUMINOSITIES ################# 
data = joblib.load('luminosities.pkl')
interp = RegularGridInterpolator((smass, Z, t_grid), data)
joblib.dump(interp, '../PMS_interp.pkl')  

