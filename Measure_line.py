# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 22:25:25 2017

@author: Qing Liu
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.visualization import quantity_support
from scipy.integrate import quad
from scipy.integrate import simps

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import exponential, delayed
from smpy.dust import Calzetti


T = 1.015*u.Gyr
tau = 1*u.Gyr

bc03 = BC('data/ssp/bc03/chab/hr/')
ssp = CSP(bc03, age=T, sfh=tau, dust=0., metal_ind=1.0)

M_s = quad(exponential, 0, T.value, args=(tau.value,))[0]
ssp *= M_s
wave = ssp.wave.value
spec = np.squeeze(ssp.SED).value

#==============================================================================
# EW Hd                 
#==============================================================================
fit_range = abs(wave-4100)<100
continuum_left = ((4100-wave)<100) & ((4100-wave)>50)
continuum_right = ((wave-4100)<100) & ((wave-4100)>50)
F_0 = (spec[continuum_left].mean() + spec[continuum_right].mean())/2.
EW_Hd = simps(wave[fit_range],1-spec[fit_range]/F_0)

plt.plot(ssp.wave,spec,ssp.wave,F_0*np.ones_like(ssp.wave))
plt.xlim(4000,4200)
plt.ylim(0,0.001)
plt.show()
               
               