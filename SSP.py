# -*- coding: utf-8 -*-
"""
Created on Sun Jul 09 22:37:39 2017

@author: Qing Liu
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import asciitable as asc

import astropy.units as u
from astropy.visualization import quantity_support
from scipy.integrate import quad

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import exponential, delayed
from smpy.dust import Calzetti

table_src = asc.read('tttt.1-7')
wave_src = table_src['col1']
spec_src = table_src['col2']

T = 1.015*u.Gyr
tau = 1*u.Gyr

bc03 = BC('data/ssp/bc03/chab/hr/')
ssp = CSP(bc03, age=T, sfh=tau, dust=0., metal_ind=1.0)

M_s = quad(exponential, 0, T.value, args=(tau.value,))[0]
ssp *= M_s
spec = np.squeeze(ssp.SED)

with quantity_support():
    Fix, Ax = plt.subplots(1, figsize=(7,5))
    Ax.semilogy(ssp.wave, spec, '--', 
                label='SMpy: M_gal = 0.64', color='firebrick')
    Ax.semilogy(wave_src, spec_src, '-', 
                label='BC03 src: M_star = 0.44', color='steelblue')
    Ax.set_ylim([1e-4,1e-2])
    Ax.set_xlim([1000,7000])
    Leg = Ax.legend(loc='best',fontsize=12)
plt.show()

eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')

print eazy_library.filternames

filters = FilterSet()
filters.addEAZYFilter(eazy_library, [2, 4, 7])  # U, V, J

synphot = Observe(ssp, filters, redshift=0.01)

mags = np.squeeze(synphot.AB)
U, V, J = mags
UV = U - V
VJ = V - J

print UV,VJ
