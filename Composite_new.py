#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 17:38:48 2018

@author: Q.Liu
"""
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, Column, vstack
from astropy.visualization import quantity_support
from scipy.integrate import quad

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import exponential, delayed, exponential_x2, delayed_x2
from smpy.dust import Calzetti

#==============================================================================
# Composite Model
#==============================================================================
bc03 = BC('data/ssp/bc03/chab/lr/')

SFH_law = exponential_x2 
    
#==============================================================================
# Test 
#==============================================================================
Ages = np.linspace(0.1, 8., 80)* u.Gyr
#SFH = np.array([3.]) * u.Gyr
SFH = np.ones(5) * 3 * u.Gyr
t_d = np.ones(len(SFH)) * 4 * u.Gyr
#weights = np.ones(len(SFH))*0.1
weights = np.array([0.01,0.05,0.1,0.2,0.3])

SFH_pars = zip(SFH,t_d,weights)
    

Dust = np.array([1.])   

models = CSP(bc03, age = Ages, sfh = SFH_pars, 
             dust = Dust, metal_ind = 1., f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti,
             neb_cont=True, neb_met=False)

SED = np.squeeze(models.SED.value)

#==============================================================================
# Mock Observation
#==============================================================================
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')

#print eazy_library.filternames

filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
    # FUV, NUV, U, B, V, R, I, J, H, K
Names = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
synphot = Observe(models, filters, redshift=0.001)

mags = synphot.AB[0]

FUV_V = mags[0]-mags[4]
NUV_V = mags[1]-mags[4]
U_V = mags[2]-mags[4]
V_J = mags[4]-mags[7]
V_K = mags[4]-mags[9]
NUV_R = mags[1]-mags[5]
R_K = mags[5]-mags[9]

#==============================================================================
# UV/Csed
#==============================================================================
theta = 34.8*u.deg
Ssed = pd.DataFrame(np.squeeze(np.sin(theta)*U_V + np.cos(theta)*V_J))
Csed = pd.DataFrame(np.squeeze(np.cos(theta)*U_V - np.sin(theta)*V_J))
UV = pd.DataFrame(np.squeeze(U_V))
VJ = pd.DataFrame(np.squeeze(V_J))
SFR = pd.DataFrame(np.squeeze(models.SFR))

plt.figure(figsize=(6,6))
ax=plt.subplot(111)
for i, tau in enumerate(SFH):
    plt.plot(Ages,UV[i],label=r'$\tau = ${0:g}'.format(tau),zorder=3)
plt.legend(loc='best',fontsize=11,frameon=True,facecolor='w')
plt.xlabel('T (Gyr)',fontsize=15)
plt.ylabel('U - V',fontsize=15)
#plt.ylabel('C$_{SED}$',fontsize=15)
plt.axhline(0.5,color='c',ls='-',lw=1,zorder=2)
plt.axhline(0.4,color='c',ls=':',lw=1,zorder=2)
plt.axhline(0.6,color='c',ls=':',lw=1,zorder=2)
plt.axvline(2.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
plt.axvline(8.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
plt.xlim(0.,11.5)
plt.ylim(0.,1.2)
plt.tight_layout()
#plt.savefig("New/Exp_UV_Csed.png",dpi=400)
plt.show()

#==============================================================================
# Make Table
#==============================================================================
make_table = False 
    
if make_table:
    fluxes = np.squeeze(synphot.fluxes[:,:,:,:,:5]).value
    
    data = Table()                   
    for i, n in enumerate(Names):
        flux = fluxes[i] * 1e10
        error = 0.01 * flux
        flux = flux.ravel()
        error = error.ravel()
        noise = [np.random.normal(0, err) for err in error]
        data.add_columns([Column(flux+noise,'F%s'%(i+1)), Column(error,'E%s'%(i+1))])

    id = Column(name='id', data=np.arange(1,len(data)+1)) 
    zspec = Column(name='zspec', data= -1 *np.ones(len(data)))  
    data.add_column(id, 0) 
    data.add_column(zspec)  

    #df = data.to_pandas()

    np.savetxt('N/Composite/comp_exp.cat', data, header=' '.join(data.colnames),
               fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
    