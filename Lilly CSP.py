# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:07:00 2017

@author: Qing Liu
"""

import numpy as np
import seaborn as sns
import asciitable as asc
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import Table, Column
from astropy.visualization import quantity_support
from astropy.cosmology import FlatLambdaCDM

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import gauss_lorentz_hermite, gauss_lorentz_hermite_prtb
from smpy.dust import Calzetti

# Go to smpy.smpy to set cosmo, the current is below:
cosmo = FlatLambdaCDM(H0=70., Om0=0.3) 

#==============================================================================
# Composite Model
#==============================================================================
bc03 = BC('data/ssp/bc03/chab/hr/')
M_seed = np.linspace(5,9,5)

#==============================================================================
# Test 
#==============================================================================
Ages = np.linspace(1.5,13.,24)* u.Gyr
                  
perturb = False
if perturb:
    A, P = 0.6, 1.0
    SFH_law = gauss_lorentz_hermite_prtb
    SFH_fit = asc.read('Lilly/Lilly SFH prtb.txt')
    SFH_pars = zip(SFH_fit.c1, SFH_fit.mu, SFH_fit.sigma,\
               SFH_fit.h13, SFH_fit.h14,\
               SFH_fit.c2, SFH_fit.x0, SFH_fit.gama,\
               SFH_fit.h23, SFH_fit.h24,\
               A*np.ones_like(M_seed), P*np.ones_like(M_seed))        
else:
    SFH_law = gauss_lorentz_hermite
    SFH_fit = asc.read('Lilly/Lilly SFH params.txt')
    SFH_pars = zip(SFH_fit.c1, SFH_fit.mu, SFH_fit.sigma, SFH_fit.h13, SFH_fit.h14,\
               SFH_fit.c2, SFH_fit.x0, SFH_fit.gama, SFH_fit.h23, SFH_fit.h24)
    

Dust = np.array([0.0])   

models = CSP(bc03, age = Ages, sfh = SFH_pars, 
             dust = Dust, metal_ind = 1., f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti)

SED = np.squeeze(np.log10(models.SED.value))

#==============================================================================
# Renormalization
#==============================================================================
use_bc = False
if use_bc:
    BC03_out = asc.read('Lilly/CSP result.txt')
    lgMs = BC03_out['lgM*'].reshape((M_seed.size, Ages.size)).T
    lgSSFR = BC03_out['lgsSFR'].reshape((M_seed.size, Ages.size)).T            
    Ms = 10**lgMs
    SSFR = 10**lgSSFR

#==============================================================================
# Mock Observation
#==============================================================================
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')

print eazy_library.filternames

filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
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
# UVJ Tracks
#==============================================================================
X, Y = V_J, U_V
plt.figure(figsize=(8,7))
with quantity_support():
    for k, Av in enumerate(Dust):
            for j,m in enumerate(M_seed):
                plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label='log M$_{today}$ = %.1f'%m,
                         lw=2, alpha=0.7)
                s = plt.scatter(X[0,:,j,k,0], Y[0,:,j,k,0], c=Ages/u.Gyr, 
                                s=15, cmap=plt.cm.viridis_r)
    col = plt.colorbar(s)
    col.set_label('Age(Gyr)')
    plt.plot([-1., 1.0, 1.6, 1.6], [1.3, 1.3, 2.01, 2.5], color='k', alpha=0.7)
    plt.legend(loc='best',fontsize=10,frameon=True,facecolor='w')
    plt.xlabel('V - J',fontsize=12)
    plt.ylabel('U - V',fontsize=12)
    plt.xlim([-0.75, 2.0])
    plt.ylim([-0.25, 2.25])
plt.show()

#==============================================================================
# Isochrone of UVJ
#==============================================================================
#plt.figure(figsize=(8,7))
#with quantity_support():
#    for i, Age in enumerate(Ages[:8]):
#        plt.plot(np.sort(X[0,i,:,k,0]), Y[0,i,:,k,0][np.argsort(X[0,i,:,k,0])], label='T = {0:.1f}'.format(Age), 
#                 lw=2, ls='--',alpha=0.9)
#    plt.legend(loc=4,fontsize=10,frameon=True,facecolor='w')
#    plt.axhline(0.5,ls='--',color='k')
#    plt.xlabel('V - J')
#    plt.ylabel('U - V')
#    plt.xlim([-0.3, 1.0])
#    plt.ylim([-0.3, 1.0])
#plt.show()

#==============================================================================
# Csed/UV vs Time
#==============================================================================
theta = 34.8*u.deg
Ssed = pd.DataFrame(np.squeeze(np.sin(theta)*U_V + np.cos(theta)*V_J))
Csed = pd.DataFrame(np.squeeze(np.cos(theta)*U_V - np.sin(theta)*V_J))
UV = pd.DataFrame(np.squeeze(U_V))

plt.figure(figsize=(7,7))
for i, m in enumerate(M_seed):
    plt.plot(Ages,UV[i],label='log M$_{today}$ = %.1f'%m)
plt.legend(loc='best',fontsize=12,frameon=True,facecolor='w')
plt.xlabel('T (Gyr)',fontsize=15)
plt.ylabel('U - V',fontsize=15)
plt.axhline(0.5,color='k',ls='--')
plt.axvline(3.,color='b',ls='--',lw=1,alpha=0.3)
plt.axvline(7.,color='b',ls='--',lw=1,alpha=0.3)
plt.xlim(-1.,13.)
plt.ylim(-0.1,1.2)
plt.show()

plt.figure(figsize=(7,7))
for i, m in enumerate(M_seed):
    plt.plot(Ages,Csed[i],label='log M$_{today}$ = %.1f'%m)
plt.legend(loc='best',fontsize=12,frameon=True,facecolor='w')
plt.xlabel('T (Gyr)',fontsize=15)
plt.ylabel('C$_{SED}$',fontsize=15)
plt.axhline(0.25,color='k',ls='--')
plt.axvline(3.,color='b',ls='--',lw=1,alpha=0.3)
plt.axvline(7.,color='b',ls='--',lw=1,alpha=0.3)
plt.xlim(-1.,13.)
plt.ylim(-0.1,1.2)
plt.show()

#==============================================================================
# Make Table
#==============================================================================
make_table = False

if make_table:
    data = Table()
    
    fluxes = np.squeeze(synphot.fluxes).value
                       
    for i, n in enumerate(Names):
        flux = (fluxes[i] * Ms[:,:]).ravel()
        error = 0.01 * flux
        noise = [np.random.normal(0, err) for err in error]
        print np.median(noise/flux)*100
        data.add_columns([Column(flux+noise,'F%s'%(i+1)), Column(error,'E%s'%(i+1))])

    id = Column(name='id', data=np.arange(1,len(data)+1)) 
    zspec = Column(name='zspec', data= -1 *np.ones(len(data)))  
    data.add_column(id, 0) 
    data.add_column(zspec)  

    df = data.to_pandas()

    np.savetxt('Lilly/Lilly_SFH.cat', data, header=' '.join(data.colnames),
               fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])