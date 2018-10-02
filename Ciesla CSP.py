# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 00:17:08 2017

@author: Qing Liu
"""

import numpy as np
import seaborn as sns
import asciitable as asc
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, Column, vstack
from astropy.visualization import quantity_support
from astropy.cosmology import FlatLambdaCDM, z_at_value

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import RKPRG
from smpy.dust import Calzetti

# Go to smpy.smpy to set cosmo, the current is below:
#cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307) 
cosmo = FlatLambdaCDM(H0=70.4, Om0=0.272)

#==============================================================================
# Ciesla 2017
#==============================================================================
bc03 = BC('data/ssp/bc03/chab/hr/')
SFH_law = RKPRG        
#M_seed = np.linspace(6.0, 10.0, 5)
M_seed = (10**np.linspace(5.,7.,5)).tolist()
tg = np.linspace(0.5,11.5,111)
Ages = tg * u.Gyr

#==============================================================================
# Test 
#==============================================================================
Meta = np.array([1])  
Dust = np.array([1.])   

models = CSP(bc03, age = Ages, sfh = M_seed, 
             dust = Dust, metal_ind = 1.0, f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti)

SED = np.squeeze(np.log10(models.SED.value))

#==============================================================================
# Renormalization
#==============================================================================
use_bc = False
if use_bc:
    
    BC03_out = asc.read('New/Ciesla/CSP result.txt')
    lgMs = BC03_out['lgM*'].reshape((len(M_seed), Ages.size)).T
    lgSSFR = BC03_out['lgsSFR'].reshape((len(M_seed), Ages.size)).T            
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
plt.figure(figsize=(6,6))
with quantity_support():
    for k, Av in enumerate(Dust):
        for j, (m,c) in enumerate(zip(M_seed,['navy','steelblue','yellowgreen','gold','orangered'])):
                plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label=r'log M$\rm_{seed}$ = %.1f'%np.log10(m),
                         c=c,lw=5, alpha=0.7,zorder=6-j)
#                s = plt.scatter(X[0,:,j,k,0], Y[0,:,j,k,0], c=Ages/u.Gyr, 
#                                s=15, cmap=plt.cm.viridis_r)
#    col = plt.colorbar(s)
#    col.set_label('Age(Gyr)')
    plt.plot([-1., 1.0, 1.6, 1.6], [1.3, 1.3, 2.01, 2.5], color='k', alpha=0.7)
    plt.legend(loc='best',fontsize=12,frameon=True,facecolor='w')
    plt.xlabel('V - J',fontsize=15)
    plt.ylabel('U - V',fontsize=15)
    plt.xlim([-0.25, 1.75])
    plt.ylim([-0.25, 2.25])
plt.tight_layout()
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


plt.figure(figsize=(6,6))
ax=plt.subplot(111)
for i, (m,c) in enumerate(zip(M_seed,['navy','steelblue','yellowgreen','gold','orangered'])):
    plt.plot(Ages,UV[i],c=c,lw=2,label=r'log M$\rm _{seed}$ = %.1f'%np.log10(m))
plt.legend(loc='best',fontsize=11,frameon=True,facecolor='w')
plt.xlabel('Cosmic Time (Gyr)',fontsize=15)
plt.ylabel('U - V',fontsize=15)
plt.axhline(0.5,color='c',ls='-',lw=2,zorder=2,alpha=0.6)
plt.axhline(0.4,color='c',ls=':',lw=1,zorder=2,alpha=0.6)
plt.axhline(0.6,color='c',ls=':',lw=1,zorder=2,alpha=0.6)
plt.axvline(2.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
plt.axvline(8.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
plt.xlim(0.,11.5)
plt.ylim(0.,1.2)


ax=plt.subplot(122)
for i, (m,c) in enumerate(zip(M_seed,['navy','steelblue','yellowgreen','gold','orangered'])):
    plt.plot(Ages,Csed[i],c=c,label=r'log M$\rm _{seed}$ = %.1f'%np.log10(m))
plt.legend(loc='best',fontsize=12,frameon=True,facecolor='w')
plt.xlabel('Cosmic Time (Gyr)',fontsize=15)
plt.ylabel('C$_{SED}$',fontsize=15)
plt.axhline(0.25,color='c',ls=':',lw=3)
plt.axvline(2.5,color='k',ls='--',lw=2,alpha=0.5)
plt.axvline(8.5,color='k',ls='--',lw=2,alpha=0.5)
plt.xlim(0.,11.5)
plt.ylim(0.,.5)
plt.tight_layout()
plt.show()

#==============================================================================
# Make Table
#==============================================================================
make_table = False
obs_error = False

def compute_SNR_obs(i,Ages=Ages,Ms=Ms):
    z = np.array([z_at_value(cosmo.age,t) for t in Ages])
    z_grid = np.vstack([z for k in range(len(M_seed))]).T
    log_M = np.log10(Ms)
    
    SNR_coef = ascii.read("New/SNR_coef_all.txt").to_pandas()    
    cof = SNR_coef.iloc[i]
    snr_pred = 10**(cof.a0 + cof.a1*z_grid + cof.a2*log_M)
    snr = snr_pred.copy()
    
    snr[tg<2.5] = snr[(tg>=2.5)&(tg<=8.5)][0]
    snr[tg>8.5] = snr[(tg>=2.5)&(tg<=8.5)][-1]
    return snr

if obs_error:
    SNR_obs = np.zeros((10,len(Ages),len(M_seed)))
    for i in range(10):
        SNR_obs[i] = compute_SNR_obs(i)

if obs_error:
    n_iter = 100
else:
    n_iter = 1
if make_table:
    data_tot = Table()
    fluxes = np.squeeze(synphot.fluxes).value
    
    for j in range(n_iter):
        data = Table()                   
        for i, n in enumerate(Names):
            flux = fluxes[i] * Ms[:,:]
            if obs_error:
                error = (1./SNR_obs[i]) * flux 
            else:
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
        data_tot = vstack([data_tot,data])

    np.savetxt('N/Ciesla/Ciesla_SFH_obs.cat', data_tot, header=' '.join(data.colnames),
               fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])