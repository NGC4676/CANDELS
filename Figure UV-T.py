# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:51:44 2017

@author: Qing Liu
"""

import numpy as np
import seaborn as sns
import asciitable as asc
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import gauss_lorentz_hermite, glh_absprtb, RKPRG, exponential, delayed
from smpy.dust import Calzetti

# Go to smpy.smpy to set cosmo, the current is below:
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307) 

#==============================================================================
# Composite Model
#==============================================================================
bc03 = BC('data/ssp/bc03/chab/hr/')

Ages = np.linspace(0.5,13.5,26)* u.Gyr
 
# Aldo
M_today = np.array([9.0,9.5,10.0,10.5,11.0,11.5])
perturb = False
if perturb:
    A, P = 0.3, 0.5
    SFH_law = glh_absprtb
    SFH_fit = asc.read('Aldo/Aldo SFH prtb.txt')
    SFH_pars = zip(SFH_fit.c1, SFH_fit.mu, SFH_fit.sigma,\
               SFH_fit.h13, SFH_fit.h14,\
               SFH_fit.c2, SFH_fit.x0, SFH_fit.gama,\
               SFH_fit.h23, SFH_fit.h24,\
               A*np.ones_like(M_today), P*np.ones_like(M_today))        
else:
    SFH_law = gauss_lorentz_hermite
    SFH_fit = asc.read('Aldo/Aldo SFH params.txt')
    SFH_pars = zip(SFH_fit.c1, SFH_fit.mu, SFH_fit.sigma, SFH_fit.h13, SFH_fit.h14,\
               SFH_fit.c2, SFH_fit.x0, SFH_fit.gama, SFH_fit.h23, SFH_fit.h24)

Dust = np.array([0.0])   

models_A = CSP(bc03, age = Ages, sfh = SFH_pars, 
             dust = Dust, metal_ind = 1., f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti,
             neb_cont=False, neb_met=False)

# Ciesla
M_seed = [10**4.5,10**5.5,10**6.5,10**7.5,10**8.5]
SFH_law = RKPRG        

models_C = CSP(bc03, age = Ages, sfh = M_seed, 
             dust = Dust, metal_ind = 1., f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti)

# Exponential
Ages_2 = np.logspace(-1., 1.3, 20)* u.Gyr
SFH = np.array([1., 2., 3., 5., 10., 20.]) * u.Gyr

SFH_law = exponential
models_exp = CSP(bc03, age = Ages_2, sfh = SFH, 
             dust = Dust, metal_ind = 1., f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti)

# Delayed
SFH_law = delayed
models_del = CSP(bc03, age = Ages_2, sfh = SFH, 
             dust = Dust, metal_ind = 1., f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti)

#==============================================================================
# Mock Observation
#==============================================================================
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')

print eazy_library.filternames

filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
Names = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']

theta = 34.8*u.deg

#Aldo
synphot_A = Observe(models_A, filters, redshift=0.001)
mags_A = synphot_A.AB[0]
FUV, NUV, U, B, V, R, I, J, H, K = mags_A
FUV_V = FUV - V
NUV_V = NUV - V
U_V = U - V
V_J = V - J
V_K = V - K
Ssed_A = pd.DataFrame(np.squeeze(np.sin(theta)*U_V + np.cos(theta)*V_J))
Csed_A = pd.DataFrame(np.squeeze(np.cos(theta)*U_V - np.sin(theta)*V_J))
UV_A = pd.DataFrame(np.squeeze(U_V))

#Ciesla
synphot_C = Observe(models_C, filters, redshift=0.001)
mags_C = synphot_C.AB[0]
FUV, NUV, U, B, V, R, I, J, H, K = mags_C
FUV_V = FUV - V
NUV_V = NUV - V
U_V = U - V
V_J = V - J
V_K = V - K

Ssed_C = pd.DataFrame(np.squeeze(np.sin(theta)*U_V + np.cos(theta)*V_J))
Csed_C = pd.DataFrame(np.squeeze(np.cos(theta)*U_V - np.sin(theta)*V_J))
UV_C = pd.DataFrame(np.squeeze(U_V))

# Exponential
synphot_exp = Observe(models_exp, filters, redshift=0.001)
mags_exp = synphot_exp.AB[0]
FUV, NUV, U, B, V, R, I, J, H, K = mags_exp
FUV_V = FUV - V
NUV_V = NUV - V
U_V = U - V
V_J = V - J
V_K = V - K

Ssed_exp = pd.DataFrame(np.squeeze(np.sin(theta)*U_V + np.cos(theta)*V_J))
Csed_exp = pd.DataFrame(np.squeeze(np.cos(theta)*U_V - np.sin(theta)*V_J))
UV_exp = pd.DataFrame(np.squeeze(U_V))

# Delayed
synphot_del = Observe(models_del, filters, redshift=0.001)
mags_del = synphot_del.AB[0]
FUV, NUV, U, B, V, R, I, J, H, K = mags_del
FUV_V = FUV - V
NUV_V = NUV - V
U_V = U - V
V_J = V - J
V_K = V - K

Ssed_del = pd.DataFrame(np.squeeze(np.sin(theta)*U_V + np.cos(theta)*V_J))
Csed_del = pd.DataFrame(np.squeeze(np.cos(theta)*U_V - np.sin(theta)*V_J))
UV_del = pd.DataFrame(np.squeeze(U_V))

#==============================================================================
# Csed/UV vs Time
#==============================================================================
with sns.axes_style("ticks"):
    plt.figure(figsize=(7,7))
    plt.fill_between(pd.Series(Ages.value), np.max(UV_C,axis=1), np.min(UV_C,axis=1), 
                 label='Ciesla17', color='steelblue', alpha=0.2)
    plt.fill_between(pd.Series(Ages.value), np.max(UV_A,axis=1), np.min(UV_A,axis=1), 
                 label='RP17', color='g', alpha=0.2)
    plt.fill_between(pd.Series(Ages_2.value), np.max(UV_exp,axis=1), np.min(UV_exp,axis=1), 
                 label='Exponential', color='firebrick', alpha=0.2)
    plt.fill_between(pd.Series(Ages_2.value), np.max(UV_del,axis=1), np.min(UV_del,axis=1), 
                 label='Delayed',color='orange', alpha=0.2)
    plt.legend(loc=4,fontsize=12,frameon=True,facecolor='w')
    
    plt.plot(Ages,UV_C.iloc[:,1],c='steelblue',lw=3,alpha=0.5)
    plt.plot(Ages,UV_A.iloc[:,3],c='g',lw=3,alpha=0.3)
    plt.plot(Ages_2,UV_exp.iloc[:,2],c='firebrick',lw=3,alpha=0.5)
    plt.plot(Ages_2,UV_del.iloc[:,2],c='orange',lw=3,alpha=0.3)
    
    plt.xlabel('T (Gyr)',fontsize=15)
    plt.ylabel('U - V',fontsize=15)
    plt.axhline(0.5,color='k',ls='--')
    plt.axvline(3.,color='b',ls='--',lw=1,alpha=0.3)
    plt.axvline(7.,color='b',ls='--',lw=1,alpha=0.3)
    plt.xlim(-1.,11.)
    plt.ylim(-0.1,1.0)
    plt.scatter(tgs, uvs, s=60, linewidth=2,
                facecolor='orange', edgecolor='k',alpha=1.,zorder=2)
    plt.errorbar(tgs, uvs,
                 yerr = [uva,uvb],fmt='o',
                 c='grey',alpha=0.5,zorder=1)
plt.show()


#with sns.axes_style("ticks"):
#    plt.figure(figsize=(7,7))
#    plt.fill_between(pd.Series(Ages), np.max(Csed_C,axis=1), np.min(Csed_C,axis=1), 
#                     label='Ciesla', alpha=0.3)
#    plt.fill_between(pd.Series(Ages), np.max(Csed_A,axis=1), np.min(Csed_A,axis=1), 
#                     label='Aldo', alpha=0.3)
#    plt.fill_between(pd.Series(Ages_2), np.max(Csed_exp,axis=1), np.min(Csed_exp,axis=1), 
#                     label='Exponential', alpha=0.3)
#    plt.fill_between(pd.Series(Ages_2), np.max(Csed_del,axis=1), np.min(Csed_del,axis=1), 
#                     label='Delayed', color='orange', alpha=0.3)
#    plt.legend(loc=4,fontsize=12,frameon=True,facecolor='w')
#    plt.xlabel('T (Gyr)',fontsize=15)
#    plt.ylabel('C$_{SED}$',fontsize=15)
#    plt.axhline(0.25,color='k',ls='--')
#    plt.axvline(3.,color='b',ls='--',lw=1,alpha=0.3)
#    plt.axvline(7.,color='b',ls='--',lw=1,alpha=0.3)
#    plt.xlim(-1.,11.)
#    plt.ylim(-0.1,0.8)
#    plt.show()