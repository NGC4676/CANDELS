# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 05:24:37 2017

@author: Qing Liu
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.visualization import quantity_support

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import exponential, delayed
from smpy.dust import Calzetti

#==============================================================================
# Composite Model
#==============================================================================
bc03 = BC('data/ssp/bc03/chab/lr/')

ages = np.logspace(8, 10, 50) * u.yr
sfh = np.array([0.1, 1, 10]) * u.Gyr
dust = np.array([0., 1.0])

models = CSP(bc03, age = ages, sfh = sfh, 
             dust = dust, metal_ind = 1., f_esc = 1.,
             sfh_law = exponential, dust_model = Calzetti)
models *= 1e10

#starburst = CSP(bc03, age=50*u.Myr, sfh=100*u.Gyr, dust=0.5, metal_ind=1.)
#starburst *= (1*u.M_sun/u.yr)/starburst.SFR   
#models +=starburst

#==============================================================================
# Plot Spectrum
#==============================================================================
#with quantity_support():
#    Fix, Ax = plt.subplots(1, figsize=(8,6))
#    for age in models.tg[:20:5]:
#        i = np.where(models.tg.value==age.value)[0][0]
#        for tau in models.tau[1:2]:
#            j = np.where(models.tau.value==tau.value)[0][0]
#            Ax.semilogy(models.wave, models.SED[0,i,j,0,0], 
#                        label=r'Age={0:.2f}, $\tau$={1:.1f}'.format(age.to(u.Gyr),tau))
#    Ax.set_ylim([1e4,1e10])
#    Ax.set_xlim([900,7000])
#    Leg = Ax.legend(loc='best',fontsize=9,frameon=True,ncol=2)
#plt.show()

#==============================================================================
# Mock Observation
#==============================================================================
eazy_library = LoadEAZYFilters('data/FILTER.RES.v8')

eazy_library.search('J')

filters = FilterSet()
filters.addEAZYFilter(eazy_library, [139, 141, 143, 125])  # U, V, I, J
filters.addTophatFilter(centre=1500*u.AA, width=150*u.AA)  # FUV

synphot = Observe(models, filters, redshift=0.)

mags = np.squeeze(synphot.AB)
UV = mags[0]-mags[1]
VI = mags[1]-mags[2]
VJ = mags[1]-mags[3]

#==============================================================================
# UVJ Diagram
#==============================================================================
with quantity_support():
    for i, tau in enumerate(sfh):
        s = plt.scatter(VJ[:,i], UV[:,i], c=np.log10(ages/u.yr), 
                         s=60, cmap=plt.cm.RdYlBu_r)
        plt.plot(VJ[:,i], UV[:,i], label=r'$\tau = ${0}'.format(tau), alpha=0.7)

    col = plt.colorbar(s)
    col.set_label(r'$\log_{10}(\rm{Age}/\rm{yr})$')
    plt.text(0.25,2,'Quiescent',color='r',fontsize=20,alpha=0.5)
    plt.text(1.25,1,'SFG',color='b',fontsize=20,alpha=0.5)
    plt.legend(loc='lower right')
    plt.xlabel('V - J')
    plt.ylabel('U - V')
    plt.plot([0., 0.69, 1.5, 1.5], [1.3, 1.3, 2.01, 2.5], color='k', alpha=0.5)
    plt.xlim([0, 2.])
    plt.ylim([0, 2.5])
plt.show()

#==============================================================================
# UVI Diagram
#==============================================================================
#with quantity_support():
#    for i, tau in enumerate(sfh):
#        s = plt.scatter(VI[:,i], UV[:,i], c=np.log10(ages/u.yr), 
#                         s=60, cmap=plt.cm.RdYlBu_r)
#        plt.plot(VI[:,i], UV[:,i], label=r'$\tau = ${0}'.format(tau), alpha=0.7)
#
#    col = plt.colorbar(s)
#    col.set_label(r'$\log_{10}(\rm{Age}/\rm{yr})$')
#    plt.text(0.,2.,'Quiescent',color='r',fontsize=20,alpha=0.5)
#    plt.text(0.7,1.,'SFG',color='b',fontsize=20,alpha=0.5)
#    plt.legend(loc='lower right')
#    plt.xlabel('V - I')
#    plt.ylabel('U - V')
#    plt.plot([-0.2, 0.45, 0.85, 0.85], [1.3, 1.3, 1.9, 2.5], color='k', alpha=0.5)
#    plt.xlim([-0.2, 1.2])
#    plt.ylim([0, 2.5])
#plt.show()

#==============================================================================
# Old + Starburst
#==============================================================================
#old_pop = CSP(bc03, age=3*u.Gyr, sfh=0.1*u.Gyr, dust=0., metal_ind=0.2)
#old_pop *= 1e11
#
#starburst = CSP(bc03, age=50*u.Myr, sfh=100*u.Gyr, dust=0.5, metal_ind=1.)
#starburst *= (100*u.M_sun/u.yr)/starburst.SFR          
#
#combined = old_pop + starburst
#
#print combined.Ms
#print combined.SFR
#
#with quantity_support():
#    Fix, Ax = plt.subplots(1, figsize=(7,5))
#    Ax.semilogy(old_pop.wave, np.squeeze(old_pop.SED), '--', 
#                label='Old Population', color='firebrick')
#    Ax.semilogy(combined.wave, np.squeeze(starburst.SED), 
#                label='Starburst', color='steelblue'), '--',
#    Ax.semilogy(combined.wave, np.squeeze(combined.SED), 
#                label='Total', color='k')
#    Ax.set_ylim([1e5,1e9])
#    Ax.set_xlim([0,2e4])
#    Leg = Ax.legend(loc='upper right',ncol=2)
    
#==============================================================================
# Exponential vs Delay
#==============================================================================
#from scipy.integrate import quad
#
#tau = 1*u.Gyr
#Age = [0.1,0.5,1,5,10]*u.Gyr
#dely = {}
#
#for i, T in enumerate(Age):
#    key = str(T.value)
#    dely[key] = CSP(bc03, age=T, sfh=tau, sfh_law = delayed, dust=0., metal_ind=1.)
#    M_s = quad(delayed,0,T.value,args=(tau.value,))[0]
#    dely[key] *= M_s
#
#    print dely[key].SFR
#    
#with quantity_support():
#    Fix, Ax = plt.subplots(1, figsize=(7,5))
#    for T in Age:
#        key = str(T.value)
#        Ax.semilogy(dely[key].wave, np.squeeze(dely[key].SED), 
#                    label='T = {0}'.format(T))
#    Ax.set_ylim([1e-7,1e-2])
#    Ax.set_xlim([0,1e4])
#    Leg = Ax.legend(loc='best',ncol=2)
#    plt.title(r'$\tau = ${0}'.format(tau),fontsize=15)
#
#UV = np.array([])
#VJ = np.array([])
#for i, T in enumerate(Age): 
#    key = str(T.value)
#    synphot = Observe(dely[key], filters, redshift=0.)
#    mags = np.squeeze(synphot.AB)
#    UV = np.append(UV, mags[0]-mags[1])
#    VJ = np.append(VJ, mags[1]-mags[3])
#
#plt.figure(figsize=(7,5))
#with quantity_support():
#    s = plt.scatter(VJ.value, UV.value, c=np.log10(Age/u.yr), 
#                    s=60, cmap=plt.cm.RdYlBu_r)
#    plt.plot(VJ.value, UV.value, label=r'$\tau = ${0}'.format(tau), alpha=0.7)
#
#    col = plt.colorbar(s)
#    col.set_label(r'$\log_{10}(\rm{Age}/\rm{yr})$')
#    plt.text(0.25,2,'Quiescent',color='r',fontsize=20,alpha=0.5)
#    plt.text(1.25,1,'SFG',color='b',fontsize=20,alpha=0.5)
#    plt.legend(loc='lower right')
#    plt.xlabel('V - J')
#    plt.ylabel('U - V')
#    plt.plot([0., 0.69, 1.5, 1.5], [1.3, 1.3, 2.01, 2.5], color='k', alpha=0.5)
#    plt.xlim([0, 2.])
#    plt.ylim([0, 2.5])
#plt.show()