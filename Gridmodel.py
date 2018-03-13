# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 00:58:25 2017

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
from scipy.integrate import quad


#==============================================================================
# Composite Grid
#==============================================================================
bc03 = BC('data/ssp/bc03/chab/hr/')

SFH = [0.1, 1, 10]*u.Gyr
#Ages = [0.1, 1, 10]*u.Gyr
Ages = np.logspace(-1,1,20)*u.Gyr
models = {}

for i, tau in enumerate(SFH):
    for j, T in enumerate(Ages):
        key = '%s-%s'%(str(i),str(j))
        models[key] = CSP(bc03, age=T, sfh=tau, sfh_law = delayed, dust=0., metal_ind=1.)
        M_s = quad(delayed, 0, T.value, args=(tau.value,))[0]
        models[key] *= M_s
        models[key] *= 1e10

        print models[key].SFR
    
with quantity_support():
    Fix, axes = plt.subplots(SFH.size, 1, figsize=(8,8))
    for i, tau in enumerate(SFH):
        ax = axes[i]
        for j, T in enumerate(Ages):
            key = '%s-%s'%(str(i),str(j))
            ax.semilogy(models[key].wave, np.squeeze(models[key].SED), 
                                label='T = {0:.1f}'.format(T))
        ax.set_ylim([1e3,1e9])
        ax.set_xlim([0,1e4])
        ax.legend(loc='best',ncol=2)
        ax.set_title(r'$\tau = ${0:.1f}'.format(tau),fontsize=15)
    plt.tight_layout()
    
    
eazy_library = LoadEAZYFilters('data/FILTER.RES.v8')
eazy_library.search('J')

filters = FilterSet()
filters.addEAZYFilter(eazy_library, [139, 141, 125])  # U, V, J
filters.addTophatFilter(centre=1500*u.AA, width=150*u.AA)  # FUV

UV = np.empty((SFH.size,Ages.size))
VJ = np.empty((SFH.size,Ages.size))
                       
for i, tau in enumerate(SFH):
    for j, T in enumerate(Ages):
        key = '%s-%s'%(str(i),str(j))

        synphot = Observe(models[key], filters, redshift=0.)

        mags = np.squeeze(synphot.AB)
        UV[i,j] = (mags[0]-mags[1]).value
        VJ[i,j] = (mags[1]-mags[2]).value

with quantity_support():
    for i, tau in enumerate(SFH):
        s = plt.scatter(VJ[i], UV[i], c=np.log10(Ages/u.yr), 
                         s=60, cmap=plt.cm.RdYlBu_r)
        plt.plot(VJ[i], UV[i], label=r'$\tau = ${0}'.format(tau), alpha=0.7)

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