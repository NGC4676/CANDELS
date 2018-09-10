#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 18:46:09 2018

@author: Q.Liu
"""

import numpy as np
import asciitable as asc
from astropy.io import ascii
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#==============================================================================
# Read model result 
#==============================================================================
lgAge = np.log10(np.linspace(0.1, 8., 80)* 1e9)
weights = np.array([0.1,0.2,0.3,0.4,0.5])

lgMs = np.ones(400)*10
lgSSFR = (np.log10(SFR)).values.ravel()
                 
lgAges = np.array([lgAge for m in weights]).T.ravel()
w_class =  np.array([weights for T in range(lgAge.size)]).ravel()

#==============================================================================
# Read FAST SED-fitting result
#==============================================================================
table = ascii.read("N/Composite/comp_exp.fout",header_start=16).to_pandas()
Ages_FAST = table.lage
SFH_FAST = table.ltau
M_FAST = table.lmass
SSFR_FAST = table.lssfr
Av_FAST = table.Av
age_cond = (lgAges>np.log10(0))&(lgAges<np.log10(9e9))

#==============================================================================
# One-One
#==============================================================================
fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = ['(Age/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges-4.9e8),lgMs,lgSSFR],
                                                 [Ages_FAST,M_FAST,SSFR_FAST])):
    color = 10**lgAges/1e9
    color = w_class
    ax = plt.subplot(2,2,i+1) 
    if (i == 0) | (i == 2):
        plt.plot(xx,xx,'k--',alpha=0.5)
        s = plt.scatter(model[age_cond], FAST[age_cond], c=color[age_cond],
                        cmap='jet', s=20, alpha=0.7,zorder=2)
        plt.scatter(model[~age_cond], FAST[~age_cond],
                    c='gray', s=20, alpha=0.5,zorder=1)
        plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize='large')
        plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize='large')
        y = pd.Series(FAST-model)[age_cond]
        y = y[abs(y<1.)] 
        if i==2:
            plt.xlim(-10.1,-7.8)
            plt.ylim(-10.1,-7.8)
        else:
            plt.xlim(8.1,10.1)
            plt.ylim(8.1,10.1)
    elif i==1:
        plt.hist(M_FAST,alpha=0.7,zorder=1)
        plt.axvline(10.0,color='k',ls='--')
        plt.xlabel(r'$\rm M_{*,FAST}$',fontsize='large')
        plt.xlim(9.5,10.5)

fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
colorbar.set_label('weight of second burst')
ax = plt.subplot(2,2,4)
for i,w in enumerate(weights):
    plt.hist(Av_FAST[i::5],alpha=0.5,zorder=1,label="w=%g"%w)
#plt.hist(Av_FAST,alpha=0.7,zorder=1)
plt.xlim(0.2,1.8)
plt.axvline(1.0,color='k',ls='--')
plt.xlabel(r'$\rm Av_{FAST}$',fontsize='large')
plt.text(0.75,0.78,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_cond].std(),
            fontsize=12,transform=ax.transAxes,color="orange")
plt.legend(loc=2, fontsize=11, frameon=True, facecolor='w')
#plt.suptitle('Double Burst Exp (Tau=0.5Gyr) vs FAST Fitting (Tau)',fontsize=18,y=0.95)
#plt.savefig("N/Composite/FAST-exp05_.png",dpi=400)
plt.show()