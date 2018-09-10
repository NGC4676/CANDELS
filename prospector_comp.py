#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:19:47 2018

@author: Q.Liu
"""

import numpy as np
import pandas as pd
import asciitable as asc
from astropy.io import ascii
import matplotlib.pyplot as plt
import seaborn as sns

#==============================================================================
# Read model result 
#==============================================================================
lgAge = np.log10(np.linspace(0.5,11.5,111)* 1e9)
M_today = np.array([9.5,10.0,10.5,10.75,11.0])
M_seed = np.linspace(5.,7.,5)

M_color = ['m','b','g','orange','firebrick']
#M_color = ['navy','steelblue','yellowgreen','gold','orangered']

BC03_out = asc.read('N/Aldo/CSP result.txt')
#BC03_out = asc.read('N/Ciesla/CSP result.txt')

lgMs = BC03_out['lgM*'].reshape((M_today.size, lgAge.size)).T.ravel()
lgSSFR = BC03_out['lgsSFR'].reshape((M_today.size, lgAge.size)).T.ravel()
                 
lgAges = np.array([lgAge for m in M_today]).T.ravel()
M_class =  np.array([M_today for T in range(lgAge.size)]).ravel()

#==============================================================================
# Read FAST SED-fitting result
#==============================================================================
table = ascii.read("N/Aldo/Aldo_exp.fout",header_start=16).to_pandas()

Ages_FAST = table.lage
SFH_FAST = table.ltau
M_FAST = table.lmass
SSFR_FAST = table.lssfr
Av_FAST = table.Av

age_cond = (lgAges>np.log10(2.e9))&(lgAges<np.log10(9.e9))

# =============================================================================
# Prospector 
# =============================================================================
table_prosp = ascii.read("N/prosp/res.txt").to_pandas()

M_prosp = np.log10(table_prosp["mass_best(M_sun)"])
M16_prosp = np.log10(table_prosp["mass_16th"])
M84_prosp = np.log10(table_prosp["mass_84th"])
Av_prosp = table_prosp["dust2_best(mag)"]
Ages_prosp = table_prosp['tage_best(Gyr)']



xx = np.linspace(3,12)
plt.figure(figsize=(7,7))
plt.scatter(lgMs[~age_cond],(M_FAST-M_prosp)[~age_cond],c="gray",s=40,alpha=0.7)
for k,(m,c) in enumerate(zip(M_today,M_color)):
    cond = age_cond&(M_class==m)
    s=plt.scatter(lgMs[cond],(M_FAST-M_prosp)[cond],
                       c=c,s=40,alpha=0.7,label=r'log M$\rm_{today}$ = %s'%m)
    plt.errorbar(lgMs[cond],(M_FAST-M_prosp)[cond],
                  yerr=[(M16_prosp-M_prosp)[cond],
                        (M84_prosp-M_prosp)[cond]], 
                        c=c,fmt='o',alpha=0.2)
plt.plot(xx,np.zeros_like(xx),"k--")
plt.xlim(7.5,11.2)
plt.ylim(-.8,.8)
plt.xlabel(r'log $(\rm{M_*}/\rm{M_{\odot}})_{model}$',fontsize=15)
plt.ylabel(r'log $(\rm{M_*}/\rm{M_{\odot}})_{FAST-prospector}$',fontsize=15)
plt.legend(loc="best",fontsize=10)
plt.savefig("N/Prospector_comp.png",dpi=300)


