# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:48:20 2017

@author: Qing Liu
"""

import re
import glob
import numpy as np
import asciitable as asc
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

SFH_law = exponential

#Ages = [0.003,0.01,0.018,0.032,0.056,0.1,0.18,0.32,0.56,1,1.78,3.16,5.62,10,12,14,20]* u.Gyr
#SFH = [0.01,0.032,0.1,0.32,1,2,3.16,4,8,10,16,50]* u.Gyr

Ages = 10**(np.linspace(7.0,10.0,13)-9)*u.Gyr
SFH = 10**(np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])-9)*u.Gyr
Meta = [1.0]   
Dust = [0.]           

models = CSP(bc03, age = Ages, sfh = SFH, 
             dust = Dust, metal_ind = Meta, f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti)
models *= 1e10

use_bc03 = False
if use_bc03:
    lgAges = np.linspace(7.0,10.0,13)
    lgSFH = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
    lgMs = np.array([])
    lgSSFR = np.array([])

    dir = glob.glob('grid_exp/*.4color') 
    values = [(f, re.findall(r'-?\d+\.?\d*e?-?\d*?',f)[0]) for f in dir]
    dtype = [('name', 'S40'), ('tau', float)]
    a = np.array(values, dtype=dtype) 
    Output = np.sort(a, order='tau')  

    for f,tau in Output:
        table = asc.read(f,names=['log-age','Mbol','Bmag','Vmag','Kmag',      
                                  'M*_liv','M_remnants','M_ret_gas',
                                  'M_galaxy','SFR','M*_tot',
                                  'M*_tot/Lb','M*_tot/Lv','M*_tot/Lk',
                                  'M*_liv/Lb','M*_liv/Lv','M*_liv/Lk'])
        lgage = table['log-age']
        lgsfr = np.log10(table['SFR'])
        lgms= np.log10(table['M*_tot'])
        lgsfr_interp = np.interp(lgAges, lgage, lgsfr)
        lgms_interp = np.interp(lgAges, lgage, lgms)
        print 10**lgms_interp
        lgMs = np.append(lgMs, lgms_interp)
        lgSSFR = np.append(lgSSFR, lgsfr_interp-lgms_interp)
    lgsSFR = lgSSFR.reshape((13,7))[None,:,:,None,None]
else:
    lgsSFR = np.log10(models.SFR.value)-9

#==============================================================================
# Mock Observation
#==============================================================================
eazy_library = LoadEAZYFilters('C:\Users\mac\Documents\Python Scripts\SPS\FILTER.RES.CANDELS')

print eazy_library.filternames

filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
    # FUV, NUV, U, B, V, R, I, J, H, K

synphot = Observe(models, filters, redshift=0.001)

mags = synphot.AB[0]
U_V = mags[2]-mags[4]
V_J = mags[4]-mags[7]
FUV_R = mags[0]-mags[5]
R_K = mags[5]-mags[9]

#==============================================================================
# UVJ Diagram
#==============================================================================
plt.figure(figsize=(8,8))
X, Y = V_J, U_V
with quantity_support():
    for i, meta in enumerate(Meta):
        for k, Av in enumerate(Dust):
            for j, tau in enumerate(SFH):
                plt.plot(X[i,:,j,k,0], Y[i,:,j,k,0],
                         label=r'$\tau = ${0:.2f}'.format(tau), alpha=0.7)
                s = plt.scatter(X[i,:,j,k,0], Y[i,:,j,k,0], 
                                c=np.log10(Ages/u.yr), 
                                s=50, cmap=plt.cm.jet)
    cb = plt.colorbar(s)
    cb.set_label(r'$\log_{10}(\rm{Age}/\rm{yr})$')
    plt.plot([-1., 1.0, 1.6, 1.6], [1.3, 1.3, 2.01, 2.5], color='k', alpha=0.7)
    plt.text(0.15,1.8,'Quiescent',color='r',fontsize=20,alpha=0.5)
    plt.text(0.8,0.6,'SFG',color='b',fontsize=20,alpha=0.5)
    plt.legend(loc='best')
    plt.xlim([-0.75, 2.0])
    plt.ylim([-0.25, 2.25])
    plt.xlabel('V - J')
    plt.ylabel('U - V')
plt.show()

#==============================================================================
# Rotated UVJ : CSED vs SSFR
#==============================================================================
theta = 34.8*u.deg
S_SED = (np.sin(theta)*U_V + np.cos(theta)*V_J).value
C_SED = (np.cos(theta)*U_V - np.sin(theta)*V_J).value
        
Colors = plt.cm.rainbow(np.linspace(0.2, 0.8, len(Meta)))
plt.figure(figsize=(8,7))
for i, (meta,c) in enumerate(zip(Meta, Colors)):
    for j, tau in enumerate(SFH.value):
        for k, Av in enumerate(Dust):
            plt.plot(C_SED[i,:,j,k,0], lgsSFR[i,:,j,k,0],
                     label=r'$\tau$=%.2f Gyr'%tau,
                     ls='--',lw=1,alpha=0.7)
            s = plt.scatter(C_SED[i,:,j,k,0], lgsSFR[i,:,j,k,0],
                            c=np.log10(Ages/u.Gyr),
                            cmap='jet', s=40, alpha=0.9)
C_SED0 = np.linspace(-1, 2, 100)
plt.plot(C_SED0,-1.95*C_SED0**2-0.82*C_SED0-8.32,'k')
col = plt.colorbar(s)
col.set_label(r'$\log_{10}(\rm{Age}/yr)$')
plt.xlabel('$C_{SED} = 0.82(U-V)-0.57(V-J)$',fontsize=14)
plt.ylabel('log $sSFR/ yr^{-1}$',fontsize=14)
plt.title('log SSFR = $-2.28 C_{SED}^{2}-0.8 C_{SED}-8.41$',
        fontsize=14,color='k')
plt.legend(loc='best')
plt.xlim(-0.2,1.2)
plt.ylim(-11.5,-8)

#==============================================================================
# Predict SSFR
#==============================================================================
lgsSFR_pred=-2.28*C_SED**2-0.8*C_SED-8.41
plt.figure(figsize=(7,7))
plt.scatter(lgsSFR,lgsSFR_pred,s=30,alpha=0.5)
xx=np.linspace(-15,-5,100)
plt.plot(xx,xx,'k-',lw=2)
plt.xlim(-15.,-5)
plt.ylim(-15.,-5)
