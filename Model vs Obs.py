# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 22:58:10 2017

@author: Qing Liu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from astropy.visualization import quantity_support

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import exponential, delayed, truncated, delayed_dstb
from smpy.dust import Calzetti

#==============================================================================
# Load Catalog
#==============================================================================
def calzetti(wave, Av):
    k = np.zeros_like(wave.value)

    w0 = [wave <= 1200 * u.AA]
    w1 = [wave < 6300 * u.AA]
    w2 = [wave >= 6300 * u.AA]
    w_u = wave.to(u.um).value

    x1 = np.argmin(np.abs(wave - 1200 * u.AA))
    x2 = np.argmin(np.abs(wave - 1250 * u.AA))

    k[w2] = 2.659 * (-1.857 + 1.040 / w_u[w2])
    k[w1] = 2.659 * (-2.156 + (1.509 / w_u[w1]) - (0.198 / w_u[w1] ** 2) + (0.011 / w_u[w1] ** 3))
    k[w0] = k[x1] + ((wave[w0] - 1200. * u.AA) * (k[x1] - k[x2]) / (wave[x1] - wave[x2]))

    k += 4.05
    k[k < 0.] = 0.

    EBV = Av / 4.05
    A_lambda = k*EBV
    return A_lambda


table_gds = Table.read('gds_all.hdf5', path='data')
table_uds = Table.read('uds_all.hdf5', path='data')
data1 = table_gds.to_pandas()
data2 = table_uds.to_pandas()
data = pd.concat([data1,data2], join='inner')

z = data.z_best
Ms = np.log10(data.M_med)
SSFR = data.ssfr_uv_corr

FUV = data.rest1600
NUV = data.rest2800
U = data.restUXbessel
V = data.restVbessel
J = data.restJpalomar
K = data.restKpalomar

Av = data.med_av
Afuv = calzetti([1597.5]*u.AA, Av)
Anuv = calzetti([2792.5]*u.AA, Av)
Au = calzetti([3593]*u.AA, Av)
Aj = calzetti([12509.6]*u.AA, Av)
Ak = calzetti([22244]*u.AA, Av)

FUV_c = FUV - Afuv
NUV_c = NUV - Anuv
U_c = U - Au
V_c = V - Av
J_c = J - Aj
K_c = K - Ak

FUVV = FUV-V
NUVV = NUV-V
UV = U-V
VJ = V-J
VK = V-K

FUVV = FUV_c-V_c
NUVV = NUV_c-V_c
UV = U_c-V_c
VJ = V_c-J_c
VK = V_c-K_c

theta = 34.8*u.deg
S_SED = pd.Series(np.sin(theta)*UV + np.cos(theta)*VJ)
C_SED = pd.Series(np.cos(theta)*UV - np.sin(theta)*VJ)

filter = (abs(Ms-10)<1.0) & (z>0.2) & (z<2.5) \
         & (data.f_f160w==0) & (data.mag_f160w_4<24.5) \
         & (data.CLASS_STAR<0.9) & (data.PhotFlag==0)
SF =  filter & (data.sf_flag == 1) 
Q = filter | (data.sf_flag == -1) 

#==============================================================================
# Composite Model
#==============================================================================
bc03 = BC('data/ssp/bc03/chab/hr/')

SFH_law = exponential
               
#==============================================================================
# Test 
#==============================================================================
Ages = np.logspace(-1.5, 1.0, 10)* u.Gyr
SFH = np.logspace(-1., 1.5, 10)* u.Gyr

#==============================================================================
# FAST
#==============================================================================
#Ages = 10**(np.linspace(7.0,10.0,13)-9)*u.Gyr
#SFH = 10**(np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])-9)*u.Gyr

Meta = np.array([1])  
Dust = np.array([0.])   

models = CSP(bc03, age = Ages, sfh = SFH, 
             dust = Dust, metal_ind = 1.0, f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti)

#==============================================================================
# Mock Observation
#==============================================================================
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')

print eazy_library.filternames

filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
    # FUV, NUV, U, B, V, R, I, J, H, K

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
# 2 in 1
#==============================================================================
#X, Y, Z, W = V_J, U_V, VJ, UV
X, Y, Z, W = V_K, FUV_V, VK, FUVV
xlab = 'V - K'
ylab = 'FUV - V'

plt.figure(figsize=(15,7))
plt.subplot(121)
for k, Av in enumerate(Dust):
        for j, tau in enumerate(SFH):
            plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label=r'$\tau = ${0:.2f}'.format(tau), 
                     lw=2, alpha=0.7)
            s = plt.scatter(X[0,:,j,k,0], Y[0,:,j,k,0], c=np.log10(Ages/u.yr), 
                            s=30, cmap=plt.cm.jet)
col = plt.colorbar(s)
col.set_label(r'$\log_{10}(\rm{Age}/\rm{yr})$')
plt.legend(loc='best')
plt.xlabel(xlab)
plt.ylabel(ylab)

plt.subplot(122)
s = plt.scatter(Z[(filter)],W[(filter)],c=z[(filter)],cmap='rainbow',s=3,alpha=0.5)
for k, Av in enumerate(Dust):
        for j, tau in enumerate(SFH):
            plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label=r'$\tau = ${0:.2f}'.format(tau), 
                     lw=2, alpha=0.7)
for i, Age in enumerate(Ages):
    plt.plot(X[0,i,:,k,0], Y[0,i,:,k,0], label='$T = ${0:.2f}'.format(Age), 
             lw=2, ls='--',alpha=0.9)
col = plt.colorbar(s)
col.set_label('Redshift')
plt.legend(loc=4,ncol=2,fontsize=9,frameon=True,facecolor='w')
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.xlim([-1.5, 2.5])
plt.ylim([-0.3, 2.0])

plt.tight_layout()
plt.show()
#==============================================================================
#  4 in 1
#==============================================================================
X, Y, Z, W = V_K, NUV_V, VK, NUVV
xlab = 'V - K'
ylab = 'NUV - V'

plt.figure(figsize=(20,16))

plt.subplot(221)
for k, Av in enumerate(Dust):
        for j, tau in enumerate(SFH):
            plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label=r'$\tau = ${0:.2f}'.format(tau), 
                     lw=2, alpha=0.7)
            s = plt.scatter(X[0,:,j,k,0], Y[0,:,j,k,0], c=np.log10(Ages/u.yr), 
                            s=30, cmap=plt.cm.jet)
col = plt.colorbar(s)
col.set_label(r'$\log_{10}(\rm{Age}/\rm{yr})$')
plt.legend(loc='best')
plt.xlabel(xlab)
plt.ylabel(ylab)

plt.subplot(222)
s = plt.scatter(Z[(filter)],W[(filter)],c=z[(filter)],cmap='rainbow',s=3,alpha=0.5)
for k, Av in enumerate(Dust):
        for j, tau in enumerate(SFH):
            plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label=r'$\tau = ${0:.2f}'.format(tau), 
                     lw=2, alpha=0.7)
for i, Age in enumerate(Ages):
    plt.plot(X[0,i,:,k,0], Y[0,i,:,k,0], label='$T = ${0:.2f}'.format(Age), 
             lw=2, ls='--',alpha=0.9)
col = plt.colorbar(s)
col.set_label('Redshift')
plt.legend(loc=4,ncol=2,fontsize=9,frameon=True,facecolor='w')
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.xlim([-1.5, 2.5])
plt.ylim([-0.3, 2.0])

plt.subplot(223)
s = plt.scatter(Z[(filter)],W[(filter)],c=z[(filter)],cmap='rainbow',s=3,alpha=0.5)
for i, Age in enumerate(Ages):
    plt.plot(X[0,i,:,k,0], Y[0,i,:,k,0], label='$T = ${0:.2f}'.format(Age), 
             lw=3, ls='-',alpha=0.9)
col = plt.colorbar(s)
col.set_label('Redshift')
plt.legend(loc=4,ncol=1,fontsize=12,frameon=True,facecolor='w')
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.xlim([-1.5, 2.5])
plt.ylim([-0.3, 2.0])

plt.subplot(224)
s = plt.scatter(Z[(filter)],W[(filter)],c=z[(filter)],cmap='rainbow',s=3,alpha=0.5)
for k, Av in enumerate(Dust):
        for j, tau in enumerate(SFH):
            plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label=r'$\tau = ${0:.2f}'.format(tau), 
                     lw=3, alpha=0.7)
col = plt.colorbar(s)
col.set_label('Redshift')
plt.legend(loc=4,ncol=1,fontsize=12,frameon=True,facecolor='w')
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.xlim([-1.5, 2.5])
plt.ylim([-0.3, 2.0])

plt.tight_layout()
plt.show()