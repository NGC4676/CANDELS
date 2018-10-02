# -*- coding: utf-8 -*-
"""
Created on Sun Jul 09 04:11:03 2017

@author: Qing Liu
"""
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.visualization import quantity_support
from scipy.integrate import quad

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import exponential, delayed
from smpy.dust import Calzetti

#==============================================================================
# Composite Model
#==============================================================================
bc03 = BC('data/ssp/bc03/chab/lr/')

SFH_law = exponential        
#==============================================================================
# Test 
#==============================================================================
Ages_e = np.logspace(-2., 1.11, 40)* u.Gyr
SFH = np.array([0.5, 1., 3., 20.]) * u.Gyr


Dust = np.array([0.])   

models = CSP(bc03, age = Ages_e, sfh = SFH, 
             dust = Dust, metal_ind = 1.0, f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti)
SED = np.squeeze(models.SED.value)
#==============================================================================
# Renormalization
#==============================================================================
#use_bc = False
#
#if use_bc:
#    M_s = 10**lgMs.reshape((SFH.size, Ages_e.size)).T
#    SSFR = 10**lgSSFR.reshape((SFH.size, Ages_e.size)).T
#else:
#    M_s = np.array([quad(SFH_law, 0, T.value, args=(tau.value,))[0] \
#                    for T in Ages_e for tau in SFH]).reshape(Ages_e.size,SFH.size)
#
#    SSFR = np.squeeze(models.SFR.value)

#==============================================================================
# Mock Observation
#==============================================================================
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')

#print eazy_library.filternames

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
# UVJ
#==============================================================================
#X_e0, Y_e0 = V_J, U_V

X_e, Y_e = V_J, U_V

plt.figure(figsize=(6,6))
with quantity_support():
    for k, Av in enumerate(Dust):
        for j, tau in enumerate(SFH):
            plt.plot(X_e[0,:,j,k,0], Y_e[0,:,j,k,0], label=r'$\tau = ${0:.1f}'.format(tau), 
                         lw=2, alpha=0.7)
    plt.plot([-1., 1.0, 1.6, 1.6], [1.3, 1.3, 2.01, 2.5], color='k', alpha=0.7)
    plt.legend(loc='best',fontsize=9,frameon=True,facecolor='w')
    plt.xlabel('V - J')
    plt.ylabel('U - V')
    plt.xlim([-0.25, 1.75])
    plt.ylim([-0.25, 2.25])
plt.tight_layout()

plt.show()

#==============================================================================
# Exp shaded region
#==============================================================================
plt.figure(figsize=(6,6))
with sns.axes_style("ticks"):
    plt.fill(np.concatenate((X_e[0,:,-1,0,0].value,
                             X_e[0,-1,:,0,0].value[::-1],
                             X_e[0,:,0,0,0].value[::-1])),
             np.concatenate((Y_e[0,:,-1,0,0].value,
                             Y_e[0,-1,:,0,0].value[::-1],
                             Y_e[0,:,0,0,0].value[::-1])),
             'grey',alpha=0.3)
#    for k, Av in enumerate(Dust):
#        for j,m in enumerate(M_today):
#            plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label='log M$_{today}$ = %.1f'%m,
#                     lw=3, alpha=1.)
#    plt.legend(loc='best',fontsize=12,frameon=True,facecolor='w')
#    zg = 1.0
#    plt.plot([-5,(0.55+0.253*zg-0.0533*zg**2)/0.88,1.6,1.6],
#          [1.3,1.3,2.158-0.253*zg+0.0533*zg**2,2.5], color='k', alpha=0.7)
#    plt.xlabel('V - J')
#    plt.ylabel('U - V')
#    plt.xlim([-0.25, 1.75])
#    plt.ylim([0., 2.])
plt.show()

#==============================================================================
# UV/Csed
#==============================================================================
theta = 34.8*u.deg
Ssed = pd.DataFrame(np.squeeze(np.sin(theta)*U_V + np.cos(theta)*V_J))
Csed = pd.DataFrame(np.squeeze(np.cos(theta)*U_V - np.sin(theta)*V_J))
UV = pd.DataFrame(np.squeeze(U_V))
VJ = pd.DataFrame(np.squeeze(V_J))
SFR = pd.DataFrame(np.squeeze(models.SFR))

# UV
colors=["firebrick","gold","m","mediumblue"]
plt.figure(figsize=(6,6))
for i, (tau,c) in enumerate(zip(SFH,colors)):
    plt.plot(Ages_e,UV_e[i],"-",c=c,lw=2,label=r'$\tau = ${0:g}'.format(tau),zorder=3,alpha=0.7)
    plt.plot(Ages_e,UV_d[i],":",c=c,lw=2,zorder=3,label="_no_legend_",alpha=0.7)
plt.legend(loc=8,fontsize=11,frameon=True,facecolor='w')
plt.xlabel('Model Age (Gyr)',fontsize=15)
plt.ylabel('U - V',fontsize=15)
#plt.ylabel('C$_{SED}$',fontsize=15)
plt.axhline(0.5,color='c',ls='-',lw=2,zorder=2,alpha=0.6)
plt.axhline(0.4,color='c',ls=':',lw=1,zorder=2,alpha=0.6)
plt.axhline(0.6,color='c',ls=':',lw=1,zorder=2,alpha=0.6)
plt.axvline(2.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
plt.axvline(8.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
plt.plot([8.8,9.4],[0.2,0.2],"k-",lw=2)
plt.text(9.6,0.2,"Exponential",va='center',fontsize=14)
plt.plot([8.8,9.4],[0.12,0.12],"k:",lw=2)
plt.text(9.6,0.12,"Delayed",va='center',fontsize=14)
plt.xlim(0.,12.)
plt.ylim(0.,1.2)
plt.tight_layout()
#plt.savefig("N/UV_exp_del.pdf")


# UV and Csed

plt.figure(figsize=(12,6))
ax=plt.subplot(121)
for i, tau in enumerate(SFH):
    plt.plot(Ages_e,UV[i],label=r'$\tau = ${0:g}'.format(tau),zorder=3)
plt.legend(loc='best',fontsize=11,frameon=True,facecolor='w')
plt.xlabel('T (Gyr)',fontsize=15)
plt.ylabel('U - V',fontsize=15)
#plt.ylabel('C$_{SED}$',fontsize=15)
plt.axhline(0.5,color='c',ls='-',lw=2,zorder=2,alpha=0.6)
plt.axhline(0.4,color='c',ls=':',lw=1,zorder=2,alpha=0.6)
plt.axhline(0.6,color='c',ls=':',lw=1,zorder=2,alpha=0.6)
plt.axvline(2.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
plt.axvline(8.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
plt.xlim(0.,11.5)
plt.ylim(0.,1.2)

ax=plt.subplot(122)
for i, tau in enumerate(SFH):
    plt.plot(Ages_e,Csed[i],label=r'$\tau = ${0:g}'.format(tau))
plt.legend(loc=8,fontsize=8,frameon=True,facecolor='w',ncol=2)
plt.xlabel('Model Age (Gyr)',fontsize=15)
#plt.ylabel('U - V',fontsize=15)
plt.ylabel('C$_{SED}$',fontsize=15)
plt.axhline(0.25,color='c',ls=':',lw=3)
plt.axvline(2.5,color='k',ls='--',lw=2,alpha=0.5)
plt.axvline(8.5,color='k',ls='--',lw=2,alpha=0.5)
plt.xlim(0.,11.5)
plt.ylim(0.,0.5)
plt.tight_layout()
#plt.savefig("N/Exp_UV_Csed.png",dpi=400)
plt.show()

# =============================================================================
# SSFR
# =============================================================================

plt.figure(figsize=(12,6))
ax=plt.subplot(121)
for i, tau in enumerate(SFH):
    plt.plot(UV[i],np.log10(SFR[i]),label=r'$\tau = ${0:.1f}'.format(tau))
plt.legend(loc='best',fontsize=11,frameon=True,facecolor='w')
plt.xlabel('U - V',fontsize=15)
plt.ylabel(r'log(sSFR/yr$^{-1}$)',fontsize=15)
plt.ylim(-11.2,-7.3)
plt.xlim(-0.25,1.74)
ax=plt.subplot(122)
for i, tau in enumerate(SFH):
    plt.plot(VJ[i],np.log10(SFR[i]),label=r'$\tau = ${0:.1f}'.format(tau))
plt.xlabel('V - J',fontsize=15)
plt.ylabel(r'log(sSFR/yr$^{-1}$)',fontsize=15)
plt.ylim(-11.2,-7.3)
plt.xlim(-0.12,1.15)
plt.legend(loc='best',fontsize=11,frameon=True,facecolor='w')
plt.tight_layout()
#plt.savefig("New/Exp_UVJ_SSFR.png",dpi=400)
plt.show()
