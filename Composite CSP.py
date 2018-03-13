# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 01:28:59 2017

@author: Qing Liu
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import astropy.units as u

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import exponential
from smpy.dust import Calzetti

bc03 = BC('data/ssp/bc03/chab/lr/')

Ages = np.logspace(-2., 1.3, 30)* u.Gyr
SFH = np.array([3.0])*u.Gyr
Dust = np.array([0.])
SFH_law = exponential

#==============================================================================
# Construct SSP
#==============================================================================
#Ages1 = np.min([Ages+0.5*u.Gyr,np.ones_like(Ages)*1],axis=0)*u.Gyr-0.5*u.Gyr
#Ages2 = np.max([Ages-1*u.Gyr,np.zeros_like(Ages)],axis=0)*u.Gyr
              
SSP_A = CSP(bc03, age = 0.1*u.Gyr, sfh = SFH, 
           dust = Dust, metal_ind = 1.0, f_esc = 1.,
           sfh_law = SFH_law, dust_model = Calzetti)

SSP_B = CSP(bc03, age = 0.4*u.Gyr, sfh = SFH, 
           dust = Dust, metal_ind = 1.0, f_esc = 1.,
           sfh_law = SFH_law, dust_model = Calzetti)

SSP_C = CSP(bc03, age = 4.0*u.Gyr, sfh = SFH, 
           dust = Dust, metal_ind = 1.0, f_esc = 1.,
           sfh_law = SFH_law, dust_model = Calzetti)

SSP_D = CSP(bc03, age = 10.0*u.Gyr, sfh = SFH, 
           dust = Dust, metal_ind = 1.0, f_esc = 1.,
           sfh_law = SFH_law, dust_model = Calzetti)

models_exp = CSP(bc03, age = Ages, sfh = SFH, 
             dust = Dust, metal_ind = 1.0, f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti)              
#==============================================================================
# Combine and Measure             
#==============================================================================
weight = np.concatenate((np.linspace(0.,0.8,9),np.linspace(0.9,1.0,6)))
X_track,Y_track = np.empty((15,SFH.size)),np.empty((15,SFH.size))

# A B C D
SSP1, SSP2 = SSP_A, SSP_D

eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')
filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
        # FUV, NUV, U, B, V, R, I, J, H, K

def Composite(SSP1, SSP2):
    synphot1 = Observe(SSP1, filters, redshift=0.001)
    synphot2 = Observe(SSP2, filters, redshift=0.001)
    mags1 = synphot1.AB[0]
    U_V1 = mags1[2]-mags1[4]
    V_J1 = mags1[4]-mags1[7]
    mags2 = synphot2.AB[0]
    U_V2 = mags2[2]-mags2[4]
    V_J2 = mags2[4]-mags2[7]
    X1, Y1 = V_J1, U_V1
    X2, Y2 = V_J2, U_V2
    for i, w in enumerate(weight):
        combined = (1-w) * SSP1 + w * SSP2
        synphot = Observe(combined, filters, redshift=0.001)

        mags = synphot.AB[0]
        U_V = mags[2]-mags[4]
        V_J = mags[4]-mags[7]
        
        X, Y = V_J, U_V
        
        X_track[i] = X.ravel()
        Y_track[i] = Y.ravel()
    return X_track, Y_track, X1, Y1, X2, Y2
 
def PlotXY(X, Y, X1, Y1, X2, Y2):

    for k, Av in enumerate(Dust):
        s=plt.scatter(X, Y, s=60, c=np.log10(SFH/u.yr),cmap='summer',alpha=0.9)
        plt.scatter(X1[0,:,:,k,0], Y1[0,:,:,k,0], s=60, label='A: 0.1 Gyr',
                    marker='s',c=np.log10(SFH/u.yr),cmap='winter_r',alpha=0.7)
        plt.scatter(X2[0,:,:,k,0], Y2[0,:,:,k,0], s=80, label='D: 10 Gyr',
                    marker='p',c=np.log10(SFH/u.yr),cmap='autumn',alpha=0.7)
    for j,tau in enumerate(SFH):
        plt.plot(X_track[:,j],Y_track[:,j], 
                 label=r'$\tau = ${0:.2f}'.format(tau),lw=2,ls='--',alpha=0.7)
    return s     

#X_track, Y_track, X1, Y1, X2, Y2 = Composite(SSP_A, SSP_D)

#for i,w in enumerate(weight):
#    plt.figure(figsize=(9,7))
#    s = PlotXY(X_track[i],Y_track[i],X1,Y1,X2,Y2)
#    cb = plt.colorbar(s)
#    cb.set_label(r'$\log_{10}(\rm{Tau}/\rm{yr})$')
#    plt.legend(loc='best',fontsize=12,frameon=True,facecolor='w')
#    plt.plot([-1., 1.0, 1.6, 1.6], [1.3, 1.3, 2.01, 2.5], color='k', alpha=0.7)
#    plt.axhline(0.5,ls='--',color='k')
#    plt.xlabel('V - J')
#    plt.ylabel('U - V')
#    plt.xlim([-0.75, 2.0])
#    plt.ylim([-0.25, 2.25])
#    plt.tight_layout()
#    #plt.savefig('taskIII/A+D %.2f.png'%w,dpi=100)
#    plt.show()

#==============================================================================
# Exponential
#==============================================================================
synphot_exp = Observe(models_exp, filters, redshift=0.001)
mags_exp = synphot_exp.AB[0]
FUV, NUV, U, B, V, R, I, J, H, K = mags_exp
FUV_V = FUV - V
NUV_V = NUV - V
U_V = U - V
V_J = V - J
V_K = V - K

#==============================================================================
# Composite Schematic
#==============================================================================
X, Y = V_J, U_V
with sns.axes_style("ticks"):
    plt.figure(figsize=(7,7))
    plt.plot(X[0,:,0,0,0], Y[0,:,0,0,0], c='m',lw=2, alpha=0.7)
    
    X_track, Y_track, X1, Y1, X2, Y2 = Composite(SSP_A, SSP_D)
    plt.plot(X_track[:,0],Y_track[:,0],lw=2,ls=':',alpha=0.9)
    plt.scatter(X1[0,:,:,0,0], Y1[0,:,:,0,0], s=75, label='A: 0.1 Gyr',
                marker='s',c='cyan',alpha=0.9)
    plt.scatter(X_track[:,0][9:-1], Y_track[:,0][9:-1], s=50,
                   marker='|',c='k',alpha=1.)
    
    X_track, Y_track, X1, Y1, X2, Y2 = Composite(SSP_B, SSP_C)
    plt.plot(X_track[:,0],Y_track[:,0],lw=2,ls=':',alpha=0.9)
    plt.scatter(X1[0,:,:,0,0], Y1[0,:,:,0,0], s=75, label='B: 0.4 Gyr',
                marker='D',c='blue',alpha=0.9)
    plt.scatter(X_track[:,0][1:10:2], Y_track[:,0][1:10:2], s=50,
                   marker='|',c='k',alpha=0.5)
    
    X_track, Y_track, X1, Y1, X2, Y2 = Composite(SSP_A, SSP_C)
    plt.plot(X_track[:,0],Y_track[:,0],lw=2,ls=':',alpha=0.9)
    plt.scatter(X2[0,:,:,0,0], Y2[0,:,:,0,0], s=125, label='C: 4 Gyr',
                marker='p',c='orange',alpha=0.9)
    plt.scatter(X_track[:,0][9:-1], Y_track[:,0][9:-1], s=50,
                   marker='|',c='k',alpha=1.)
    
    X_track, Y_track, X1, Y1, X2, Y2 = Composite(SSP_B, SSP_D)
    plt.plot(X_track[:,0],Y_track[:,0],lw=2,ls=':',alpha=0.9)
    plt.scatter(X2[0,:,:,0,0], Y2[0,:,:,0,0], s=125, label='D: 10 Gyr',
                    marker='H',c='red',alpha=0.9)
    plt.scatter(X_track[:,0][1:10:2], Y_track[:,0][1:10:2], s=50,
                   marker='|',c='k',alpha=0.5)

    plt.axhline(0.5,ls='--',color='k',alpha=0.7)
    plt.legend(loc='best',fontsize=14,frameon=True,facecolor='w')

    plt.scatter(1.,0.02,marker='|',s=100,c='k',alpha=0.5)
    plt.text(1.1,0.,'[0.1,0.3,0.5,0.7,0.9]',alpha=0.5)
    plt.scatter(1.,-0.08,marker='|',s=100,c='k',alpha=1.)
    plt.text(1.1,-0.1,'[0.92,0.94,0.96,0.98]',alpha=1.)
    
    zg = 1.0
    plt.plot([-5,(0.55+0.253*zg-0.0533*zg**2)/0.88,1.6,1.6],
      [1.3,1.3,2.158-0.253*zg+0.0533*zg**2,2.5], color='k', alpha=0.7)
    
    plt.xlabel('V - J',fontsize=12)
    plt.ylabel('U - V',fontsize=12)
    plt.xlim([-0.75, 2.0])
    plt.ylim([-0.25, 2.25])
    plt.savefig('Composite/Composite Schematic.png',dpi=200,bbox_inches='tight')
    plt.show()
    