# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:14:22 2017

@author: Qing Liu
"""

import re
import glob
import numpy as np
import asciitable as asc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#==============================================================================
# Read model result 
#==============================================================================
Ages = np.array([0.1,0.4,4.0,10.0])*1e9
SFH = np.array([0.1, 0.5, 1.0, 3.0, 5.0, 20.0])*1e9
lgSFH = np.log10(([0.1, 0.5, 1.0, 3.0, 5.0, 20.0]))
lgAges = np.log10(np.array([0.1,0.4,4.0,10.0])*1e9)
Dust = np.array([1.0])
weight = np.concatenate((np.linspace(0.,0.8,9),np.linspace(0.9,1.0,6)))

lgMs = np.array([])
lgSSFR = np.array([])
dir = glob.glob('Composite/*.4color') 
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
Ms = 10**lgMs.reshape((6,4)).T
SSFR = 10**lgSSFR.reshape((6,4)).T

def SetValue(Value,isSFH=False):
    X = np.zeros((Ages.size, SFH.size, Dust.size))
    if isSFH:
        for i in range(Ages.size):
            for j in range(Dust.size):
                X[i,:,j] = np.log10(Value)
    else:
        for i in range(Dust.size):
            X[:,:,i] = np.log10(Value)
    lgValue = X.ravel()
    return lgValue

lgAges = np.array([np.log10(Ages) for i in range(SFH.size)\
                                  for i in range(Dust.size)]).T.ravel()
lgSFH = SetValue(SFH,isSFH=True)
lgM = SetValue(Ms)
lgSSFR = SetValue(SSFR)

Av = np.array([Dust for i in range(weight.size)\
                    for i in range(SFH.size)]).ravel()                         
#==============================================================================
# A:0 B:1 C:2 D:3                  
#==============================================================================
sp1,sp2 = (0, 2)   
              
Ages_w = np.array([])
SFH_w = np.array([])
Ms_w = np.array([])
SSFR_w = np.array([])
for j,w in enumerate(weight):
    Ages_w = np.append(Ages_w,[(1-w) * Ages[sp1] + w * Ages[sp2] for i in range(SFH.size) for j in range(Dust.size)])
    SFH_w = np.append(SFH_w,np.array(((1-w) * SFH + w * SFH).tolist()*Dust.size).reshape((Dust.size,SFH.size)).T.ravel())
    Ms_w = np.append(Ms_w,np.array(((1-w) * Ms[sp1,:] + w * Ms[sp2,:]).tolist()*Dust.size).reshape((Dust.size,SFH.size)).T.ravel())
    SSFR_w = np.append(SSFR_w,np.array(((1-w) * SSFR[sp1,:] + w * SSFR[sp2,:]).tolist()*Dust.size).reshape((Dust.size,SFH.size)).T.ravel())
lgAges_w = np.log10(Ages_w) 
lgSFH_w = np.log10(SFH_w) 
lgMs_w = np.log10(Ms_w) 
lgSSFR_w = np.log10(SSFR_w)
Weights = np.array([weight for i in range(SFH.size) for j in range(Dust.size)]).T.ravel()
Compositeness = 4*(1-Weights)*Weights
#==============================================================================
# Read FAST SED-fitting result
#==============================================================================
table = asc.read('Composite/composite_ds_AC_exp.fout')
Ages_FAST = table.lage
SFH_FAST = table.ltau
M_FAST = table.lmass
SSFR_FAST = table.lssfr
Av_FAST = table.Av

#==============================================================================
# Plot FAST vs Model
#==============================================================================
#fig,axes = plt.subplots(figsize=(9,8), nrows=2, ncols=2)
#xx=np.linspace(-20,20,20)
#legends = [r'log(Age/yr)$_{\rm{weighted}}$',
#           r'log$(\rm{M_*}/\rm{M_{\odot}})_{weighted}$',r'log(sSFR/yr$^{-1}$)$_{\rm{weighted}}$']
#for i, (l, model, FAST) in enumerate(zip(legends,[lgAges_w,lgMs_w,lgSSFR_w],
#                                       [Ages_FAST,M_FAST,SSFR_FAST])):
#    ax = plt.subplot(2,2,i+1) 
#    plt.plot(xx,xx,'k--',alpha=0.5)
#    s = plt.scatter(model, FAST, c=Weights,
#                    cmap='jet', label=l, s=20, alpha=0.7)
#    plt.xlabel('Model',fontsize=12)
#    plt.ylabel('FAST',fontsize=12)
#    if i==2:
#        plt.xlim(-14,-7)
#        plt.ylim(-14,-7)
#    elif i==1:
#        plt.xlim(-3.,1.)
#        plt.ylim(-3.,1.)
#    else:
#        plt.xlim(6.,11.)
#        plt.ylim(6.,11.)
#    plt.legend(loc=4, fontsize=12, frameon=True, facecolor='w')
#fig.subplots_adjust(right=0.82)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
#colorbar = fig.colorbar(s, cax=cbar_ax)
#colorbar.set_label('Weight of Old SSP')
#ax = plt.subplot(2,2,4)
#plt.hist(Av_FAST,alpha=0.7,label='Av')
#plt.axvline(1.0,color='k',ls='--')
#plt.xlabel('FAST',fontsize=12)
#plt.suptitle('Composite Tau A+C vs FAST Fitting',fontsize=15,y=0.92)
#plt.show() 

#==============================================================================
# Residual Plot
#==============================================================================
fig,axes = plt.subplots(figsize=(11,10), nrows=2, ncols=2)
legends = [r'log(Age/yr)$_{\rm{weighted}}$',
           r'log$(\rm{M_*}/\rm{M_{\odot}})_{weighted}$',r'log(sSFR/yr$^{-1}$)$_{\rm{weighted}}$']
for i, (l, model, FAST) in enumerate(zip(legends,[lgAges_w,lgMs_w,lgSSFR_w],
                                       [Ages_FAST,M_FAST,SSFR_FAST])):
    ax = plt.subplot(2,2,i+1) 
    s = plt.scatter(model, FAST-model, c=Compositeness,
                    cmap='viridis', label=l, s=20, alpha=0.7)
    plt.xlabel('Model',fontsize=12)
    plt.ylabel('FAST - Model',fontsize=12)
    if i==2:
        plt.xlim(-12.5,-7.5)
    elif i==1:
        plt.xlim(-3.,1.)
    else:
        plt.xlim(7.,11.)
    plt.ylim(-3.,3.)
    plt.axhline(0.0,color='k',ls='--',alpha=0.7)
    plt.text(0.1,0.1,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
             fontsize=14,transform=ax.transAxes)
fig.subplots_adjust(right=0.82)
cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
colorbar = fig.colorbar(s, cax=cbar_ax)
colorbar.set_label('Compositeness', fontsize=15)
ax = plt.subplot(2,2,4)
s = plt.scatter(Av_FAST,SSFR_FAST-lgSSFR_w,c=Compositeness,s=20,cmap='viridis')
plt.axvline(1.0,color='k',ls='--')
plt.ylim(-2.,1.)
plt.xlabel('FAST Av',fontsize=12)
plt.ylabel('$SSFR_{FAST} - SSFR_{model}$',fontsize=12)
plt.text(0.1,0.1,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1.).std(),
             fontsize=14,transform=ax.transAxes)
plt.legend(loc=4, fontsize=12, frameon=True, facecolor='w')
plt.suptitle('Composite Tau A+C vs FAST Fitting',fontsize=15,y=0.92)
plt.show()
