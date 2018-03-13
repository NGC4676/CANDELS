# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 22:59:01 2017

@author: Qing Liu
"""
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfc
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70.4, Om0=0.272)
t_start = (cosmo.hubble_time - cosmo.lookback_time(z=5)).value
t_stop = (cosmo.hubble_time - cosmo.lookback_time(z=0.3)).value
m_MS = 3e11

def MS_Sch(M, z):
    r = np.log10(1+z)
    m = np.log10(M/1e9)
    lgSFR = m - 0.5 + 1.5*r - 0.3* \
            (np.max((np.zeros_like(m), m-0.36-2.5*r),axis=0))**2
    return 10**lgSFR
         
def RKPRG(x, m_seed):
    A = 6e-3*np.exp(-np.log10(m_seed)/-0.84)
    mu = 47.39*np.exp(-np.log10(m_seed)/3.12)
    sigma = 17.08*np.exp(-np.log10(m_seed)/2.96)
    r_s = -0.56*np.log10(m_seed) + 7.03 
    res = A*(np.sqrt(np.pi)/2.) \
            *np.exp( (sigma/(r_s*2.))**2 - (x-mu)/r_s ) \
            *sigma*erfc( sigma/(r_s*2.) - (x-mu)/sigma )
    return res

T = np.linspace(t_start, 12.5, 100)
M_seed = np.logspace(6.0, 10.0, 5)

SFR = np.empty((M_seed.size, T.size))
dM = np.empty_like(SFR)
Ms = np.empty_like(SFR)
for i, m_seed in enumerate(M_seed):
    SFR[i] = RKPRG(T, m_seed)
    dM[i] = np.array([np.trapz(SFR[i][:(k+1)], x=T[:(k+1)]*1e9) for k in range(T.size)])
    Ms[i] = dM[i] + m_seed
 
Data = {}
for k, m in enumerate(M_seed):
    t, sfr = T, SFR[k]
    data = pd.DataFrame({'t':t, 'SFR':sfr})
    Data['%s'%np.log10(m)]  = data 
        
#==============================================================================
# Fig 5
#==============================================================================
#plt.semilogy(T, RKPRG(T, 1e8))
#plt.xlim(1,10)
#plt.ylim(1,2e2)
#plt.show()

#==============================================================================
# Fig 1 in Ciesla
#==============================================================================
#fig = plt.figure(figsize=(8,6))
#with sns.axes_style("ticks"):
#    for i, m_seed in enumerate(M_seed):
#        m,sfr = Ms[i], SFR[i]
#        plt.loglog(m[m<m_MS], sfr[m<m_MS],'.',
#                   label='log M$_{seed}$ = %s'%np.log10(m_seed))
#    m_plot = np.logspace(6.0, 13.0, 100)
#    plt.plot(m_plot,MS_Sch(m_plot,0.0),
#             'k--',alpha=0.7)
#    plt.plot(m_plot,MS_Sch(m_plot,1.0),
#             'k--',alpha=0.5)
#    plt.plot(m_plot,MS_Sch(m_plot,5.0),
#             'k--',alpha=0.3)
#    plt.legend(loc=4,fontsize=12,frameon=True,facecolor='w')
#    plt.text(1e8,2e-2,'z = 0')
#    plt.text(4e7,5e-2,'z = 1')
#    plt.text(1e7,1e-1,'z = 5')
#    plt.xlim(1e6,3e11)
#    plt.ylim(1e-3,1e3)
#    plt.xlabel('M* (M$_\odot$)',fontsize=15)
#    plt.ylabel('SFR (M$_\odot$/yr)',fontsize=15)
#plt.show()


#==============================================================================
# Fig 2 in Ciesla
#==============================================================================
fig = plt.figure(figsize=(8,10))
with sns.axes_style("ticks"):
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    for i, m_seed in enumerate(M_seed):
        m,sfr = Ms[i], SFR[i]
        ax1.semilogy(T[m<m_MS], m[m<m_MS],
                     label='log M$_{seed}$ = %s'%np.log10(m_seed))
        ax2.semilogy(T[m<m_MS], sfr[m<m_MS],
                     label='log M$_{seed}$ = %s'%np.log10(m_seed))
    for ax in [ax1,ax2]:
        ax.set_xlim(1.,12.5)
        ax.set_xlabel('t (Gyr)',fontsize=15)
        ax.legend(loc=4,fontsize=12,frameon=True,facecolor='w')
    ax1.set_ylim(2e5, 2e12)
    ax1.set_ylabel('M* (M$_\odot$)',fontsize=15)
    ax2.set_ylim(1e-3, 1e4)
    ax2.set_ylabel('SFR (M$_\odot$/yr)',fontsize=15)
plt.show()

#fig = plt.figure(figsize=(8,6))
#with sns.axes_style("ticks"):
#    from scipy.interpolate import interp1d
#    for i, m_seed in enumerate(M_seed):
#        m,sfr = Ms[i], SFR[i]
#        f = interp1d(T[m<m_MS],np.log10((sfr[m<m_MS])),fill_value='extrapolate')
#        T_interp = T[m>=m_MS]
#        SFR_interp = f(T_interp)   # use interpolation function returned by `interp1d`
#        plt.semilogy(T[m<m_MS],(sfr[m<m_MS]), '-', T_interp, 10**SFR_interp, '--')
#plt.show()

#==============================================================================
# Save SFH as txt
#==============================================================================
#for m in Data:
#    plt.plot(Data[m].t,Data[m].SFR)
#    tab=np.vstack((Data[m].t*1e9,Data[m].SFR)).T
#    np.savetxt('Ciesla/bc03_M%s.txt'%m,tab,fmt='%.7e',header='t(yr) SFR(Msol/yr)')