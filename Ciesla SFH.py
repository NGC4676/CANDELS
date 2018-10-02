# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 22:59:01 2017

@author: Qing Liu
"""

import numpy as np
import seaborn as sns
import pandas as pd
import asciitable as asc
import matplotlib.pyplot as plt
from scipy.special import erfc
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70.4, Om0=0.272)
t_start = (cosmo.hubble_time - cosmo.lookback_time(z=10.)).value
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
M_seed = 10**np.linspace(5., 7., 5)

SFR = np.empty((M_seed.size, T.size))
dM = np.empty_like(SFR)
Ms = np.empty_like(SFR)
for i, m_seed in enumerate(M_seed):
    SFR[i] = RKPRG(T, m_seed)
    dM[i] = np.array([np.trapz(SFR[i][:(k+1)], x=T[:(k+1)]*1e9) for k in range(T.size)])
    Ms[i] = dM[i] + m_seed
 
Data_c = {}
for k, m in enumerate(M_seed):
    t, sfr, lgMs = T, SFR[k], np.log10(Ms[k])
    data = pd.DataFrame({'t':t, 'SFR':sfr, "lgMs":lgMs})
    Data_c['%s'%np.log10(m)]  = data 
        
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
fig = plt.figure(figsize=(11,5))
with sns.axes_style("ticks"):
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for i, (m_seed,c) in enumerate(zip(M_seed,['navy','steelblue','yellowgreen','gold','orangered'])):
        m,sfr = Ms[i], SFR[i]
        ax1.semilogy(T[m<m_MS], m[m<m_MS],c=c,
                     label='log M$_{seed}$ = %s'%np.log10(m_seed))
        ax2.semilogy(T[m<m_MS], sfr[m<m_MS],c=c,
                     label='log M$_{seed}$ = %s'%np.log10(m_seed))
    for ax in [ax1,ax2]:
        ax.set_xlim(1.,12.5)
        ax.set_xlabel('t (Gyr)',fontsize=15)
        ax.legend(loc=4,fontsize=10,frameon=True,facecolor='w')
    ax1.set_ylim(2e5, 2e12)
    ax1.set_ylabel('M* (M$_\odot$)',fontsize=15)
    ax2.set_ylim(1e-3, 1e2)
    ax2.set_ylabel('SFR (M$_\odot$/yr)',fontsize=15)
plt.tight_layout()
plt.show()

# =============================================================================
#  Show ALdo and Ciesla in one Fig.
# =============================================================================
M_today = np.array([9.5,10.0,10.5,10.75,11.0])
Data_a = {}
for m in M_today.astype('str'):
    table = asc.read('Aldo/galaxies/gal_%s.dat'%m)
    t, z, lgMh, lgMs, SFR = table.col1, table.col2, table.col3, table.col4, table.col5
    data = pd.DataFrame({'t':t, 'z':z, 'lgMh':lgMh, 'lgMs':lgMs, 'SFR':SFR})
    Data_a['%s'%m]  = data

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch

# Papovich15 AM
Ms_pop = np.array([9.48,9.7,9.88,10.06,10.21,10.35,10.47,10.6])
z_pop = np.array([2.5,2.1,1.85,1.55,1.25,1.0,0.8,0.45])
t_pop=cosmo.age(z_pop)

bb = mtransforms.Bbox([[0.36, 0.28], [0.988, 0.36]])
def draw_box(ax,bb):  
    p_fancy = FancyBboxPatch((bb.xmin, bb.ymin),
                             abs(bb.width), abs(bb.height),
                             transform=ax.transAxes,
                             boxstyle="square,pad=0",
                             fc="w",ec="gray",alpha=.5,zorder=3) 
    ax.add_patch(p_fancy)
    ax.axvline(x=8.2,ymin=0.31,ymax=0.38,color='gray',lw=1,alpha=0.7,zorder=3)  
    plt.text(0.45,0.32,"AM-SFH",color='k',va="center",ha="center",
             transform=ax.transAxes,fontsize=12,alpha=0.8)    
    plt.text(0.8,0.32,"MS-SFH",color='k',va="center",ha="center",
             transform=ax.transAxes,fontsize=12,alpha=0.8) 
    
# SFR/Ms of Aldo and Ciesla
csfont = {'fontname':'helvetica'}
fig = plt.figure(figsize=(6,11))
ax = plt.subplot(211)
for i,(m,c) in enumerate(zip(M_today.astype('str'),['m','b','g','orange','firebrick'])):
    ax.semilogy(Data_a[m].t, Data_a[m]["SFR"],c=c,
            #label=r'log M$\rm_{today}$ = %s'%m,
            label='RP%d'%(i+1),alpha=0.7,zorder=3)
for i, (m,c) in enumerate(zip(np.log10(M_seed).astype(str),['navy','steelblue','yellowgreen','gold','orangered'])):
#        ms,sfr = Ms[i], SFR[i]
    ax.semilogy(Data_c[m].t, Data_c[m]["SFR"],c=c,ls="-.",
                 #label=r'log M$\rm_{seed}$ = %s'%m,
                 label='C%d'%(i+1),zorder=2)
ax.set_xlim(0.,12.)
ax.set_xlabel('Cosmic Time (Gyr)',fontsize=15)
#ax.legend(loc=4,fontsize=12,frameon=True,facecolor='w')
ax.set_ylabel('SFR (M$_\odot$/yr)',fontsize=15)
plt.axvline(2.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
plt.axvline(8.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
leg=plt.legend(loc="best",ncol=2, title="AM-SFH       MS-SFH",
          fontsize=12,frameon=True,facecolor='w',framealpha=0.8)  
leg.get_title().set_fontsize(14)
#draw_box(ax,bb)

ax = plt.subplot(212)
patch = PatchCollection([Rectangle((5.725, 3e8), 0.5, 1.5e11)],
                 linestyle=":", linewidth=1.5,
                 facecolor="None", edgecolor="gray", alpha=0.8)
for i,(m,c) in enumerate(zip(M_today.astype('str'),['m','b','g','orange','firebrick'])):
    ax.semilogy(Data_a[m].t, 10**Data_a[m]["lgMs"],c=c,
            #label=r'log M$\rm_{z=1}$ = %s'%m,
            label='RP%d'%(i+1),alpha=0.7,zorder=3)
for i, (m,c) in enumerate(zip(np.log10(M_seed).astype(str),['navy','steelblue','yellowgreen','gold','orangered'])):
    ax.semilogy(Data_c[m].t, 10**Data_c[m]["lgMs"],c=c,ls="-.",
                 #label=r'log M$\rm_{z=1}$ = %s'%m,
                 label='C%d'%(i+1),zorder=2)
plt.scatter(t_pop, 10**Ms_pop, s=80, c="k", marker="s", alpha=0.7,zorder=4)

ax.add_collection(patch)
plt.text(5.97,10**11.4,"z$=$1",color='gray',va="center",ha="center",fontsize=12,alpha=0.8,**csfont)    
plt.scatter(0.5, 2e11, s=80, c="k", marker="s", alpha=0.7,zorder=4)
plt.text(0.85,2e11,"Papovich+15 MW",color='k',va="center",fontsize=12,alpha=0.8,**csfont)    

#draw_box(ax,bb)

plt.axvline(2.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
plt.axvline(8.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)    
leg=plt.legend(loc="best",ncol=2, title="AM-SFH       MS-SFH",
           fontsize=12,frameon=True,facecolor='w',framealpha=0.8)
leg.get_title().set_fontsize(14)
ax.set_xlim(0.,12.)
ax.set_xlabel('Cosmic Time (Gyr)',fontsize=15)
ax.set_ylabel('M$_*$ (M$_{\odot}$)',fontsize=15)
fig.subplots_adjust(left=0.15,right=0.95,bottom=0.075,top=0.975)
plt.savefig("N/AM+MS-SFH-M.pdf",dpi=400)
       
    
# Only SFH    
with sns.axes_style("ticks"):
    fig = plt.figure(figsize=(7,6))
    ax = plt.subplot(111)
    for i,(m,c) in enumerate(zip(M_today.astype('str'),['m','b','g','orange','firebrick'])):
        ax.semilogy(Data_a[m].t, Data_a[m]["SFR"],c=c,
                label=r'$RP%d$'%(i+1),alpha=0.7,zorder=3)
    for i, (m,c) in enumerate(zip(np.log10(M_seed).astype(str),['navy','steelblue','yellowgreen','gold','orangered'])):
#        ms,sfr = Ms[i], SFR[i]
        ax.semilogy(Data_c[m].t, Data_c[m]["SFR"],c=c,ls="-.",
                     label=r'$C%d$'%(i+1),zorder=2)
    ax.set_xlim(-0.2,12.)
    ax.set_xlabel('Cosmic Time (Gyr)',fontsize=12)
    ax.set_ylabel('SFR (M$_\odot$/yr)',fontsize=12)
    plt.axvline(2.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
    plt.axvline(8.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
    plt.legend(loc="best",ncol=2,title="AM-SFH         MS-SFH",
              fontsize=11,frameon=True,facecolor='w',framealpha=0.8)  
    plt.tight_layout()
#plt.savefig("N/AM+MS-SFH.pdf",dpi=400)

#==============================================================================
# Save SFH as txt
#==============================================================================
#for m in Data_c:
#    plt.plot(Data_c[m].t,Data_c[m].SFR)
#    tab=np.vstack((Data_c[m].t*1e9,Data_c[m].SFR)).T
#    np.savetxt('Ciesla/bc03_M%s.txt'%m,tab,fmt='%.7e',header='t(yr) SFR(Msol/yr)')