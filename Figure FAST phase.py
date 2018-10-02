# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:32:48 2017

@author: Qing Liu
"""

import numpy as np
import pandas as pd
import asciitable as asc
from astropy.io import ascii
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import asciitable as asc
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, least_squares

lgAge = np.log10(np.linspace(0.5,11.5,111)* 1e9)
#M_today = np.array([9.0,9.5,10.0,10.5,11.0])
M_today = np.array([9.5,10.0,10.5,10.75,11.0])
M_color = ['m','b','g','orange','firebrick']

Phases = [np.pi/4.,3*np.pi/4.,5*np.pi/4.,7*np.pi/4.]
Phases_rd = np.round(Phases,decimals=1)
phase_start = lgAge[0]
    

Data = {}
for m in M_today.astype('str'):
    table = asc.read('Aldo/galaxies/gal_%s.dat'%m)
    t, z, lgMh, lgMs, SFR = table.col1, table.col2, table.col3, table.col4, table.col5
    data = pd.DataFrame({'t':t, 'z':z, 'lgMh':lgMh, 'lgMs':lgMs, 'SFR':SFR})
    Data['%s'%m]  = data
        
new_tick_locs = np.array([.1, .3, .5, .7, .9])
def tick_func(x_ticks):
    ticks = [Data[m].z[np.argmin(abs(Data[m].t-x))] for x in x_ticks]
    return ["%.1f"%z for z in ticks]

def MS_Sch(M, z):
    r = np.log10(1+z)
    m = np.log10(M/1e9)
    lgSFR = m - 0.5 + 1.5*r - 0.3* \
            (np.max((np.zeros_like(m), m-0.36-2.5*r),axis=0))**2
    return 10**lgSFR
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307) 
cosmo = FlatLambdaCDM(H0=70.4, Om0=0.27) 

#==============================================================================
# Read model result 
#==============================================================================
A, P = 0.3, 2.5
p_id = 1 
ConstP = False

#BC03_out = asc.read('New/Perturb/CSP result ST.txt')
BC03_out = asc.read('N/Aldo/ST/ST%s_p%s CSP result.txt'%(A,Phases_rd[p_id]))

lgMs = BC03_out['lgM*'].reshape((M_today.size, lgAge.size)).T.ravel()
lgSSFR = BC03_out['lgsSFR'].reshape((M_today.size, lgAge.size)).T.ravel()
lgSFR = lgSSFR + lgMs
                 
lgAges = np.array([lgAge for m in M_today]).T.ravel()
M_class =  np.array([M_today for T in range(lgAge.size)]).ravel()

#SSFR_model=lgSSFR.reshape((lgAge.size,M_today.size)).T
#==============================================================================
# Read FAST SED-fitting result
#==============================================================================
#table = ascii.read('New/Perturb/Aldo_ST_exp.fout',header_start=16).to_pandas()
table = ascii.read('N/Aldo/ST/Aldo_exp_ST0.3_p%s.fout'%p_id,header_start=16).to_pandas()

Ages_FAST = table.lage
SFH_FAST = table.ltau
M_FAST = table.lmass
SSFR_FAST = table.lssfr
SFR_FAST = table.lsfr
Av_FAST = table.Av    
chi2_FAST = table.chi2

age_cond = (lgAges>np.log10(2e9))&(lgAges<np.log10(9.e9))
age_cond2 = (lgAge>np.log10(2e9))&(lgAge<np.log10(9.e9))

#table2 = ascii.read('New/Perturb/Aldo_ST_exp_obs.fout',header_start=16).to_pandas()
table2 = ascii.read('N/Aldo/ST/Aldo_exp_ST0.3_p%s_obs.fout'%p_id,header_start=16).to_pandas()
Ages_FAST2 = table2.lage
SFH_FAST2 = table2.ltau
M_FAST2 = table2.lmass
SSFR_FAST2 = table2.lssfr
SFR_FAST2 = table2.lsfr
Av_FAST2 = table2.Av    
chi2_FAST2 = table2.chi2

#==============================================================================
# Plot
#==============================================================================
for m, mi in zip(M_today.astype('str'),range(M_today.size)):
    C = 0.3
    t = np.linspace(0.5,13.0,126)
    t = np.linspace(0.5,11.5,111)
    fig = plt.figure(figsize=(8,3))
    ax = plt.subplot(111)
    sfr = np.interp(t, Data[m].t[::-1], Data[m].SFR[::-1])
    plt.semilogy(t, sfr, 'k-',
                 label='log M$_{today}$ = %s'%m,alpha=0.7)
    if ConstP:
        sfr_prtb = sfr*10**(-C*np.sin(2*np.pi*t/P + 3*np.pi/4.))
        s=plt.scatter(t, sfr_prtb,
                      c=np.mod(2*np.pi*t/P + 3*np.pi/4.,2*np.pi) /np.pi,
                      marker='o',s=25,cmap='gnuplot_r',alpha=0.8)
        for n in np.arange(0.4,5.4,1.):
            plt.axvline(n*P,color='skyblue',ls='--',alpha=0.7)
    else:
        sfr_prtb = sfr*10**(-C*np.sin(5*np.pi*np.log(t) + 3*np.pi/4.))
        s=plt.scatter(t[(t>=2) & (t<=9)], sfr_prtb[(t>=2) & (t<=9)],
#                      c=np.mod(5*np.pi*np.log(t[(t>=2) & (t<=9)]) + 3*np.pi/4.,2*np.pi) /np.pi,
                        c=(np.log10(sfr_prtb)-np.log10(sfr))[(t>=2) & (t<=9)],
                      marker='o',s=25,cmap='RdYlBu',alpha=0.9)
        s=plt.scatter(t[(t<2) | (t>9)], sfr_prtb[(t<2) | (t>9)],
                      c="gray",marker='o',s=15,alpha=0.9)
        for n in np.arange(0.15,2.55,0.4):
            plt.axvline(np.e**n,color='steelblue',ls='--',alpha=0.5)

    if ConstP:
        peaks=[29,54,78,104]
    else:
        peaks=[21,33,52,81]
#    plt.scatter(t[peaks[0]],sfr_prtb[peaks[0]],s=30,color='k',marker='x')
#    plt.scatter(t[peaks[1]],sfr_prtb[peaks[1]],s=30,color='k',marker='x')
#    plt.scatter(t[peaks[2]],sfr_prtb[peaks[2]],s=30,color='k',marker='x')
#    plt.scatter(t[peaks[3]],sfr_prtb[peaks[3]],s=30,color='k',marker='x')

    plt.xlabel('t (Gyr)')
    plt.ylabel('SFR (M$\odot$/yr)')
#    plt.legend(loc=4, fontsize=10, frameon=True, facecolor='w')
    plt.text(0.75,0.08,'log M$_{today}$ = %s'%m,fontsize=12,transform=ax.transAxes)
    plt.xlabel('Cosmic Time (Gyr)')
    plt.ylabel('SFR (M$\odot$/yr)')

    labels = [r'$\rm \Delta{\ }log(M_*/M_{\odot}$)',
              r'$\rm \Delta{\ }log(SFR/M_{\odot}yr^{-1})$', 
              r'$\rm \Delta{\ }log(SSFR/yr^{-1})$', r'$\rm \Delta{\ }A_v$']
    for i, (yl, model, FAST) in enumerate(zip(labels, [
                                                       lgMs,lgSFR,lgSSFR,np.ones_like(Av_FAST)],
                                                      [M_FAST,SFR_FAST,SSFR_FAST,Av_FAST])):
        ax = fig.add_axes([0.125,0.96+0.4*i, 0.775, 0.36])
        age_cond_m = age_cond[mi::5]
        
        if ConstP:
            if i==0:
                plt.plot(10**(lgAge-9), (FAST-model)[mi::5],alpha=0.7)
            else:
                s = plt.scatter(10**(lgAge-9), (FAST-model)[mi::5], 
                    c=np.mod(2*np.pi*10**(lgAge-9)/P + 3*np.pi/4.,2*np.pi) /np.pi,
                    cmap='gnuplot_r', s=20, alpha=0.7)
            for n in np.arange(0.4,5.4,1.):
                plt.axvline(n*P,color='r',ls='--',alpha=0.3)
        else:
            s = plt.scatter(10**(lgAge[age_cond2]-9), (FAST-model)[mi::5][age_cond_m], 
#                            c=np.mod(5*np.pi*np.log(10**(lgAge[age_cond2]-9)) + 3*np.pi/4.,2*np.pi) /np.pi,
                            c=(np.log10(sfr_prtb)-np.log10(sfr))[age_cond2],
                            cmap='RdYlBu', s=25, alpha=0.9)
            s = plt.scatter(10**(lgAge[~age_cond2]-9), (FAST-model)[mi::5][~age_cond_m], 
                            c="gray",s=25, alpha=0.9)
            for n in np.arange(0.15,2.55,0.4):
                plt.axvline(np.e**n,color='steelblue',ls='--',alpha=0.5)
        y = pd.Series((FAST-model)[mi::5][age_cond_m])
        y = y[abs(y)<1.]
        y2 = pd.Series((FAST-model)[mi::5])
#        y2 = y2[abs(y2)<1.]
        plt.text(0.54,0.08,'$\sigma$ = %.2f'%y.std(),
                fontsize=12,transform=ax.transAxes,color="orange") 
#        plt.text(0.8,0.08,'$\sigma$ = %.2f'%y2.std(),
#                fontsize=12,transform=ax.transAxes,color="gray")        
        plt.axhline(0.,color='k',ls='--',alpha=0.5)
        plt.xlim(0.,12.)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.ylabel(yl)    

    #ax = fig.add_axes([0.125,0.95+0.28*4, 0.775, 0.25])
    #lgSSFR_pred=-2.28*Csed**2-0.8*Csed-8.41
    #plt.plot(Ages,(SSFR_FAST.reshape((76,6))-lgSSFR_pred)[mi],c='m',alpha=0.5)
    #plt.ylabel('($\Delta$ lgSSFR)$_{mod - mod/Csed}$')
    #plt.plot(Ages,(UV_st-UV)[mi],c='m',alpha=0.5)
    #plt.ylabel('(U-V)$_{model}$')
    #plt.plot(Ages,(Csed_2P5-Csed)[mi],c='k',alpha=0.5)
    #plt.ylabel('$(C_{sed})_{model}$')
    #for n in np.arange(0.75,5.75,1.):
    #    plt.axvline(n*P,color='r',ls='--',alpha=0.3)
    #plt.setp(ax.get_xticklabels(), visible=False)
    #plt.xlim(-0.5,13.5)
    
    plt.suptitle('$\Delta$ : FAST$-$model',y=0.95,fontsize=12)
    
    #plt.savefig('New/Perturb/ST_exp_M%s.png'%m,dpi=400,bbox_inches='tight')
    plt.show()
    
# =============================================================================
# Observed Noise
# =============================================================================
S=50
#S=100
for k,(m, mi,c) in enumerate(zip(M_today.astype('str'),range(M_today.size),M_color)):
    C = 0.3
    t = np.linspace(0.5,11.5,111)
    
    fig = plt.figure(figsize=(8,6))
    ax=plt.subplot(212)
    ax2=plt.subplot(211)
    axins = inset_axes(ax, 1.4, 1., bbox_to_anchor=(0.55,0.65),  
                       bbox_transform=ax.transAxes)
    plt.tick_params(axis='both', which='major', labelsize=8) 
    axins2 = inset_axes(ax, 1.4, 1., bbox_to_anchor=(0.55,0.65),  
                       bbox_transform=ax2.transAxes)
    plt.tick_params(axis='both', which='major', labelsize=8)  
    
    sfr = np.interp(t, Data[m].t[::-1], Data[m].SFR[::-1])    
    
    if ConstP:
        sfr_prtb = sfr*10**(-C*np.sin(2*np.pi*t/P + Phases[p_id]))
        s=ax.scatter(t, sfr_prtb,
                      c=np.mod(2*np.pi*t/P + Phases[p_id],2*np.pi) /np.pi,
                      marker='o',s=15,cmap='gnuplot_r',alpha=0.8)
        for n in np.arange(0.4,5.4,1.):
            ax.axvline(n*P,color='r',ls='--',alpha=0.3)
    else:
        sfr_prtb = sfr*10**(-C*np.sin(5*np.pi*np.log(t) + Phases[p_id]))
        s=ax.scatter(t[(t>=2) & (t<=9)], sfr_prtb[(t>=2) & (t<=9)],
#                      c=np.mod(5*np.pi*np.log(t[(t>=2) & (t<=9)]) + 3*np.pi/4.,2*np.pi) /np.pi,
                        c=(np.log10(sfr_prtb)-np.log10(sfr))[(t>=2) & (t<=9)],
                      marker='o',s=60,cmap='RdYlBu',alpha=0.8)
        s=ax.scatter(t[(t<2) | (t>9)], sfr_prtb[(t<2) | (t>9)],
                      c="gray",marker='o',s=60,alpha=0.8)
        for n in np.arange(0.15,2.55,0.4):
            ax.axvline(np.e**n,color='steelblue',ls='--',alpha=0.5)
            ax2.axvline(np.e**n,color='steelblue',ls='--',alpha=0.5)
            axins.axvline(np.e**n,color='steelblue',ls='--',alpha=0.5,lw=1)
            axins2.axvline(np.e**n,color='steelblue',ls='--',alpha=0.5,lw=1)

    ax.semilogy(t, sfr, '--', color=c, lw=3,alpha=0.7)
    ax.semilogy(t, sfr_prtb, '-', color=c, lw=3.,alpha=0.7)
    ax.set_xlim(0,11.5)
    ax.set_ylim(sfr.min()/2,sfr.max()*5)
    ax.text(0.75,0.1,'M$_{today}$ = 10$^{%s}$'%m,transform=ax.transAxes,fontsize=15)
    ax.set_xlabel('Cosmic Time (Gyr)')
    ax.set_ylabel('SFR (M$\odot$/yr)$_{model}$')
    
    axins.plot(t, sfr, '--', color=c, alpha=0.5)
    axins.plot(t, sfr_prtb, '-', color=c, alpha=0.5)
    axins.set_xlim(0.,11.5)
    
    ax2.set_xlim(0,11.5)
    ax2.semilogy(t, sfr, '--', color=c, lw=3,alpha=0.7)
    ax2.scatter(10**np.tile(lgAge,S)/1e9,10**SFR_FAST2[mi::5],facecolors="None",edgecolors=c,s=3,alpha=0.3,zorder=0)
    ax2.set_ylabel('SFR (M$\odot$/yr)$_{FAST}$')
    ax2.set_ylim(sfr.min()/2,sfr.max()*5)   
    axins2.scatter(10**np.tile(lgAge,S)/1e9,10**SFR_FAST2[mi::5],facecolors="None",edgecolors=c,s=1,alpha=0.1,zorder=0)
    axins2.plot(t, sfr, '--', color=c, alpha=0.5)
    axins2.set_xlim(0.,11.5)

    
    if ConstP:
        peaks=[29,54,78,104]
    else:
        peaks=[21,33,52,81]
#    plt.scatter(t[peaks[0]],sfr_prtb[peaks[0]],s=30,color='k',marker='x')
#    plt.scatter(t[peaks[1]],sfr_prtb[peaks[1]],s=30,color='k',marker='x')
#    plt.scatter(t[peaks[2]],sfr_prtb[peaks[2]],s=30,color='k',marker='x')
#    plt.scatter(t[peaks[3]],sfr_prtb[peaks[3]],s=30,color='k',marker='x')
        
    colors = np.tile((np.log10(sfr_prtb)-np.log10(sfr)),S)
    labels = ['$\Delta$ log($M_*$/$M_{\odot}$)', 
              '$\Delta$ log(SSFR/$yr^{-1})$', 
              '$\Delta{\ }A_v$']
    for i, (yl, model, FAST,FAST_0) in enumerate(zip(labels, [lgMs,lgSSFR,np.ones_like(lgMs)],
                                                      [M_FAST2,SSFR_FAST2,Av_FAST2],
                                                      [M_FAST,SSFR_FAST,Av_FAST])):
        ax = fig.add_axes([0.125,0.96+0.2*i, 0.775, 0.18])
        if ConstP:
            if i==0:
                plt.plot(10**(lgAge-9), (FAST-model)[mi::5],alpha=0.7)
            else:
                s = plt.scatter(10**(lgAge-9), (FAST-model)[mi::5], 
                    c=np.mod(2*np.pi*10**(lgAge-9)/P + Phases[p_id],2*np.pi) /np.pi,
                    cmap='gnuplot_r', s=20, alpha=0.7)
            for n in np.arange(0.4,5.4,1.):
                plt.axvline(n*P,color='r',ls='--',alpha=0.3)
        else:
            models = np.tile(model,S)
            lgAgeS = np.tile(lgAge,S)
            age_conds = np.tile(age_cond,S)
            age_cond2s = np.tile(age_cond2,S)
            
            phase = np.mod(5*np.pi*np.log(10**(lgAgeS[age_cond2s]-9)) + Phases[p_id],2*np.pi) /np.pi
            s = plt.scatter(10**(lgAgeS[age_cond2s]-9), (FAST-models)[mi::5][age_conds[mi::5]], 
                            c=colors[age_cond2s],                            
                            cmap='RdYlBu', s=40, alpha=0.5)
            s = plt.scatter(10**(lgAgeS[~age_cond2s]-9), (FAST-models)[mi::5][~age_conds[mi::5]], 
                            c="gray",s=40, alpha=0.5)

            plt.plot(10**(lgAge-9), (FAST_0-model)[mi::5], 
                            c="gold", ls="-", lw=3,alpha=0.9)
            
            for n in np.arange(0.15,2.55,0.4):
                plt.axvline(np.e**n,color='steelblue',ls='--',alpha=0.5)
            if (i==0)|(i==1):
                plt.ylim(-1.,.75)        
        y = pd.Series((FAST-models)[mi::5][age_condms])
        y = y[abs(y)<1.]
#        y2 = pd.Series((FAST-models)[mi::5])
#        y2 = y2[abs(y2)<1.]
        plt.text(0.54,0.08,'$\sigma$ = %.2f'%y.std(),
                fontsize=12,transform=ax.transAxes,color="orange") 
#        plt.text(0.8,0.08,'$\sigma$ = %.2f'%y2.std(),
#                fontsize=12,transform=ax.transAxes,color="gray")  
        plt.axhline(0.,color='k',ls='--',alpha=0.7)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.ylabel(yl,fontsize=10)   
        plt.xlim(0,11.5)

    #ax = fig.add_axes([0.125,0.95+0.28*4, 0.775, 0.25])
    #lgSSFR_pred=-2.28*Csed**2-0.8*Csed-8.41
    #plt.plot(Ages,(SSFR_FAST.reshape((76,6))-lgSSFR_pred)[mi],c='m',alpha=0.5)
    #plt.ylabel('($\Delta$ lgSSFR)$_{mod - mod/Csed}$')
    #plt.plot(Ages,(UV_st-UV)[mi],c='m',alpha=0.5)
    #plt.ylabel('(U-V)$_{model}$')
    #plt.plot(Ages,(Csed_2P5-Csed)[mi],c='k',alpha=0.5)
    #plt.ylabel('$(C_{sed})_{model}$')
    #for n in np.arange(0.75,5.75,1.):
    #    plt.axvline(n*P,color='r',ls='--',alpha=0.3)
    #plt.setp(ax.get_xticklabels(), visible=False)
    #plt.xlim(-0.5,13.5)
    
    plt.suptitle('$\Delta$ : FAST$-$model',y=0.95,fontsize=12)
    
    #plt.savefig('New/Perturb/OBS_ST_exp_M%s.png'%m,dpi=400,bbox_inches='tight')
    plt.show()
    
    
    
# =============================================================================
# Different phase
# =============================================================================
A, P = 0.3, 2.5
p_id = 0 

BC03_outs = [asc.read('N/Aldo/ST/ST%s_p%s CSP result.txt'%(A,Phases_rd[p_id])) \
             for p_id in [0,1,2,3]]
tables = [ascii.read('N/Aldo/ST/Aldo_exp_ST%s_p%s.fout'%(A,p_id),header_start=16).to_pandas() \
          for p_id in [0,1,2,3]]
table2s = [ascii.read('N/Aldo/ST/Aldo_exp_ST%s_p%s_obs.fout'%(A,p_id),header_start=16).to_pandas() \
           for p_id in [0,1,2,3]]


S=100
for k,(m, mi,c) in enumerate(zip(M_today.astype('str'),range(M_today.size),M_color)):
    if k!=3: continue
    fig = plt.figure(figsize=(8,6))
    ax=plt.subplot(212)
    ax2=plt.subplot(211)
    axins = inset_axes(ax, 1.4, 1., bbox_to_anchor=(0.55,0.65),  
                       bbox_transform=ax.transAxes)
    plt.tick_params(axis='both', which='major', labelsize=8)
    axi = [fig.add_axes([0.125,0.96+0.2*i, 0.775, 0.18]) for i in range(3)]
#    axins2 = inset_axes(ax, 1.4, 1., bbox_to_anchor=(0.55,0.65),  
#                       bbox_transform=ax2.transAxes)
#    plt.tick_params(axis='both', which='major', labelsize=8)  
    
    for p_id in [0,1,2,3]:
        
        lgMs = BC03_outs[p_id]['lgM*'].reshape((M_today.size, lgAge.size)).T.ravel()
        lgSSFR = BC03_outs[p_id]['lgsSFR'].reshape((M_today.size, lgAge.size)).T.ravel()
        lgSFR = lgSSFR + lgMs
                         
        lgAges = np.array([lgAge for k in M_today]).T.ravel()
        M_class =  np.array([M_today for T in range(lgAge.size)]).ravel()  
                
        Ages_FAST = tables[p_id].lage
        SFH_FAST = tables[p_id].ltau
        M_FAST = tables[p_id].lmass
        SSFR_FAST = tables[p_id].lssfr
        SFR_FAST = tables[p_id].lsfr
        Av_FAST = tables[p_id].Av    
        chi2_FAST = tables[p_id].chi2
        
        age_cond = (lgAges>np.log10(2e9))&(lgAges<np.log10(9.e9))
        age_cond2 = (lgAge>np.log10(2e9))&(lgAge<np.log10(9.e9))
        
        Ages_FAST2 = table2s[p_id].lage
        SFH_FAST2 = table2s[p_id].ltau
        M_FAST2 = table2s[p_id].lmass
        SSFR_FAST2 = table2s[p_id].lssfr
        SFR_FAST2 = table2s[p_id].lsfr
        Av_FAST2 = table2s[p_id].Av    
        chi2_FAST2 = table2s[p_id].chi2
    
        C = 0.3
        t = np.linspace(0.5,11.5,111)
        
        sfr = np.interp(t, Data["%s"%m].t[::-1], Data["%s"%m].SFR[::-1])    
        
        if ConstP:
            sfr_prtb = sfr*10**(-C*np.sin(2*np.pi*t/P + Phases[p_id]))
            s=ax.scatter(t, sfr_prtb,
                          c=np.mod(2*np.pi*t/P + 3*np.pi/4.,2*np.pi) /np.pi,
                          marker='o',s=15,cmap='gnuplot_r',alpha=0.8)
            for n in np.arange(0.4,5.4,1.):
                ax.axvline(n*P,color='r',ls='--',alpha=0.3)
        else:
            sfr_prtb = sfr*10**(-C*np.sin(5*np.pi*np.log(t) + Phases[p_id]))
            s=ax.scatter(t[(t>=2) & (t<=9)], sfr_prtb[(t>=2) & (t<=9)],
                            c=(np.log10(sfr_prtb)-np.log10(sfr))[(t>=2) & (t<=9)],
                          marker='o',s=10,cmap='RdYlBu',alpha=0.5)
            s=ax.scatter(t[(t<2) | (t>9)], sfr_prtb[(t<2) | (t>9)],
                          c="gray",marker='o',s=10,alpha=0.5)

        ax.semilogy(t, sfr, '--', color=c, lw=1,alpha=0.3)
        ax.semilogy(t, sfr_prtb, '-', color=c, lw=1,alpha=0.3)
        if k==0:
            ax.set_xlim(0,11.5)
            ax.set_ylim(sfr.min()/2,sfr.max()*5)
            ax.text(0.72,0.1,'M$_{today}$ = 10$^{%s}$'%m,transform=ax.transAxes,fontsize=15)
            ax.set_xlabel('Cosmic Time (Gyr)')
            ax.set_ylabel('SFR (M$\odot$/yr)$_{model}$')
        
        axins.plot(t, sfr, '--', color=c, alpha=0.5)
        axins.plot(t, sfr_prtb, '-', color=c, alpha=0.5)
        axins.set_xlim(0.,11.5)
        
        ax2.set_xlim(0,11.5)
        ax2.semilogy(t, sfr, '--', color=c, lw=3,alpha=0.7)
        ts= 10**np.tile(lgAge,S)/1e9
        sfr_FAST = 10**SFR_FAST2[mi::5]
        
        colors = np.tile((np.log10(sfr_prtb)-np.log10(sfr)),S)
        
        ax2.scatter(ts[(ts>=2) & (ts<=9)],sfr_FAST[(ts>=2) & (ts<=9)],
                    c=colors[(ts>=2) & (ts<=9)],
                    marker='o',cmap='RdYlBu',s=5,alpha=0.3,zorder=0)
        ax2.scatter(ts[(ts<2) | (ts>9)], sfr_FAST[(ts<2) | (ts>9)],
                      c="gray",marker='o',s=5,alpha=0.3)
        ax2.set_ylabel('SFR (M$\odot$/yr)$_{FAST}$')
        ax2.set_ylim(sfr.min()/2,sfr.max()*5)   
#        axins2.scatter(10**np.tile(lgAge,S)/1e9,10**SFR_FAST2[mi::5],
#                       facecolors="None",edgecolors=c,s=1,alpha=0.1,zorder=0)
#        axins2.plot(t, sfr, '--', color=c, alpha=0.5)
#        axins2.set_xlim(0.,11.5)
    
        if ConstP:
            peaks=[29,54,78,104]
        else:
            peaks=[21,33,52,81]
    
            
#        colors = np.tile((np.log10(sfr_prtb)-np.log10(sfr)),S)
        labels = ['$\Delta$ log($M_*$/$M_{\odot}$)', 
                  '$\Delta$ log(SSFR/$yr^{-1})$', 
                  '$\Delta{\ }A_v$']
        for i, (yl, model, FAST,FAST_0) in enumerate(zip(labels, [lgMs,lgSSFR,np.ones_like(lgMs)],
                                                          [M_FAST2,SSFR_FAST2,Av_FAST2],
                                                          [M_FAST,SSFR_FAST,Av_FAST])):
            if ConstP:
                if i==0:
                    plt.plot(10**(lgAge-9), (FAST-model)[mi::5],alpha=0.7)
                else:
                    s = plt.scatter(10**(lgAge-9), (FAST-model)[mi::5], 
                        c=np.mod(2*np.pi*10**(lgAge-9)/P + Phases[p_id],2*np.pi) /np.pi,
                        cmap='gnuplot_r', s=20, alpha=0.7)
                for n in np.arange(0.4,5.4,1.):
                    plt.axvline(n*P,color='r',ls='--',alpha=0.3)
            else:
                models = np.tile(model,S)
                lgAgeS = np.tile(lgAge,S)
                age_conds = np.tile(age_cond,S)
                age_cond2s = np.tile(age_cond2,S)
                
                phase = np.mod(5*np.pi*np.log(10**(lgAgeS[age_cond2s]-9)) + Phases[p_id],2*np.pi) /np.pi
                s = axi[i].scatter(10**(lgAgeS[age_cond2s]-9), (FAST-models)[age_conds][mi::5], 
                                c=colors[age_cond2s],                            
                                cmap='RdYlBu', s=5, alpha=0.3)
                s = axi[i].scatter(10**(lgAgeS[~age_cond2s]-9), (FAST-models)[~age_conds][mi::5], 
                                c="gray",s=5, alpha=0.3)
    
                axi[i].plot(10**(lgAge-9), (FAST_0-model)[mi::5], 
                                c="gold", ls="-", lw=1,alpha=0.7)

                if (i==0)|(i==1):
                    plt.ylim(-1.,.75)        
            y = pd.Series((FAST-models)[age_conds][mi::5])
            y = y[abs(y)<1.]
            y2 = pd.Series((FAST-models)[mi::5])
            y2 = y2[abs(y2)<1.]
            axi[i].text(0.54,0.08,'$\sigma$ = %.2f'%y.std(),
                    fontsize=12,transform=axi[i].transAxes,color="orange") 
            axi[i].text(0.8,0.08,'$\sigma$ = %.2f'%y2.std(),
                    fontsize=12,transform=axi[i].transAxes,color="gray")  
            axi[i].axhline(0.,color='k',ls='--',alpha=0.7)
            axi[i].set_ylabel(yl,fontsize=10)   
            axi[i].set_xlim(0,11.5)
            if i==1: 
                axi[i].set_ylim(-1.,1.)
            else:
                axi[i].set_ylim(-.5,.5)
            plt.setp(axi[i].get_xticklabels(), visible=False)

    plt.suptitle('$\Delta$ : FAST$-$model',y=0.95,fontsize=12)
        
    #plt.savefig('N/Aldo/ST/OBS_ST_exp_M%s.png'%m,dpi=300,bbox_inches='tight')
    plt.show()
    
# =============================================================================
# Snapshot
# =============================================================================
snap_t = 5*1e9

snap_cond = (lgAges>np.log10(snap_t-0.5e9))&(lgAges<np.log10(snap_t+0.5e9))
snap_cond2 = (lgAge>np.log10(snap_t-0.5e9))&(lgAge<np.log10(snap_t+0.5e9))
snap_conds = np.tile(snap_cond,S)
snap_cond2s = np.tile(snap_cond2,S)

m = "10.75"
mi = 3
for p_id in [0,1,2,3]:
    
    lgMs = BC03_outs[p_id]['lgM*'].reshape((M_today.size, lgAge.size)).T.ravel()
    lgSSFR = BC03_outs[p_id]['lgsSFR'].reshape((M_today.size, lgAge.size)).T.ravel()
    lgSFR = lgSSFR + lgMs
                     
    lgAges = np.array([lgAge for k in M_today]).T.ravel()
    M_class =  np.array([M_today for T in range(lgAge.size)]).ravel()  
            
    Ages_FAST = tables[p_id].lage
    SFH_FAST = tables[p_id].ltau
    M_FAST = tables[p_id].lmass
    SSFR_FAST = tables[p_id].lssfr
    SFR_FAST = tables[p_id].lsfr
    Av_FAST = tables[p_id].Av    
    chi2_FAST = tables[p_id].chi2
    
    age_cond = (lgAges>np.log10(2e9))&(lgAges<np.log10(9.e9))
    age_cond2 = (lgAge>np.log10(2e9))&(lgAge<np.log10(9.e9))
    
    Ages_FAST2 = table2s[p_id].lage
    SFH_FAST2 = table2s[p_id].ltau
    M_FAST2 = table2s[p_id].lmass
    SSFR_FAST2 = table2s[p_id].lssfr
    SFR_FAST2 = table2s[p_id].lsfr
    Av_FAST2 = table2s[p_id].Av    
    chi2_FAST2 = table2s[p_id].chi2

    sfr = np.interp(t, Data["%s"%m].t[::-1], Data["%s"%m].SFR[::-1])    
    sfr_prtb = sfr*10**(-C*np.sin(5*np.pi*np.log(t) + Phases[p_id]))
    
    colors = np.tile((np.log10(sfr_prtb)-np.log10(sfr)),S)

    plt.scatter((M_FAST2-np.tile(lgMs,S))[snap_conds][mi::5],
                (SFR_FAST2-np.tile(lgSFR,S))[snap_conds][mi::5],  
                c=colors[snap_cond2s], s=10,cmap='RdYlBu',alpha=0.5)
    plt.axhline(0,ls="--")
    plt.axvline(0,ls="--")
#    plt.xlim(-0.4,0.4)
#    plt.ylim(-.8,.8)
    

# =============================================================================
# Different phase + snapshot
# =============================================================================
A, P = 0.3, 2.5
p_id = 0 

BC03_outs = [asc.read('N/Aldo/ST/ST%s_p%s CSP result.txt'%(A,Phases_rd[p_id])) \
             for p_id in [0,1,2,3]]
tables = [ascii.read('N/Aldo/ST/Aldo_exp_ST%s_p%s.fout'%(A,p_id),header_start=16).to_pandas() \
          for p_id in [0,1,2,3]]
table2s = [ascii.read('N/Aldo/ST/Aldo_exp_ST%s_p%s_obs.fout'%(A,p_id),header_start=16).to_pandas() \
           for p_id in [0,1,2,3]]

#snap
snap_t = 5*1e9

S=100
snap_cond = (lgAges>np.log10(snap_t-0.5e9))&(lgAges<np.log10(snap_t+0.5e9))
snap_cond2 = (lgAge>np.log10(snap_t-0.5e9))&(lgAge<np.log10(snap_t+0.5e9))
snap_conds = np.tile(snap_cond,S)
snap_cond2s = np.tile(snap_cond2,S)

m = "10.75"
mi = 3

# Plot
for k,(m, mi,c) in enumerate(zip(M_today.astype('str'),range(M_today.size),M_color)):
    if k!=3: continue
    fig = plt.figure(figsize=(8,6))
    ax=plt.subplot(212)
    axins = inset_axes(ax, 1.4, 1., bbox_to_anchor=(0.55,0.65),  
                       bbox_transform=ax.transAxes)
    plt.tick_params(axis='both', which='major', labelsize=8)
    axi = [fig.add_axes([0.125,0.5+0.21*i, 0.775, 0.18]) for i in range(3)]
#    axins2 = inset_axes(ax, 1.4, 1., bbox_to_anchor=(0.55,0.65),  
#                       bbox_transform=ax2.transAxes)
#    plt.tick_params(axis='both', which='major', labelsize=8)  
    y_all = pd.Series([])
    y2_all = pd.Series([])
    for p_id in [0,1,2,3]:
        
        lgMs = BC03_outs[p_id]['lgM*'].reshape((M_today.size, lgAge.size)).T.ravel()
        lgSSFR = BC03_outs[p_id]['lgsSFR'].reshape((M_today.size, lgAge.size)).T.ravel()
        lgSFR = lgSSFR + lgMs
                         
        lgAges = np.array([lgAge for k in M_today]).T.ravel()
        M_class =  np.array([M_today for T in range(lgAge.size)]).ravel()  
                
        Ages_FAST = tables[p_id].lage
        SFH_FAST = tables[p_id].ltau
        M_FAST = tables[p_id].lmass
        SSFR_FAST = tables[p_id].lssfr
        SFR_FAST = tables[p_id].lsfr
        Av_FAST = tables[p_id].Av    
        chi2_FAST = tables[p_id].chi2
        
        age_cond = (lgAges>np.log10(2e9))&(lgAges<np.log10(9.e9))
        age_cond2 = (lgAge>np.log10(2e9))&(lgAge<np.log10(9.e9))
        
        Ages_FAST2 = table2s[p_id].lage
        SFH_FAST2 = table2s[p_id].ltau
        M_FAST2 = table2s[p_id].lmass
        SSFR_FAST2 = table2s[p_id].lssfr
        SFR_FAST2 = table2s[p_id].lsfr
        Av_FAST2 = table2s[p_id].Av    
        chi2_FAST2 = table2s[p_id].chi2
    
        C = 0.3
        t = np.linspace(0.5,11.5,111)
        
        sfr = np.interp(t, Data["%s"%m].t[::-1], Data["%s"%m].SFR[::-1])
        sfr_prtb = sfr*10**(-C*np.sin(5*np.pi*np.log(t) + Phases[p_id]))
        s=ax.scatter(t[(t>=2) & (t<=9)], sfr_prtb[(t>=2) & (t<=9)],
                        c=(np.log10(sfr_prtb)-np.log10(sfr))[(t>=2) & (t<=9)],
                      marker='o',s=8,cmap='RdYlBu',alpha=0.9,zorder=2)
        s=ax.scatter(t[(t<2) | (t>9)], sfr_prtb[(t<2) | (t>9)],
                      c="gray",marker='o',s=8,alpha=0.9,zorder=2)

        ax.semilogy(t, sfr, '--', color=c, lw=1.5,alpha=0.7,zorder=1)
        ax.semilogy(t, sfr_prtb, '-', color=c, lw=1.5,alpha=0.7,zorder=1)
        
        if p_id==3:
            ax.set_xlim(0,11.5)
            ax.set_ylim(sfr.min()/2,sfr.max()*5)
            ax.text(0.68,0.1,'M$_{today}$/M$_{\odot}$ = 10$^{%s}$'%m,transform=ax.transAxes,fontsize=14)
            ax.set_xlabel('Cosmic Time (Gyr)')
            ax.set_ylabel('SFR (M$\odot$/yr)$_{model}$')
        
        axins.plot(t, sfr, '--', color=c, alpha=0.5)
        axins.plot(t, sfr_prtb, '-', color=c, alpha=0.5)
        axins.set_xlim(0.,11.5)

        colors = np.tile((np.log10(sfr_prtb)-np.log10(sfr)),S)    

#        axins2.scatter(10**np.tile(lgAge,S)/1e9,10**SFR_FAST2[mi::5],
#                       facecolors="None",edgecolors=c,s=1,alpha=0.1,zorder=0)
#        axins2.plot(t, sfr, '--', color=c, alpha=0.5)
#        axins2.set_xlim(0.,11.5)
    
        peaks=[21,33,52,81]
              
        labels = ['$\Delta$ log($M_*$/$M_{\odot}$)', 
                  '$\Delta$ log(SSFR/$yr^{-1})$', 
                  '$\Delta{\ }A_v$']
        for i, (yl, model, FAST,FAST_0) in enumerate(zip(labels, [lgMs,lgSSFR,np.ones_like(lgMs)],
                                                          [M_FAST2,SSFR_FAST2,Av_FAST2],
                                                          [M_FAST,SSFR_FAST,Av_FAST])):
            models = np.tile(model,S)
            lgAgeS = np.tile(lgAge,S)
            age_conds = np.tile(age_cond,S)
            age_cond2s = np.tile(age_cond2,S)
            
            phase = np.mod(5*np.pi*np.log(10**(lgAgeS[age_cond2s]-9)) + Phases[p_id],2*np.pi) /np.pi
            s = axi[i].scatter(10**(lgAgeS[age_cond2s]-9), (FAST-models)[age_conds][mi::5], 
                            c=colors[age_cond2s],                            
                            cmap='RdYlBu', s=3, alpha=0.3,zorder=1)
            s = axi[i].scatter(10**(lgAgeS[~age_cond2s]-9), (FAST-models)[~age_conds][mi::5], 
                            c="gray",s=3, alpha=0.3,zorder=1)

            axi[i].plot(10**(lgAge-9), (FAST_0-model)[mi::5], 
                            c="gold", ls="-", lw=1,alpha=0.7)

            if (i==0)|(i==1):
                plt.ylim(-1.,.75)        
            y = pd.Series((FAST-models)[age_conds][mi::5])
            y = y[abs(y)<1.]
            y2 = pd.Series((FAST-models)[mi::5])
            y2 = y2[abs(y2)<1.]
            y_all = y_all.append(y)
            y2_all = y2_all.append(y2)
            if p_id==3:
                axi[i].text(0.54,0.08,'$\sigma$ = %.2f'%y_all.std(),
                            fontsize=12,transform=axi[i].transAxes,color="orange") 
                axi[i].text(0.8,0.08,'$\sigma$ = %.2f'%y2_all.std(),
                            fontsize=12,transform=axi[i].transAxes,color="gray") 
                axi[i].axhline(0.,color='k',ls='--',alpha=0.9)
                axi[i].set_ylabel(yl,fontsize=10)   
                axi[i].set_xlim(0,11.5)
            if i==1: 
                axi[i].set_ylim(-1.,1.)
            else:
                axi[i].set_ylim(-.5,.5)
            plt.setp(axi[i].get_xticklabels(), visible=False)

    #plt.suptitle('$\Delta$ : FAST$-$model',y=0.95,fontsize=12)
    for j,(snap_t,T) in enumerate(zip([3*1e9,5*1e9,7*1e9],["3","5","7"])):
        snap_cond = (lgAges>np.log10(snap_t-0.5e9))&(lgAges<np.log10(snap_t+0.5e9))
        snap_cond2 = (lgAge>np.log10(snap_t-0.5e9))&(lgAge<np.log10(snap_t+0.5e9))
        snap_conds = np.tile(snap_cond,S)
        snap_cond2s = np.tile(snap_cond2,S)
        ax2=fig.add_axes((0.14+j*0.28,1.18,0.18,0.2))
        ax2.scatter((M_FAST2-np.tile(lgMs,S))[snap_conds][mi::5],
                    (SFR_FAST2-np.tile(lgSFR,S))[snap_conds][mi::5],  
                    c=colors[snap_cond2s], s=5,cmap='RdYlBu',alpha=0.3)
        ax2.axhline(0,color="k",ls="--",lw=1,alpha=0.9)
        ax2.axvline(0,color="k",ls="--",lw=1,alpha=0.9) 
        ax2.text(0.06,0.15,'t$=%s$ Gyr'%T,transform=ax2.transAxes,color="g",fontsize=10,alpha=0.5)
        if  j==0:
            ax2.set_ylabel('$\Delta$ log(SFR/M$\odot$ yr$^{-1}$)')
        ax2.set_xlabel('$\Delta$ log($M_*$/$M_{\odot}$)')
        xx = np.linspace(ax2.get_xlim()[0],ax2.get_xlim()[1],10)
        ax2.plot(xx,-xx,color="k",lw=1,ls="--",alpha=0.5)
        
    #plt.savefig('N/Aldo/ST/OBS_ST_phase_exp_M%s.png'%m,dpi=300,bbox_inches='tight')
    plt.show()
    