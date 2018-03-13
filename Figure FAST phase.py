# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:32:48 2017

@author: Qing Liu
"""

import numpy as np
import pandas as pd
import asciitable as asc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import asciitable as asc
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, least_squares

lgAge = np.log10(np.linspace(0.5,11.5,111)* 1e9)
M_today = np.array([9.0,9.5,10.0,10.5,11.0,11.5])

phase_start = lgAge[0]

# Real
multiple = False
if multiple:
    lgAge = np.log10(np.linspace(2.,8.,61)* 1e9)
    M_today = np.linspace(9.5,11.5,5)
    

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

#==============================================================================
# Read model result 
#==============================================================================
A, P = 0., 2.5
ConstP = True

BC03_out = asc.read('Aldo/Perturb/Final/CSP/CSP result.txt')
#BC03_out = asc.read('Aldo/Perturb/Final/CSP/P%.1f CSP result.txt'%P)

lgMs = BC03_out['lgM*'].reshape((M_today.size, lgAge.size)).T.ravel()
lgSSFR = BC03_out['lgsSFR'].reshape((M_today.size, lgAge.size)).T.ravel()
lgSFR = lgSSFR + lgMs
                 
lgAges = np.array([lgAge for m in M_today]).T.ravel()
M_class =  np.array([M_today for T in range(lgAge.size)]).ravel()

if multiple:
    N_sample = 10
    lgMs = np.concatenate([lgMs for k in range(N_sample)])
    lgSSFR = np.concatenate([lgSSFR for k in range(N_sample)])
    lgSFR = np.concatenate([(lgSSFR+lgMs) for k in range(N_sample)])
    lgAges = np.concatenate([lgAges for k in range(N_sample)])
    M_class = np.concatenate([M_class for k in range(N_sample)])

#SSFR_model=lgSSFR.reshape((lgAge.size,M_today.size)).T
#==============================================================================
# Read FAST SED-fitting result
#==============================================================================
table = asc.read('Aldo/Perturb/Final/Aldo_exp.fout')
#table = asc.read('Aldo/Perturb/Final/Aldo_exp_P%.1f.fout'%P)

Ages_FAST = table.lage
SFH_FAST = table.ltau
M_FAST = table.lmass
SSFR_FAST = table.lssfr
SFR_FAST = table.lsfr
Av_FAST = table.Av    
chi2_FAST = table.chi2

#==============================================================================
# Plot
#==============================================================================
for m, mi in zip(M_today.astype('str'),range(M_today.size)):
    C = 0.
    t = np.linspace(0.5,13.0,126)
    t = np.linspace(0.5,11.5,111)
    fig = plt.figure(figsize=(8,3))

    sfr = np.interp(t, Data[m].t[::-1], Data[m].SFR[::-1])
    plt.semilogy(t, sfr, 'b-',
                 label='log M$_{today}$ = %s'%m,alpha=0.5)
    if ConstP:
        sfr_prtb = sfr*10**(-C*np.sin(2*np.pi*t/P + 3*np.pi/4.))
        s=plt.scatter(t, sfr_prtb,
                      c=np.mod(2*np.pi*t/P + 3*np.pi/4.,2*np.pi) /np.pi,
                      marker='o',s=15,cmap='gnuplot_r',alpha=0.8)
        for n in np.arange(0.4,5.4,1.):
            plt.axvline(n*P,color='r',ls='--',alpha=0.3)
    else:
        sfr_prtb = sfr*10**(-C*np.sin(5*np.pi*np.log(t) + 3*np.pi/4.))
        s=plt.scatter(t, sfr_prtb,
                      c=np.mod(5*np.pi*np.log(t) + 3*np.pi/4.,2*np.pi) /np.pi,
                      marker='o',s=15,cmap='gnuplot_r',alpha=0.8)
        for n in np.arange(0.15,2.55,0.4):
            plt.axvline(np.e**n,color='r',ls='--',alpha=0.3)

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
    plt.legend(loc=4, fontsize=10, frameon=True, facecolor='w')
    plt.xlim(-0.5,12.5)
    plt.xlabel('Time since Big Bang (Gyr)')
    plt.ylabel('SFR (M$\odot$/yr)')

    labels = ['$\chi^2$','lgT$_{FAST}$','$\Delta$lg($M_*$/$M_{\odot}$)',
              '$\Delta$lg(SFR/$M_{\odot}yr^{-1})$', '$\Delta$lg(SSFR/$yr^{-1})$', '$\Delta Av$']
    for i, (yl, model, FAST) in enumerate(zip(labels, [np.zeros_like(Av_FAST),np.zeros_like(Av_FAST),
                                                       lgMs,lgSFR,lgSSFR,np.ones_like(Av_FAST)],
                                                      [chi2_FAST,Ages_FAST,M_FAST,SFR_FAST,SSFR_FAST,Av_FAST])):
        ax = fig.add_axes([0.125,0.96+0.4*i, 0.775, 0.36])
        if ConstP:
            if i==0:
                plt.plot(10**(lgAge-9), (FAST-model)[mi::6],alpha=0.7)
            else:
                s = plt.scatter(10**(lgAge-9), (FAST-model)[mi::6], 
                    c=np.mod(2*np.pi*10**(lgAge-9)/P + 3*np.pi/4.,2*np.pi) /np.pi,
                    cmap='gnuplot_r', s=20, alpha=0.7)
            for n in np.arange(0.4,5.4,1.):
                plt.axvline(n*P,color='r',ls='--',alpha=0.3)
        else:
            if i==0:
                plt.plot(10**(lgAge-9), (FAST-model)[mi::6],alpha=0.7)
            else:
                s = plt.scatter(10**(lgAge-9), (FAST-model)[mi::6], 
                            c=np.mod(5*np.pi*np.log(10**(lgAge-9)) + 3*np.pi/4.,2*np.pi) /np.pi,
                            cmap='gnuplot_r', s=20, alpha=0.7)
            for n in np.arange(0.15,2.55,0.4):
                plt.axvline(np.e**n,color='r',ls='--',alpha=0.3)
        plt.axhline(0.,color='k',ls='--',alpha=0.7)
        plt.xlim(-0.5,12.5)
        plt.setp(ax.get_xticklabels(), visible=False)
        if i>1:                
            plt.ylim(-0.7,0.7)
        elif i==0:
            plt.ylim(0.,1.)
        else:
            plt.ylim(7.5,11.)
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
    
    plt.suptitle('$\Delta$:FAST-model',y=0.95,fontsize=10)
    
#    plt.savefig('ST M%s.png'%m,dpi=200,bbox_inches='tight')
    plt.show()