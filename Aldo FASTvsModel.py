# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 23:37:04 2017

@author: Qing Liu
"""
import numpy as np
import pandas as pd
import asciitable as asc
import matplotlib.pyplot as plt
import seaborn as sns

#==============================================================================
# Read model result 
#==============================================================================
# Tachella 16
#lgAge = np.log10(np.linspace(1.,8.5,76)* 1e9)
#M_today = np.array([9.0,9.5,10.0,10.5,11.0])

# Constant P
#lgAge = np.log10(np.linspace(0.5,10.,100)* 1e9)
#M_today = np.array([9.0,9.5,10.0,10.5,11.0,11.5])

# W/ fluctuation
lgAge = np.log10(np.linspace(1.,8.5,76)* 1e9)
M_today = np.array([9.0,9.5,10.0,10.5,11.0])

phase_start = lgAge[0]

# Real
multiple = False
if multiple:
    lgAge = np.log10(np.linspace(2.,8.,61)* 1e9)
    M_today = np.linspace(9.5,11.5,5)

A, P = 0.3, 2.5
BC03_out = asc.read('Aldo/ST/%.1f CSP result.txt'%A)
#BC03_out = asc.read('Aldo/Perturb/ConstP/P%.1f CSP result.txt'%P)
#BC03_out = asc.read('Aldo/0.3 CSP result.txt')

lgMs = BC03_out['lgM*'].reshape((M_today.size, lgAge.size)).T.ravel()
lgSSFR = BC03_out['lgsSFR'].reshape((M_today.size, lgAge.size)).T.ravel()
                 
lgAges = np.array([lgAge for m in M_today]).T.ravel()
M_class =  np.array([M_today for T in range(lgAge.size)]).ravel()

if multiple:
    N_sample = 10
    lgMs = np.concatenate([lgMs for k in range(N_sample)])
    lgSSFR = np.concatenate([lgSSFR for k in range(N_sample)])
    lgAges = np.concatenate([lgAges for k in range(N_sample)])
    M_class = np.concatenate([M_class for k in range(N_sample)])

#SSFR_model=lgSSFR.reshape((lgAge.size,M_today.size)).T
#==============================================================================
# Read FAST SED-fitting result
#==============================================================================
table = asc.read('Aldo/ST/Aldo_exp_ST%.1f.fout'%A)
#table = asc.read('Aldo/Perturb/ConstP/Aldo_exp_P%.1f.fout'%P)
#table = asc.read('Aldo/Aldo_exp.fout')

Ages_FAST = table.lage
SFH_FAST = table.ltau
M_FAST = table.lmass
SSFR_FAST = table.lssfr
Av_FAST = table.Av

#SSFR_fit = SSFR_FAST.reshape((lgAge.size,M_today.size)).T
#Av_fit = Av_FAST.reshape((lgAge.size,M_today.size)).T
#==============================================================================
# Residual
#==============================================================================
#fig,axes = plt.subplots(figsize=(10,10), nrows=2, ncols=2)
#xx=np.linspace(-20,20,20)
#legends = ['log(T/yr)', r'log$(\rm{M_*}/\rm{M_{\odot}})$', r'log(sSFR/yr$^{-1}$)']
#for i, (l, model, FAST) in enumerate(zip(legends,[lgAges,lgMs,lgSSFR],
#                                                 [Ages_FAST,M_FAST,SSFR_FAST])):
#    ax = plt.subplot(2,2,i+1) 
#    plt.plot(xx,np.zeros_like(xx),'k--',alpha=0.7)
#    s = plt.scatter(model, (FAST-model), c=lgAges,
#                    cmap='jet', label=l, s=20, alpha=0.7)
#    plt.xlabel('Model',fontsize=12)
#    plt.ylabel('FAST - Model',fontsize=12)
#    if i==2:
#        plt.xlim(-12.,-7.)
#    elif i==1:
#        plt.xlim(2.,14.)
#    else:
#        plt.xlim(8.5,10.5)
#        plt.xlabel('Time since Big Bang',fontsize=12)
#        plt.ylabel('FAST Age - Time',fontsize=12)
#    plt.ylim(-3.,3.)
#    plt.legend(loc=4, fontsize=12, frameon=True, facecolor='w')
#fig.subplots_adjust(bottom=0.2)
#cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
#colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
#colorbar.set_label('log (Age/yr)')
#ax = plt.subplot(2,2,4)
#plt.hist(Av_FAST,alpha=0.7,label='Av')
#plt.axvline(1.0,color='k',ls='--',alpha=0.7)
#plt.xlabel('FAST',fontsize=12)
#plt.legend(loc='best', fontsize=12, frameon=True, facecolor='w')
#plt.suptitle('AM-SFH Models (p:%.2f dex) vs FAST Fitting with Tau Templates'%A,fontsize=15,y=0.92)
#plt.show()

with sns.axes_style("ticks"):
    fig,axes = plt.subplots(figsize=(11,10), nrows=2, ncols=2)
    labels = ['log($T_{BB}$ / yr)', 'log($M_{*,model}$ / $M_{\odot}$)', 'log($sSFR_{model}$ / $yr^{-1})$']
    for i, (l, model, FAST) in enumerate(zip(labels,[lgAges,lgMs,lgSSFR],
                                                     [Ages_FAST,M_FAST,SSFR_FAST])):
#        color = np.mod(5*np.pi*np.log(10**(lgAges-9))+np.pi,2*np.pi) /np.pi
        color = M_class
        if multiple:
            model = model[M_FAST>9.]
            FAST = FAST[M_FAST>9.]
            color = M_class[M_FAST>9.]
        ax = plt.subplot(2,2,i+1) 
        plt.axhline(0.0,color='k',ls='--',alpha=0.7)
        s = plt.scatter(model, (FAST-model), c=color,
                        cmap='jet', label=l, s=20, alpha=0.7)
        plt.xlabel(l,fontsize=12)
        plt.ylabel('FAST $-$ Model',fontsize=12)
        plt.text(0.1,0.1,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
                 fontsize=15,transform=ax.transAxes)
        if i==2:
            plt.xlim(-12.,-7.)
            plt.ylim(-.5,.5)
        elif i==1:
            plt.xlim(8.,12.)
            plt.ylim(-.2,.2)
        else:
            plt.xlim(9.0,10.0)
            plt.ylabel('T$_{FAST}$ $-$ T$_{BB}$',fontsize=12)
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
    colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
    colorbar.set_label('$M_{today}$')
#    colorbar.set_label('$Phase$')
    ax = plt.subplot(2,2,4)
    if multiple:
        s = plt.scatter(Av_FAST[M_FAST>9.],(SSFR_FAST-lgSSFR)[M_FAST>9.],c=color,s=20,cmap='jet')
    else:       
        s = plt.scatter(Av_FAST,SSFR_FAST-lgSSFR,c=color,s=20,cmap='jet')
    plt.axvline(1.0,color='k',ls='--')
    plt.xlabel('FAST Av',fontsize=12)
    plt.ylabel('$SSFR_{FAST} - SSFR_{model}$',fontsize=12)
    with sns.axes_style("white"):
        nx = fig.add_axes([0.8, 0.25, 0.08, 0.08])
        nx.set_yticks([])
        nx.hist(Av_FAST,alpha=0.7)
        nx.axvline(1.0,color='k',ls='--',alpha=0.7)
    plt.suptitle('AM-SFH Models (Period:%.1f Gyr) vs FAST Fitting with Tau Templates'%P,fontsize=15,y=0.92)
    plt.show()
    
#==============================================================================
# One-One
#==============================================================================
fig,axes = plt.subplots(figsize=(10,10), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = ['log(T/Gyr)', r'log$(\rm{M_*}/\rm{M_{\odot}})$', r'log(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[lgAges,lgMs,lgSSFR],
                                                 [Ages_FAST,M_FAST,SSFR_FAST])):
    color = lgAges
#    color = np.mod(5*np.pi*np.log(10**(lgAges-9))+np.pi,2*np.pi) /np.pi
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.5)
    if multiple:
        model = model[M_FAST>9.]
        FAST = FAST[M_FAST>9.]
        color = lgAges[M_FAST>9.]
    s = plt.scatter(model, FAST, c=color,
                    cmap='jet', label=l, s=20, alpha=0.7)
    plt.xlabel('Model',fontsize=12)
    plt.ylabel('FAST',fontsize=12)
    if i==2:
        plt.xlim(-12.,-7.)
        plt.ylim(-12.,-7.)
    elif i==1:
        plt.xlim(8.,12.)
        plt.ylim(8.,12.)
    else:
        plt.xlim(8.,11.)
        plt.ylim(8.,11.)
        plt.xlabel('Age of Model',fontsize=12)
        plt.ylabel('Age return by FAST',fontsize=12)
    plt.legend(loc=4, fontsize=12, frameon=True, facecolor='w')
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
colorbar.set_label('log (Age/yr)')
#colorbar.set_label('Phase')
ax = plt.subplot(2,2,4)
plt.hist(Av_FAST,alpha=0.7,label='Av')
plt.axvline(1.0,color='k',ls='--')
plt.xlabel('FAST',fontsize=12)
plt.legend(loc='best', fontsize=12, frameon=True, facecolor='w')
plt.suptitle('AM-SFH Models (Period:%.1f Gyr) vs FAST Fitting with Tau Templates'%P,fontsize=15,y=0.92)
plt.show()

#==============================================================================
# phase
#==============================================================================
#with sns.axes_style("ticks"):
#    fig,axes = plt.subplots(figsize=(11,10), nrows=2, ncols=2)
#    labels = ['log($T_{BB}$ / yr)', 'log($M_{*,model}$ / $M_{\odot}$)', 'log($sSFR_{model}$ / $yr^{-1})$']
#    for i, (l, model, FAST) in enumerate(zip(labels,[lgAges,lgMs,lgSSFR],
#                                                     [Ages_FAST,M_FAST,SSFR_FAST])):
#        color = np.mod(5*np.pi*np.log(10**(lgAges-9))+np.pi,2*np.pi) /np.pi
#        #color = M_class
#        if multiple:
#            model = model[M_FAST>9.]
#            FAST = FAST[M_FAST>9.]
#            color = M_class[M_FAST>9.]
#        ax = plt.subplot(2,2,i+1) 
#        plt.axhline(0.0,color='k',ls='--',alpha=0.7)
#        #s = plt.scatter(model, (FAST-model), c=color,
#         #               cmap='jet', label=l, s=20, alpha=0.7)
#        for (phase,m) in zip(np.arange(0.,2.,0.5),['o','v','^','s']):
#            cond = (color>phase)&(color<phase+0.5)
#            plt.scatter(model[cond], (FAST-model)[cond],
#                        label=l, marker=m,s=40, alpha=0.7)
#        
#        plt.xlabel(l,fontsize=12)
#        plt.ylabel('FAST $-$ Model',fontsize=12)
#        plt.text(0.1,0.1,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
#                 fontsize=15,transform=ax.transAxes)
#        if i==2:
#            plt.xlim(-11.,-8.)
#            plt.ylim(-.6,.6)
#        elif i==1:
#            plt.xlim(7.,11.)
#            plt.ylim(-.6,.6)
#        else:
#            plt.xlim(9.0,10.0)
#            plt.ylabel('T$_{FAST}$ $-$ T$_{BB}$',fontsize=12)
#    #fig.subplots_adjust(bottom=0.2)
#    #cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
#    #colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
##    colorbar.set_label('$M_{today}$')
#    #colorbar.set_label('$Phase$')
#    ax = plt.subplot(2,2,4)
#    if multiple:
#        s = plt.scatter(Av_FAST[M_FAST>9.],(SSFR_FAST-lgSSFR)[M_FAST>9.],c=color,s=20,cmap='jet')
#    else:       
#        for (phase,m) in zip(np.arange(0.,2.,0.5),['o','v','^','s']):
#            cond = (color>phase)&(color<phase+0.5)
#            s = plt.scatter(Av_FAST[cond],(SSFR_FAST-lgSSFR)[cond],s=30,marker=m, alpha=0.7)
#    plt.axvline(1.0,color='k',ls='--')
#    plt.xlabel('FAST Av',fontsize=12)
#    plt.ylabel('$SSFR_{FAST} - SSFR_{model}$',fontsize=12)
#    with sns.axes_style("white"):
#        nx = fig.add_axes([0.8, 0.25, 0.08, 0.08])
#        nx.set_yticks([])
#        nx.hist(Av_FAST,alpha=0.7)
#        nx.axvline(1.0,color='k',ls='--',alpha=0.7)
#    plt.suptitle('AM-SFH Models (p:%.2f dex) vs FAST Fitting with Tau Templates'%A,fontsize=15,y=0.92)
#    plt.show()
    