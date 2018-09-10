#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:03:57 2018

@author: Q.Liu
"""
import numpy as np
import asciitable as asc
from astropy.io import ascii
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

lgAge = np.log10(np.linspace(0.5,11.5,111)* 1e9)

M_today = np.array([9.0,9.5,10.0,10.5,11.0])
M_seed = 10**np.linspace(5.,7.,5)

lgAges = np.array([lgAge for m in M_today]).T.ravel()
age_cond = (lgAges>np.log10(2.e9))&(lgAges<np.log10(9.e9))
age_young = lgAges<np.log10(2.e9)
age_old = lgAges>np.log10(9.e9)


M_class_A =  np.array([M_today for T in range(lgAge.size)]).ravel()
M_class_C =  np.array([np.log10(M_seed) for T in range(lgAge.size)]).ravel()
#==============================================================================
# Read FAST SED-fitting result AM-SFH
#==============================================================================
phase_start = lgAge[0]

A, P = 0.3, 2.5
#BC03_out = asc.read('New/Perturb/CSP result ST.txt')
#table_A = ascii.read('New/Perturb/Aldo_ST_exp.fout',header_start=16).to_pandas()

BC03_out = asc.read('New/Aldo/CSP result.txt')
table_A = ascii.read("New/Aldo/Aldo_exp.fout",header_start=16).to_pandas()
#table_A = ascii.read("New/Aldo/Aldo_exp_obs.fout",header_start=16).to_pandas()

lgMs_A = BC03_out['lgM*'].reshape((M_today.size, lgAge.size)).T.ravel()
lgSSFR_A = BC03_out['lgsSFR'].reshape((M_today.size, lgAge.size)).T.ravel()

Ages_FAST_A = table_A.lage
SFH_FAST_A = table_A.ltau
M_FAST_A = table_A.lmass
SSFR_FAST_A = table_A.lssfr
Av_FAST_A = table_A.Av

#==============================================================================
# Read FAST SED-fitting result MS-SFH
#==============================================================================
lgAge = np.log10(np.linspace(0.5,11.5,111)* 1e9)

BC03_out = asc.read('New/Ciesla/CSP result.txt')
table_C = ascii.read("New/Ciesla/Ciesla_exp.fout",header_start=16).to_pandas()
#table_C = ascii.read("New/Ciesla/Ciesla_exp_obs.fout",header_start=16).to_pandas()

lgMs_C = BC03_out['lgM*'].reshape((M_seed.size, lgAge.size)).T.ravel()
lgSSFR_C = BC03_out['lgsSFR'].reshape((M_seed.size, lgAge.size)).T.ravel()
                 
Ages_FAST_C = table_C.lage
SFH_FAST_C = table_C.ltau
M_FAST_C = table_C.lmass
SSFR_FAST_C = table_C.lssfr
Av_FAST_C = table_C.Av

#==============================================================================
# Residual
#==============================================================================
with sns.axes_style("ticks"):
    fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
    for i, (model_A,model_C,FAST_A,FAST_C) \
            in enumerate(zip([np.log10(10**lgAges-4.9e8),lgMs_A,lgSSFR_A],
                             [np.log10(10**lgAges-4.9e8),lgMs_C,lgSSFR_C],
                             [Ages_FAST_A,M_FAST_A,SSFR_FAST_A],
                             [Ages_FAST_C,M_FAST_C,SSFR_FAST_C])):
        color = 10**lgAges/1e9
        ax = plt.subplot(2,2,i+1) 
        plt.axhline(0.,color='k',ls='--')
        plt.axvline(0.,color='k',ls='--')
        s = plt.scatter((FAST_A-model_A)[age_cond], (FAST_C-model_C)[age_cond], c=color[age_cond],cmap="jet",
                s=25, alpha=0.5,zorder=2)
        plt.scatter((FAST_A-model_A)[~age_cond], (FAST_C-model_C)[~age_cond],
            c='gray', s=25, alpha=0.3,zorder=1)
        plt.scatter((FAST_A-model_A)[age_cond].median(), (FAST_C-model_C)[age_cond].median(),
            c='gold', edgecolors='k', lw=2, s=50, alpha=0.9,zorder=3)
#        y = pd.Series(FAST-model)[age_cond]
#        y = y[abs(y<1.)]
#        plt.text(0.05,0.78,'$\sigma$ = %.2f'%y.std(),
#                fontsize=12,transform=ax.transAxes,color="orange") 
#        plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
#                fontsize=12,transform=ax.transAxes,color="gray")
        if i==2:
            plt.xlim(-.61,.4)
            plt.ylim(-.61,.4)
            plt.ylabel(r'$\rm SSFR_{FAST,MS} - SSFR_{model,MS}$',fontsize=12)
            plt.xlabel(r'$\rm SSFR_{FAST,AM} - SSFR_{model,AM}$',fontsize=12)
        elif i==1:
            plt.xlim(-.15,.1)
            plt.ylim(-.15,.1)
            plt.ylabel(r'$\rm M_{*,FAST,MS} - M_{*,model,MS}$',fontsize=12)
            plt.xlabel(r'$\rm M_{*,FAST,AM} - M_{*,model,AM}$',fontsize=12)
        else:
            plt.xlim(-.8,0.25)
            plt.ylim(-1.1,0.25)
            plt.xlabel(r"$\rm Age_{FAST,AM}-\rm Age_{model,AM}$")
            plt.ylabel(r"$\rm Age_{FAST,MS}-\rm Age_{model,MS}$")
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
    colorbar.set_label('Cosmic Time (Gyr)')
    ax = plt.subplot(2,2,4)

    s = plt.scatter(Av_FAST_A[age_cond],Av_FAST_C[age_cond],
                    c=color[age_cond],cmap="jet",s=25,alpha=0.5,zorder=2)
    plt.scatter(Av_FAST_A[~age_cond],Av_FAST_C[~age_cond],
                c="gray",s=25,alpha=0.3,zorder=1)
    plt.scatter(Av_FAST_A[age_cond].median(), Av_FAST_C[age_cond].median(),
                c='gold', edgecolors='k', lw=2,s=50, alpha=0.8,zorder=3)
    plt.axhline(1.0,color='k',ls='--')
    plt.axvline(1.0,color='k',ls='--')
    plt.ylabel(r'$\rm Av_{FAST,MS}$',fontsize='large')
    plt.xlabel(r'$\rm Av_{FAST,AM}$',fontsize=12)
    plt.ylim(0.8,1.2)
    plt.xlim(0.8,1.2)
    plt.subplots_adjust(wspace=0.25,hspace=0.2)
#    plt.text(0.05,0.78,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_cond].std(),
#        fontsize=12,transform=ax.transAxes,color="orange")
#    plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
#             fontsize=12,transform=ax.transAxes,color="gray")
    plt.suptitle('Residual-Residual of FAST (Tau) / MS-SFH / AM-SFH',fontsize=16,y=0.95)
    #plt.savefig("New/Aldo/Res-Res_FAST-exp_AM-MS.png",dpi=400)
    plt.show()
    
#==============================================================================
# Real Noise Residual
#==============================================================================
from scipy.ndimage import gaussian_filter
S = 50

with sns.axes_style("ticks"):
    fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
    for i, (model_A,model_C,FAST_A,FAST_C) \
            in enumerate(zip([np.log10(10**lgAges-4.9e8),lgMs_A,lgSSFR_A],
                             [np.log10(10**lgAges-4.9e8),lgMs_C,lgSSFR_C],
                             [Ages_FAST_A,M_FAST_A,SSFR_FAST_A],
                             [Ages_FAST_C,M_FAST_C,SSFR_FAST_C])):
        models_A = np.tile(model_A,S)
        models_C = np.tile(model_C,S)
        age_conds = np.tile(age_cond,S)
        age_youngs = np.tile(age_young,S)
        age_olds = np.tile(age_old,S)
        color = np.tile(10**lgAges/1e9,S)
        ax = plt.subplot(2,2,i+1) 
        plt.axhline(0.,color='k',ls='--')
        plt.axvline(0.,color='k',ls='--')
        s = plt.scatter((FAST_A-models_A)[age_conds], (FAST_C-models_C)[age_conds], 
                        marker="o",c=color[age_conds],cmap="jet",s=10, alpha=0.5,zorder=2)
        plt.scatter((FAST_A-models_A)[age_youngs], (FAST_C-models_C)[age_youngs],
                    marker="^",c='gray', s=10, alpha=0.3,zorder=1)
        plt.scatter((FAST_A-models_A)[age_olds], (FAST_C-models_C)[age_olds],
                    marker="s",c='gray', s=10, alpha=0.3,zorder=1)
        plt.scatter((FAST_A-models_A)[age_conds].median(), (FAST_C-models_C)[age_conds].median(),
                    c='gold', edgecolors='k', lw=2, s=50, alpha=0.9,zorder=3)
#        y = pd.Series(FAST-model)[age_cond]
#        y = y[abs(y<1.)]
#        plt.text(0.05,0.78,'$\sigma$ = %.2f'%y.std(),
#                fontsize=12,transform=ax.transAxes,color="orange") 
#        plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
#                fontsize=12,transform=ax.transAxes,color="gray")
        if i==2:
            plt.xlim(-2.,1.)
            plt.ylim(-2.,1.)
            plt.ylabel(r'$\rm SSFR_{FAST,MS} - SSFR_{model,MS}$',fontsize=12)
            plt.xlabel(r'$\rm SSFR_{FAST,AM} - SSFR_{model,AM}$',fontsize=12)
        elif i==1:
#            plt.xlim(-.2,.2)
#            plt.ylim(-.2,.2)
            plt.ylabel(r'$\rm M_{*,FAST,MS} - M_{*,model,MS}$',fontsize=12)
            plt.xlabel(r'$\rm M_{*,FAST,AM} - M_{*,model,AM}$',fontsize=12)
        else:
#            plt.xlim(-1.1,0.4)
#            plt.ylim(-1.1,0.4)
            plt.xlabel(r"$\rm Age_{FAST,AM}-\rm Age_{model,AM}$")
            plt.ylabel(r"$\rm Age_{FAST,MS}-\rm Age_{model,MS}$")
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
    colorbar.set_label('Cosmic Time (Gyr)')
    ax = plt.subplot(2,2,4)

    s = plt.scatter(Av_FAST_A[age_conds],Av_FAST_C[age_conds],
                    marker="o",c=color[age_conds],cmap="jet",s=10,alpha=0.5,zorder=2)
    plt.scatter(Av_FAST_A[age_youngs],Av_FAST_C[age_youngs],
                marker="^",c="gray",s=10,alpha=0.3,zorder=1)
    plt.scatter(Av_FAST_A[age_olds],Av_FAST_C[age_olds],
                marker="s",c="gray",s=10,alpha=0.3,zorder=1)
    plt.scatter(Av_FAST_A[age_conds].median(), Av_FAST_C[age_conds].median(),
                c='gold', edgecolors='k', lw=2,s=50, alpha=0.9,zorder=3)
    plt.axhline(1.0,color='k',ls='--')
    plt.axvline(1.0,color='k',ls='--')
    plt.ylabel(r'$\rm Av_{FAST,MS}$',fontsize='large')
    plt.xlabel(r'$\rm Av_{FAST,AM}$',fontsize=12)
    #plt.ylim(0.8,1.2)
    #plt.xlim(-.4,.4)
    plt.subplots_adjust(wspace=0.25,hspace=0.2)
#    plt.text(0.05,0.78,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_cond].std(),
#        fontsize=12,transform=ax.transAxes,color="orange")
#    plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
#             fontsize=12,transform=ax.transAxes,color="gray")
    plt.suptitle('Residual-Residual of FAST (Tau) / MS-SFH / AM-SFH',fontsize=16,y=0.95)
#    plt.savefig("New/Aldo/Res-Res_OBS_FAST-exp_AM-MS.png",dpi=400)
    plt.show()
    