#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 18:34:04 2018

@author: Q.Liu
"""

import numpy as np
import pandas as pd
import asciitable as asc
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns

def compute_stat(res,t,t1,t2,t3,t4,t5):
    Y_0 = np.vstack((res,t))  
    Y = Y_0[:,abs(Y_0[0])<1]
    y1 = pd.Series(Y[0][(Y[1]>t1)&(Y[1]<t2)])
    y2 = pd.Series(Y[0][(Y[1]>t2)&(Y[1]<t3)])
    y3 = pd.Series(Y[0][(Y[1]>t3)&(Y[1]<t4)])
    y4 = pd.Series(Y[0][(Y[1]>t4)&(Y[1]<t5)])
    Stds = [y.std() for y in [y1,y2,y3,y4]]
    Meds = [y.median() for y in [y1,y2,y3,y4]]
    return Stds, Meds

#==============================================================================
# Read model result 
#==============================================================================
lgAge = np.log10(np.linspace(0.5,11.5,111)* 1e9)
M_par = np.array([9.5,10.0,10.5,10.75,11.0])
#M_par = np.linspace(5.,7.,5)

M_color = ['m','b','g','orange','firebrick']
#M_color = ['navy','steelblue','yellowgreen','gold','orangered']

BC03_out = asc.read('N/Aldo/CSP result.txt')
#BC03_out = asc.read('N/Ciesla/CSP result.txt')

lgMs = BC03_out['lgM*'].reshape((M_par.size, lgAge.size)).T.ravel()
lgSSFR = BC03_out['lgsSFR'].reshape((M_par.size, lgAge.size)).T.ravel()
                 
lgAges = np.array([lgAge for m in M_par]).T.ravel()
M_class =  np.array([M_par for T in range(lgAge.size)]).ravel()

#==============================================================================
# Read FAST SED-fitting result
#==============================================================================
table = ascii.read("N/Aldo/Aldo_exp.fout",header_start=16).to_pandas()
#table = ascii.read("N/Ciesla/Ciesla_exp.fout",header_start=16).to_pandas()


Ages_FAST = table.lage
SFH_FAST = table.ltau
M_FAST = table.lmass
SSFR_FAST = table.lssfr
Av_FAST = table.Av

age_cond = (lgAges>np.log10(2.e9))&(lgAges<np.log10(9.e9))


S = 100
table2 = ascii.read("N/Aldo/Aldo_exp_obs.fout",header_start=16).to_pandas()
#table2 = ascii.read("N/Ciesla/Ciesla_exp_obs.fout",header_start=16).to_pandas()
Ages_FAST2 = table2.lage
SFH_FAST2 = table2.ltau
M_FAST2 = table2.lmass
SSFR_FAST2 = table2.lssfr
Av_FAST2 = table2.Av
M_class2 = np.tile(M_class,S)

table2d = ascii.read("N/Aldo/Aldo_del_obs.fout",header_start=16).to_pandas()
#table2d = ascii.read("N/Ciesla/Ciesla_del_obs.fout",header_start=16).to_pandas()
Ages_FAST2d = table2d.lage
SFH_FAST2d = table2d.ltau
M_FAST2d = table2d.lmass
SSFR_FAST2d = table2d.lssfr
Av_FAST2d = table2d.Av

# =============================================================================
# Compute
# =============================================================================
z_bin = np.array([2.25,1.75,1.25,0.75])
fig,axes = plt.subplots(figsize=(10,6), nrows=2, ncols=3,sharex=True)
for i, (model, FAST) in enumerate(zip([lgMs,lgSSFR,np.ones_like(lgMs)],
                                      [M_FAST2,SSFR_FAST2,Av_FAST2])):
    ax1, ax2 = axes[0,i],axes[1,i]
    models = np.tile(model,S)
    model_ages = np.tile(lgAges,S)
    age_conds = np.tile(age_cond,S)
    for m,c in zip(M_par,M_color):    
        stds, meds = compute_stat((FAST-models)[M_class2==m], 
                                  model_ages[M_class2==m],
                                  9.42,9.52,9.63,9.77,9.936)
        ax1.plot(z_bin, meds,"o-",color=c,ms=6,alpha=0.8)
        ax2.plot(z_bin, stds,"o-",color=c,ms=6,alpha=0.8)
    ax1.set_xlim(2.5,0.5)
plt.subplots_adjust(wspace=0.3,hspace=0.001)

# =============================================================================
# Plot
# =============================================================================
z_bin = np.array([2.25,1.75,1.25,0.75])

fig = plt.figure(figsize=(10,6))
gs = mpl.gridspec.GridSpec(2, 3, height_ratios=[1, 1],width_ratios=[1,1,1])
for i, (model, FAST) in enumerate(zip([lgMs,lgSSFR,np.ones_like(lgMs)],
                                      [M_FAST2,SSFR_FAST2,Av_FAST2])):
    ax1, ax2 = plt.subplot(gs[i]),plt.subplot(gs[3+i])
    models = np.tile(model,S)
    model_ages = np.tile(lgAges,S)
    age_conds = np.tile(age_cond,S)
    for j,(m,c) in enumerate(zip(M_par,M_color)):    
        stds, meds = compute_stat((FAST-models)[M_class2==m], 
                                  model_ages[M_class2==m],
                                  9.42,9.52,9.63,9.77,9.936)
        s=ax1.plot(z_bin, meds,"o--",color=c,ms=7,lw=1.,mfc="w",
                   #label=r'M$\rm_{seed}$=10$^{%s}$'%m,
                   label="RP%d"%j,alpha=0.8,zorder=1)
        ax2.plot(z_bin, stds,"o--",color=c,ms=7,lw=1.,mfc="w",alpha=0.8)
        
        if i==0: 
            ax1.scatter(0.625,0.155,marker="o",facecolors="none",edgecolors="k",linewidths=0.5,transform=ax1.transAxes)
            ax1.scatter(0.65,0.085,marker="^",c="k",linewidths=0.5,transform=ax1.transAxes)
            ax1.text(0.7,0.065,"Delayed",fontsize=9,transform=ax1.transAxes)
            ax1.text(0.665,0.14,"Exponential",fontsize=9,transform=ax1.transAxes)

            ax1.legend(loc=2,fontsize=10,labelspacing=0.7,frameon=False)
#            ax2.legend(loc=1,fontsize=10,labelspacing=0.5,frameon=False)

    ax1.set_xlim(2.5,0.5)
    ax2.set_xlim(2.5,0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)

for i, (l,model, FAST) in enumerate(zip(['M$_*$','sSFR','Av'],
                                        [lgMs,lgSSFR,np.ones_like(lgMs)],
                                        [M_FAST2d,SSFR_FAST2d,Av_FAST2d])):
    ax1, ax2 = plt.subplot(gs[i]),plt.subplot(gs[3+i])
    models = np.tile(model,S)
    model_ages = np.tile(lgAges,S)
    age_conds = np.tile(age_cond,S)
    for j,(m,c) in enumerate(zip(M_par,M_color)):    
        stdsd, medsd = compute_stat((FAST-models)[M_class2==m], 
                                  model_ages[M_class2==m],
                                  9.42,9.52,9.63,9.77,9.936)
        ax1.plot(z_bin, medsd,"^-",color=c,ms=7,lw=1.,alpha=0.8,zorder=2)
        ax2.plot(z_bin, stdsd,"^-",color=c,ms=7,lw=1.,alpha=0.8,zorder=2)

    ax1.hlines(0,0.5,2.5,linestyles="dotted",linewidth=1,alpha=0.7,zorder=0)
    ax1.set_xlim(2.5,0.5)
    ax2.set_xlabel("z",fontsize="large")

    if i==0:
        ax1.set_ylabel("Median Offset",fontsize="large")
        ax2.set_ylabel("$\sigma$",fontsize="large")

    ax2.annotate(l, (0., 0.), xytext=(3,3), rotation=0, size=13,
                        textcoords='offset points', xycoords='axes fraction',
                        va='bottom', bbox=dict(facecolor='none',pad=3.),zorder=1)
plt.subplots_adjust(left=0.1,right=0.95,bottom=0.1,top=0.95,
                    wspace=0.3,hspace=0.001)

plt.savefig("N/Aldo/Stat_Fitting_AM.pdf",dpi=300)
#plt.savefig("N/Ciesla/Stat_Fitting_MS.pdf",dpi=300)
   