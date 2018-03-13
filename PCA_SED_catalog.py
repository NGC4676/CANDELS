#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 00:41:16 2018

@author: mac
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from sklearn.decomposition import PCA
from wpca import EMPCA

sns.set_style('ticks')


def calzetti(wave, Av):
    k = np.zeros_like(wave.value)

    w0 = [wave <= 1200 * u.AA]
    w1 = [wave < 6300 * u.AA]
    w2 = [wave >= 6300 * u.AA]
    w_u = wave.to(u.um).value

    x1 = np.argmin(np.abs(wave - 1200 * u.AA))
    x2 = np.argmin(np.abs(wave - 1250 * u.AA))

    k[w2] = 2.659 * (-1.857 + 1.040 / w_u[w2])
    k[w1] = 2.659 * (-2.156 + (1.509 / w_u[w1]) - (0.198 / w_u[w1] ** 2) + (0.011 / w_u[w1] ** 3))
    k[w0] = k[x1] + ((wave[w0] - 1200. * u.AA) * (k[x1] - k[x2]) / (wave[x1] - wave[x2]))

    k += 4.05
    k[k < 0.] = 0.

    EBV = Av / 4.05
    A_lambda = k*EBV
    return A_lambda

def assign_weights(grid_SED):
    W = grid_SED.copy()
    for k in range(len(grid_SED)):
        w = np.isnan(grid_SED[k]).sum() 
        W[k] = 1 - w/10.
    #W[W==W]=1
    W[grid_SED!=grid_SED]=0
    return W

# =============================================================================
#  Read
# =============================================================================

table_gds = Table.read('gds_all.hdf5', path='data')
table_uds = Table.read('uds_all.hdf5', path='data')
data1 = table_gds.to_pandas()
data2 = table_uds.to_pandas()
data = pd.concat([data1,data2], join='inner')

Mgrid = np.arange(9.25,11.,0.5)
zgrid = np.arange(0.75,2.5,0.5)

# =============================================================================
# UDS observed SEDs
# =============================================================================
uds_lamb = [3838.,4448.,5470.,6276.,7671.,9028.,
            5919.,8060.,12471.,15396.,10201.,21484.,
            35569.,45020.,57450.,79158.]

uds_band = ['CFHT_U_FLUX','SUBARU_B_FLUX',
            'SUBARU_V_FLUX','SUBARU_R_FLUX',
            'SUBARU_i_FLUX','SUBARU_z_FLUX',
            'ACS_F606W_FLUX','ACS_F814W_FLUX',
            'WFC3_F125W_FLUX','WFC3_F160W_FLUX',
            'HAWKI_Y_FLUX','HAWKI_Ks_FLUX',
            'IRAC_CH1_SEDS_FLUX','IRAC_CH2_SEDS_FLUX',
            'IRAC_CH3_SPUDS_FLUX','IRAC_CH4_SPUDS_FLUX']

uds_band=[b for _,b in sorted(zip(uds_lamb,uds_band))]
uds_lamb=np.sort(uds_lamb)

z_best = data2.z_best
Ms = np.log10(data2.M_med)

filter = (abs(Ms-10)<1.0) & (z_best>0.2) & (z_best<2.5) \
         & (data2.f_f160w==0) & (data2.mag_f160w_4<24.5) \
         & (data2.CLASS_STAR<0.9) & (data2.PhotFlag==0) 
SF =  filter & (data2.sf_flag == 1) 
Q = filter | (data2.sf_flag == -1) 

Fang = data2[SF]


# =============================================================================
#  UDS Grid SED PCA
# =============================================================================

#fig, axes = plt.subplots(nrows=4, ncols=4,sharey=True,sharex=True,figsize=(12, 10))
#for i, mg in enumerate(Mgrid):
#    for j, zg in enumerate(zgrid):
#        print mg, zg
#        
#        clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
#                & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
#        cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
#        
#        ax = axes[i,j]
#        
#        grid_SED = np.zeros_like(uds_lamb)
#        
#        for ind,gal in Fang[cond].iterrows():
#            uds_flux = np.array([gal[b] for b in uds_band])/(gal.M_med/10**10)
#            uds_flux[uds_flux<0]=np.nan
#            #if True in np.isnan(uds_flux): continue
#            grid_SED = np.vstack([grid_SED,uds_flux])
#            #ax.semilogy(uds_lamb/(1+gal.z_best),uds_flux,c='grey',alpha=0.1)
#        
#        for ind,gal in Fang[cond].iterrows():
#            uds_flux = np.array([gal[b] for b in uds_band])/(gal.M_med/10**10)
#            ax.semilogy(uds_lamb/(1+gal.z_best),uds_flux*(1+gal.z_best),c="darkred",alpha=0.1)
#            
#        grid_SED = grid_SED[1:]
#        
#        W = assign_weights(grid_SED)
#        kwds = {'weights': W}
#        pca = EMPCA(n_components=1).fit(grid_SED, **kwds)
#        
#        ax.semilogy(uds_lamb/(1+Fang[cond].z_best.median()),(1+Fang[cond].z_best.median())*\
#                    (pca.components_[0]+pca.mean_),lw=3,c='r')
#        
#for i, zg in enumerate(zgrid):
#    axes[0, i].annotate('$%.1f<$z$<%.1f$'%(zg-0.25,zg+0.25), 
#                        (0.5, 1), xytext=(0, 15), rotation=0,
#                        textcoords='offset points', xycoords='axes fraction',
#                        ha='center', va='bottom', size=15)
#for j, mg in enumerate(Mgrid):
#    axes[j, 3].annotate('$%.1f<$log M$<%.1f$'%(mg-0.25,mg+0.25), 
#                        (1., 0), xytext=(20, 5), rotation=270,
#                        textcoords='offset points', xycoords='axes fraction',
#                        ha='center', va='bottom', size=15)
#    
#plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace = 0.22,hspace = 0.2)
##plt.savefig("Feb/Grid_obsSED_PCA_UDS-z2.pdf",dpi=400)
#plt.show()


# =============================================================================
# GDS observed SEDs
# =============================================================================

gds_lamb = [3738.,3750.,4319.,5919.,7693.,8060.,
            9036.,9867.,10545.,12471.,15396.,
            21667.,21484.,35569.,45020.,57450.,79158.]

gds_band = ['CTIO_U_FLUX','VIMOS_U_FLUX',
            'ACS_F435W_FLUX','ACS_F606W_FLUX',
            'ACS_F775W_FLUX','ACS_F814W_FLUX',
            'ACS_F850LP_FLUX','WFC3_F098M_FLUX',
            'WFC3_F105W_FLUX','WFC3_F125W_FLUX','WFC3_F160W_FLUX',
            'ISAAC_KS_FLUX','HAWKI_KS_FLUX',
            'IRAC_CH1_FLUX','IRAC_CH2_FLUX',
            'IRAC_CH3_FLUX','IRAC_CH4_FLUX']

gds_band=[b for _,b in sorted(zip(gds_lamb,gds_band))]
gds_lamb=np.sort(gds_lamb)

z_best = data1.z_best
Ms = np.log10(data1.M_med)

filter = (abs(Ms-10)<1.0) & (z_best>0.2) & (z_best<2.5) \
         & (data1.f_f160w==0) & (data1.mag_f160w_4<24.5) \
         & (data1.CLASS_STAR<0.9) & (data1.PhotFlag==0) 
SF =  filter & (data1.sf_flag == 1) 
Q = filter | (data1.sf_flag == -1) 

Fang = data1[SF]

# =============================================================================
# GDS grid SED PCA
# =============================================================================
#PC_SED = np.empty((4,4))
#fig, axes = plt.subplots(nrows=4, ncols=4,sharey=True, sharex=True, figsize=(12, 10))
#for i, mg in enumerate(Mgrid):
#    for j, zg in enumerate(zgrid):
#        print mg, zg
#
#        clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
#                & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
#        cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
#
#        ax = axes[i,j]
#
#        grid_SED = np.zeros_like(gds_lamb)
#
#        for ind,gal in Fang[cond].iterrows():
#            gds_flux = np.array([gal[b] for b in gds_band])/(gal.M_med/10**10)
#            gds_flux[gds_flux<0]=np.nan
#            #if True in np.isnan(gds_flux): continue
#            grid_SED = np.vstack([grid_SED,gds_flux])
#            #ax.semilogy(gds_lamb,gds_flux,c="grey",alpha=0.1)
#            ax.semilogy(gds_lamb/(1+gal.z_best),gds_flux*(1+gal.z_best),c="steelblue",alpha=0.1)
#            
#        grid_SED = grid_SED[1:]
#        
#        #W = grid_SED.copy()
#        #W[W==W]=1
#        #W[W!=W]=0
#        W = assign_weights(grid_SED)
#        kwds = {'weights': W}
#        pca = EMPCA(n_components=1).fit(grid_SED, **kwds)
#        
#        #ax.semilogy(gds_lamb,
#         #           pca.components_[0]*pca.explained_variance_ratio_[0]+\
#          #          +pca.mean_,lw=3,c='k')
#        
#        ax.semilogy(gds_lamb/(1+Fang[cond].z_best.median()),
#                    (1+Fang[cond].z_best.median())*(pca.components_[0]+pca.mean_),
#                    lw=3,c='navy')
#
#for i, zg in enumerate(zgrid):
#    axes[0, i].annotate('$%.1f<$z$<%.1f$'%(zg-0.25,zg+0.25), 
#                        (0.5, 1), xytext=(0, 15), rotation=0,
#                        textcoords='offset points', xycoords='axes fraction',
#                        ha='center', va='bottom', size=15)
#
#for j, mg in enumerate(Mgrid):
#    axes[j, 3].annotate('$%.1f<$log M$<%.1f$'%(mg-0.25,mg+0.25), 
#                        (1., 0.5), xytext=(25, 0), rotation=270,
#                        textcoords='offset points', xycoords='axes fraction',
#                        ha='center', va='center', size=15)
#
#plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace = 0.22,hspace = 0.2)
##plt.savefig("Feb/Grid_obsSED_PCA_GDS-z2.pdf",dpi=400)
#plt.show()


# =============================================================================
# Interpolate to Rest-frame and then do EMPCA
# =============================================================================
from smpy.smpy import LoadEAZYFilters, FilterSet
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')
filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  

xrf = [filt.lambda_c.value for filt in filters.filters]
PC_SED = np.zeros((4,4,len(xrf)))

fig, axes = plt.subplots(nrows=4, ncols=4,sharey=True, sharex=True, figsize=(12, 10))
for i, mg in enumerate(Mgrid):
    for j, zg in enumerate(zgrid):
        print mg, zg

        clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
                & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
        cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean

        ax = axes[i,j]

        grid_SED = np.zeros_like(xrf)
        W = np.zeros_like(xrf)
        
        for ind,gal in Fang[cond].iterrows():
            gds_flux = np.array([gal[b] for b in gds_band])/(gal.M_med/10**10)
            gds_flux[gds_flux<0]=np.nan
            ax.semilogy(gds_lamb/(1+gal.z_best),gds_flux*(1+gal.z_best),c="steelblue",alpha=0.1)
            
            xp,yp = gds_lamb/(1+gal.z_best),gds_flux*(1+gal.z_best)
            yrf = pd.Series(np.interp(xrf,xp,yp))
            
            if np.isnan(yrf).sum()>5: continue
            w = np.ones_like(xrf)*np.isnan(yrf).sum() 
            W = np.vstack([W, 1 - w/10.])
            
            ax.semilogy(xrf,yrf.interpolate(method="akima"),c="gold",alpha=0.1)
            grid_SED = np.vstack([grid_SED,yrf.values])
            
        grid_SED = grid_SED[1:]
        W = W[1:]
        W[grid_SED!=grid_SED]=0
        
        kwds = {'weights': W}
        pca = EMPCA(n_components=1).fit(grid_SED, **kwds)
        pc_SED = pca.components_[0]+pca.mean_
        PC_SED[i,j] = pc_SED
        ax.semilogy(xrf,pc_SED,lw=3,c='gold')

for i, zg in enumerate(zgrid):
    axes[0, i].annotate('$%.1f<$z$<%.1f$'%(zg-0.25,zg+0.25), 
                        (0.5, 1), xytext=(0, 15), rotation=0,
                        textcoords='offset points', xycoords='axes fraction',
                        ha='center', va='bottom', size=15)

for j, mg in enumerate(Mgrid):
    axes[j, 3].annotate('$%.1f<$log M$<%.1f$'%(mg-0.25,mg+0.25), 
                        (1., 0.5), xytext=(25, 0), rotation=270,
                        textcoords='offset points', xycoords='axes fraction',
                        ha='center', va='center', size=15)

plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace = 0.22,hspace = 0.2)
#plt.savefig("Feb/Grid_obsSED_EMPCA_GDS-RF.pdf",dpi=400)
plt.show()

# =============================================================================
# rf PCA
# =============================================================================
#fig, axes = plt.subplots(nrows=4, ncols=4,sharey=True, sharex=True, figsize=(12, 10))
#for i, mg in enumerate(Mgrid):
#    for j, zg in enumerate(zgrid):
#        print mg, zg
#
#        clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
#                & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
#        cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
#
#        ax = axes[i,j]
#
#        grid_SED = np.zeros_like(xrf)
#        
#        for ind,gal in Fang[cond].iterrows():
#            gds_flux = np.array([gal[b] for b in gds_band])/(gal.M_med/10**10)
#            gds_flux[gds_flux<0]=np.nan
#            ax.semilogy(gds_lamb/(1+gal.z_best),gds_flux*(1+gal.z_best),c="steelblue",alpha=0.1)
#            
#            xp,yp = gds_lamb/(1+gal.z_best),gds_flux*(1+gal.z_best)
#            yrf = pd.Series(np.interp(xrf,xp,yp))
#            
#            if np.isnan(yrf).sum()>5: continue
#            
#            ax.semilogy(xrf,yrf.interpolate(method="akima"),c="gold",alpha=0.1)
#            grid_SED = np.vstack([grid_SED,yrf.interpolate(method="akima").values])
#            
#        grid_SED = pd.DataFrame(grid_SED[1:])
#    
#        pca = PCA(n_components=1).fit(grid_SED.dropna())
#        ax.semilogy(xrf,(pca.components_[0]+pca.mean_),lw=3,c='gold')
#
#for i, zg in enumerate(zgrid):
#    axes[0, i].annotate('$%.1f<$z$<%.1f$'%(zg-0.25,zg+0.25), 
#                        (0.5, 1), xytext=(0, 15), rotation=0,
#                        textcoords='offset points', xycoords='axes fraction',
#                        ha='center', va='bottom', size=15)
#
#for j, mg in enumerate(Mgrid):
#    axes[j, 3].annotate('$%.1f<$log M$<%.1f$'%(mg-0.25,mg+0.25), 
#                        (1., 0.5), xytext=(25, 0), rotation=270,
#                        textcoords='offset points', xycoords='axes fraction',
#                        ha='center', va='center', size=15)
#
#plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace = 0.22,hspace = 0.2)
#plt.savefig("Feb/Grid_obsSED_PCA_GDS-RF.pdf",dpi=400)
#plt.show()

# =============================================================================
# noise
# =============================================================================
#for i, mg in enumerate(Mgrid):
#    for j, zg in enumerate(zgrid):
#        print mg, zg
#
#        clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
#                & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
#        cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
#        
#        pc_SED = PC_SED[i,j]
#        rf_SED = np.zeros_like(xrf)
#        
#        for ind,gal in Fang[cond].iterrows():
#            gds_flux = np.array([gal[b] for b in gds_band])/(gal.M_med/10**10)
#            gds_flux[gds_flux<0]=np.nan
#            ax.semilogy(gds_lamb/(1+gal.z_best),gds_flux*(1+gal.z_best),c="steelblue",alpha=0.1)
#            
#            xp,yp = gds_lamb/(1+gal.z_best),gds_flux*(1+gal.z_best)
#            yrf = pd.Series(np.interp(xrf,xp,yp))
#            
#            if np.isnan(yrf).sum()>5: continue
#            
#            rf_SED = np.vstack([rf_SED, np.log10(yrf.interpolate(method="akima"))-np.log10(pc_SED)])
#            
#        rf_SED = rf_SED[1:]
#
#
#df_SED=pd.DataFrame(rf_SED)
#noise = np.vstack([np.random.normal(scale=df_SED.std()) for i in range(100)])
#
#plt.plot(xrf,df_SED.iloc[1])
#for k in range(100):
#    plt.plot(xrf,df_SED.iloc[1]-df_SED.mean()+np.random.normal(scale=df_SED.std()),alpha=0.1,c="grey")

