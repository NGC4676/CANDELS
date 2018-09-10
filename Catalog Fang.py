# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 01:07:00 2017

@author: Qing Liu
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

sns.set_style('ticks')

def rotate_UVJ(V_J,U_V,theta = 34.8*u.deg):
    Ssed = np.sin(theta)*U_V + np.cos(theta)*V_J
    Csed = np.cos(theta)*U_V - np.sin(theta)*V_J
    return Ssed, Csed

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

table_gds = Table.read('gds_all.hdf5', path='data')
table_uds = Table.read('uds_all.hdf5', path='data')
data1 = table_gds.to_pandas()
data2 = table_uds.to_pandas()
data = pd.concat([data1,data2], join='inner')

z_best = data.z_best
Ms = np.log10(data.M_med)
SSFR = data.ssfr_uv_corr

FUV = data.rest1600
NUV = data.rest2800
U = data.restUXbessel
V = data.restVbessel
J = data.restJpalomar
K = data.restKpalomar

Av = data.med_av
Afuv = calzetti([1597.5]*u.AA, Av)
Anuv = calzetti([2792.5]*u.AA, Av)
Au = calzetti([3593]*u.AA, Av)
Aj = calzetti([12509.6]*u.AA, Av)
Ak = calzetti([22244]*u.AA, Av)

FUV_c = FUV - Afuv
NUV_c = NUV - Anuv
U_c = U - Au
V_c = V - Av
J_c = J - Aj
K_c = K - Ak

FUVV = FUV-V
NUVV = NUV-V
UV = U-V
VJ = V-J
VK = V-K

FUVV = FUV_c-V_c
NUVV = NUV_c-V_c
UV = U_c-V_c
VJ = V_c-J_c
VK = V_c-K_c

theta = 34.8*np.pi/180.
S_SED = np.sin(theta)*UV + np.cos(theta)*VJ
C_SED = np.cos(theta)*UV - np.sin(theta)*VJ

filter = (abs(Ms-10)<1.0) & (z_best>0.5) & (z_best<2.5) \
         & (data.f_f160w==0) & (data.mag_f160w_4<24.5) \
         & (data.CLASS_STAR<0.9) & (data.PhotFlag==0) 
SF =  filter & (data.sf_flag == 1) 
Q = filter | (data.sf_flag == -1) 

#==============================================================================
# UVJ grid Fig 13
#==============================================================================
from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import exponential
from smpy.dust import Calzetti

bc03 = BC('data/ssp/bc03/chab/lr/')
models_exp = CSP(bc03, age = np.logspace(-2.5, 1.3, 25)* u.Gyr,
                 sfh = 3.0*u.Gyr, dust = 0., metal_ind = 1.0, f_esc = 1.,
                 sfh_law = exponential, dust_model = Calzetti) 
  
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')
filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames))) 

synphot_exp = Observe(models_exp, filters, redshift=0.001)
mags_exp = np.squeeze(synphot_exp.AB)
UV_tau = mags_exp[2] - mags_exp[4]
VJ_tau = mags_exp[4] - mags_exp[7]

# Fang Catalog UVJ grid
Fang = data[filter]
VJ, UV = VJ[filter], UV[filter]
S_SED = np.sin(theta)*UV + np.cos(theta)*VJ
C_SED = np.cos(theta)*UV - np.sin(theta)*VJ

# =============================================================================

Mgrid = np.arange(9.25,11.,0.5)
zgrid = np.arange(0.75,2.5,0.5)

fig,axes = plt.subplots(figsize=(9,9), nrows=4, ncols=4,sharex=True,sharey=True)
for i, zg in enumerate(zgrid):
    for j, mg in enumerate(Mgrid):
        #print zg, mg
        clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
                & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
        cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
        ax = axes[i,j]
        ax.plot(VJ_tau, UV_tau, c='m',lw=2, alpha=0.7)
        s = ax.scatter(VJ[cond], UV[cond], c=Fang.ssfr_uv_corr[cond],
                        s=5, cmap='jet_r',alpha=0.5)
        ax.plot([-5,(0.55+0.253*zg-0.0533*zg**2)/0.88,1.6,1.6],
                 [1.3,1.3,2.158-0.253*zg+0.0533*zg**2,2.5], color='k', alpha=0.7)
        ax.set_xlim([-0.5, 1.75])
        ax.set_ylim([-0.25, 2.])
        ax.grid()
        if ((i==3) & (j==1)) | ((i==1) & (j==2)):
            plt.setp(ax.spines.values(), color="gold", linewidth=4)

for i, mg in enumerate(Mgrid):
    axes[0, i].annotate('$10^{%.1f}<$M$<10^{%.1f}$'%(mg-0.25,mg+0.25), 
                        (0.5, 1), xytext=(0, 15), rotation=0,
                        textcoords='offset points', xycoords='axes fraction',
                        ha='center', va='bottom', size=12)
for j, zg in enumerate(zgrid):
    axes[j, 3].annotate('$%.1f<$z$<%.1f$'%(zg-0.25,zg+0.25), 
                        (1., 0.5), xytext=(25, 0), rotation=270,
                        textcoords='offset points', xycoords='axes fraction',
                        ha='center', va='center', size=12)
cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
colorbar.set_label('log sSFR',fontsize=12)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.15,top=0.95,wspace=0.1,hspace=0.1)
plt.savefig("N/CANDELS_grid.pdf")


#==============================================================================
# MW UVJ
#==============================================================================
from astropy.cosmology import FlatLambdaCDM, z_at_value
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307)
T1, T2, T3 = 3., 5., 7.
z1, z2, z3 = z_at_value(cosmo.age,T1*u.Gyr), z_at_value(cosmo.age,T2*u.Gyr), z_at_value(cosmo.age,T3*u.Gyr)

# Papovich15 AM
UV_pop = np.array([0.6,0.6,0.9,1.1,1.4,1.6,1.7,1.9])
VJ_pop = np.array([0.3,0.5,0.8,1.0,1.2,1.2,1.3,1.3])
Ms_pop = np.array([9.48,9.7,9.88,10.06,10.21,10.35,10.47,10.6])
z_pop = np.array([2.5,2.1,1.85,1.55,1.25,1.0,0.8,0.45])
M1 = np.interp(z1, z_pop[::-1], Ms_pop[::-1])
M2 = np.interp(z2, z_pop[::-1], Ms_pop[::-1])
M3 = np.interp(z3, z_pop[::-1], Ms_pop[::-1])

## Plot
#fig = plt.figure(figsize=(13,5))
#plt.suptitle('Evolution of UVJ for MW-mass progenitors',fontsize=20, y=1.02)
#for i, (tg,zg,mg) in enumerate([(T1,z1,M1),(T2,z2,M2),(T3,z3,M3)]):
#    print 'z:',zg,'M:',mg
#    clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
#            & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
#    cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
#    ax = plt.subplot(1, 3, i+1)
#    plt.plot(VJ_tau, UV_tau, c='m',lw=2, alpha=0.7)
#    s = plt.scatter(VJ[cond], UV[cond], c=Fang.ssfr_uv_corr[cond],
#                    s=10, cmap='jet_r')
#    plt.scatter(VJ[cond].median(), UV[cond].median(), s=60,
#                facecolor='orange', edgecolor='k',linewidth=2,alpha=1.)
#    plt.axhline(0.5,color='k',ls='--')
#    plt.arrow(1.2,0.8,0.673,0.491,fc='k', ec='k',
#              lw=1, head_width=0.05, head_length=0.1)
#    plt.text(1.4,0.8,r'$\it \Delta A_{v} = 1$',fontsize=12)
#    plt.plot([-5,(0.55+0.253*zg-0.0533*zg**2)/0.88,1.6,1.6],
#              [1.3,1.3,2.158-0.253*zg+0.0533*zg**2,2.5], color='k', alpha=0.7)
#    plt.text(1.2,0.0,'T = %.1f Gyr'%tg,fontsize=15)
#    plt.text(-0.2,2.0,'z$\sim$%.1f'%zg,fontsize=15)
#    plt.text(-0.2,1.8,'M$\sim$%.1f'%mg,fontsize=15)
#    plt.xlim([-0.5, 2.0])
#    plt.ylim([-0.25, 2.25])
#cbar_ax = fig.add_axes([0.2, -0.08, 0.6, 0.04])
#cb = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
#t_ax = fig.add_axes([0.05, -0.01, 0.9, 0.03])
#t_ax.arrow(0., 0.5, 1., 0., fc='k', ec='k', lw = 3,
#         head_width=0.8, head_length=0.03, overhang = 0.3,
#         length_includes_head= True, clip_on = False)
#t_ax.axis('off')
#t_ax.text(1.01,0.,r'$\bf t$',fontsize=18)
#cb.set_label('SSFR$_{UV,corr}$',fontsize=12)
#plt.tight_layout()
#plt.show()



#### SF / non-SF ####
def UVJ_bond(VJ, UV, zg):
    x1 = (0.55+0.253*zg-0.0533*zg**2)/0.88
    x2 = 1.6
    y1 = 1.3
    y2 = 2.158-0.253*zg+0.0533*zg**2
    if VJ<x1:
        if UV<1.3: return 1 
        else: return 0
    elif VJ<1.6:
        if UV<y2+(VJ-x2)*(y2-y1)/(x2-x1): return 1
        else: return 0
    else: return 0
vUVJ_bond = np.vectorize(UVJ_bond)

fig = plt.figure(figsize=(13,6))
#plt.suptitle('Evolution of UVJ for MW-mass progenitors',fontsize=20, y=0.97)
for i, (tg,zg,mg) in enumerate([(T1,z1,M1),(T2,z2,M2),(T3,z3,M3)]):
    print 'z:',zg,'M:',mg
    clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
            & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
    cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
    SF = vUVJ_bond(VJ,UV,zg)  & (Fang.ssfr_uv_corr>-9.75)
    ax = plt.subplot(1, 3, i+1)
    plt.plot(VJ_tau, UV_tau, c='m',lw=2, alpha=0.7,zorder=2)
    s = plt.scatter(VJ[cond & SF], UV[cond & SF], c=Fang.ssfr_uv_corr[cond & SF],
                    s=10, cmap='jet_r',zorder=1,alpha=0.7)
    plt.scatter(VJ[cond & ~SF], UV[cond & ~SF],
                    s=10, c='grey',zorder=1,alpha=0.7)
    plt.scatter(VJ[cond & SF].median(), UV[cond & SF].median(), s=60,
                facecolor='greenyellow', edgecolor='k',linewidth=2,alpha=1.,zorder=3)
    plt.axhline(0.5,color='k',ls='--')
    plt.arrow(1.1,0.8,0.673,0.491,fc='k', ec='k',
              lw=1, head_width=0.05, head_length=0.1)
    plt.text(1.4,0.8,r'$\it \Delta A_{v} = 1$',fontsize=12)
    plt.plot([-5,(0.55+0.253*zg-0.0533*zg**2)/0.88,1.6,1.6],
              [1.3,1.3,2.158-0.253*zg+0.0533*zg**2,2.5], color='k', alpha=0.7)
    plt.text(1.,0.0,r'$\bf T = %d{\ }Gyr$'%tg,fontsize=15)
    plt.text(-0.2,2.0,'z$\sim$%.1f'%zg,fontsize=15)
    plt.text(-0.2,1.8,'M$\sim$%.1f'%mg,fontsize=15)
    plt.xlim([-0.5, 2.0])
    plt.ylim([-0.25, 2.25])
    plt.xlabel(r"$\rmV-J$",fontsize=12)
    plt.ylabel(r"$\rmU-V$",fontsize=12)
    cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.04])
    cb = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
    cb.set_clim(-10.55,-8.45)
t_ax = fig.add_axes([0.05, 0.16, 0.9, 0.03])
t_ax.arrow(0.15, 0.5, .7, 0., fc='k', ec='k', lw = 3,
         head_width=0.8, head_length=0.03, overhang = 0.3,
         length_includes_head= True, clip_on = False)
t_ax.axis('off')
t_ax.text(0.86,0.,r'$\bf Cosmic{\ }Time$',fontsize=18)
cb.set_label('sSFR$_{UV,corr}$',fontsize=12)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.3,top=0.95,wspace=0.25,hspace=0.1)
plt.savefig("N/MW_UVJ_evo.png")
plt.show()

# =============================================================================
#  UV-T
# =============================================================================
tgs = np.linspace(2.5,8.,8)
tgs = np.linspace(2.8,8.2,7)

# SFG
uvs = np.array([])
uvm = np.array([])
uva = np.array([])
uvb = np.array([])
vjs = np.array([])
vjm = np.array([])
vja = np.array([])
vjb = np.array([])
fig = plt.figure(figsize=(6,5))
plt.title('U-V Evolution for MW-mass progenitors',fontsize=20, y=1.02)
for i, (tg) in enumerate(tgs):
    zg = z_at_value(cosmo.age,tg*u.Gyr)
    mg = np.interp(zg, z_pop[::-1], Ms_pop[::-1])
    print 'z:',zg,'M:',mg
    clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
            & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
    cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
    SF = vUVJ_bond(VJ,UV,zg)  #& (Fang.ssfr_uv_corr>-9.75)
    
    UV_err = np.array([[UV[cond & SF].quantile(0.3)],[UV[cond & SF].quantile(0.7)]])
    VJ_err = np.array([[VJ[cond & SF].quantile(0.3)],[VJ[cond & SF].quantile(0.7)]])
    t_err = np.array([[cosmo.age(zg+0.25).value],[cosmo.age(zg-0.25).value]])
    plt.scatter(tg, UV[cond & SF].median(), s=60,
                facecolor='orange', edgecolor='k',linewidth=2,alpha=1.,zorder=2)
    plt.scatter(tg, VJ[cond & SF].median(), s=60,
                facecolor='r', edgecolor='k',linewidth=2,alpha=1.,zorder=2)
    plt.errorbar(tg, UV[cond & SF].median(),
                 xerr = abs(t_err - tg),
                 yerr = abs(UV_err - UV[cond & SF].median()),
                 c='grey',fmt='',alpha=1.,zorder=1)
    uvs = np.append(uvs, UV[cond & SF].median())
    uvm = np.append(uvm, UV[cond & SF].mean())
    uva = np.append(uva, UV[cond & SF].median()-UV[cond & SF].quantile(0.3))
    uvb = np.append(uvb, UV[cond & SF].quantile(0.7)-UV[cond & SF].median())
    vjs = np.append(vjs, VJ[cond & SF].median())
    vjm = np.append(vjm, VJ[cond & SF].mean())
    vja = np.append(vja, VJ[cond & SF].median()-VJ[cond & SF].quantile(0.3))
    vjb = np.append(vjb, VJ[cond & SF].quantile(0.7)-VJ[cond & SF].median())
plt.axhline(0.5,color='k',ls='--')
#plt.axvline(3.,color='steelblue',ls='--')
#plt.axvline(7.,color='steelblue',ls='--')
plt.axvline(2.5,color='steelblue',ls='--')
plt.axvline(8.5,color='steelblue',ls='--')
plt.xlim(0.,11.)
plt.ylim(0.,1.2)
plt.tight_layout()
plt.show()

# Pure SFG
uvs_SF = np.array([])
uvm_SF = np.array([])
uva_SF = np.array([])
uvb_SF = np.array([])
vjs_SF = np.array([])
vjm_SF = np.array([])
vja_SF = np.array([])
vjb_SF = np.array([])
fig = plt.figure(figsize=(6,5))
plt.title('U-V Evolution for MW-mass progenitors',fontsize=20, y=1.02)
for i, (tg) in enumerate(tgs):
    zg = z_at_value(cosmo.age,tg*u.Gyr)
    mg = np.interp(zg, z_pop[::-1], Ms_pop[::-1])
    print 'z:',zg,'M:',mg
    clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
            & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
    cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
    SF = vUVJ_bond(VJ,UV,zg)  & (Fang.ssfr_uv_corr>-9.75)
    
    UV_err = np.array([[UV[cond & SF].quantile(0.3)],[UV[cond & SF].quantile(0.7)]])
    VJ_err = np.array([[VJ[cond & SF].quantile(0.3)],[VJ[cond & SF].quantile(0.7)]])
    t_err = np.array([[cosmo.age(zg+0.25).value],[cosmo.age(zg-0.25).value]])
    plt.scatter(tg, UV[cond & SF].median(), s=60,
                facecolor='orange', edgecolor='k',linewidth=2,alpha=1.,zorder=2)
    plt.scatter(tg, VJ[cond & SF].median(), s=60,
                facecolor='r', edgecolor='k',linewidth=2,alpha=1.,zorder=2)
    plt.errorbar(tg, UV[cond & SF].median(),
                 xerr = abs(t_err - tg),
                 yerr = abs(UV_err - UV[cond & SF].median()),
                 c='grey',fmt='',alpha=1.,zorder=1)
    uvs_SF = np.append(uvs_SF, UV[cond & SF].median())
    uvm_SF = np.append(uvm_SF, UV[cond & SF].mean())
    uva_SF = np.append(uva_SF, UV[cond & SF].median()-UV[cond & SF].quantile(0.3))
    uvb_SF = np.append(uvb_SF, UV[cond & SF].quantile(0.7)-UV[cond & SF].median())
    vjs_SF = np.append(vjs_SF, VJ[cond & SF].median())
    vjm_SF = np.append(vjm_SF, VJ[cond & SF].mean())
    vja_SF = np.append(vja_SF, VJ[cond & SF].median()-VJ[cond & SF].quantile(0.3))
    vjb_SF = np.append(vjb_SF, VJ[cond & SF].quantile(0.7)-VJ[cond & SF].median())
plt.axhline(0.5,color='k',ls='--')
#plt.axvline(3.,color='steelblue',ls='--')
#plt.axvline(7.,color='steelblue',ls='--')
plt.axvline(2.5,color='steelblue',ls='--')
plt.axvline(8.5,color='steelblue',ls='--')
plt.xlim(0.,11.)
plt.ylim(0.,1.2)
plt.tight_layout()
plt.show()

# All
tgs = np.linspace(2.5,8.,8)
tgs = np.linspace(2.8,8.2,7)
uvs_a = np.array([])
uvm_a = np.array([])
uva_a = np.array([])
uvb_a = np.array([])
vjs_a = np.array([])
vjm_a = np.array([])
vja_a = np.array([])
vjb_a = np.array([])
fig = plt.figure(figsize=(6,5))
plt.title('U-V Evolution for MW-mass progenitors',fontsize=20, y=1.02)
for i, (tg) in enumerate(tgs):
    zg = z_at_value(cosmo.age,tg*u.Gyr)
    mg = np.interp(zg, z_pop[::-1], Ms_pop[::-1])
    print 'z:',zg,'M:',mg
    clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
            & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
    cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
    SF = True
    
    UV_err = np.array([[UV[cond & SF].quantile(0.3)],[UV[cond & SF].quantile(0.7)]])
    t_err = np.array([[cosmo.age(zg+0.25).value],[cosmo.age(zg-0.25).value]])
    plt.scatter(tg, UV[cond & SF].median(), s=60,
                facecolor='orange', edgecolor='k',linewidth=2,alpha=1.,zorder=2)
    plt.scatter(tg, VJ[cond & SF].median(), s=60,
                facecolor='r', edgecolor='k',linewidth=2,alpha=1.,zorder=2)
    plt.errorbar(tg, UV[cond & SF].median(),
                 xerr = abs(t_err - tg),
                 yerr = abs(UV_err - UV[cond & SF].median()),
                 c='grey',fmt='',alpha=1.,zorder=1)
    uvs_a = np.append(uvs_a, UV[cond & SF].median())
    uvm_a = np.append(uvm_a, UV[cond & SF].mean())
    uva_a = np.append(uva_a, UV[cond & SF].median()-UV[cond & SF].quantile(0.3))
    uvb_a = np.append(uvb_a, UV[cond & SF].quantile(0.7)-UV[cond & SF].median())
    vjs_a = np.append(vjs_a, VJ[cond & SF].median())
    vjm_a = np.append(vjm_a, VJ[cond & SF].mean())
    vja_a = np.append(vja_a, VJ[cond & SF].median()-VJ[cond & SF].quantile(0.3))
    vjb_a = np.append(vjb_a, VJ[cond & SF].quantile(0.7)-VJ[cond & SF].median())

plt.axhline(0.5,color='k',ls='--')
#plt.axvline(3.,color='steelblue',ls='--')
#plt.axvline(7.,color='steelblue',ls='--')
plt.axvline(2.5,color='steelblue',ls='--')
plt.axvline(8.5,color='steelblue',ls='--')
plt.xlim(0.,11.)
plt.ylim(0.,1.2)
plt.tight_layout()
plt.show()

# Then Plot tgs, uvs on Figure UV-T!


# =============================================================================
# 
# =============================================================================
cseds = np.array([])
cseda = np.array([])
csedb = np.array([])
fig = plt.figure(figsize=(6,5))
plt.title('U-V Evolution for MW-mass progenitors',fontsize=20, y=1.02)
for i, (tg) in enumerate(tgs):
    zg = z_at_value(cosmo.age,tg*u.Gyr)
    mg = np.interp(zg, z_pop[::-1], Ms_pop[::-1])
    print 'z:',zg,'M:',mg
    clean = (Fang.ssfr_uv_corr < Fang.ssfr_uv_corr.quantile(.975)) \
            & (Fang.ssfr_uv_corr > Fang.ssfr_uv_corr.quantile(.025))    
    cond = (abs(np.log10(Fang.M_med)-mg)<0.25) & (abs(Fang.z_best-zg)<0.25) & clean
    SF = vUVJ_bond(VJ,UV,zg)  #& (Fang.ssfr_uv_corr>-9.75)
    
    C_SED_err = np.array([[C_SED[cond & SF].quantile(0.3)],[C_SED[cond & SF].quantile(0.7)]])
    t_err = np.array([[cosmo.age(zg+0.15*zg).value],[cosmo.age(zg-0.15*zg).value]])
    plt.scatter(tg, C_SED[cond & SF].median(), s=60,
                facecolor='orange', edgecolor='k',linewidth=2,alpha=1.,zorder=2)
    plt.errorbar(tg, C_SED[cond & SF].median(),
                 xerr = abs(t_err - tg),
                 yerr = abs(C_SED_err - C_SED[cond & SF].median()),
                 c='grey',fmt='',alpha=1.,zorder=1)
    cseds = np.append(cseds, C_SED[cond & SF].median())
    cseda = np.append(cseda, C_SED[cond & SF].median()-C_SED[cond & SF].quantile(0.3))
    csedb = np.append(csedb, C_SED[cond & SF].quantile(0.7)-C_SED[cond & SF].median())
plt.axhline(0.25,color='k',ls='--')
plt.axvline(3.,color='steelblue',ls='--')
plt.axvline(7.,color='steelblue',ls='--')
plt.xlim(0.,11.)
plt.ylim(0.,.5)
plt.tight_layout()
plt.show()

#==============================================================================
# 3D Flux-C_SED-S_SED
#==============================================================================
#from mpl_toolkits.mplot3d import Axes3D
#S_SED = pd.Series(np.sin(theta)*UV + np.cos(theta)*VJ)
#C_SED = pd.Series(np.cos(theta)*UV - np.sin(theta)*VJ)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(C_SED, S_SED, FUV[filter],alpha=0.3,s=10)
#ax.view_init(30, 45)
#
#
#import plotly.plotly as py
#import plotly.graph_objs as go
#
#clean = (S_SED < S_SED.quantile(.975)) & (S_SED > S_SED.quantile(.025))
#aes = go.Scatter3d(x = C_SED[clean].get_values(), 
#                   y = S_SED[clean].get_values(), 
#                   z = pd.Series(U[filter].get_values())[clean],
#    mode = 'markers', marker = dict(color = pd.Series(z_best[filter].get_values())[clean], 
#                                    size = '3', 
#                                    colorscale = 'hot'))
#
#fig = go.Figure(data=[aes]) 
#
#py.iplot(fig,filename='U-Csed-Ssed')

#==============================================================================
# FF calibration Fig 14
#==============================================================================
#H, xbins, ybins = np.histogram2d(C_SED[data.sn>3],
#                                 SSFR[data.sn>3],
#                                 bins=(np.linspace(-0.2, 1.2, 100),
#                                       np.linspace(-11.5, -8., 100)))
#fig, ax = plt.subplots(figsize=(5, 6))																																			
#ax.imshow(np.log10(H).T, origin='lower',
#          extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
#          cmap='jet', interpolation='nearest',
#          aspect='auto')
#C_SED0 = np.linspace(-1, 1.5, 100)
#plt.plot(C_SED0,-2.28*C_SED0**2-0.8*C_SED0-8.41,'k',lw=4)
#plt.xlabel('$C_{SED} = 0.82(U-V)-0.57(V-J)$',fontsize=14)
#plt.ylabel('log sSFR / $yr^{-1}$',fontsize=14)
#plt.xlim(-0.2,1.2)
#plt.ylim(-11.5,-8.)
