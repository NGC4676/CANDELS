# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:33:10 2017

@author: Qing Liu
"""

import numpy as np
import asciitable as asc
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, Column, vstack
from astropy.visualization import quantity_support
from astropy.cosmology import FlatLambdaCDM, z_at_value

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import gauss_lorentz_hermite, glh_absprtb, glh_ST
from smpy.dust import Calzetti

# Go to smpy.smpy to set cosmo, the current is below:
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307) 
cosmo = FlatLambdaCDM(H0=70.4, Om0=0.27) 

#==============================================================================
# Rodriguez-Puebla 2017
#==============================================================================
bc03 = BC('data/ssp/bc03/chab/hr/')
#M_today = np.array([9.0,9.5,10.0,10.5,11.0])
M_today = np.array([9.5,10.0,10.5,10.75,11.0])
tg = np.linspace(0.5,11.5,111)
Ages = tg * u.Gyr

Phases = [np.pi/4.,3*np.pi/4.,5*np.pi/4.,7*np.pi/4.]
Phases_rd = np.round(Phases,decimals=1)
#==============================================================================
# Real Error
#==============================================================================                  
perturb = False
emission = False
ConstP = False

if perturb:
    if ConstP:
        A, P = 0.3, 2.5
        SFH_law = glh_absprtb
    else:
        A, p_id = 0.3, 1
        SFH_law = glh_ST
        SFH_fit = asc.read('N/Aldo/ST/Aldo_SFH_params_ST.txt')
        SFH_pars = zip(SFH_fit.c1, SFH_fit.mu, SFH_fit.sigma,\
                       SFH_fit.h13, SFH_fit.h14,\
                       SFH_fit.c2, SFH_fit.x0, SFH_fit.gama,\
                       SFH_fit.h23, SFH_fit.h24,\
                       A*np.ones_like(M_today), SFH_fit["p%s"%p_id])        
else:
    SFH_law = gauss_lorentz_hermite
#    SFH_fit = asc.read('Aldo/Aldo SFH params.txt')
    SFH_fit = asc.read('N/Aldo/Aldo_SFH_params.txt')
    SFH_pars = zip(SFH_fit.c1, SFH_fit.mu, SFH_fit.sigma, SFH_fit.h13, SFH_fit.h14,\
               SFH_fit.c2, SFH_fit.x0, SFH_fit.gama, SFH_fit.h23, SFH_fit.h24)
    

Dust = np.array([1])   

models = CSP(bc03, age = Ages, sfh = SFH_pars, 
             dust = Dust, metal_ind = 1., f_esc = 1.,
             sfh_law = SFH_law, dust_model = Calzetti,
             neb_cont=True, neb_met=False)

SED = np.squeeze(models.SED.value)


#==============================================================================
# Show Spectrum
#==============================================================================
iT = 30 
plt.figure(figsize=(8,4))
for j in range(5):
    plt.semilogy(models.wave,SED[iT,j,:],lw=1)
plt.fill_between(np.linspace(3e3,4.2e3,10), 1, 1e-7,facecolor='grey', alpha=0.3)
plt.fill_between(np.linspace(4.8e3,6.9e3,10), 1, 1e-7,facecolor='grey', alpha=0.3)
plt.fill_between(np.linspace(1.1e4,1.34e4,10), 1, 1e-7,facecolor='grey', alpha=0.3)
plt.xlabel('Wavelength',fontsize=15)
plt.ylabel('Flux',fontsize=15)
plt.text(1.2e4,1e-3,'T = %.1f Gyr'%Ages[iT].value,fontsize=15)
plt.xlim(1e3,1.5e4)
plt.ylim(3e-6,5e-3)

#for k in range(220):
#    print (models.ta[k+1]-models.ta[k]).to('Gyr'),models.ta[k].to('Gyr')

#==============================================================================
# Renormalization
#==============================================================================
use_bc = False
if use_bc:
    if perturb:
        BC03_out = asc.read('N/Aldo/ST/ST%s_p%s CSP result.txt'%(A,Phases_rd[p_id]))
    else:
        BC03_out = asc.read('N/Aldo/CSP result.txt')
    lgMs = BC03_out['lgM*'].reshape((M_today.size, Ages.size)).T
    lgSSFR = BC03_out['lgsSFR'].reshape((M_today.size, Ages.size)).T            
    Ms = 10**lgMs
    SSFR = 10**lgSSFR

#==============================================================================
# Mock Observation
#==============================================================================
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')

print eazy_library.filternames

filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
Names = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
Redshifts = np.arange(0.5,2.5,0.5)
synphot = Observe(models, filters, redshift=0.001)

mags = synphot.AB[:,:,:,:,:5][0]
FUV, NUV, U, B, V, R, I, J, H, K = mags
FUV_V = FUV - V
NUV_V = NUV - V
U_V = U - V
V_J = V - J
V_K = V - K

#==============================================================================
# UVJ Tracks
#==============================================================================
X, Y = V_J, U_V
plt.figure(figsize=(6,6))
with quantity_support():
    for k, Av in enumerate(Dust):
        for j, (m,c) in enumerate(zip(M_today,['m','b','g','orange','firebrick'])):
                plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label=r'log M$\rm_{today}$ = %g'%m,
                         c=c,lw=5, alpha=0.9,zorder=6-j)
#                s = plt.scatter(X[0,:,j,k,0], Y[0,:,j,k,0], c=Ages/u.Gyr, 
#                                s=15, cmap=plt.cm.viridis_r)
#    col = plt.colorbar(s)
#    col.set_label('Age(Gyr)')
    zg = 1.0
    plt.plot([-5,(0.55+0.253*zg-0.0533*zg**2)/0.88,1.6,1.6],
              [1.3,1.3,2.158-0.253*zg+0.0533*zg**2,2.5], color='k', alpha=0.7,zorder=1)
    plt.legend(loc='best',fontsize=11,frameon=True,facecolor='w')
    plt.xlabel('V - J',fontsize=12)
    plt.ylabel('U - V',fontsize=12)
    plt.xlim([-0.25, 1.75])
    plt.ylim([-0.25, 2.25])
plt.tight_layout()
plt.show()

#==============================================================================
# Single Perturbed UVJ color code in Av
#==============================================================================
#X, Y = V_J, U_V
#plt.figure(figsize=(9,6))
#with quantity_support():
#    for k, Av in enumerate(Dust):
#            for j,m in enumerate(M_today):
#                plt.subplot(2,3,j+1)
#                plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label='log M$_{today}$ = %.1f'%m,
#                         lw=2, alpha=0.7)
##                s = plt.scatter(X[0,:,j,k,0], Y[0,:,j,k,0], c=Av_fit[j], 
##                                 s=30, cmap='rainbow_r')
#                
##                col = plt.colorbar(s)
##                col.set_label('$Av_{FAST}$')
#                j+=1
#                plt.plot([-1., 1.0, 1.6, 1.6], [1.3, 1.3, 2.01, 2.5], color='k', alpha=0.7)
#                plt.legend(loc='best',fontsize=10,frameon=True,facecolor='w')
#                plt.xlabel(' ',fontsize=12)
#                plt.ylabel(' ',fontsize=12)
#                plt.xlim([-0.5, 1.8])
#                plt.ylim([-0.25, 2.25])
#                
#plt.tight_layout()
#plt.show()

#==============================================================================
# Exp shaded region
#==============================================================================
#if ConstP:
#    phase=np.mod(2*np.pi*Ages.value/P + 3*np.pi/4.,2*np.pi) /np.pi
#else:
#    phase=np.mod(5*np.pi*np.log(Ages.value) + 3*np.pi/4.,2*np.pi) /np.pi
#        
#fig=plt.figure(figsize=(10,6))
#with sns.axes_style("ticks"):
#    for i, m in enumerate(M_today):
#        ax = plt.subplot(2,3,i+1)
#
#        plt.fill(np.concatenate((X_e[0,:,-1,0,0].value,
#                         X_e[0,-1,:,0,0].value[::-1],
#                         X_e[0,:,0,0,0].value[::-1])),
#                 np.concatenate((Y_e[0,:,-1,0,0].value,
#                         Y_e[0,-1,:,0,0].value[::-1],
#                         Y_e[0,:,0,0,0].value[::-1])),
#                 'grey',alpha=0.3)
#        plt.fill(np.concatenate((X_e0[0,:,-1,0,0].value,
#                         X_e0[0,-1,:,0,0].value[::-1],
#                         X_e0[0,:,0,0,0].value[::-1])),
#                 np.concatenate((Y_e0[0,:,-1,0,0].value,
#                         Y_e0[0,-1,:,0,0].value[::-1],
#                         Y_e0[0,:,0,0,0].value[::-1])),
#                 'grey',alpha=0.5)
#
#        plt.plot(X[0,:,i,0,0], Y[0,:,i,0,0],
#                label='log M$_{today}$ = %.1f'%m,lw=1, alpha=0.5)
#        s=plt.scatter(X[0,:,i,0,0], Y[0,:,i,0,0], c=phase, cmap='gnuplot_r', s=10, alpha=0.7)
#
#        plt.arrow(1.0,0.8,0.67,0.49,fc='k', ec='k',
#              lw=1, head_width=0.05, head_length=0.1)  
#        plt.text(1.2,0.8,r'$\it \Delta A_{v} = 1$',fontsize=8)
#        plt.legend(loc=4,fontsize=8,frameon=True,facecolor='w')
#
#        zg = 1.0
#        plt.plot([-5,(0.55+0.253*zg-0.0533*zg**2)/0.88,1.6,1.6],
#              [1.3,1.3,2.158-0.253*zg+0.0533*zg**2,2.5], color='k', alpha=0.7)
#
#        plt.xlabel('V - J')
#        plt.ylabel('U - V')
#        plt.xlim([0., 1.8])
#        plt.ylim([0., 2.25])
#        if ConstP:
#            peaks=[29,54,78,104]
#        else:
#            peaks=[21,33,52,81]
#        plt.scatter(X[0,peaks[0],i,0,0],Y[0,peaks[0],i,0,0],s=20,color='k',marker='x')
#        plt.scatter(X[0,peaks[1],i,0,0],Y[0,peaks[1],i,0,0],s=20,color='k',marker='x')
#        plt.scatter(X[0,peaks[2],i,0,0],Y[0,peaks[2],i,0,0],s=20,color='k',marker='x')
#        plt.scatter(X[0,peaks[3],i,0,0],Y[0,peaks[3],i,0,0],s=20,color='k',marker='x')
#
#        cbar_ax = fig.add_axes([0.2, -0.02, 0.6, 0.03])
#        colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
#        colorbar.set_label('Phase')
#plt.tight_layout()
#plt.show()

#==============================================================================
# Csed/UV vs Time
#==============================================================================
theta = 34.8*u.deg
Ssed = pd.DataFrame(np.squeeze(np.sin(theta)*U_V + np.cos(theta)*V_J))
Csed = pd.DataFrame(np.squeeze(np.cos(theta)*U_V - np.sin(theta)*V_J))
UV = pd.DataFrame(np.squeeze(U_V))
VJ = pd.DataFrame(np.squeeze(V_J))
SFR = pd.DataFrame(np.squeeze(models.SFR)[:,:5])

# =============================================================================
# Aldo+Ciesla
# =============================================================================
# U-V

def plot_UV():
    plt.legend(loc=4,fontsize=12,frameon=True,facecolor='w')
    plt.xlabel('Cosmic Time (Gyr)',fontsize=15)
    plt.ylabel('U - V',fontsize=15)
    plt.axhline(0.5,color='c',ls='-',lw=2,zorder=2,alpha=0.6)
    plt.axhline(0.4,color='c',ls=':',lw=1,zorder=2,alpha=0.6)
    plt.axhline(0.6,color='c',ls=':',lw=1,zorder=2,alpha=0.6)
    plt.axvline(2.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
    plt.axvline(8.5,color='k',ls='--',lw=1,alpha=0.5,zorder=1)
    plt.xlim(0.,11.5)
    plt.ylim(0.,1.2)

csfont = {'fontname':'helvetica'}

fig=plt.figure(figsize=(11,5))
ax=plt.subplot(121)
for i, (m,c) in enumerate(zip(M_today,['m','b','g','orange','firebrick'])):
    plt.plot(Ages,UV_a[i],color=c,lw=2,# label=r'log M$\rm_{today}$ = %g'%m
             label=r"RP%d"%(i+1),zorder=3)
plt.text(0.3,1.1,"AM-SFH",va='center',fontsize=15,**csfont)
plot_UV()
ax=plt.subplot(122)
for i, (m,c) in enumerate(zip(M_seed,['navy','steelblue','yellowgreen','gold','orangered'])):
    plt.plot(Ages,UV_c[i],c=c,lw=2,#label=r'log M$\rm _{seed}$ = %.1f'%np.log10(m)
             label=r"C%d"%(i+1),zorder=3)
plot_UV()
plt.text(0.3,1.1,"MS-SFH",va='center',fontsize=15,**csfont)
fig.subplots_adjust(left=0.075,right=0.975,bottom=0.125,top=0.95,wspace=0.25)
plt.savefig("N/AM+MS_UV.pdf")
#plt.show()


# UVJ
def plot_UVJ():
    plt.fill(np.concatenate((X_e[0,:,-1,0,0].value,
                         X_e[0,-1,:,0,0].value[::-1],
                         X_e[0,:,0,0,0].value[::-1])),
         np.concatenate((Y_e[0,:,-1,0,0].value,
                         Y_e[0,-1,:,0,0].value[::-1],
                         Y_e[0,:,0,0,0].value[::-1])),
         'grey',alpha=0.3)
    zg = 1.0
    plt.plot([-5,(0.55+0.253*zg-0.0533*zg**2)/0.88,1.6,1.6],
              [1.3,1.3,2.158-0.253*zg+0.0533*zg**2,2.5], color='k', alpha=0.7,zorder=1)
    leg=plt.legend(loc='best',fontsize=12,frameon=True,facecolor='w')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)
    plt.xlabel('V - J',fontsize=15)
    plt.ylabel('U - V',fontsize=15)
    plt.xlim([-0.25, 1.75])
    plt.ylim([-0.25, 2.25])
    
fig=plt.figure(figsize=(11,5))
ax=plt.subplot(121)
with quantity_support():
    for k, Av in enumerate(Dust):
        for j, (m,c) in enumerate(zip(M_today,['m','b','g','orange','firebrick'])):
                plt.plot(X_a[0,:,j,k,0], Y_a[0,:,j,k,0], label=r"RP%d"%(j+1),
                         c=c,lw=5, alpha=0.7,zorder=6-j)
    plot_UVJ()
plt.text(0.4,0.4,"AM-SFH",va='center',fontsize=15,**csfont)

ax=plt.subplot(122)                
with quantity_support():
    for k, Av in enumerate(Dust):
        for j, (m,c) in enumerate(zip(M_seed,['navy','steelblue','yellowgreen','gold','orangered'])):
                plt.plot(X_c[0,:,j,k,0], Y_c[0,:,j,k,0], label=r"C%d"%(j+1),
                         c=c,lw=5, alpha=0.7,zorder=6-j)
    plot_UVJ()
plt.text(0.4,0.4,"MS-SFH",va='center',fontsize=15,**csfont)
fig.subplots_adjust(left=0.075,right=0.975,bottom=0.125,top=0.95,wspace=0.25)
plt.savefig("N/AM+MS_UVJ.pdf")

# =============================================================================

# U-V

plt.figure(figsize=(6,6))
ax=plt.subplot(111)
for i, (m,c) in enumerate(zip(M_today,['m','b','g','orange','firebrick'])):
    plt.plot(Ages,UV[i],color=c,lw=2,# label=r'log M$\rm_{today}$ = %g'%m
             label=r"RP %d"%(i+1),zorder=3)
plot_UV()
plt.tight_layout()
#plt.savefig("N/AM_UV.pdf",dpi=400)
#plt.show()

#CSED

for i, (m,c) in enumerate(zip(M_today.astype('str'),['m','b','g','orange','firebrick'])):
    plt.plot(Ages,Csed[i],color=c,label=r'log M$\rm_{today}$ = %s'%m)
plt.legend(loc='best',fontsize=10,frameon=True,facecolor='w')
plt.xlabel('Cosmic Time (Gyr)',fontsize=15)
plt.ylabel('C$_{SED}$',fontsize=15)
plt.axhline(0.25,color='c',ls=':',lw=3)
plt.axvline(2.5,color='k',ls='--',lw=2,alpha=0.5)
plt.axvline(8.5,color='k',ls='--',lw=2,alpha=0.5)
plt.xlim(0.,11.5)
plt.ylim(0.,0.5)
plt.tight_layout()
#plt.savefig("New/Aldo/AM_UV_Csed.png",dpi=400)
plt.show()


plt.figure(figsize=(12,6))
ax=plt.subplot(121)
for i, (m,c) in enumerate(zip(M_today.astype('str'),['m','b','g','orange','firebrick'])):
    plt.plot(UV[i],np.log10(SFR[i]),label=r'log M$\rm_{today}$ = %s'%m)
plt.legend(loc='best',fontsize=11,frameon=True,facecolor='w')
plt.xlabel('U - V',fontsize=15)
plt.ylabel(r'log(sSFR/yr$^{-1}$)',fontsize=15)
plt.ylim(-11.2,-7.3)
plt.xlim(-0.25,1.74)
ax=plt.subplot(122)
for i, (m,c) in enumerate(zip(M_today.astype('str'),['m','b','g','orange','firebrick'])):
    plt.plot(VJ[i],np.log10(SFR[i]),label=r'log M$\rm_{today}$ = %s'%m)
plt.xlabel('V - J',fontsize=15)
plt.ylabel(r'log(sSFR/yr$^{-1}$)',fontsize=15)
plt.ylim(-11.2,-7.3)
plt.xlim(-0.12,1.15)
plt.legend(loc='best',fontsize=11,frameon=True,facecolor='w')
plt.tight_layout()
#plt.savefig("New/Exp_UVJ_SSFR.png",dpi=400)
plt.show()

# =============================================================================
# F-F Calibration
# =============================================================================
#plt.figure(figsize=(7,7))
#for i, (m,c) in enumerate(zip(M_today,['m','b','g','orange','firebrick'])):
#    plt.scatter(Csed[i],np.log10(SSFR[:,i]), c=c,
#                label=r'log M$\rm_{today}$ = %.1f'%m,alpha=0.7)
#C_SED0 = np.linspace(-1, 2, 100)
#plt.plot(C_SED0,-2.28*C_SED0**2-0.8*C_SED0-8.41,'k')
#plt.legend(loc="best")
#plt.xlim(-0.2,1.0)
#plt.ylim(-11.,-7.5)

#==============================================================================
# Make Table
#==============================================================================
make_table = True 
obs_error = False

def compute_SNR_obs(i,Ages=Ages,Ms=Ms):
    z = np.array([z_at_value(cosmo.age,t) for t in Ages])
    z_grid = np.vstack([z for k in range(len(M_today))]).T
    log_M = np.log10(Ms)
    
    SNR_coef = ascii.read("New/SNR_coef_all.txt").to_pandas()    
    cof = SNR_coef.iloc[i]
    snr_pred = 10**(cof.a0 + cof.a1*z_grid + cof.a2*log_M)
    snr = snr_pred.copy()
    
    snr[tg<2.5] = snr[(tg>=2.5)&(tg<=8.5)][0]
    snr[tg>8.5] = snr[(tg>=2.5)&(tg<=8.5)][-1]
    return snr

if obs_error:
    SNR_obs = np.zeros((10,len(Ages),len(M_today)))
    for i in range(10):
        SNR_obs[i] = compute_SNR_obs(i)
    
if obs_error:
    n_iter = 100
else:
    n_iter = 1
    
if make_table:
    data_tot = Table()
    fluxes = np.squeeze(synphot.fluxes[:,:,:,:,:5]).value
    
    for j in range(n_iter):
        data = Table()                   
        for i, n in enumerate(Names):
            flux = fluxes[i] * Ms[:,:]
            if obs_error:
                error = (1./SNR_obs[i]) * flux 
            else:
                error = 0.01 * flux
            flux = flux.ravel()
            error = error.ravel()
            noise = [np.random.normal(0, err) for err in error]
            data.add_columns([Column(flux+noise,'F%s'%(i+1)), Column(error,'E%s'%(i+1))])

        id = Column(name='id', data=np.arange(1,len(data)+1)) 
        zspec = Column(name='zspec', data= -1 *np.ones(len(data)))  
        data.add_column(id, 0) 
        data.add_column(zspec)  

        data_tot = vstack([data_tot,data])
        #df = data.to_pandas()

    if perturb:
        np.savetxt('N/Aldo/ST/Aldo_SFH_ST%s_p%s_obs.cat'%(A,p_id), data_tot, 
                    header=' '.join(data_tot.colnames),
                   fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
#        np.savetxt('Aldo/Perturb/Final/Aldo_SFH_P%.1f.cat'%P, table, header=' '.join(table.colnames),
#                   fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
    else:
        np.savetxt('N/Aldo/Aldo_SFH_obs.cat', data_tot, header=' '.join(data_tot.colnames),
                   fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
    
    
# =============================================================================
# different phase
# =============================================================================
   
for p_id in [0,2,3]:
    if perturb:
        if ConstP:
            A, P = 0.3, 2.5
            SFH_law = glh_absprtb
        else:
            A, p_id = 0.3, p_id
            SFH_law = glh_ST
            SFH_fit = asc.read('N/Aldo/ST/Aldo_SFH_params_ST.txt')
            SFH_pars = zip(SFH_fit.c1, SFH_fit.mu, SFH_fit.sigma,\
                           SFH_fit.h13, SFH_fit.h14,\
                           SFH_fit.c2, SFH_fit.x0, SFH_fit.gama,\
                           SFH_fit.h23, SFH_fit.h24,\
                           A*np.ones_like(M_today), SFH_fit["p%s"%p_id])        
    else:
        SFH_law = gauss_lorentz_hermite
    #    SFH_fit = asc.read('Aldo/Aldo SFH params.txt')
        SFH_fit = asc.read('N/Aldo/Aldo_SFH_params.txt')
        SFH_pars = zip(SFH_fit.c1, SFH_fit.mu, SFH_fit.sigma, SFH_fit.h13, SFH_fit.h14,\
                   SFH_fit.c2, SFH_fit.x0, SFH_fit.gama, SFH_fit.h23, SFH_fit.h24)
        
    
    Dust = np.array([1])   
    
    models = CSP(bc03, age = Ages, sfh = SFH_pars, 
                 dust = Dust, metal_ind = 1., f_esc = 1.,
                 sfh_law = SFH_law, dust_model = Calzetti,
                 neb_cont=True, neb_met=False)
    
    SED = np.squeeze(models.SED.value)
    use_bc = True
    if use_bc:
        if perturb:
            BC03_out = asc.read('N/Aldo/ST/ST%s_p%s CSP result.txt'%(A,Phases_rd[p_id]))
        else:
            BC03_out = asc.read('N/Aldo/CSP result.txt')
        lgMs = BC03_out['lgM*'].reshape((M_today.size, Ages.size)).T
        lgSSFR = BC03_out['lgsSFR'].reshape((M_today.size, Ages.size)).T            
        Ms = 10**lgMs
        SSFR = 10**lgSSFR
        
    eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')
    
    print eazy_library.filternames
    
    filters = FilterSet()
    filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
    Names = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
    Redshifts = np.arange(0.5,2.5,0.5)
    synphot = Observe(models, filters, redshift=0.001)
    
    mags = synphot.AB[:,:,:,:,:5][0]
    FUV, NUV, U, B, V, R, I, J, H, K = mags
    FUV_V = FUV - V
    NUV_V = NUV - V
    U_V = U - V
    V_J = V - J
    V_K = V - K
    
    X, Y = V_J, U_V
    plt.figure(figsize=(6,6))
    with quantity_support():
        for k, Av in enumerate(Dust):
            for j, (m,c) in enumerate(zip(M_today,['m','b','g','orange','firebrick'])):
                    plt.plot(X[0,:,j,k,0], Y[0,:,j,k,0], label=r'log M$\rm_{today}$ = %g'%m,
                             c=c,lw=5, alpha=0.9,zorder=6-j)
    #                s = plt.scatter(X[0,:,j,k,0], Y[0,:,j,k,0], c=Ages/u.Gyr, 
    #                                s=15, cmap=plt.cm.viridis_r)
    #    col = plt.colorbar(s)
    #    col.set_label('Age(Gyr)')
        zg = 1.0
        plt.plot([-5,(0.55+0.253*zg-0.0533*zg**2)/0.88,1.6,1.6],
                  [1.3,1.3,2.158-0.253*zg+0.0533*zg**2,2.5], color='k', alpha=0.7,zorder=1)
        plt.legend(loc='best',fontsize=11,frameon=True,facecolor='w')
        plt.xlabel('V - J',fontsize=12)
        plt.ylabel('U - V',fontsize=12)
        plt.xlim([-0.25, 1.75])
        plt.ylim([-0.25, 2.25])
    plt.tight_layout()
    plt.show()
    
    obs_error = False
    
    if obs_error:
        SNR_obs = np.zeros((10,len(Ages),len(M_today)))
        for i in range(10):
            SNR_obs[i] = compute_SNR_obs(i)
    
    if obs_error:
        n_iter = 100
    else:
        n_iter = 1
    
    if make_table:
        data_tot = Table()
        fluxes = np.squeeze(synphot.fluxes[:,:,:,:,:5]).value
    
        for j in range(n_iter):
            data = Table()                   
            for i, n in enumerate(Names):
                flux = fluxes[i] * Ms[:,:]
                if obs_error:
                    error = (1./SNR_obs[i]) * flux 
                else:
                    error = 0.01 * flux
                flux = flux.ravel()
                error = error.ravel()
                noise = [np.random.normal(0, err) for err in error]
                data.add_columns([Column(flux+noise,'F%s'%(i+1)), Column(error,'E%s'%(i+1))])
    
            id = Column(name='id', data=np.arange(1,len(data)+1)) 
            zspec = Column(name='zspec', data= -1 *np.ones(len(data)))  
            data.add_column(id, 0) 
            data.add_column(zspec)  
    
            data_tot = vstack([data_tot,data])
            #df = data.to_pandas()
    
        if perturb:
            np.savetxt('N/Aldo/ST/Aldo_SFH_ST%s_p%s.cat'%(A,p_id), data_tot, 
                        header=' '.join(data_tot.colnames),
                       fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
    #        np.savetxt('Aldo/Perturb/Final/Aldo_SFH_P%.1f.cat'%P, table, header=' '.join(table.colnames),
    #                   fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
        elif emission:
            np.savetxt('Aldo/Aldo_SFH_elnm.cat', data_tot, header=' '.join(data_tot.colnames),
                       fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
        else:
            np.savetxt('N/Aldo/Aldo_SFH_obs.cat', data_tot, header=' '.join(data_tot.colnames),
                       fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
    
    obs_error = True
    
    if obs_error:
        SNR_obs = np.zeros((10,len(Ages),len(M_today)))
        for i in range(10):
            SNR_obs[i] = compute_SNR_obs(i)
        
    if obs_error:
        n_iter = 100
    else:
        n_iter = 1
        
    if make_table:
        data_tot = Table()
        fluxes = np.squeeze(synphot.fluxes[:,:,:,:,:5]).value
        
        for j in range(n_iter):
            data = Table()                   
            for i, n in enumerate(Names):
                flux = fluxes[i] * Ms[:,:]
                if obs_error:
                    error = (1./SNR_obs[i]) * flux 
                else:
                    error = 0.01 * flux
                flux = flux.ravel()
                error = error.ravel()
                noise = [np.random.normal(0, err) for err in error]
                data.add_columns([Column(flux+noise,'F%s'%(i+1)), Column(error,'E%s'%(i+1))])
    
            id = Column(name='id', data=np.arange(1,len(data)+1)) 
            zspec = Column(name='zspec', data= -1 *np.ones(len(data)))  
            data.add_column(id, 0) 
            data.add_column(zspec)  
    
            data_tot = vstack([data_tot,data])
            #df = data.to_pandas()
    
        if perturb:
            np.savetxt('N/Aldo/ST/Aldo_SFH_ST%s_p%s_obs.cat'%(A,p_id), data_tot, 
                        header=' '.join(data_tot.colnames),
                       fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
    #        np.savetxt('Aldo/Perturb/Final/Aldo_SFH_P%.1f.cat'%P, table, header=' '.join(table.colnames),
    #                   fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
        elif emission:
            np.savetxt('Aldo/Aldo_SFH_elnm.cat', data_tot, header=' '.join(data_tot.colnames),
                       fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
        else:
            np.savetxt('N/Aldo/Aldo_SFH_obs.cat', data_tot, header=' '.join(data_tot.colnames),
                       fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
        