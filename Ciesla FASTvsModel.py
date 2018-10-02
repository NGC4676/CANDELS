# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:28:09 2017

@author: Qing Liu
"""

import numpy as np
import asciitable as asc
from astropy.io import ascii
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams["font.family"] = "Times New Roman"
csfont = {'fontname':'helvetica'}
from scipy.integrate import simps
import seaborn as sns
import pandas as pd

#==============================================================================
# Read model result 
#==============================================================================
lgAge = np.log10(np.linspace(0.5,11.5,111)* 1e9)
M_seed = np.linspace(5.,7.,5)
M_color = ['navy','steelblue','yellowgreen','gold','orangered']

BC03_out = asc.read('N/Ciesla/CSP result.txt')
lgMs = BC03_out['lgM*'].reshape((M_seed.size, lgAge.size)).T.ravel()
lgSSFR = BC03_out['lgsSFR'].reshape((M_seed.size, lgAge.size)).T.ravel()
                 
lgAges = np.array([lgAge for m in M_seed]).T.ravel()
M_class =  np.array([M_seed for T in range(lgAge.size)]).ravel()

## Set Mass Weighted Age (completed)
lgAges_mw = np.zeros(M_seed.size*lgAge.size)
for i in range(5):
    t = 10**lgAge/1e9 - 0.5
    m = 10**lgMs[i::5]
    dt = 0.1
#    lgAge_mw = np.log10(np.convolve(dm,t)[:lgAge.size]/m)+9
    lgAge_mw = np.log10(np.cumsum(m*dt)/m)+9
    lgAges_mw[i::5] = lgAge_mw
#==============================================================================
# Read FAST SED-fitting result
#==============================================================================
table = ascii.read("N/Ciesla/Ciesla_exp.fout",header_start=16).to_pandas()

Ages_FAST = table.lage
SFH_FAST = table.ltau
M_FAST = table.lmass
SSFR_FAST = table.lssfr
Av_FAST = table.Av

age_cond = (lgAges>np.log10(2.e9))&(lgAges<np.log10(9.e9))

def get_age_mw(t_f,tau_l,sfh):
    t_l = np.linspace(0,t_f,50)
    dt = t_f/50.
    if sfh=="exp":
        sfr_l = np.exp(-t_l/tau_l)
    elif sfh=="del":
        sfr_l = t_l*np.exp(-t_l/tau_l)
    m_l = np.array([simps(sfr_l[:(k+1)],t_l[:(k+1)]) for k in range(len(t_l))])
    age_mw = np.sum(m_l[1:]*dt)/m_l[-1]
    return age_mw

## Set Mass Weighted Age (completed)
Ages_mw_FAST = np.zeros_like(Ages_FAST)
for i in range(5):
    t = 10**Ages_FAST[i::5]/1e9
    tau = 10**SFH_FAST[i::5]/1e9
#    lgAge_mw_FAST = np.log10(np.convolve(dm,dt)[:lgAge.size]/m)+9
    t_mw_FAST = np.array([get_age_mw(t_f,tau_l,"exp") for (t_f,tau_l) in zip(t,tau)])
    lgAge_mw_FAST = np.log10(t_mw_FAST)+9
    Ages_mw_FAST[i::5] = lgAge_mw_FAST


S = 100
table2 = ascii.read("N/Ciesla/Ciesla_exp_obs.fout",header_start=16).to_pandas()
Ages_FAST2 = table2.lage
SFH_FAST2 = table2.ltau
M_FAST2 = table2.lmass
SSFR_FAST2 = table2.lssfr
Av_FAST2 = table2.Av
M_class2 = np.tile(M_class,S)

## Set Mass Weighted Age (completed)
#Ages_mw_FAST2 = np.zeros_like(Ages_FAST2)
#for i in range(5):
#    t = 10**Ages_FAST2[i::5]/1e9
#    tau = 10**SFH_FAST2[i::5]/1e9
##    lgAge_mw_FAST = np.log10(np.convolve(dm,dt)[:lgAge.size]/m)+9
#    t_mw_FAST2 = np.array([get_age_mw(t_f,tau_l,"exp") for (t_f,tau_l) in zip(t,tau)])
#    lgAge_mw_FAST2 = np.log10(t_mw_FAST2)+9
#    Ages_mw_FAST2[i::5] = lgAge_mw_FAST2
#    print i
#np.savetxt("N/Ciesla/Ages_mw_Ciesla_exp_obs.txt",Ages_mw_FAST2)

Ages_mw_FAST2 = np.loadtxt("N/Aldo/Ages_mw_Aldo_exp_obs.txt")

#==============================================================================
# One-One
#==============================================================================
fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = ['(Age/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges-4.9e8),lgMs,lgSSFR],
                                                 [Ages_FAST,M_FAST,SSFR_FAST])):
    color = 10**lgAges/1e9
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.5)
    s = plt.scatter(model[age_cond], FAST[age_cond], c=color[age_cond],
                    cmap='jet', s=20, alpha=0.7,zorder=2)
    plt.scatter(model[~age_cond], FAST[~age_cond],
                c='gray', s=20, alpha=0.5,zorder=1)
    plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize='large')
    plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize='large')
    y = pd.Series(FAST-model)[age_cond]
    print len(y[abs(y)>1.])
    y = y[abs(y<1.)]
    plt.text(0.05,0.78,'$\sigma$ = %.2f'%y.std(),
            fontsize=12,transform=ax.transAxes,color="orange") 
    plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
            fontsize=12,transform=ax.transAxes,color="gray") 
    if i==2:
        plt.xlim(-10.1,-7.8)
        plt.ylim(-10.1,-7.8)
    elif i==1:
        plt.xlim(5.5,11.5)
        plt.ylim(5.5,11.5)
    else:
        plt.xlim(8.1,10.1)
        plt.ylim(8.1,10.1)
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
colorbar.set_label('Cosmic Time (Gyr)')
ax = plt.subplot(2,2,4)
plt.hist(Av_FAST,label='Av all',alpha=0.7,zorder=1)
plt.hist(Av_FAST[age_cond],label='Av colored',alpha=0.7,zorder=2)
plt.xlim(0.8,1.2)
plt.axvline(1.0,color='k',ls='--')
plt.text(0.75,0.78,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_cond].std(),
            fontsize=12,transform=ax.transAxes,color="orange")
plt.text(0.75,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
            fontsize=12,transform=ax.transAxes,color="gray")
plt.xlabel(r'$\rm Av_{FAST}$',fontsize='large')
plt.legend(loc=2, fontsize=11, frameon=True, facecolor='w')
plt.suptitle('MS-SFH Models vs FAST Fitting (Tau Templates)',fontsize=18,y=0.95)
#plt.savefig("New/Ciesla/FAST-exp_MS-SFH.png",dpi=400)
plt.show()

# =============================================================================
# One-One M scatter 
# =============================================================================
fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = [r'(Age$\rm_{m}$/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges_mw),lgMs,lgSSFR],
                                                 [Ages_mw_FAST,M_FAST,SSFR_FAST])):
    
#    color = np.mod(5*np.pi*np.log(10**(lgAges-9))+np.pi,2*np.pi) /np.pi
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.75)
    for j,(m,c) in enumerate(zip(M_seed,M_color)):
        s = plt.scatter(model[age_cond&(M_class==m)], FAST[age_cond&(M_class==m)], 
                              c=c, s=15, alpha=0.5,zorder=2, label="C%d"%(j+1))
                              #label=r'M$\rm _{seed}$ = 10$^{%.1f}$'%m)
    plt.scatter(model[~age_cond], FAST[~age_cond],
                c='gray', s=15, alpha=0.3,zorder=1,label=None)
    plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize=12)
    plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize=12)
    for k,(m,c) in enumerate(zip(M_seed,M_color)):
        y = pd.Series(FAST-model)[age_cond&(M_class==m)]
        y = y[abs(y)<1.]
        if i==0:
            plt.legend(loc=4,fontsize=10,labelspacing=0.5,frameon=True)
#            plt.plot(lgAge,np.log10(10**lgAge-0.75e9),'--',color="brown",alpha=0.3,zorder=0)    
        plt.text(0.05,0.92-0.08*k,'$\sigma$ = %.2f'%y.std(),
                 fontsize=9,transform=ax.transAxes,color=c,**csfont) 
#    plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
#            fontsize=12,transform=ax.transAxes,color="gray")
    if i==2:
        plt.xlim(-10.2,-7.7)
        plt.ylim(-10.2,-7.7)
        
        axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44),  
                           bbox_transform=ax.transAxes)
        for m,c in zip(M_seed,M_color):
            axins.scatter(model[age_cond&(M_class==m)], (FAST-model)[age_cond&(M_class==m)], 
                                c=c, s=3, alpha=0.5,zorder=2)
        axins.scatter(model[~age_cond], (FAST-model)[~age_cond],
            c='gray', s=3, alpha=0.3,zorder=1)
        plt.axhline(0.0,color='k',ls='--',alpha=0.75)
        plt.tick_params(axis='both', which='major', labelsize=8)
        axins.set_xlim(-10.2,-7.7)  
        axins.set_ylim(-0.4,0.4)
        
    elif i==1:
        plt.xlim(7.4,11.2)
        plt.ylim(7.4,11.2)
    
        axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44), 
                           bbox_transform=ax.transAxes)
        for m,c in zip(M_seed,M_color):
            axins.scatter(model[age_cond&(M_class==m)], (FAST-model)[age_cond&(M_class==m)], 
                                c=c, s=2, alpha=0.5,zorder=2)
        axins.scatter(model[~age_cond], (FAST-model)[~age_cond],
            c='gray', s=2, alpha=0.3,zorder=1)
        plt.axhline(0.0,color='k',ls='--',alpha=0.75)
        plt.tick_params(axis='both', which='major', labelsize=8)
        axins.set_xlim(7.4,11.2)
        axins.set_ylim(-0.2,0.2)
        
    else:
        plt.xlim(8.0,10.0)
        plt.ylim(8.0,10.0)
        
ax = plt.subplot(2,2,4)
for k,(m,c) in enumerate(zip(M_seed,M_color)):
    s = plt.scatter((FAST-model)[age_cond&(M_class==m)], Av_FAST[age_cond&(M_class==m)],
                            c=c, cmap='jet', s=15, alpha=0.5,zorder=2)
    plt.text(0.05,0.92-0.08*k,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_cond&(M_class==m)].std(),
             fontsize=9,transform=ax.transAxes,color=c,**csfont)
plt.scatter((FAST-model)[~age_cond], Av_FAST[~age_cond],
                c='gray', s=15, alpha=0.1,zorder=1)
plt.xlim(-0.4,0.4)
plt.ylim(0.8,1.2)
plt.axhline(1.0,color='k',ls='--',alpha=0.75)
plt.xlabel('$\Delta$ log (sSFR/yr$^{-1}$)'+r' $\rm_{FAST-Model}$',fontsize=12)
plt.ylabel(r'$\rm Av_{FAST}$',fontsize=12)
#plt.legend(loc=1, fontsize=11, frameon=True, facecolor='w')
axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44), 
                           bbox_transform=ax.transAxes)
for k,(m,c) in enumerate(zip(M_seed,M_color)):
    plt.hist(Av_FAST[age_cond&(M_class==m)],bins=5,color=c,histtype="step",alpha=0.5,zorder=2,linewidth=2)
plt.axvline(1.0,color='k',ls='--',alpha=0.75)
plt.text(0.1,0.8,r"Av$\rm _{FAST}$", alpha=0.8,fontsize=10,transform=axins.transAxes)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.tick_params(axis='y', which='major', labelsize=0)
plt.xlim(0.8,1.2)
fig.subplots_adjust(left=0.1,right=0.95,bottom=0.1,top=0.95,wspace=0.25)
#plt.savefig("N/Ciesla/FAST-exp_MS-SFH_m.pdf",dpi=400)


#==============================================================================
# Residual
#==============================================================================
with sns.axes_style("ticks"):
    fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
    for i, (model, FAST) in enumerate(zip([np.log10(10**lgAges-4.9e8),lgMs,lgSSFR],
                                           [Ages_FAST,M_FAST,SSFR_FAST])):
        color = M_class
        ax = plt.subplot(2,2,i+1) 
        plt.axhline(0.0,color='k',ls='--',alpha=0.7)
        s = plt.scatter(model[age_cond], (FAST-model)[age_cond], c=color[age_cond],
                cmap='rainbow', s=20, alpha=0.5,zorder=2)
        plt.scatter(model[~age_cond], (FAST-model)[~age_cond],
            c='gray', s=20, alpha=0.7,zorder=1)
        y = pd.Series(FAST-model)[age_cond]
        print len(y[abs(y)>1.])
        y = y[abs(y<1.)]
        plt.text(0.05,0.78,'$\sigma$ = %.2f'%y.std(),
                fontsize=12,transform=ax.transAxes,color="orange") 
        plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
                fontsize=12,transform=ax.transAxes,color="gray")
        if i==2:
            plt.xlim(-10.1,-7.8)
            plt.ylim(-.4,.4)
            plt.ylabel(r'$\rm SSFR_{FAST} - SSFR_{model}$',fontsize=12)
            plt.xlabel(r'$\rm SSFR_{model}$',fontsize=12)
        elif i==1:
            plt.xlim(5.5,11.5)
            plt.ylim(-.2,.1)
            plt.ylabel(r'$\rm M_{*,FAST} - M_{*,model}$',fontsize=12)
            plt.xlabel(r'$\rm M_{*,model}$',fontsize=12)
        else:
            plt.xlim(8.1,10.1)
            plt.ylim(-1.1,.1)
            plt.xlabel(r"$\rm Age_{model}$")
            plt.ylabel(r"$\rm Age_{FAST}-\rm Age_{model}$")
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
    colorbar.set_label(r'$\rm M_{seed}$')
    ax = plt.subplot(2,2,4)
    s = plt.scatter((SSFR_FAST-lgSSFR)[age_cond], Av_FAST[age_cond],c=color[age_cond],s=20,cmap='rainbow',alpha=0.5,zorder=2)
    plt.scatter((SSFR_FAST-lgSSFR)[~age_cond],Av_FAST[~age_cond],c="gray",s=20,alpha=0.5,zorder=1)
    plt.axhline(1.0,color='k',ls='--')
    plt.ylabel(r'$\rm Av_{FAST}$',fontsize='large')
    plt.xlabel(r'$\rm SSFR_{FAST} - SSFR_{model}$',fontsize=12)
    plt.ylim(0.8,1.2)
    plt.xlim(-.4,.4)
    plt.subplots_adjust(wspace=0.25,hspace=0.2)
    with sns.axes_style("white"):
        nx = fig.add_axes([0.8, 0.25, 0.08, 0.08])
        nx.set_yticks([])
        nx.hist(Av_FAST,alpha=0.7)
        nx.hist(Av_FAST[age_cond],alpha=0.7)
        nx.axvline(1.0,color='k',ls='--',alpha=0.7)
        plt.text(0.05,0.78,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_cond].std(),
                fontsize=12,transform=ax.transAxes,color="orange")
        plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
                 fontsize=12,transform=ax.transAxes,color="gray")
    plt.suptitle('FAST Fitting (Tau) $-$ MS-SFH Models',fontsize=18,y=0.95)
    #plt.savefig("New/Ciesla/Residual_FAST-exp_MS-SFH.pdf")
    plt.show()
    
    
# =============================================================================
# Multiple Noise
# =============================================================================
from scipy.ndimage import gaussian_filter
S = 50
fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = ['(Age/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges-4.9e8),lgMs,lgSSFR],
                                                 [Ages_FAST,M_FAST,SSFR_FAST])):
    models = np.tile(model,S)
    age_conds = np.tile(age_cond,S)
    color = np.tile(10**lgAges/1e9,S)
    
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.9)
    plt.plot(xx,xx+.5,'k--',alpha=0.5)
    plt.plot(xx,xx-.5,'k--',alpha=0.5)
    
    s = plt.scatter(models[age_conds], FAST[age_conds], c=color[age_conds],
                    cmap='jet', s=5, alpha=0.7,zorder=2)
    plt.scatter(models[~age_conds], FAST[~age_conds],
                c='gray', s=5, alpha=0.3,zorder=1)
    
    if i==2:
        H, xbins, ybins = np.histogram2d(models, FAST,
			bins=(np.linspace(-10.25,-7.25, 50), np.linspace(-10.25,-7.25, 50)))
        XH = np.sort(pd.Series(H[H!=0].ravel()))
        Hsum = XH.sum()
        XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.01,0.32]]
        levels = [XH[k] for k in XH_levels]
        plt.contour(gaussian_filter(H, sigma=.8, order=0).T, levels, 
                    extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
                    linewidths=1,colors='black',linestyles='solid') 

    plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize='large')
    plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize='large')
    y = pd.Series(FAST-models)[age_conds]
    print len(y[abs(y)>1.])
    y = y[abs(y)<1.]
    y2 = pd.Series(FAST-models)
    y2 = y2[abs(y2)<1.]
    plt.text(0.05,0.78,'$\sigma$ = %.2f'%y.std(),
            fontsize=12,transform=ax.transAxes,color="orange") 
    plt.text(0.05,0.88,'$\sigma$ = %.2f'%y2.std(),
            fontsize=12,transform=ax.transAxes,color="gray")
    if i==2:
        plt.xlim(-10.25,-7.7)
        plt.ylim(-10.25,-7.7)
    elif i==1:
        plt.xlim(5.4,11.5)
        plt.ylim(5.4,11.5)
    else:
        plt.xlim(8.6,10.1)
        plt.ylim(7.4,10.1)
        
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
colorbar.set_label('Cosmic Time (Gyr)')

ax = plt.subplot(2,2,4)
plt.hist(Av_FAST,bins=20,label='Av all',alpha=0.7,zorder=1)
plt.xlim(0.6,1.4)
plt.hist(Av_FAST[age_conds],bins=20,label='Av colored',alpha=0.7,zorder=2)
plt.axvline(1.0,color='k',ls='--')
plt.text(0.05,0.78,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_conds].std(),
            fontsize=12,transform=ax.transAxes,color="orange")
plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
            fontsize=12,transform=ax.transAxes,color="gray")
plt.xlabel(r'$\rm Av_{FAST}$',fontsize='large')
plt.legend(loc=1, fontsize=11, frameon=True, facecolor='w')

plt.suptitle('MS-SFH Models vs FAST Fitting (Tau Templates)',fontsize=18,y=0.95)
#plt.savefig("New/Ciesla/OBS_FAST-exp_MS-SFH.pdf")
plt.show()


# =============================================================================
# Multiple Noise M scatter
# =============================================================================

fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = [r'(Age$\rm_{m}$/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges_mw),lgMs,lgSSFR],
                                                 [Ages_mw_FAST2,M_FAST2,SSFR_FAST2])):
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.75)
    models = np.tile(model,S)
    age_conds = np.tile(age_cond,S)
    for j,(m,c) in enumerate(zip(M_seed,M_color)):
        s = plt.scatter(models[age_conds&(M_class2==m)], FAST[age_conds&(M_class2==m)], 
                              c=c, cmap='jet', s=8, alpha=0.3,zorder=2,label="C%d"%(j+1))
                              #label=r'M$\rm _{seed}$ = 10$^{%.1f}$'%m)
    plt.scatter(models[~age_conds], FAST[~age_conds],
                c='gray', s=8, alpha=0.1,zorder=1,label=None)
    plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize='large')
    plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize='large')
    for k,(m,c) in enumerate(zip(M_seed,M_color)):
        y = pd.Series(FAST-models)[age_conds&(M_class2==m)]
        y = y[abs(y)<1.]
        if i==0:
#            plt.plot(lgAge,np.log10(10**lgAge-0.75e9),'--',color="brown",alpha=0.3,zorder=0)
            plt.legend(loc=4,fontsize=10,labelspacing=0.5,handletextpad=0.1,frameon=True)  
        plt.text(0.05,0.92-0.08*k,'$\sigma$ = %.2f'%y.std(),
             fontsize=9,transform=ax.transAxes,color=c,**csfont) 
#    plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
#            fontsize=12,transform=ax.transAxes,color="gray")
    if i==2:
        plt.xlim(-10.2,-7.7)
        plt.ylim(-10.2,-7.7)
        axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44), 
                           bbox_transform=ax.transAxes)
        for m,c in zip(M_seed,M_color):
            axins.scatter(models[age_conds&(M_class2==m)], (FAST-models)[age_conds&(M_class2==m)], 
                                c=c, s=1, alpha=0.05,zorder=2)
        axins.scatter(models[~age_conds], (FAST-models)[~age_conds],
            c='gray', s=1, alpha=0.03,zorder=1)
        plt.axhline(0.0,color='k',ls='--',alpha=0.75)
        plt.tick_params(axis='both', which='major', labelsize=8)
        axins.set_xlim(-10.2,-7.7)
        axins.set_ylim(-1.,0.6)
    elif i==1:
        plt.xlim(7.4,11.2)
        plt.ylim(7.4,11.2)
        axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44), 
                           bbox_transform=ax.transAxes)
        for m,c in zip(M_seed,M_color):
            axins.scatter(models[age_conds&(M_class2==m)], (FAST-models)[age_conds&(M_class2==m)], 
                                c=c, s=1, alpha=0.05,zorder=2)
        axins.scatter(models[~age_conds], (FAST-models)[~age_conds],
            c='gray', s=1, alpha=0.03,zorder=1)
        plt.axhline(0.0,color='k',ls='--',alpha=0.75)
        plt.tick_params(axis='both', which='major', labelsize=8)
        axins.set_xlim(7.4,11.2)
        axins.set_ylim(-.4, .2)
    else:
        plt.xlim(7.8,9.8)
        plt.ylim(7.8,9.8)
        
ax = plt.subplot(2,2,4)
for k,(m,c) in enumerate(zip(M_seed,M_color)):
    s = plt.scatter((FAST-models)[age_conds&(M_class2==m)], Av_FAST2[age_conds&(M_class2==m)],
                            c=c, cmap='jet', s=8, alpha=0.3,zorder=2)
    plt.text(0.05,0.92-0.08*k,'$\sigma$ = %.2f'%pd.Series(Av_FAST2-1)[age_conds&(M_class2==m)].std(),
             fontsize=9,transform=ax.transAxes,color=c,**csfont)
plt.scatter((FAST-models)[~age_conds], Av_FAST2[~age_conds],
                c='gray', s=8, alpha=0.1,zorder=1)
plt.xlim(-1.,1.)
plt.ylim(0.7,1.3)
plt.axhline(1.0,color='k',ls='--',alpha=0.75)
#plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
#         fontsize=12,transform=ax.transAxes,color="gray")
plt.xlabel('$\Delta$ log (sSFR/yr$^{-1}$)'+r' $\rm_{FAST-Model}$',fontsize=12)
plt.ylabel(r'$\rm Av_{FAST}$',fontsize=12)
#plt.legend(loc=1, fontsize=11, frameon=True, facecolor='w')
axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44), 
                           bbox_transform=ax.transAxes)
for k,(m,c) in enumerate(zip(M_seed,M_color)):
    plt.hist(Av_FAST2[age_conds&(M_class2==m)],color=c,histtype="step",alpha=0.5,zorder=2,linewidth=2)
plt.axvline(1.0,color='k',ls='--',alpha=0.75)
plt.text(0.06,0.84,r"Av$\rm _{FAST}$", alpha=0.8,fontsize=10,transform=axins.transAxes)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.tick_params(axis='y', which='major', labelsize=0)
plt.xlim(0.7,1.3)
fig.subplots_adjust(left=0.1,right=0.95,bottom=0.1,top=0.95,wspace=0.25)
#plt.savefig("N/Ciesla/FAST-exp_MS-SFH_m_obs.pdf",dpi=350)


# =============================================================================
# AM vs MS
# =============================================================================
M_seed = np.array([9.0,9.5,10.0,10.5,11.0])

phase_start = lgAge[0]

A, P = 0.3, 2.5
#BC03_out = asc.read('New/Perturb/CSP result ST.txt')
BC03_out = asc.read('New/Aldo/CSP result.txt')

lgMs = BC03_out['lgM*'].reshape((M_seed.size, lgAge.size)).T.ravel()
lgSSFR = BC03_out['lgsSFR'].reshape((M_seed.size, lgAge.size)).T.ravel()
                 
lgAges = np.array([lgAge for m in M_seed]).T.ravel()
M_class =  np.array([M_seed for T in range(lgAge.size)]).ravel()

#SSFR_model=lgSSFR.reshape((lgAge.size,M_seed.size)).T