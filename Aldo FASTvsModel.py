# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 23:37:04 2017

@author: Qing Liu
"""
import numpy as np
import pandas as pd
import asciitable as asc
from astropy.io import ascii
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import simps
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
lgAge = np.log10(np.linspace(0.5,11.5,111)* 1e9)
#M_today = np.array([9.0,9.5,10.0,10.5,11.0])
M_today = np.array([9.5,10.0,10.5,10.75,11.0])
M_color = ['m','b','g','orange','firebrick']


A, P = 0.3, 2.5
#BC03_out = asc.read('New/Perturb/CSP result ST.txt')
BC03_out = asc.read('N/Aldo/CSP result.txt')

lgMs = BC03_out['lgM*'].reshape((M_today.size, lgAge.size)).T.ravel()
lgSSFR = BC03_out['lgsSFR'].reshape((M_today.size, lgAge.size)).T.ravel()
                 
lgAges = np.array([lgAge for m in M_today]).T.ravel()
M_class =  np.array([M_today for T in range(lgAge.size)]).ravel()


## Set Mass Weighted Age (completed)
lgAges_mw = np.zeros(M_today.size*lgAge.size)
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
#table = ascii.read('New/Perturb/Aldo_ST_exp.fout',header_start=16).to_pandas()
table = ascii.read("N/Aldo/Aldo_exp.fout",header_start=16).to_pandas()

Ages_FAST = table.lage
SFH_FAST = table.ltau
M_FAST = table.lmass
SSFR_FAST = table.lssfr
Av_FAST = table.Av

def get_age_mw(t_f,tau_l):
    t_l = np.linspace(0,t_f,50)
    dt = t_f/50.
    sfr_l = np.exp(-t_l/tau_l)
    m_l = np.array([simps(sfr_l[:(k+1)],t_l[:(k+1)]) for k in range(len(t_l))])
    age_mw = np.sum(m_l[1:]*dt)/m_l[-1]
    return age_mw

## Set Mass Weighted Age (completed)
Ages_mw_FAST = np.zeros_like(Ages_FAST)
for i in range(5):
    t = 10**Ages_FAST[i::5]/1e9
    tau = 10**SFH_FAST[i::5]/1e9
#    lgAge_mw_FAST = np.log10(np.convolve(dm,dt)[:lgAge.size]/m)+9
    t_mw_FAST = np.array([get_age_mw(t_f,tau_l) for (t_f,tau_l) in zip(t,tau)])
    lgAge_mw_FAST = np.log10(t_mw_FAST)+9
    Ages_mw_FAST[i::5] = lgAge_mw_FAST


age_cond = (lgAges>np.log10(2.e9))&(lgAges<np.log10(9.e9))

S = 100
table2 = ascii.read("N/Aldo/Aldo_exp_obs.fout",header_start=16).to_pandas()
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
#    t_mw_FAST2 = np.array([get_age_mw(t_f,tau_l) for (t_f,tau_l) in zip(t,tau)])
#    lgAge_mw_FAST2 = np.log10(t_mw_FAST2)+9
#    Ages_mw_FAST2[i::5] = lgAge_mw_FAST2
#    print i
#np.savetxt("N/Aldo/Ages_mw_Aldo_exp_obs.txt",Ages_mw_FAST2)

Ages_mw_FAST2 = np.loadtxt("N/Aldo/Ages_mw_Aldo_exp_obs.txt")


#==============================================================================
# Residual
#==============================================================================

with sns.axes_style("ticks"):
    fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
    for i, (model, FAST) in enumerate(zip([np.log10(10**lgAges-4.9e8),lgMs,lgSSFR],
                                           [Ages_FAST,M_FAST,SSFR_FAST])):
#        color = np.mod(5*np.pi*np.log(10**(lgAges-9))+np.pi,2*np.pi) /np.pi
        color = M_class
        
        ax = plt.subplot(2,2,i+1) 
        plt.axhline(0.0,color='k',ls='--',alpha=0.7)
        s = plt.scatter(model[age_cond], (FAST-model)[age_cond], c=color[age_cond],
                cmap='rainbow', s=20, alpha=0.5,zorder=2)
        plt.scatter(model[~age_cond], (FAST-model)[~age_cond],
            c='gray', s=20, alpha=0.5,zorder=1)
        y = pd.Series(FAST-model)[age_cond]
        y = y[abs(y<1.)]
        plt.text(0.05,0.78,'$\sigma$ = %.2f'%y.std(),
                fontsize=12,transform=ax.transAxes,color="orange") 
        plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
                fontsize=12,transform=ax.transAxes,color="gray")
        if i==2:
            plt.xlim(-10.9,-7.6)
            plt.ylim(-.4,.4)
            #plt.ylim(-1.,1.)
            plt.ylabel(r'$\rm SSFR_{FAST} - SSFR_{model}$',fontsize=12)
            plt.xlabel(r'$\rm SSFR_{model}$',fontsize=12)
        elif i==1:
            plt.xlim(5.5,11.5)
            plt.ylim(-.16,.16)
            #plt.ylim(-.4,.4)
            plt.ylabel(r'$\rm M_{*,FAST} - M_{*,model}$',fontsize=12)
            plt.xlabel(r'$\rm M_{*,model}$',fontsize=12)
        else:
            plt.xlim(8.5,10.2)
            plt.ylim(-.75,.25)
            #plt.ylim(-1.5,.4)
            plt.xlabel(r"$\rm Age_{model}$")
            plt.ylabel(r"$\rm Age_{FAST}-\rm Age_{model}$")
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
    colorbar.set_label('$M_{today}$')
#    colorbar.set_label('$Phase$')
    ax = plt.subplot(2,2,4)
          
    s = plt.scatter((SSFR_FAST-lgSSFR)[age_cond],Av_FAST[age_cond],c=color[age_cond],s=20,cmap='rainbow',alpha=0.5,zorder=2)
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
        nx.set_xlim(0.8,1.2)
    plt.text(0.05,0.78,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_cond].std(),
        fontsize=12,transform=ax.transAxes,color="orange")
    plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
             fontsize=12,transform=ax.transAxes,color="gray")
    plt.suptitle('FAST Fitting (Tau) $-$ AM-SFH Models',fontsize=18,y=0.95)
    #plt.savefig("New/Aldo/Residual_FAST-exp_AM-SFH.png",dpi=400)
    plt.show()
    
#==============================================================================
# One-One
#==============================================================================
fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = ['(Age/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges),lgMs,lgSSFR],
                                                 [Ages_FAST,M_FAST,SSFR_FAST])):
    color = 10**lgAges/1e9
#    color = np.mod(5*np.pi*np.log(10**(lgAges-9))+np.pi,2*np.pi) /np.pi
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.5)
        
    s = plt.scatter(model[age_cond], FAST[age_cond], c=color[age_cond],
                    cmap='jet', s=20, alpha=0.5,zorder=2)
    plt.scatter(model[~age_cond], FAST[~age_cond],
                c='gray', s=20, alpha=0.3,zorder=1)
    plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize='large')
    plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize='large')
    y = pd.Series(FAST-model)[age_cond]
    y = y[abs(y<1.)]
    plt.text(0.05,0.78,'$\sigma$ = %.2f'%y.std(),
            fontsize=12,transform=ax.transAxes,color="orange") 
    plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
            fontsize=12,transform=ax.transAxes,color="gray")
    if i==2:
        plt.xlim(-10.8,-7.8)
        plt.ylim(-10.8,-7.8)
    elif i==1:
        plt.xlim(7.4,11.2)
        plt.ylim(7.4,11.2)
    else:
        plt.xlim(8.4,10.1)
        plt.ylim(8.4,10.1)
        plt.xlabel('log (Cosmic Time/yr)',fontsize='large')
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
colorbar.set_label('Cosmic Time (Gyr)')
#colorbar.set_label('Phase')
ax = plt.subplot(2,2,4)
plt.hist(Av_FAST,label='Av all',alpha=0.7,zorder=1)
plt.hist(Av_FAST[age_cond],label='Av colored',alpha=0.7,zorder=2)
plt.xlim(0.8,1.2)
plt.axvline(1.0,color='k',ls='--')
plt.text(0.05,0.78,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_cond].std(),
        fontsize=12,transform=ax.transAxes,color="orange")
plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
         fontsize=12,transform=ax.transAxes,color="gray")
plt.xlabel(r'$\rm Av_{FAST}$',fontsize='large')
plt.legend(loc=1, fontsize=11, frameon=True, facecolor='w')
fig.subplots_adjust(wspace=0.2,hspace=0.2)
plt.suptitle('AM-SFH Models vs FAST Fitting (Tau Templates)',fontsize=18,y=0.95)
#plt.savefig("N/Aldo/FAST-exp_AM-SFH.png",dpi=400)
plt.show()

# =============================================================================
# One-One M scatter 
# =============================================================================
fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = [r'(Age$\rm_{mw}$/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges_mw),lgMs,lgSSFR],
                                                 [Ages_mw_FAST,M_FAST,SSFR_FAST])):
    
#    color = np.mod(5*np.pi*np.log(10**(lgAges-9))+np.pi,2*np.pi) /np.pi
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.75)
    for m,c in zip(M_today,M_color):
        s = plt.scatter(model[age_cond&(M_class==m)], FAST[age_cond&(M_class==m)], 
                              c=c, s=10, alpha=0.5,zorder=2,
                              label=r'M$\rm _{today}$ = 10$^{%.1f}$'%m)
    plt.scatter(model[~age_cond], FAST[~age_cond],
                c='gray', s=10, alpha=0.3,zorder=1,label=None)
    plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize='large')
    plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize='large')
    for k,(m,c) in enumerate(zip(M_today,M_color)):
        y = pd.Series(FAST-model)[age_cond&(M_class==m)]
        y = y[abs(y<1.)]
        if i==0:
            plt.legend(loc=4,fontsize=10,labelspacing=0.5,frameon=True)
#            plt.plot(lgAge,np.log10(10**lgAge-0.75e9),'--',color="brown",alpha=0.3,zorder=0)   
        plt.text(0.05,0.92-0.08*k,'$\sigma$ = %.2f'%y.std(),
                 fontsize=9,transform=ax.transAxes,color=c) 
    if i==2:
        plt.xlim(-10.8,-7.8)
        plt.ylim(-10.8,-7.8)
        
        axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44),  
                           bbox_transform=ax.transAxes)
        for m,c in zip(M_today,M_color):
            axins.scatter(model[age_cond&(M_class==m)], (FAST-model)[age_cond&(M_class==m)], 
                                c=c, s=2, alpha=0.5,zorder=2)
        axins.scatter(model[~age_cond], (FAST-model)[~age_cond],
            c='gray', s=2, alpha=0.3,zorder=1)
        plt.axhline(0.0,color='k',ls='--',alpha=0.75)
        plt.tick_params(axis='both', which='major', labelsize=8)
        axins.set_xlim(-10.8,-7.8)  
        axins.set_ylim(-0.4,0.55)
        
    elif i==1:
        plt.xlim(7.4,11.2)
        plt.ylim(7.4,11.2)
    
        axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44), 
                           bbox_transform=ax.transAxes)
        for m,c in zip(M_today,M_color):
            axins.scatter(model[age_cond&(M_class==m)], (FAST-model)[age_cond&(M_class==m)], 
                                c=c, s=2, alpha=0.5,zorder=2)
        axins.scatter(model[~age_cond], (FAST-model)[~age_cond],
            c='gray', s=2, alpha=0.3,zorder=1)
        plt.axhline(0.0,color='k',ls='--',alpha=0.75)
        plt.tick_params(axis='both', which='major', labelsize=8)
        axins.set_xlim(7.4,11.2)
        axins.set_ylim(-0.2,0.2)
        
    else:
#        plt.xlim(9.,10.15)
#        plt.ylim(8.45,10.15)
#        plt.xlabel('log (Cosmic Time/yr)',fontsize='large')
        plt.xlim(8.15,10.)
        plt.ylim(8.15,10.)
        
ax = plt.subplot(2,2,4)
for k,(m,c) in enumerate(zip(M_today,M_color)):
    s = plt.scatter((FAST-model)[age_cond&(M_class==m)], Av_FAST[age_cond&(M_class==m)],
                            c=c, cmap='jet', s=10, alpha=0.5,zorder=2)
    plt.text(0.05,0.92-0.08*k,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_cond&(M_class==m)].std(),
             fontsize=9,transform=ax.transAxes,color=c)
plt.scatter((FAST-model)[~age_cond], Av_FAST[~age_cond],
                c='gray', s=5, alpha=0.1,zorder=1)
plt.xlim(-0.45,0.45)
plt.ylim(0.8,1.2)
plt.axhline(1.0,color='k',ls='--',alpha=0.75)
#plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
#         fontsize=12,transform=ax.transAxes,color="gray")
plt.xlabel('$\Delta$ log (sSFR/yr$^{-1}$)'+r' $\rm_{FAST-Model}$',fontsize='large')
plt.ylabel(r'$\rm Av_{FAST}$',fontsize='large')
#plt.legend(loc=1, fontsize=11, frameon=True, facecolor='w')
axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44), 
                           bbox_transform=ax.transAxes)
for k,(m,c) in enumerate(zip(M_today,M_color)):
    plt.hist(Av_FAST[age_cond&(M_class==m)],bins=5,color=c,histtype="step",alpha=0.5,zorder=2,linewidth=2)
plt.axvline(1.0,color='k',ls='--',alpha=0.75)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.tick_params(axis='y', which='major', labelsize=0)
plt.xlim(0.8,1.2)
#plt.legend(loc=1, fontsize=11, frameon=True, facecolor='w')
#plt.suptitle('AM-SFH Models vs FAST Fitting (Tau Templates)',fontsize=18,y=0.95)
fig.subplots_adjust(left=0.1,right=0.95,bottom=0.1,top=0.95,wspace=0.25)
#plt.savefig("N/Aldo/FAST-exp_AM-SFH_m.pdf",dpi=400)
plt.show()

# =============================================================================
# Multiple Noise 
# =============================================================================
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
                    cmap='jet', s=10, alpha=0.3,zorder=2)
    plt.scatter(models[~age_conds], FAST[~age_conds],
                c='gray', s=10, alpha=0.1,zorder=1)
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
    
#    if i==2:
#        H, xbins, ybins = np.histogram2d(models, FAST,
#			bins=(np.linspace(-10.7,-7.7, 50), np.linspace(-10.7,-7.7, 50)))
#        XH = np.sort(pd.Series(H[H!=0].ravel()))
#        Hsum = XH.sum()
#        XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.01,0.32]]
#        levels = [XH[k] for k in XH_levels]
#        plt.contour(gaussian_filter(H, sigma=.8, order=0).T, levels, 
#                    extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
#                    linewidths=1,colors='black',linestyles='solid') 
    
    if i==2:
        plt.xlim(-10.7,-7.9)
        plt.ylim(-10.7,-7.9)
    elif i==1:
        plt.xlim(6.9,11.5)
        plt.ylim(6.9,11.5)
    else:
        plt.xlim(8.6,10.1)
        plt.ylim(7.4,10.1)
        
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
colorbar.set_label('Cosmic Time (Gyr)')

ax = plt.subplot(2,2,4)
plt.hist(Av_FAST,bins=20,label='Av all',alpha=0.7,zorder=1)
plt.xlim(0.7,1.3)
plt.hist(Av_FAST[age_conds],bins=20,label='Av',alpha=0.7,zorder=2)
plt.axvline(1.0,color='k',ls='--')
plt.text(0.05,0.78,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_conds].std(),
            fontsize=12,transform=ax.transAxes,color="orange")
plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
            fontsize=12,transform=ax.transAxes,color="gray")
plt.xlabel(r'$\rm Av_{FAST}$',fontsize='large')
plt.legend(loc=1, fontsize=11, frameon=True, facecolor='w')
fig.subplots_adjust(wspace=0.2,hspace=0.2)
plt.suptitle('AM-SFH Models vs FAST Fitting (Tau Templates)',fontsize=18,y=0.95)
#plt.savefig("New/Aldo/OBS_FAST-exp_AM-SFH.png",dpi=400)
plt.show()

# =============================================================================
# Multiple Noise M scatter
# =============================================================================
fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = [r'(Age$\rm_{mw}$/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges_mw),lgMs,lgSSFR],
                                                 [Ages_mw_FAST2,M_FAST2,SSFR_FAST2])):
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.75)
    models = np.tile(model,S)
    age_conds = np.tile(age_cond,S)
    for m,c in zip(M_today,M_color):
        s = plt.scatter(models[age_conds&(M_class2==m)], FAST[age_conds&(M_class2==m)], 
                              c=c, cmap='jet', s=5, alpha=0.3,zorder=2,
                              label=r'M$\rm _{today}$ = 10$^{%.1f}$'%m)
    plt.scatter(models[~age_conds], FAST[~age_conds],
                c='gray', s=5, alpha=0.1,zorder=1,label=None)
    plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize='large')
    plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize='large')
    for k,(m,c) in enumerate(zip(M_today,M_color)):
        y = pd.Series(FAST-models)[age_conds&(M_class2==m)]
        y = y[abs(y)<1.]
        if i==0:
#            plt.plot(lgAge,np.log10(10**lgAge-0.75e9),'--',color="brown",alpha=0.3,zorder=0)
            plt.legend(loc=4,fontsize=10,labelspacing=0.5,frameon=True)
        plt.text(0.05,0.92-0.08*k,'$\sigma$ = %.2f'%y.std(),
             fontsize=9,transform=ax.transAxes,color=c) 
#    plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
#            fontsize=12,transform=ax.transAxes,color="gray")
    if i==2:
        plt.xlim(-10.6,-8.0)
        plt.ylim(-10.6,-8.0)
        axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44), 
                           bbox_transform=ax.transAxes)
        for m,c in zip(M_today,M_color):
            axins.scatter(models[age_conds&(M_class2==m)], (FAST-models)[age_conds&(M_class2==m)], 
                                c=c, s=1, alpha=0.1,zorder=2)
        axins.scatter(models[~age_conds], (FAST-models)[~age_conds],
            c='gray', s=1, alpha=0.1,zorder=1)
        plt.axhline(0.0,color='k',ls='--',alpha=0.75)
        plt.tick_params(axis='both', which='major', labelsize=8)
        axins.set_xlim(-10.2,-7.7)
        axins.set_ylim(-.6,0.6)
    elif i==1:
        plt.xlim(7.4,11.2)
        plt.ylim(7.4,11.2)
        axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44), 
                           bbox_transform=ax.transAxes)
        for m,c in zip(M_today,M_color):
            axins.scatter(models[age_conds&(M_class2==m)], (FAST-models)[age_conds&(M_class2==m)], 
                                c=c, s=1, alpha=0.1,zorder=2)
        axins.scatter(models[~age_conds], (FAST-models)[~age_conds],
            c='gray', s=1, alpha=0.1,zorder=1)
        plt.axhline(0.0,color='k',ls='--',alpha=0.75)
        plt.tick_params(axis='both', which='major', labelsize=8)
        axins.set_xlim(7.4,11.2)
        axins.set_ylim(-.4, .2)
    else:
        plt.xlim(8.0,10.0)
        plt.ylim(8.0,10.0)

ax = plt.subplot(2,2,4)
for k,(m,c) in enumerate(zip(M_today,M_color)):
    s = plt.scatter((FAST-models)[age_conds&(M_class2==m)], Av_FAST2[age_conds&(M_class2==m)],
                            c=c, cmap='jet', s=5, alpha=0.1,zorder=2)
    plt.text(0.05,0.92-0.08*k,'$\sigma$ = %.2f'%pd.Series(Av_FAST2-1)[age_conds&(M_class2==m)].std(),
             fontsize=9,transform=ax.transAxes,color=c)
plt.scatter((FAST-models)[~age_conds], Av_FAST2[~age_conds],
                c='gray', s=5, alpha=0.1,zorder=1)
plt.xlim(-0.8,0.8)
plt.ylim(0.7,1.3)
plt.axhline(1.0,color='k',ls='--',alpha=0.75)
#plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
#         fontsize=12,transform=ax.transAxes,color="gray")
plt.xlabel('$\Delta$ log (sSFR/yr$^{-1}$)'+r' $\rm_{FAST-Model}$',fontsize='large')
plt.ylabel(r'$\rm Av_{FAST}$',fontsize='large')
#plt.legend(loc=1, fontsize=11, frameon=True, facecolor='w')
axins = inset_axes(ax, 1.4, 1.2, bbox_to_anchor=(0.98,0.44), 
                           bbox_transform=ax.transAxes)
for k,(m,c) in enumerate(zip(M_today,M_color)):
    plt.hist(Av_FAST2[age_conds&(M_class2==m)],color=c,histtype="step",alpha=0.5,zorder=2,linewidth=2)
plt.axvline(1.0,color='k',ls='--',alpha=0.75)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.tick_params(axis='y', which='major', labelsize=0)
plt.xlim(0.7,1.3)
#plt.suptitle('MS-SFH Models vs FAST Fitting (Tau Templates)',fontsize=18,y=0.95)
fig.subplots_adjust(left=0.1,right=0.95,bottom=0.1,top=0.95,wspace=0.25)
#plt.savefig("N/Aldo/FAST-exp_MS-SFH_m_obs.pdf",dpi=350)


#==============================================================================
# phase
#==============================================================================
#m=50
#with sns.axes_style("ticks"):
#    fig,axes = plt.subplots(figsize=(11,10), nrows=2, ncols=2)
#    labels = ['log($T_{BB}$ / yr)', 'log($M_{*,model}$ / $M_{\odot}$)', 'log($sSFR_{model}$ / $yr^{-1})$']
#    for i, (l, model, FAST) in enumerate(zip(labels,[lgAges,lgMs,lgSSFR],
#                                                     [Ages_FAST,M_FAST,SSFR_FAST])):
#        color = np.mod(5*np.pi*np.log(10**(lgAges-9))+np.pi,2*np.pi) /np.pi
#        #color = M_class

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
#    for (phase,m) in zip(np.arange(0.,2.,0.5),['o','v','^','s']):
#        cond = (color>phase)&(color<phase+0.5)
#        s = plt.scatter(Av_FAST[cond],(SSFR_FAST-lgSSFR)[cond],s=30,marker=m, alpha=0.7)
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


# =============================================================================
# All
# =============================================================================

S = 100
fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = ['(Age/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges),lgMs,lgSSFR],
                                                 [Ages_FAST2,M_FAST2,SSFR_FAST2])):
    models = np.tile(model,S)
    age_conds = np.tile(age_cond,S)
    color = np.tile(10**lgAges/1e9,S)

    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.9)
    plt.plot(xx,xx+.5,'k--',alpha=0.5)
    plt.plot(xx,xx-.5,'k--',alpha=0.5)
    s = plt.scatter(models[age_conds], FAST[age_conds], c=color[age_conds],
                    cmap='jet', s=10, alpha=0.3,zorder=2)
    plt.scatter(models[~age_conds], FAST[~age_conds],
                c='gray', s=10, alpha=0.1,zorder=1)
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
ax = plt.subplot(2,2,4)
sns.distplot(Av_FAST2,bins=20,label='Av all',ax=ax)
sns.distplot(Av_FAST2[age_conds],bins=20,label='Av colored',ax=ax)
plt.text(0.05,0.78,'$\sigma$ = %.2f'%pd.Series(Av_FAST2-1)[age_conds].std(),
            fontsize=12,transform=ax.transAxes,color="orange")
plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(Av_FAST2-1).std(),
            fontsize=12,transform=ax.transAxes,color="gray")
plt.legend(loc=1, fontsize=11, frameon=True, facecolor='w')

fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
colorbar.set_label('Cosmic Time (Gyr)')


for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges),lgMs,lgSSFR],
                                                 [Ages_FAST,M_FAST,SSFR_FAST])):
    color = 10**lgAges/1e9
#    color = np.mod(5*np.pi*np.log(10**(lgAges-9))+np.pi,2*np.pi) /np.pi
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.5)
    s = plt.scatter(model[age_cond], FAST[age_cond], c=color[age_cond],
                    cmap='jet', s=20, edgecolors="k",alpha=0.5,zorder=2)
    plt.scatter(model[~age_cond], FAST[~age_cond],
                c='gray', s=20, edgecolors="k",alpha=0.3,zorder=1)
    plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize='large')
    plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize='large')
    y = pd.Series(FAST-model)[age_cond]
    y = y[abs(y<1.)]
    plt.text(0.05,0.58,'$\sigma$ = %.2f'%y.std(),
            fontsize=12,transform=ax.transAxes,color="orange") 
    plt.text(0.05,0.68,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
            fontsize=12,transform=ax.transAxes,color="gray")
    if i==2:
        plt.xlim(-10.9,-7.6)
        plt.ylim(-10.9,-7.6)
    elif i==1:
        plt.xlim(6.9,11.5)
        plt.ylim(6.9,11.5)
    else:
        plt.xlim(8.6,10.1)
        plt.ylim(7.5,10.1)
ax = plt.subplot(2,2,4)
sns.distplot(Av_FAST,label='Av all (obs)',ax=ax)
sns.distplot(Av_FAST[age_cond],label='Av colored (obs)',ax=ax)
plt.xlim(0.75,1.25)
plt.ylim(0.,8.5)
plt.axvline(1.0,color='k',ls='--')
plt.text(0.05,0.58,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1)[age_cond].std(),
        fontsize=12,transform=ax.transAxes,color="orange")
plt.text(0.05,0.68,'$\sigma$ = %.2f'%pd.Series(Av_FAST-1).std(),
         fontsize=12,transform=ax.transAxes,color="gray")
plt.xlabel(r'$\rm Av_{FAST}$',fontsize='large')
plt.legend(loc=1, fontsize=9, frameon=True, facecolor='w')
fig.subplots_adjust(wspace=0.2,hspace=0.2)
plt.suptitle('AM-SFH Models vs FAST Fitting (Tau Templates)',fontsize=18,y=0.95)
#plt.savefig("N/FAST-exp_AM-SFH.png",dpi=300)
plt.show()    

# =============================================================================
# All M scatter
# =============================================================================

fig,axes = plt.subplots(figsize=(10,9), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = ['(Age/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges),lgMs,lgSSFR],
                                                 [Ages_FAST2,M_FAST2,SSFR_FAST2])):
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.9)
    plt.plot(xx,xx+.5,'k--',alpha=0.5)
    plt.plot(xx,xx-.5,'k--',alpha=0.5)
    
    models = np.tile(model,S)
    age_conds = np.tile(age_cond,S)
    
    for m,c in zip(M_today,M_color):
        s = plt.scatter(models[age_conds&(M_class2==m)], FAST[age_conds&(M_class2==m)], 
                              c=c, cmap='jet', s=15, alpha=0.1,zorder=2)
    plt.scatter(models[~age_conds], FAST[~age_conds],
                c='gray', s=15, alpha=0.1,zorder=1)
    plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize='large')
    plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize='large')
    for k,(m,c) in enumerate(zip(M_today,M_color)):
        y = pd.Series(FAST-models)[age_conds&(M_class2==m)]
        y = y[abs(y<1.)]
        plt.text(0.05,0.92-0.08*k,'$\sigma$ = %.2f'%y.std(),
                 fontsize=9,transform=ax.transAxes,color=c) 
#    plt.text(0.05,0.88,'$\sigma$ = %.2f'%pd.Series(FAST-model).std(),
#            fontsize=12,transform=ax.transAxes,color="gray")
    if i==2:
        plt.xlim(-10.8,-7.8)
        plt.ylim(-10.8,-7.8)
    elif i==1:
        plt.xlim(7.4,11.2)
        plt.ylim(7.4,11.2)
    else:
        plt.xlim(8.4,10.1)
        plt.ylim(7.5,10.1)
        plt.xlabel('log (Cosmic Time/yr)',fontsize='large')
ax = plt.subplot(2,2,4)
for k,(m,c) in enumerate(zip(M_today,M_color)):
    plt.hist(Av_FAST2[age_conds&(M_class2==m)],color=c,density=True,alpha=0.3,zorder=2)
    plt.text(0.05,0.88-0.08*k,'$\sigma$ = %.2f'%pd.Series(Av_FAST2-1)[age_conds&(M_class2==m)].std(),
             fontsize=9,transform=ax.transAxes,color=c)
plt.xlim(0.75,1.25)
plt.axvline(1.0,color='k',ls='--')
plt.xlabel(r'$\rm Av_{FAST}$',fontsize='large')

legends = ['(Age/yr)', r'$(\rm{M_*}/\rm{M_{\odot}})$', r'(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[np.log10(10**lgAges),lgMs,lgSSFR],
                                                 [Ages_FAST,M_FAST,SSFR_FAST])):
    ax = plt.subplot(2,2,i+1) 
    for m,c in zip(M_today,M_color):
        s = plt.scatter(model[age_cond&(M_class==m)], FAST[age_cond&(M_class==m)], 
                              c="None", edgecolors="k", s=15, alpha=0.5,zorder=3)
    plt.scatter(model[~age_cond], FAST[~age_cond],
                c="None", edgecolors="k", s=15, alpha=0.1,zorder=1)
    plt.xlabel('log'+l+r'$\rm_{Model}$',fontsize='large')
    plt.ylabel('log'+l+r'$\rm_{FAST}$',fontsize='large')
    for k,(m,c) in enumerate(zip(M_today,M_color)):
        y = pd.Series(FAST-model)[age_cond&(M_class==m)]
        y = y[abs(y<1.)]
        plt.text(0.22,0.92-0.08*k,'(%.2f)'%y.std(),
                 fontsize=9,transform=ax.transAxes,color=c) 

fig.subplots_adjust(wspace=0.25,hspace=0.2)
plt.suptitle('AM-SFH Models vs FAST Fitting (Tau Templates)',fontsize=18,y=0.95)
plt.savefig("N/FAST-exp_AM-SFH_all_m.png",dpi=350)
plt.show()    