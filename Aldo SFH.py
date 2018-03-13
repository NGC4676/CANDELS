# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 06:10:53 2017

@author: Qing Liu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import asciitable as asc
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, least_squares


M_today = np.array([9.0,9.5,10.0,10.5,11.0,11.5])
#M_today = np.linspace(9.5,11.5,9)

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
#==============================================================================
# Main Sequence Aldo
#==============================================================================
A, P = 0.0, 1.
z_interp  = np.array([0.5, 1., 2., 5.])

with sns.axes_style("ticks"):
    plt.figure(figsize=(9,6))
            
    m_plot = np.logspace(6.0, 13.0, 100)
    plt.semilogy(np.log10(m_plot), MS_Sch(m_plot,0.5),
             'k--',alpha=1.0)
    plt.semilogy(np.log10(m_plot), MS_Sch(m_plot,1.0),
             'k--',alpha=0.8)
    plt.semilogy(np.log10(m_plot), MS_Sch(m_plot,2.0),
             'k--',alpha=0.6)
    plt.semilogy(np.log10(m_plot), MS_Sch(m_plot,5.0),
             'k--',alpha=0.4)
    
    for m in Data:
        
        early = (Data[m].z>z_interp[0])
        late = (Data[m].lgMh>11.5) & (Data[m].z>z_interp[0])
        
        plt.semilogy(Data[m].lgMs[early], (Data[m].SFR*\
                     (1+A*np.sin( 2*np.pi*Data[m].t/P)))[early],
                     label='log M$_{today}$ = %s'%m,lw=3,alpha=0.7)
        
        plt.semilogy(Data[m].lgMs[late], (Data[m].SFR*\
                     (1+A*np.sin(2*np.pi*Data[m].t/P)))[late], 
                     c='grey',lw=3)
        
        Ms_interp = np.interp(z_interp, Data[m].z, Data[m].lgMs)
        SFR_interp = np.interp(z_interp, Data[m].z, Data[m].SFR)
        s = plt.scatter(Ms_interp, SFR_interp, c=z_interp, cmap='rainbow')
    
    plt.text(11.,3e2,'z = 5')
    plt.text(11.5,1.5e2,'z = 2')
    plt.text(11.5,5e1,'z = 1')
    plt.text(11.3,1e1,'z = 0.5')
    plt.xlim(6,12.)
    plt.ylim(1e-2,5e2)
    cb = plt.colorbar(s)
    cb.set_label('redshift')
    plt.xlabel('log (M$_*$/M$_\odot$)')
    plt.ylabel('SFR (M$_\odot$/yr)')
    #plt.legend(loc='best',fontsize=9,ncol=2)
    plt.show()

#==============================================================================
# Main Sequence Aldo Perturb
#==============================================================================
#A, P = 0.3, 0.1
#z_interp  = np.array([0.,1.,2.,5.])
#Ms_interp = np.empty((M_today.size, z_interp.size))
#SFR_interp = np.empty((M_today.size, z_interp.size))
#with sns.axes_style("ticks"):
#    plt.figure(figsize=(9,6))
#    for i, m in enumerate(M_today.astype('str')):
#        Ms_interp[i] = np.interp(z_interp, Data[m].z, Data[m].lgMs)
#        SFR_interp[i] = np.interp(z_interp, Data[m].z, Data[m].SFR)
#        for k in range(1):
#            SFR_prtb = Data[m].SFR*(1+A*np.sin( 2*np.pi*Data[m].t/P + np.random.rand()*2*np.pi ))
#            sfr_prtb = np.interp(z_interp, Data[m].z, SFR_prtb)
#            plt.semilogy(Data[m].lgMs, SFR_prtb, label='log M$_{today}$ = %s'%m)
#        s = plt.scatter(Ms_interp[i], SFR_interp[i], c=z_interp, cmap='jet')
#    for j in range(z_interp.size):
#        plt.fill_between(Ms_interp[:,j],SFR_interp[:,j]*(1-A),SFR_interp[:,j]*(1+A),alpha=0.5)
#    m_plot = np.logspace(6.0, 13.0, 100)
#    plt.semilogy(np.log10(m_plot), MS_Sch(m_plot,0.0),
#             'k--',alpha=0.9)
#    plt.semilogy(np.log10(m_plot), MS_Sch(m_plot,1.0),
#             'k--',alpha=0.7)
#    plt.semilogy(np.log10(m_plot), MS_Sch(m_plot,2.0),
#             'k--',alpha=0.5)
#    plt.semilogy(np.log10(m_plot), MS_Sch(m_plot,5.0),
#             'k--',alpha=0.3)
#    plt.text(10.5,5e2,'z = 5')
#    plt.text(11.5,5e2,'z = 2')
#    plt.text(11.5,5e1,'z = 1')
#    plt.text(11.5,5e0,'z = 0')
#    cb = plt.colorbar(s)
#    cb.set_label('redshift')
#    plt.xlim(6,12.)
#    plt.ylim(1e-3,1e3)
#    plt.legend(loc=2,fontsize=8,ncol=2)
#    plt.xlabel('log (M$_*$/M$_\odot$)')
#    plt.ylabel('SFR (M$_\odot$/yr)')
#plt.show()
#print ((np.log10((1+A)*SFR_interp[:,j])-np.log10((1-A)*SFR_interp[:,j]))/2.).mean()

#==============================================================================
# Sinus Absolute SFH
#==============================================================================
#B, P = 0.3, 2.5
#t = np.linspace(0.5,13.0,100)
#fig, axes = plt.subplots(figsize=(10,6), nrows=2, ncols=3)
#for i, m in enumerate(M_today.astype('str')):
#    ax = plt.subplot(2,3,i+1)
#    sfr = np.interp(t, Data[m].t[::-1], Data[m].SFR[::-1])
#    plt.semilogy(t, sfr,
#             'b-',label='Aldo',alpha=0.5)
#    plt.semilogy(t, sfr*10**(-B*np.sin(2*np.pi*t/P+3*np.pi/4.)),
#             'k-',label='Perturbed Aldo',alpha=0.8)
#    plt.legend(loc='best',fontsize=10)
#    plt.title('log M$_{today}$ = %s'%m)
#    plt.xlim(-0.5,13.)
#plt.suptitle('A = %.1f dex  P = %.1f Gyr'%(B,P),fontsize=20,y=1.02)
#plt.tight_layout()
#plt.show()
#
#writeBC = False
#if writeBC:
#    for m in Data:
#        sfr = np.interp(t, Data[m].t[::-1], Data[m].SFR[::-1])
#        plt.plot(t,sfr*10**(-B*np.sin(2*np.pi*t/P+3*np.pi/4.)))
#        tab = np.vstack( (t*1e9, sfr*10**(-B*np.sin(2*np.pi*t/P + 3*np.pi/4.))) ).T
#        np.savetxt('Aldo/Perturb/Final/bc03/bc03_M%s_%.1f_P%.1f.txt'%(m,B,P),tab,fmt='%.7e',header='t(yr) SFR(Msol/yr)')

#==============================================================================
# Tachella 2016 SFH
#==============================================================================
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307) 

C = 0.3
t = np.linspace(0.5,11.5,111)
fig, axes = plt.subplots(figsize=(10,8), nrows=2, ncols=3)
for i, m in enumerate(M_today.astype('str')):
    ax = plt.subplot(3,3,i+1)
    sfr = np.interp(t, Data[m].t[::-1], Data[m].SFR[::-1])
    plt.semilogy(t, sfr, 'b-',
                 label='log M$_{today}$ = %s'%m,alpha=0.5)
    R = 0.
    for n in range(1):
         s=plt.scatter(t, sfr*10**(-C*np.sin(5*np.pi*np.log(t) + 3*np.pi/4.)+\
                             R*np.random.normal(0.,0.3,t.size)),
                        c=np.mod(5*np.pi*np.log(t)+np.pi + 3*np.pi/4.,2*np.pi) /np.pi,
                        marker='o',s=15,cmap='jet',alpha=0.8)
    if i>5:
        plt.xlabel('t (Gyr)')
    if np.mod(i,3)==0:
        plt.ylabel('SFR (M$\odot$/yr)')
    plt.legend(loc=4, fontsize=10, frameon=True, facecolor='w')
    plt.xlim(-0.5,13.5)
#plt.suptitle('A = %.1f dex'%(C),fontsize=20,y=1.02)
plt.tight_layout()
cbar_ax = fig.add_axes([0.2, 0.25, 0.6, 0.03])
colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
colorbar.set_label('Phase')
plt.show()

writeBC = False
if writeBC:
    for m in Data:
        sfr = np.interp(t, Data[m].t[::-1], Data[m].SFR[::-1])
        plt.plot(t,sfr*10**(-C*np.sin(5*np.pi*np.log(t)+3*np.pi/4.)))
        tab = np.vstack( ( t*1e9, sfr*10**(-C*np.sin(5*np.pi*np.log(t)+3*np.pi/4.)) ) ).T
        np.savetxt('Aldo/Perturb/Final/bc03_M%s_%.1f_ST.txt'%(m,C),tab,fmt='%.7e',header='t(yr) SFR(Msol/yr)')
        
## MS   
      
#from scipy.integrate import simps
#t = np.linspace(0.5,13.0,100)
#fig, axes = plt.subplots(figsize=(10,8), nrows=2, ncols=3)
#for i, m in enumerate(M_today.astype('str')):
#    ax = plt.subplot(3,3,i+1)
#    sfr = np.interp(t, Data[m].t[::-1], Data[m].SFR[::-1])
#    ms = np.array([simps(sfr[:n+1],t[:n+1]) for n in range(t.size)])
#    plt.loglog(ms, sfr, 'b-',
#                 label='log M$_{today}$ = %s'%m,alpha=0.5)
#    for n in range(1):
#        sfr_2 = sfr*10**(-C*np.sin(5*np.pi*np.log(t)))
#        ms_2 = np.array([simps(sfr_2[:n+1],t[:n+1]) for n in range(t.size)])        
#        s=plt.scatter(ms_2, sfr_2,
#                        c=np.mod(5*np.pi*np.log(t)+np.pi,2*np.pi) /np.pi,
#                        marker='o',s=15,cmap='jet',alpha=0.8)
#    if i>5:
#        plt.xlabel('t (Gyr)')
#    if np.mod(i,3)==0:
#        plt.ylabel('SFR (M$\odot$/yr)')
#    plt.legend(loc=4, fontsize=10, frameon=True, facecolor='w')
#    plt.xlim(1e-4,1e3)
#plt.tight_layout()
#cbar_ax = fig.add_axes([0.2, 0.25, 0.6, 0.03])
#colorbar = fig.colorbar(s, orientation='horizontal',cax=cbar_ax)
#colorbar.set_label('Phase')
#plt.show()
    
#==============================================================================
# Plotting quantities in Aldo table
#==============================================================================
with sns.axes_style("ticks"):
    fig,axes = plt.subplots(figsize=(9,8), nrows=2, ncols=2)
    for (m,c) in zip(M_today.astype('str'),['m','b','steelblue','g','orange','firebrick']):
        for i, (ax, col, ylab) in enumerate(zip(axes.ravel(),['lgMs','lgMh','SFR','SFR'],
                                      ['log M$_*$(M$_\odot$)','log M$_{halo}$(M$_\odot$)',
                                      'log SFR(M$_\odot$/yr)','log sSFR(yr$^{-1}$)'])):
            if i == 2:
                ax.plot(Data[m].t, np.log10(Data[m][col]),c=c,
                        label='log M$_{today}$ = %s'%m,alpha=0.7)
            elif i == 3:
                ax.plot(Data[m].t, np.log10(Data[m][col])-Data[m].lgMs,c=c,
                        label='log M$_{today}$ = %s'%m,alpha=0.7)
            else:
                ax.plot(Data[m].t, Data[m][col],c=c,
                        label='log M$_{today}$ = %s'%m,alpha=0.7)
            ax.set_ylabel(ylab)
    plt.legend(loc=10,bbox_to_anchor=(1.3, 1.2),
              fontsize=12,frameon=True,facecolor='w')
    new_tick_locs = np.array([.2, .4, .6, .7])
    for ax in axes.ravel():
        ax2 = ax.twiny()
        x_ticks = (ax.get_xlim()[1]-ax.get_xlim()[0])*new_tick_locs
        ax2.set_xticks(new_tick_locs)
        ax2.set_xticklabels(tick_func(x_ticks))
        ax.set_xlabel('t (Gyr)')
        ax2.set_xlabel('z')
    plt.tight_layout()

#==============================================================================
# sSFR History
#==============================================================================
#with sns.axes_style("ticks"):
#    plt.figure(figsize=(8,6))
#    for m in M_today.astype('str'):
#        plt.plot(Data[m].t, np.log10(Data[m].SFR)-Data[m].lgMs,label='log M$_{today}$ = %s'%m)
#    plt.legend(loc='best',fontsize=12,frameon=True,facecolor='w')
#    plt.xlabel('t (Gyr)', fontsize=15)
#    plt.ylabel('log sSFR(yr$^{-1}$)', fontsize=15)
#    #plt.title('Aldo SFH along cosmic time', fontsize=15)
#    plt.tight_layout()
    
#==============================================================================
# Save SFH as txt
#==============================================================================
#for m in Data:
#    plt.plot(Data[m].t,Data[m].SFR)
#    tab=np.vstack((Data[m].t[::-1]*1e9,Data[m].SFR[::-1])).T
#    np.savetxt('Aldo/bc03_M%s.txt'%m,tab,fmt='%.7e',header='t(yr) SFR(Msol/yr)')
   
#==============================================================================
# Profile
#==============================================================================
def gaussian( x, params ):
    (c, mu, sigma) = params
    res =   c * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) )
    return res
def lorentzian( x, params ):
    (c, x0, gama) = params
    res =   c * gama / ((x-x0)**2+gama**2) / np.pi
    return res
def hermite( x, params):
    (h3, h4) = params
    res = (1 + h3*(8*x**3.-12*x)/np.sqrt(6*8.) + \
               h4*(16*x**4-48.*x**2.+12)/np.sqrt(24*16.))
    return res

def gaussian_hermite(x, params):
    (c, mu, sigma, h3, h4) = params
    res =   gaussian(x, (c, mu, sigma)) * hermite(x, (h3, h4))
    return res
def lorentzian_hermite(x, params):
    (c, x0, gama, h3, h4) = params
    res =   lorentzian(x, (c, x0, gama)) * hermite(x, (h3, h4))
    return res
 
#==============================================================================
# Double Gaussian Fitting
#==============================================================================
#def double_gaussian(x, params):
#    (c1, mu1, sigma1, c2, mu2, sigma2) = params
#    res =   gaussian(x, (c1, mu1, sigma1)) + gaussian(x, (c2, mu2, sigma2))
#    return res
#def double_gaussian_fit(params,x,y):
#        fit = double_gaussian(x, params)
#        return (fit - y)
# 
#plt.figure(figsize=(11,6))    
#for i,m in enumerate(M_today.astype('str')):
#    t,y = Data[m].t,Data[m].SFR
#    param_guess = [y.mean(), t.mean(), 0.5, 1.0, 5.0, 0.5]
#    fit = least_squares(double_gaussian_fit, x0=param_guess, args=(t,y),
#                        bounds=(0., np.inf))
#    y_fit = double_gaussian(t, fit.x)
#    print fit
#    ax = plt.subplot(2,3,i+1)
#    plt.plot( t, y, 'k-')
#    plt.plot( t, y_fit, 'k--' )
#    plt.plot( t, gaussian(t,fit.x[:3]), 'k--',alpha=0.5)
#    plt.plot( t, gaussian(t,fit.x[3:]), 'k--',alpha=0.5)
#    plt.title('log M$_{today}$ = %s'%m)
#plt.tight_layout()
#plt.show()

#==============================================================================
# Gauss Lorentzian
#==============================================================================
#def gauss_lorentzian(x, params):
#    (c1, mu, sigma, c2, x0, gama) = params
#    res = gaussian(x, (c1, mu, sigma)) + lorentzian(x, (c2, x0, gama))
#    return res
#def gauss_lorentzian_fit(params,x,y):
#        fit = gauss_lorentzian(t, params)
#        return (fit - y)
# 
#plt.figure(figsize=(11,6))    
#for i,m in enumerate(M_today.astype('str')):
#    t,y = Data[m].t,Data[m].SFR
#    param_guess = [y.mean(),3.0,0.5, 1.0,10.0,0.5]
#    fit = leastsq(gauss_lorentzian_fit, x0=param_guess, args=(t,y))
#    y_fit = gauss_lorentzian(t, fit[0])
#    print fit
#    ax = plt.subplot(2,3,i+1)
#    plt.plot( t, y, 'k-')
#    plt.plot( t, y_fit, 'k--' )
#    plt.plot( t, gaussian(t,fit[0][:3]), 'k--',alpha=0.5)
#    plt.plot( t, lorentzian(t,fit[0][3:]), 'k--',alpha=0.5)
#    plt.title('log M$_{today}$ = %s'%m)
#plt.tight_layout()
#plt.show()

#==============================================================================
# Gaussian Hermite6th Fitting
#==============================================================================
#def gaussian_hermite6(x, params):
#    (c, mu, sigma, h3, h4, h5, h6) = params
#    res =   gaussian(x, (c, mu, sigma)) * \
#            (1 + h3*(8*x**3.-12*x)/np.sqrt(6*8.) + \
#                 h4*(16*x**4-48.*x**2.+12)/np.sqrt(24*16.) + \
#                 h5*(32*x**5.-160*x*3.+120*x)/np.sqrt(120*32.)+
#                 h6*(64*x**6-480*x**4+720*x**2-120)/np.sqrt(720*64.))
#    return res
#def gaussian_hermite6_fit(params,x,y):
#        fit = gaussian_hermite6(t, params)
#        return (fit - y)
# 
#plt.figure(figsize=(11,6))    
#for i,m in enumerate(M_today.astype('str')):
#    t,y = Data[m].t,np.log10(Data[m].SFR)-np.log10(Data[m].SFR).min()
#    param_guess = [y.mean(),5.0,0.5, 0.,0.,0.,0.]
#    fit = leastsq(gaussian_hermite6_fit, x0=param_guess, args=(t,y))
#    y_fit = gaussian_hermite6(t, fit[0])
#    print fit
#    ax = plt.subplot(2,3,i+1)
#    plt.plot( t, y, 'k-')
#    plt.plot( t, y_fit, 'k--' )
#    plt.title('log M$_{today}$ = %s'%m)
#plt.tight_layout()
#plt.show()

#==============================================================================
# Gauss-Lorentz-Herimite4th
#==============================================================================
#def gauss_lorentz_hermite(x, params):
#    (c1, mu, sigma, h13, h14, c2, x0, gama, h23, h24) = params
#    res = gaussian_hermite(x,(c1, mu, sigma, h13, h14)) + lorentzian_hermite(x, (c2, x0, gama, h23, h24)) 
#    return res
#
#def gauss_lorentz_hermite_fit(params,x,y):
#        fit = gauss_lorentz_hermite(x, params)
#        return (fit - y)
# 
#plt.figure(figsize=(11,6)) 
#sfh_par = np.empty((6,10))   
#for i, m in enumerate(M_today.astype('str')):
#    t,y = Data[m].t,Data[m].SFR
#    param_guess = [1.0, 5.0, 0.5, 0.0, 0.0, 1.0, 10.0, 0.5, 0.0, 0.0]
#    param_guess = [1.0, 5.0, 0.5, 0.0, 0.0, 1.0, 7.0, 0.5, 0.0, 0.0]
#    param_guess = [1.0, 3.0, 0.5, 0.0, 0.0, 1.0, 7.0, 0.5, 0.0, 0.0]
#    fit = leastsq(gauss_lorentz_hermite_fit, x0=param_guess, args=(t,y))
#    y_fit = gauss_lorentz_hermite(t, fit[0])
#    sfh_par[i] = fit[0]
#    print fit
#    ax = plt.subplot(2,3,i+1)
#    plt.plot( t, y, 'k-')
#    plt.plot( t, y_fit, 'k--' )
#    plt.plot( t, gaussian_hermite(t,fit[0][:5]), 'k--',alpha=0.5)
#    plt.plot( t, lorentzian_hermite(t,fit[0][5:]), 'k--',alpha=0.5)
#    plt.title('log M$_{today}$ = %s'%m)
#plt.tight_layout()
#plt.show()
