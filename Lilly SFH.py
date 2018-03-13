# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 01:49:39 2017

@author: Qing Liu
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from scipy.optimize import least_squares

cosmo = FlatLambdaCDM(H0=70., Om0=0.3) 

def rsSFR(m_star, Time):
    if Time.size==1:
        z = z_at_value(cosmo.age, Time)
    else:
        z = np.array([z_at_value(cosmo.age, t) for t in Time])
    rsSFR_1 = 0.07 * (m_star/10**10.5)**(-0.1) * (1.+z)**3.
    rsSFR_2 = 0.3 * (m_star/10**10.5)**(-0.1) * (1.+z)**(5/3.)
    rsSFR = np.concatenate((rsSFR_2[z>2],rsSFR_1[z<2]))
    if rsSFR.size == 1:
        return rsSFR[0]*(1./u.Gyr)
    else:
        return rsSFR*(1./u.Gyr)

T = np.linspace(0.5, 13., 26) * u.Gyr                
plt.semilogy(T, rsSFR(1e10,T), c='r')

#==============================================================================
# Constructing SFH
#==============================================================================
Time_grids  = np.linspace(1.5, 13., 200) * u.Gyr
                         
def Mass_growth(m_seed,Time_grids):
    m = np.empty(Time_grids.size)
    sSFR = np.empty(Time_grids.size)*(1./u.Gyr)
    
    m[0] = m_seed
    sSFR[0] = rsSFR(m[0], Time_grids[0])
    dT = Time_grids[1]-Time_grids[0]
    for k, t in enumerate(Time_grids[:-1]):
        m[k+1] = m[k] + sSFR[k]*m[k]*dT
        sSFR[k+1] = rsSFR(m[k+1], t)
    SFR = (m*sSFR).value/1e9   #Msol/yr
    return m, sSFR, SFR

M,sSFR,SFR = Mass_growth(1e6,Time_grids)

#==============================================================================
# Fitting
#==============================================================================
#from scipy.special import erfc
#def RKPRG(x, params):
#    (A, mu, sigma, r_s) = params
#    res = A*(np.sqrt(np.pi)/2.) \
#            *np.exp( (sigma/(r_s*2.))**2 - (x-mu)/r_s ) \
#            *sigma*erfc( sigma/(r_s*2.) - (x-mu)/sigma )
#    return res
#def RKPRG_fit(params,x,y):
#        fit = RKPRG(x, params)
#        return (fit - y)
#
#param_guess = [100.,4.,1.,2.]
#
#M_seed = np.logspace(5,9,5)
#param_fit = np.empty((M_seed.size,4))
#plt.figure()
#for i, m_seed in enumerate(M_seed):
#    M,sSFR,SFR = Mass_growth(m_seed,Time_grids)
#    param_guess = [100.,4.,1.,2.]
#    fit = least_squares(RKPRG_fit, x0=param_guess, 
#                    args=(Time_grids.value,SFR),
#                    bounds=(0., np.inf))
#    param_fit[i] = fit.x
#    SFR_fit = RKPRG(Time_grids.value, fit.x)
#
#    plt.plot( Time_grids, SFR, '-')
#    plt.plot( Time_grids, SFR_fit, '--' )
#
#fig =  plt.figure(figsize=(6,6))
#for j in range(4):
#    ax = plt.subplot(2,2,j+1)
#    ax.semilogx(M_seed,param_fit[:,j])

#==============================================================================
# 4th order Hermitian                 
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
def gauss_lorentz_hermite(x, params):
    (c1, mu, sigma, h13, h14, c2, x0, gama, h23, h24) = params
    res = gaussian_hermite(x,(c1, mu, sigma, h13, h14)) + lorentzian_hermite(x, (c2, x0, gama, h23, h24)) 
    return res

def gauss_lorentz_hermite_fit(params,x,y):
        fit = gauss_lorentz_hermite(x, params)
        return (fit - y)

M_seed = np.logspace(5,9,5)
plt.figure(figsize=(8,6))
param_fit = np.empty((M_seed.size,10))
for i, m_seed in enumerate(M_seed):
    M,sSFR,SFR = Mass_growth(m_seed,Time_grids)
    param_guess = [100.0, 3.0, 1.0, 0.0, 0.0, 100.0, 7.0, 1.0, 0.0, 0.0]
    fit = least_squares(gauss_lorentz_hermite_fit, x0=param_guess, 
                    args=(Time_grids.value,SFR),
                    bounds=(0., np.inf))
    param_fit[i] = fit.x
    SFR_fit = gauss_lorentz_hermite(Time_grids.value, fit.x)
    plt.plot( Time_grids, SFR, '-')
    plt.plot( Time_grids, SFR_fit, '--' )
                  
#np.savetxt('Lilly/Lilly SFH params.txt', param_fit)