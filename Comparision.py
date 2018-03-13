# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 00:20:05 2017

@author: Qing Liu
"""
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.visualization import quantity_support
from scipy.integrate import quad

from smpy.ssp import BC
from smpy.sfh import exponential, delayed
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe

Ages = [0.003, 0.01, 0.03, 0.1, 0.3, 1., 2., 3., 4., 6., 8., 10., 12., 14.]* u.Gyr
SFH = [0.01, 0.03, 0.1, 0.3, 1., 2., 4., 8., 16. , 50.] * u.Gyr

with open('Exp_grid.pkl','r') as f: 
    models = pickle.load(f)

M_s = np.array([quad(exponential, 0, T.value, args=(tau.value,))[0] \
                for T in Ages for tau in SFH]).reshape(Ages.size,SFH.size)

SED = models.SED * M_s[None,:,:,None,None,None]
SFR = models.SFR * M_s[None,:,:,None,None]
sSFR = np.squeeze(np.log10(models.SFR.value))
     
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')

print eazy_library.filternames

filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
    # FUV, NUV, U, B, V, R, I, J, H, K

synphot = Observe(models, filters, redshift=0.001)

mags = np.squeeze(synphot.AB)
U_V = mags[2]-mags[4]
V_J = mags[4]-mags[7]
FUV_R = mags[0]-mags[5]
R_K = mags[5]-mags[9]

#==============================================================================
# Comparision with Xinyi
#==============================================================================
with open('color.pkl','r') as f: 
    Color_xy = pickle.load(f)

# Plot 
plt.figure(figsize=(8,7))

X, Y = Color_xy['VJ'], Color_xy['UV']
with quantity_support():
    for j, tau in enumerate(SFH):
        plt.plot(X[:,j], Y[:,j], label=r'$\tau = ${0:.1f}'.format(tau), alpha=0.7)
        s = plt.scatter(X[:,j], Y[:,j], c=np.log10(Ages/u.yr), 
                         s=50, cmap=plt.cm.RdYlBu_r)
X2, Y2 = V_J, U_V
with quantity_support():
    for j, tau in enumerate(SFH):
        plt.plot(X2[:,j], Y2[:,j], label=r'$\tau = ${0:.1f}'.format(tau), alpha=0.7)
        s = plt.scatter(X2[:,j], Y2[:,j], c=np.log10(Ages/u.yr), 
                         s=50, cmap=plt.cm.RdYlBu_r)
xx=np.linspace(-2.5,2.5,100)
plt.plot(xx,xx,'k-',lw=2)
for b in np.arange(-2.,5.,1.):
    plt.plot(xx,-xx+b,'k--',lw=1,alpha=0.5)
plt.text(1.1,-.9,'Xinyi',fontsize=20)
plt.text(-.9,1.1,'Qing',fontsize=20)
plt.xlim(-1.6,2.6)
plt.ylim(-1.6,2.6)
plt.title('Exponential SFH',fontsize=15)
cb = plt.colorbar(s)
cb.set_label(r'$\log_{10}(\rm{Age}/\rm{yr})$')


