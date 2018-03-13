# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:57:33 2017

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
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import exponential, delayed
from smpy.dust import Calzetti

Ages = [0.003, 0.01, 0.03, 0.1, 0.3, 1., 2., 3., 4., 6., 8., 10., 12., 14.]* u.Gyr
SFH = [0.01, 0.03, 0.1, 0.3, 1., 2., 4., 8., 16. , 50.] * u.Gyr

with open('Exp_grid.pkl','r') as f: 
    models_e = pickle.load(f)
with open('Delay_grid.pkl','r') as f: 
    models_d = pickle.load(f)

M_s_e = np.array([quad(exponential, 0, T.value, args=(tau.value,))[0] \
                for T in Ages for tau in SFH]).reshape(Ages.size,SFH.size)
M_s_d = np.array([quad(delayed, 0, T.value, args=(tau.value,))[0] \
                for T in Ages for tau in SFH]).reshape(Ages.size,SFH.size)

SED_e = models_e.SED * M_s_e[None,:,:,None,None,None]
SFR_e = models_e.SFR * M_s_e[None,:,:,None,None]
sSFR_e = np.squeeze(np.log10(models_e.SFR.value))

SED_d = models_d.SED * M_s_d[None,:,:,None,None,None]
SFR_d = models_d.SFR * M_s_d[None,:,:,None,None]
sSFR_d = np.squeeze(np.log10(models_d.SFR.value))
      
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')

print eazy_library.filternames

filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
    # FUV, NUV, U, B, V, R, I, J, H, K

synphot_e = Observe(models_e, filters, redshift=0.001)
synphot_d = Observe(models_d, filters, redshift=0.001)

mags_e = np.squeeze(synphot_e.AB)
Colors_e = {'UV':mags_e[2]-mags_e[4],
            'VJ':mags_e[4]-mags_e[7],
            'FUVR':mags_e[0]-mags_e[5],
            'RK':mags_e[5]-mags_e[9]}
mags_d = np.squeeze(synphot_d.AB)
Colors_d = {'UV':mags_d[2]-mags_d[4],
            'VJ':mags_d[4]-mags_d[7],
            'FUVR':mags_d[0]-mags_d[5],
            'RK':mags_d[5]-mags_d[9]}

#==============================================================================
# Plot
#==============================================================================
fig, (ax1, ax2) = plt.subplots(figsize=(11,6), nrows=1, ncols=2)
with quantity_support():
    
    ax1=plt.subplot(121)
    X, Y = Colors_e['VJ'],Colors_e['UV']
    for j, tau in enumerate(SFH):
        plt.plot(X[:,j], Y[:,j], label=r'$\tau = ${0:.2f}'.format(tau), alpha=0.5)
        s = plt.scatter(X[:,j], Y[:,j], c=sSFR_e[:,j], 
                         s=50, cmap=plt.cm.RdYlBu)
    plt.xlabel('V-J',fontsize=15)
    plt.ylabel('U-V',fontsize=15)
    plt.title('Exponential',fontsize=15)
    plt.legend(fontsize=10)
    
    ax2=plt.subplot(122)
    X, Y = Colors_d['VJ'],Colors_d['UV']
    for j, tau in enumerate(SFH):
        plt.plot(X[:,j], Y[:,j], label=r'$\tau = ${0:.2f}'.format(tau), alpha=0.5)
        plt.scatter(X[:,j], Y[:,j], c=sSFR_d[:,j], 
                         s=50, cmap=plt.cm.RdYlBu)
    plt.xlabel('V-J',fontsize=15)
    plt.ylabel('U-V',fontsize=15)
    plt.title('Delay',fontsize=15)
    plt.legend(fontsize=10)
    
    fig.subplots_adjust(bottom=0.22)
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.05])
    colorbar = fig.colorbar(s, cax=cbar_ax, orientation='horizontal')
    colorbar.set_label(r'$\log_{10}(\rm{sSFR}/\rm{yr^{-1}})$')
plt.show()


#==============================================================================
# Fang Fig.14
#==============================================================================
#X, Y = V_J, U_V
#plt.figure(figsize=(10,6))
#with quantity_support():
#    for i in range(2):
#        plt.subplot(1,2,i+1)
#        if i==0:
#            for j, tau in enumerate(SFH):
#                plt.plot(X[1,:,j], Y[1,:,j], label=r'$\tau = ${0:.1f}'.format(tau), 
#                         lw=3, alpha=0.7)
#                plt.scatter(X0[1,0,:],Y0[1,0,:],s=50,c='steelblue')
#                plt.scatter(X0[1,1,:],Y0[1,1,:],s=50,c='firebrick')
#                plt.text(-0.5,0.7,'1 Gyr',color='steelblue',fontsize=20,alpha=0.3)
#                plt.text(1.0,0.8,'3 Gyr',color='firebrick',fontsize=20,alpha=0.3)
#        elif i==1:
#            for k, meta in enumerate(Meta):
#                plt.plot(X[k,:,1], Y[k,:,1], label=r'$Z = ${0:.1f}Z$_\odot$'.format(meta), 
#                         lw=3, alpha=0.7)
#                plt.scatter(X0[:,0,1],Y0[:,0,1],s=50,c='steelblue')
#                plt.scatter(X0[:,1,1],Y0[:,1,1],s=50,c='firebrick')
#                plt.text(-0.6,0.6,'1 Gyr',color='steelblue',fontsize=20,alpha=0.3)
#                plt.text(1.0,0.8,'3 Gyr',color='firebrick',fontsize=20,alpha=0.3)
#        plt.plot([-1., 0.69, 1.5, 1.5], [1.3, 1.3, 2.01, 2.5], color='k', alpha=0.7)
#        plt.legend(loc='best',fontsize=12,frameon=True,facecolor='w')
#        plt.xlabel('V - J')
#        plt.ylabel('U - V')
#        plt.xlim([-0.75, 2.0])
#        plt.ylim([-0.25, 2.15])
#plt.show()

#==============================================================================
# Show SFH
#==============================================================================
#with quantity_support():
#    for i, tau in enumerate(SFH):
#        plt.semilogx(Ages,SFR[:,i],label=r'$\tau = ${0:.1f}'.format(tau), alpha=0.7)
#    plt.legend(loc='best')
#    plt.xlim([1e-1, 1e1])
#plt.show()

#==============================================================================
# Matplotlib 3D Track
#==============================================================================
#from mpl_toolkits.mplot3d import Axes3D
#
#fig = plt.figure(figsize=(10,8))
#ax = fig.add_subplot(111, projection='3d')
#with quantity_support():
#    for i, tau in enumerate(SFH):
#        plt.plot(VJ[:,i], UV[:,i], np.squeeze(sSFR)[:,i],
#                 label=r'$\tau = ${0:.1f}'.format(tau), alpha=0.7)
#    ax.view_init(45, 15)
    
#==============================================================================
# Plotly 3D track
#==============================================================================
#import plotly.plotly as py
#import plotly.graph_objs as go
#
#def UVJ_3D(x, y, z, c='Viridis'):
#    trace = go.Scatter3d(x=x, y=y, z=z, 
#                         marker=dict(size=4, 
#                                     color=z,
#                                     colorscale=c),
#                         line=dict(color='steelblue',width=3))  
#    return trace
#
#
#data = [UVJ_3D(VJ[:,i].value, 
#               UV[:,i].value, 
#               np.log(np.squeeze(sSFR)[:,i].value)) \
#                for i in range(SFH.size)]

#Data = np.append(data,data2).tolist()

#fig = go.Figure(data=data)
#
#py.iplot(fig, filename='Delay-3d', height=700, validate=False)