# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 00:41:09 2017

@author: Qing Liu
"""

import numpy as np
import pickle
import asciitable as asc
import seaborn as sns
import matplotlib.pyplot as plt

#==============================================================================
# Read model result 
#==============================================================================

#with open('grid_exp.pkl','r') as f: 
#    Data = pickle.load(f)

SFH = Data['SFH']
Ages = Data['Ages']
Dust = Data['Dust']

models= Data['Models']
Ms = Data['M*']
SSFR = Data['SSFR']

def SetValue(Value,isSFH=False):
    X = np.zeros((Ages.size, SFH.size, Dust.size))
    if isSFH:
        for i in range(Ages.size):
            for j in range(Dust.size):
                X[i,:,j] = np.log10(Value)
    else:
        for i in range(Dust.size):
            X[:,:,i] = np.log10(Value)
    lgValue = X.ravel()
    return lgValue

lgAges = np.array([np.log10(Ages) for i in range(SFH.size)\
                                  for i in range(Dust.size)]).T.ravel()
lgSFH = SetValue(SFH,isSFH=True)
lgM = SetValue(Ms)
lgSSFR = SetValue(SSFR)

Av = np.array([Dust for i in range(Ages.size)\
                    for i in range(SFH.size)]).ravel()

#==============================================================================
# Read FAST SED-fitting result
#==============================================================================
table = asc.read('dust_exp.fout')
Ages_FAST = table.lage-9
SFH_FAST = table.ltau-9
M_FAST = table.lmass
SSFR_FAST = table.lssfr

Av_FAST = table.Av

#==============================================================================
# Plot FAST vs Model
#==============================================================================
fig,axes = plt.subplots(figsize=(9,8), nrows=2, ncols=2)
xx=np.linspace(-20,20,20)
legends = ['log(Age/Gyr)',r'log($\tau$/Gyr)',
           r'log($\rm{M_*}/\rm{M_\odot}$)',r'log(sSFR/yr$^{-1}$)']
for i, (l, model, FAST) in enumerate(zip(legends,[lgAges,lgSFH,lgM,lgSSFR],
                                       [Ages_FAST,SFH_FAST,M_FAST,SSFR_FAST])):
    ax = plt.subplot(2,2,i+1) 
    plt.plot(xx,xx,'k--',alpha=0.5)
    s = plt.scatter(model, FAST, c=lgAges,
                    cmap='viridis', label=l, s=30, alpha=0.7)
    plt.xlabel('Model',fontsize=12)
    plt.ylabel('FAST',fontsize=12)
    if i==3:
        plt.xlim(-15,-5)
        plt.ylim(-15,-5)
    elif i==2:
        plt.xlim(-5.,2.)
        plt.ylim(-5.,2.)
    else:
        plt.xlim(-5.,5.)
        plt.ylim(-5.,5.)
    plt.legend(loc=4, fontsize=12, frameon=True, facecolor='w')
fig.subplots_adjust(right=0.82)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
colorbar = fig.colorbar(s, cax=cbar_ax)
colorbar.set_label(r'$\log_{10}(\rm{Age}/\rm{Gyr})$')
plt.suptitle('Exp SMpy Model vs FAST Fitting',fontsize=15,y=0.92)

#==============================================================================
# Av confusion matrix
#==============================================================================
#import itertools
#from sklearn.metrics import confusion_matrix
#cnf_matrix = confusion_matrix(Av.astype('str'), Av_FAST.astype('str'))
#
#def plot_confusion_matrix(cm, classes=None,
#                          title='Confusion matrix',cmap=plt.cm.Blues):
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, cm[i, j],
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#    plt.tight_layout()
#    ticks = np.arange(len(classes))
#    plt.xticks(ticks,classes)
#    plt.yticks(ticks,classes)
#    plt.ylabel('True')
#    plt.xlabel('Predicted')
#
#with sns.axes_style("white"):
#    plt.figure()
#    plot_confusion_matrix(cnf_matrix, classes=np.linspace(0,1,9),
#                          title='Confusion matrix for Av')
#    plt.show()