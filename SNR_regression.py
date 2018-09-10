#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:51:18 2018

@author: Q.Liu
"""

from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import ascii

# =============================================================================
# Read
# =============================================================================
RF_band = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']

df_flux1 = ascii.read("EAZY_uds-all_MC250.beta.flux").to_pandas()
df_flux2 = ascii.read("EAZY_gds-all_MC250.beta.flux").to_pandas()
df_flux2.obj+=df_flux1.obj.max()

df_flux = pd.concat([df_flux1,df_flux2])

gp = df_flux.groupby("obj")

phot = gp[RF_band]
param = gp["z_spec","log_M"]

figure()
scatter(gp.z_spec.mean(), log10(gp.V.mean()/gp.V.std()), 
        c=gp.log_M.mean(), cmap="rainbow", s=10, alpha=0.3)
ylabel(r"log (S/N)$\rm _{V,rest-frame}$")
xlabel("z")
cb = colorbar()
cb.set_label("log M")

# =============================================================================
# Regression
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

regr = LinearRegression()

SNR = phot.mean()/phot.std()
X = param.mean()

X_train, X_test, y_train, y_test = train_test_split(X, log10(SNR), test_size=0.2)

regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)

for i in range(10):
    t = RF_band[i]
    print t
    print r2_score(y_test[t], y_pred[:,i])
    
SNR_coef = pd.DataFrame({"a1":regr.coef_[:,0],"a2":regr.coef_[:,1],"a0":regr.intercept_})

figure(figsize=(8,12))
for k in range(10):
    ax=subplot(5,2,k+1)
    cof = SNR_coef.iloc[k]
    xp = cof.a0+cof.a1*gp.z_spec.mean()+cof.a2*gp.log_M.mean()
    xx = linspace(xp.min()-0.01,xp.max()+0.01,10)
    s = scatter(xp, log10(SNR).iloc[:,k],
            c=gp.z_spec.mean(), cmap="rainbow", s=10, alpha=0.3)
    plot(xx,xx,ls="--",c="k")
    text(0.1,0.85,RF_band[k],transform=ax.transAxes,fontsize=15)
    colorbar()
tight_layout()

#SNR_coef.to_csv("New/SNR_coef_all.txt",sep=" ",index=False)
    

