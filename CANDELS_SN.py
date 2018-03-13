# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 17:40:44 2017

@author: Qing Liu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table

table_uds = Table.read('uds_merged_v1.1.fits')
data_uds = table_uds.to_pandas()
table_gds = Table.read('gds_merged_v1.1.fits')
data_gds = table_gds.to_pandas()

z_cat = np.arange(0.5,3.0,0.5)
M_cat = np.arange(9.0,11.5,0.25)
NS_H_uds = np.zeros((z_cat.size, M_cat.size))
NS_H_gds = np.zeros((z_cat.size, M_cat.size))

for i, z in enumerate(z_cat):
    for j, m in enumerate(M_cat):
        Data = data_uds[(abs(data_uds.zbest-z)<0.1) & (abs(np.log10(data_uds.M_med)-m)<0.25)]
        NS_H_uds[i,j] = (Data.FLUXERR_PETRO_F160W/ Data.FLUX_PETRO_F160W).median() 
        
for i, z in enumerate(z_cat):
    for j, m in enumerate(M_cat):
        Data = data_gds[(abs(data_gds.zbest-z)<0.1) & (abs(data_gds.M_med-m)<0.25)]
        NS_H_gds[i,j] = (Data.FLUXERR_PETRO_F160W/ Data.FLUX_PETRO_F160W).median()        

for col in data_uds.columns:
    if 'ERR' in col:
        print col
print ''        
for col in data_gds.columns:
    if 'ERR' in col:
        print col

#==============================================================================
# Huang
#==============================================================================
data = data_gds

band = 'VIMOS_U_FLUX'
clean = (data[band] < data[band].quantile(.99)) \
        & (data[band] > data[band].quantile(.01))
x = data[clean][band]
y = data[clean][band]/data[clean][band+'ERR']
plt.scatter(x,y,s=10,alpha=0.3)
fit =  np.polyfit(x,y,4)
fit_fn = np.poly1d(fit)
x_plot = np.linspace(0,1.8,100)
plt.plot(x_plot, fit_fn(x_plot), '.k')
