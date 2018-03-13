#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:17:50 2018

@author: Q.Liu
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from astropy.io import ascii

# =============================================================================
# Read CANDELS catalog
# =============================================================================
table_gds = Table.read('gds_all.hdf5', path='data')
table_uds = Table.read('uds_all.hdf5', path='data')
data1 = table_gds.to_pandas()
data2 = table_uds.to_pandas()
data = pd.concat([data1,data2], join='inner')

Mgrid = np.arange(9.25,11.,0.5)
zgrid = np.arange(0.75,2.5,0.5)

uds_lamb = [3838.,4448.,5470.,6276.,7671.,9028.,
            5919.,8060.,12471.,15396.,10201.,21484.,
            35569.,45020.,57450.,79158.]

uds_band = ['CFHT_U_FLUX','SUBARU_B_FLUX',
            'SUBARU_V_FLUX','SUBARU_R_FLUX',
            'SUBARU_i_FLUX','SUBARU_z_FLUX',
            'ACS_F606W_FLUX','ACS_F814W_FLUX',
            'WFC3_F125W_FLUX','WFC3_F160W_FLUX',
            'HAWKI_Y_FLUX','HAWKI_Ks_FLUX',
            'IRAC_CH1_SEDS_FLUX','IRAC_CH2_SEDS_FLUX',
            'IRAC_CH3_SPUDS_FLUX','IRAC_CH4_SPUDS_FLUX']

uds_band_id = [72,78,79,81,82,83,4,6,203,205,271,267,18,19,20,21]

uds_band = [b for _,b in sorted(zip(uds_lamb,uds_band))]
uds_band_id = [b for _,b in sorted(zip(uds_lamb,uds_band_id))]
uds_lamb = np.sort(uds_lamb)

z_best = data2.z_best
Ms = np.log10(data2.M_med)

filter = (abs(Ms-10)<1.0) & (z_best>0.2) & (z_best<2.5) \
         & (data2.f_f160w==0) & (data2.mag_f160w_4<24.5) \
         & (data2.CLASS_STAR<0.9) & (data2.PhotFlag==0) 
SF =  filter & (data2.sf_flag == 1) 
Q = filter | (data2.sf_flag == -1) 

Fang = data2[SF]

# =============================================================================
# Make EAZY catalog
# =============================================================================
Fang_rand = Fang.sample(300)
make_table = True
N_mc = 200
if make_table:
    df_all = pd.DataFrame()
    for k in range(N_mc):
        data = Table()                       
        for (band,id) in zip(uds_band,uds_band_id):
            flux = Fang_rand[band]
            error = Fang_rand[band+"ERR"]
            err = np.copy(error)
            err[err<0] = 0
            noise = np.random.normal(0, err)
            data.add_columns([Column(flux+noise,'F%s'%id), Column(error,'E%s'%id)])
    
         
        obj = Column(name='obj', data=np.arange(1,len(data)+1)) 
        run = Column(name='run', data=[k+1]*len(data)) 
        z_spec = Column(name='z_spec', data=Fang_rand.z_best) 
        log_M = Column(name='log_M', data=np.log10(Fang_rand.M_med))  
        data.add_column(obj, 0)
        data.add_column(run, 1)
        data.add_column(z_spec)  
        data.add_column(log_M)
    
        df = data.to_pandas()
        df_all = pd.concat([df_all,df])
        
    Data = Table.from_pandas(df_all)
    
    eazyid = Column(name='id', data=np.arange(1,(len(data))*N_mc+1))    
    Data.add_column(eazyid, 0)

    np.savetxt('EAZY_catalog/EAZY_uds-300.cat', Data, header=' '.join(Data.colnames),
               fmt=['%d']*3+['%.5e' for i in range(len(uds_band)*2)]+['%.4f']*2)
