# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 17:29:36 2017

@author: Qing Liu
"""

import re
import glob
import numpy as np
import asciitable as asc
import matplotlib.pyplot as plt

#==============================================================================
# BC03 .4color Aldo SFH
#==============================================================================
# FAST
lgAges = np.log10(np.linspace(0.5,11.5,111)* 1e9)
M_today = np.array([9.0,9.5,10.0,10.5,11.0,11.5])

# FAST Real Error
#lgAges = np.log10(np.linspace(2.,8.,61)* 1e9)
#M_today = np.linspace(9.5,11.5,5)
#==============================================================================
lgMs = np.array([])
lgSSFR = np.array([])
lgT = np.array([])
M_class = np.array([])

dir = glob.glob('Aldo/Perturb/Final/4color/M*0.3_P0.5.spec.4color') 
values = [(f, re.findall(r'-?\d+\.?\d*e?-?\d*?',f)[1]) for f in dir]
dtype = [('name', 'S80'), ('M', float)]
a = np.array(values, dtype=dtype) 
filelist = np.sort(a, order='M')  

for f, m in filelist:
    table = asc.read(f,names=['log-age','Mbol','Bmag','Vmag','Kmag',      
                              'M*_liv','M_remnants','M_ret_gas',
                              'M_galaxy','SFR','M*_tot',
                              'M*_tot/Lb','M*_tot/Lv','M*_tot/Lk',
                              'M*_liv/Lb','M*_liv/Lv','M*_liv/Lk'])
    lgt = table['log-age']
    lgsfr = np.log10(table['SFR'])
    lgms= np.log10(table['M*_tot'])
    lgsfr_interp = np.interp(lgAges, lgt, lgsfr)
    lgms_interp = np.interp(lgAges, lgt, lgms)
    print 10**lgms_interp
    plt.plot(lgt,lgms)
    lgMs = np.append(lgMs, lgms_interp)
    lgSSFR = np.append(lgSSFR, lgsfr_interp-lgms_interp)
    lgT = np.append(lgT, lgAges)
    M_class = np.append(M_class, [m for i in range(lgAges.size)])

info = np.vstack((M_class, lgT, lgMs, lgSSFR)).T
np.savetxt('Aldo/Perturb/Final/P0.5 CSP result.txt',info,fmt='%.7e',header='M_today lgT lgM* lgsSFR')

    