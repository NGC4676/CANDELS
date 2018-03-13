# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:17:57 2017

@author: Qing Liu
"""

import re
import glob
import numpy as np
import asciitable as asc
import matplotlib.pyplot as plt
#==============================================================================
# BC03 .4color Tau|Delay SFH
#==============================================================================
# FAST
lgAges = np.linspace(7.0,10.0,13)
lgAges = np.log10(np.array([0.1,0.4,4.0,10.0])*1e9)
lgSFH = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
lgSFH = np.array([0.1, 0.5, 1.0, 3.0, 5.0, 20.0])

lgMs = np.array([])
lgSSFR = np.array([])
lgT = np.array([])
lgTau = np.array([])

dir = glob.glob('Composite/*.4color') 
values = [(f, re.findall(r'-?\d+\.?\d*e?-?\d*?',f)[0]) for f in dir]
dtype = [('name', 'S40'), ('tau', float)]
a = np.array(values, dtype=dtype) 
Output = np.sort(a, order='tau')  

for f,tau in Output:
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
    lgTau = np.append(lgTau, [tau for i in range(lgAges.size)])
    
info = np.vstack((lgT, lgTau, lgMs, lgSSFR)).T
np.savetxt('CSP Tau.txt',info,fmt='%.7e',header='lgT lgTau lgM* lgsSFR')
