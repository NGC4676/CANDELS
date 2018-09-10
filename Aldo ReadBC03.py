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
#M_today = np.array([9.0,9.5,10.0,10.5,11.0])
M_today = np.array([9.5,10.0,10.5,10.75,11.0])


# FAST Real Error
#lgAges = np.log10(np.linspace(2.,8.,61)* 1e9)
#M_today = np.linspace(9.5,11.5,5)
#==============================================================================

perturb = True
A, p_id = 0.3, 1
Phases = [np.pi/4.,3*np.pi/4.,5*np.pi/4.,7*np.pi/4.]
Phases_rd = np.round(Phases,decimals=1)

if perturb:    
    dir = glob.glob('N/Aldo/ST/M_BC/M*p%s_ST%s.spec.4color'%(Phases_rd[p_id],A)) 
else:
    dir = glob.glob('N/Aldo/M*.spec.4color') 

values = [(f, re.findall(r'-?\d+\.?\d*e?-?\d*?',f)[0]) for f in dir]
dtype = [('name', 'S80'), ('M', float)]
a = np.array(values, dtype=dtype) 
filelist = np.sort(a, order='M')  

lgMs = np.array([])
lgSSFR = np.array([])
lgT = np.array([])
M_class = np.array([])

plt.figure(figsize=(6,5))
for f, m in filelist:
    table = asc.read(f,names=['log-age','Mbol','Bmag','Vmag','Kmag',      
                              'M*_liv','M_remnants','M_ret_gas',
                              'M_galaxy','SFR','M*_tot',
                              'M*_tot/Lb','M*_tot/Lv','M*_tot/Lk',
                              'M*_liv/Lb','M*_liv/Lv','M*_liv/Lk'])
    lgt = table['log-age']
    lgsfr = np.log10(table['SFR'])
    lgms= np.log10(table['M*_tot'])
    lgms_tot =  np.log10(table['M_galaxy'])
    lgsfr_interp = np.interp(lgAges, lgt, lgsfr)
    lgms_interp = np.interp(lgAges, lgt, lgms)
    lgms_tot_interp = np.interp(lgAges, lgt, lgms_tot)
    #plt.plot(lgt,lgms)
    lgMs = np.append(lgMs, lgms_interp)
    lgSSFR = np.append(lgSSFR, lgsfr_interp-lgms_interp)
    lgT = np.append(lgT, lgAges)
    M_class = np.append(M_class, [m for i in range(lgAges.size)])
    plt.plot(10**(lgAges-9),lgms_tot_interp-lgms_interp,label=r"log M$_{today}=%.1f$"%m)
plt.xlabel("t (Gyr)",fontsize=15)
plt.ylabel(r"$\rm log(M_{tot})-log(M_{*,liv}+M_{rem})$",fontsize=15)
#plt.legend(loc="best",fontsize=12)
#plt.savefig("New/BC03_mass_difference",dpi=200)

info = np.vstack((M_class, lgT, lgMs, lgSSFR)).T
if perturb:
    np.savetxt('N/Aldo/ST/ST%s_p%s CSP result.txt'%(A,Phases_rd[p_id]),info,fmt='%.7e',header='M_today lgT lgM* lgsSFR')
else:
    np.savetxt('N/Aldo/CSP result.txt',info,fmt='%.7e',header='M_today lgT lgM* lgsSFR')
    
