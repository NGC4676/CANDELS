# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:40:31 2017

@author: Qing Liu
"""

import re
import glob
import numpy as np
import seaborn as sns
import asciitable as asc
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import Table, Column

from smpy.ssp import BC
from smpy.smpy import CSP, LoadEAZYFilters, FilterSet, Observe
from smpy.sfh import exponential
from smpy.dust import Calzetti

bc03 = BC('data/ssp/bc03/chab/hr/')

lgAges = np.log10(np.array([0.1,0.4,4.0,10.0])*1e9)
SFH = np.array([0.1, 0.5, 1.0, 3.0, 5.0, 20.0])*u.Gyr
Dust = np.array([1.])
SFH_law = exponential
#==============================================================================
# SSP
#==============================================================================              
SP_A = CSP(bc03, age = 0.1*u.Gyr, sfh = SFH, 
           dust = Dust, metal_ind = 1.0, f_esc = 1.,
           sfh_law = SFH_law, dust_model = Calzetti)

SP_B = CSP(bc03, age = 0.4*u.Gyr, sfh = SFH, 
           dust = Dust, metal_ind = 1.0, f_esc = 1.,
           sfh_law = SFH_law, dust_model = Calzetti)

SP_C = CSP(bc03, age = 4.0*u.Gyr, sfh = SFH, 
           dust = Dust, metal_ind = 1.0, f_esc = 1.,
           sfh_law = SFH_law, dust_model = Calzetti)

SP_D = CSP(bc03, age = 10.0*u.Gyr, sfh = SFH, 
           dust = Dust, metal_ind = 1.0, f_esc = 1.,
           sfh_law = SFH_law, dust_model = Calzetti)
#==============================================================================
# Read BC03  
#==============================================================================
lgMs = np.array([])
lgSSFR = np.array([])
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
    lgage = table['log-age']
    lgsfr = np.log10(table['SFR'])
    lgms= np.log10(table['M*_tot'])
    lgsfr_interp = np.interp(lgAges, lgage, lgsfr)
    lgms_interp = np.interp(lgAges, lgage, lgms)
    print 10**lgms_interp
    lgMs = np.append(lgMs, lgms_interp)
    lgSSFR = np.append(lgSSFR, lgsfr_interp-lgms_interp)
Ms = 10**lgMs.reshape((6,4)).T
#==============================================================================
# Combine and Measure             
#==============================================================================
weight = np.concatenate((np.linspace(0.,0.8,9),np.linspace(0.9,1.0,6)))
X_track,Y_track = np.empty((15,SFH.size)),np.empty((15,SFH.size))

#==============================================================================
# Choose SSP
#==============================================================================
# A:0 B:1 C:2 D:3
SP1, SP2 = SP_A, SP_C
sp1, sp2 = 0, 2
#==============================================================================
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')
filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
Names = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
synphot1 = Observe(SP1, filters, redshift=0.001)
synphot2 = Observe(SP2, filters, redshift=0.001)
fluxes1 = np.squeeze(synphot1.fluxes).value
fluxes2 = np.squeeze(synphot2.fluxes).value
                    
data = Table()

for i, n in enumerate(Names):
    flux = np.array([])
    for k,w in enumerate(weight):
        flux = np.append(flux, ((1-w) * fluxes1[i] * Ms[sp1,:] + \
                                w * fluxes2[i] * Ms[sp2,:]).ravel())
    err = 0.01 * flux
    data.add_columns([Column(flux,'F%s'%(i+1)), Column(err,'E%s'%(i+1))])

id = Column(name='id', data=np.arange(1,len(data)+1)) 
zspec = Column(name='zspec', data= -1 *np.ones(len(data)))  
data.add_column(id, 0) 
data.add_column(zspec)  

df = data.to_pandas()
np.savetxt('Composite/composite_ds_AC_exp.cat', data, header=' '.join(data.colnames),
               fmt=['%d']+['%.5e' for k in range(20)]+['%.2f'])
