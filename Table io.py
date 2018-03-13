# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 23:16:26 2017

@author: Qing Liu
"""

import pickle
import numpy as np

from astropy.table import Table, Column

from smpy.smpy import LoadEAZYFilters, FilterSet, Observe

#with open('grid_exp.pkl','r') as f: 
#    Data = pickle.load(f)

SFH = Data['SFH']
Ages = Data['Ages']
models= Data['Models']
Ms = Data['M*']
    
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')
filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
    # FUV, NUV, U, B, V, R, I, J, H, K
Names = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']

synphot = Observe(models, filters, redshift=0.001)
fluxes = np.squeeze(synphot.fluxes).value

#==============================================================================
# Make Table
#==============================================================================
data = Table()

for i, n in enumerate(Names):
    flux = (fluxes[i] * Ms[:,:,None]).ravel()
    err = 0.2 * flux
    data.add_columns([Column(flux,'F%s'%(i+1)), Column(err,'E%s'%(i+1))])

id = Column(name='id', data=np.arange(1,len(data)+1)) 
zspec = Column(name='zspec', data= -1 *np.ones(len(data)))  
data.add_column(id, 0) 
data.add_column(zspec)  

df = data.to_pandas()
#df.insert(1, 'T', np.vstack([Ages for i in range(SFH.size)]).T.ravel())
#df.insert(2, 'tau', np.concatenate([SFH for i in range(Ages.size)]))

np.savetxt('dust_exp.cat', data, header=' '.join(data.colnames),
           fmt=['%d']+['%.5e' for i in range(20)]+['%.2f'])
