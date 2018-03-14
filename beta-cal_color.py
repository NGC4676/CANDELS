from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps
from astropy.table import Table, Column
from astropy.io import ascii
from smpy.smpy import LoadEAZYFilters, FilterSet
import time

# CANDELS filter
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')
filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))
print "Reading CANDELS rest-frame filters...OK!"
bands_rf = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
c_AAs     = 2.99792458e18

# EAZY output
#EAZY_catalog = ascii.read("EAZY_uds-all_MC250.cat")
#EAZY_catalog = ascii.read("EAZY_uds-100_MC100.cat")
z_spec = EAZY_catalog["z_spec"]
log_M = EAZY_catalog["log_M"]
objid = EAZY_catalog["obj"]
run = EAZY_catalog["run"]
print "Reading EAZY output...OK!"

# Calculate rest-frame colors
print "Start calculating rest-frmae colors..."
lam_rf = [filt.lambda_c.value for filt in filters.filters]
flux = np.zeros((len(z_spec),10))
N_obj = objid.max()

lams_s = np.empty((len(z_spec),1603))
specs = np.empty((len(z_spec),1603))

start = time.time()

for d,z in enumerate(z_spec):
	temp_sed = ascii.read("./OUTPUT/%d.temp_sed"%(d+1))
	lams_s[d] = temp_sed["lambda"]/(1+z)
	specs[d] = temp_sed["tempflux"]*(1+z)
	if mod(d,N_obj)==0: print "N Monte-Carlo: %d"%(d/N_obj+1)

for i,filt in enumerate(filters.filters):
	lam_f = filt.wave.value
        res_f = filt.response
	res_intp  = np.interp(lams_s,lam_f,res_f)
	I1 = simps(specs*res_intp*lams_s,lams_s,axis=1)
        I2 = simps(res_intp/lams_s,lams_s,axis=1)
        fnu = I1/I2/ c_AAs
	flux[:,i] = fnu

end = time.time()

print "Total Time used: %.2fs"%(end-start)
	
df_flux = pd.DataFrame(flux,columns=bands_rf)
df_flux.insert(0,"id",arange(1,len(z_spec)+1))
df_flux.insert(1,"obj",objid)
df_flux.insert(2,"run",run)
df_flux["z_spec"] = z_spec
df_flux["log_M"] = log_M

#df_flux.to_csv("run_uds-100_MC100.beta.flux",sep=" ",index=False)
#df_flux.to_csv("run_uds-all_MC250.beta.flux",sep=" ",index=False)
