from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps
from astropy.table import Table, Column
from astropy.io import ascii
from smpy.smpy import LoadEAZYFilters, FilterSet

def compute_flux_density(temp_sed, filt):
	lam_s = temp_sed["lambda"]/(1+z)
	spec = temp_sed.tempflux*(1+z)
	lam_f = filt.wave
	res_f = filt.response

	res_intp  = np.interp(lam_s,lam_f,res_f)        #Interpolate to common wavelength axis
	I1 = simps(spec*res_intp*lam_s,lam_s)
	I2 = simps(res_intp/lam_s,lam_s)
	fnu = I1/I2/ c_AAs
	return fnu

# CANDELS filter
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')
filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))
print "Reading CANDELS rest-frame filters...OK!"
bands_rf = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
c_AAs     = 2.99792458e18

# EAZY output
EAZY_catalog = ascii.read("EAZY_uds-all_MC200.cat")
z_spec = EAZY_catalog["z_spec"]
log_M = EAZY_catalog["log_M"]
objid = EAZY_catalog["obj"]
run = EAZY_catalog["run"]
print "Reading EAZY output...OK!"

# Calculate rest-frame colors
print "Start calculating rest-frmae colors..."
lam_rf = [filt.lambda_c.value for filt in filters.filters]
flux = np.zeros((len(z_spec),10))
N_obj = len(objid)

for d,z in enumerate(z_spec):

	#obs_sed = ascii.read("./OUTPUT/%d.obs_sed"%(d+1)).to_pandas()
	temp_sed = ascii.read("./OUTPUT/%d.temp_sed"%(d+1)).to_pandas()

	flux[d] = [compute_flux_density(temp_sed,filt) for i,filt in enumerate(filters.filters)]
	
#	semilogy(obs_sed["lambda"]/(1+z),obs_sed.flux_cat*(1+z))
#	semilogy(temp_sed["lambda"]/(1+z),temp_sed.tempflux*(1+z),ls="--")
	
	if mod(d,N_obj)==0: print "N Monte-Carlo: %d"%(d/N_obj+1)
	
df_flux = pd.DataFrame(flux,columns=bands_rf)
df_flux.insert(0,"id",arange(1,len(z_spec)+1))
df_flux.insert(1,"obj",objid)
df_flux.insert(2,"run",run)
df_flux["z_spec"] = z_spec
df_flux["log_M"] = log_M

df_flux.to_csv("run_all-MC200.flux",sep=" ",index=False)
