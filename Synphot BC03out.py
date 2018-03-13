# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:05:38 2017

@author: Qing Liu
"""

import glob
import numpy as np
import asciitable as asc
from scipy.interpolate import griddata

import astropy.units as u
from astropy import constants as c
from astropy import cosmology as cos
from astropy.table import Table, Column
cosmo = cos.FlatLambdaCDM(H0=70, Om0=0.3)

from smpy.smpy import LoadEAZYFilters, FilterSet
from smpy.misc import tau_madau

eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')

print eazy_library.filternames

filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  

class observe(object):    
    def __init__(self, SED, wave, Filters, redshift,
                 madau=True, units=u.uJy):
        """
        
        Parameters
        ----------
        SED : '~smpy.CSP' object
            Built
        Filters : '~smpy.FilterSet' object
            Filter set through which to observe the set of models included
            in SED object
        redshift : float of numpy.array
            Redshift(s) at which models are to be observed
        madau : boolean
            Apply IGM absorption following Madau 1999 prescription
        units : '~astropy.units.Quantity'
            Desired output units, must be in spectral flux density equivalent        
        """
        self.F = Filters
        self.redshifts = np.array(redshift, ndmin=1)
        self.wave = wave
        
        self.fluxes = np.zeros(np.append([len(self.redshifts),
                                          len(self.F.filters)],
                                          SED.shape[:-1])) * units
        self.AB = np.zeros_like(self.fluxes.value) * u.mag
        self.wl = np.zeros(len(self.F.filters)) * u.AA
        self.fwhm = np.zeros(len(self.F.filters)) * u.AA
        
        self.dl = cosmo.luminosity_distance(self.redshifts).cgs
        self.dl[self.redshifts == 0] = 10 * c.pc
        
        
        for i, z in enumerate(self.redshifts):
            self.lyman_abs = np.ones(len(self.wave))
            if madau:
                self.lyman_abs = np.clip(tau_madau(self.wave, z), 0., 1.)
            
            for j, filter in enumerate(self.F.filters):
                self.wl[j] = filter.lambda_c
                self.fwhm[j] = filter.fwhm
                self.fluxes[i, j] = self.calcflux(SED, filter, z, 
                                                  self.dl[i], units)
            
        
        # Convert spectral flux density to AB magnitudes
        self.AB = (-2.5 * np.log10(self.fluxes.to(u.Jy) / 
                   (3631 * u.Jy))) * u.mag
    
    def calcflux(self, SED, filt, z, dl, units):
        """ Convolve synthetic SEDs with a given filter
        
            Arguments
            ---------
                SED : numpy.array
                    Grid of synthetic spectra
                filt : '~smpy.Filter' class
                    Filter through which to convolve SED grid
                z : float
                    Redshift at which models are to be observed
                dl : '~astropy.units.Quantity'
                    Luminosity distance corresponding to redshift(z) in given
                    cosmology.
                units : '~astropy.units'
                    Desired output flux units (in spectral flux density)
            Returns
            -------
                Flux : '~astropy.units.Quantity'
                    Spectral flux density, with exact units as given by 'units'
        """
        # Find SED wavelength entries within filter range
        wff = np.logical_and(filt.wave[0] < self.wave, 
                             self.wave < filt.wave[-1])
        wft = self.wave[wff]
        
        # Interpolate to find throughput values at new wavelength points
        tpt = griddata(filt.wave, filt.response, wft)
        
        # Join arrays and sort w.r.t to wf
        # Also replace units stripped by concatenate
        wf = np.array(np.concatenate((filt.wave, wft))) * u.AA
        tp = np.concatenate((filt.response, tpt))
        
        order = np.argsort(wf)
        wf = wf[order]
        tp = tp[order]
                
        # Interpolate redshifted SED and LyAbs at new wavelength points
        sed = griddata(self.wave * (1 + z), SED.T, wf).T * SED.unit
        lyabs = griddata(self.wave, self.lyman_abs, wf)
        
        # Calculate f_nu mean
        # Integrate SED through filter, as per BC03 Fortran
        # As: f_nu=int(dnu Fnu Rnu/h*nu)/int(dnu Rnu/h*nu)
        # ie: f_nu=int(dlm Flm Rlm lm / c)/int(dlm Rlm/lm)
        top = np.trapz(sed * lyabs[None, None, None, None, None, :] * tp * wf /
                       c.c.to(u.AA / u.s), wf)
        bottom = np.trapz(tp / wf, wf)
        area = (4 * np.pi * (dl ** 2))
        Flux = top / bottom / (1 + z) / area
        
        return Flux.to(units)

Flux = np.zeros(10)
dir=glob.glob('FAST/*.spec')   
for f in dir:
    table = asc.read(f)
    wave = table['col1']*u.AA
    SED = np.vstack([table[col_age] for col_age in table.dtype.names[1:]])*u.L_sun/u.AA
    synshot = observe(SED,wave,filters, redshift=0.0001)
    flux = np.squeeze(synshot.fluxes.value).T
    Flux = np.vstack((Flux,flux))
Flux=Flux[1:]

data = Table()
Names = ['1', '2', '3', '4', '5', '6', '7', '8','9','10']  
for i, n in enumerate(Names):
    err = 0.01 * Flux[:,i]
    data.add_columns([Column(Flux[:,i], 'F'+n), Column(err, 'E'+n)])
id = Column(name='id', data=np.arange(1,len(data)+1)) 
zspec = Column(name='zspec', data=-1 *np.ones(len(data)))  
data.add_column(id, 0) 
data.add_column(zspec)  

np.savetxt('BC03_exp.cat',data,fmt='%.5e')

