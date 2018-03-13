# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:17:39 2017

@author: Qing Liu
"""

import numpy as np
import asciitable as asc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from astropy import constants as c
from astropy import units as u        
from astropy import cosmology as cos
cosmo = cos.FlatLambdaCDM(H0=67.8, Om0=0.307)
      
#==============================================================================
# 
#==============================================================================
Ages = np.linspace(0.5,11.5,111)
Ages = np.linspace(0.5,3.5,61)
M_today = np.array([9.0,9.5,10.0,10.5,11.0,11.5])
SED_grid = np.empty((Ages.size,M_today.size,2022))

for i in range(Ages.size):
    for j in range(M_today.size):
        table  = asc.read('BEST_FITS/Aldo_SFH_P2.5_%d.fit'
                          %(i*M_today.size+j+1))
        wl, fl = table['col1'], table['col2']
        table_i  = asc.read('BEST_FITS/Aldo_SFH_P2.5_%d.input_res.fit'
                            %(i*M_today.size+j+1))
        wave, flux = table_i['col1'], table_i['col2']
        SED_grid[i,j,:] = fl
#        plt.figure(figsize=(8,4))
#        plt.plot(wl,fl,'k',alpha=0.7)
#        plt.plot(wave,flux/1e29,'rs',alpha=0.9)
#        plt.title(r'T = %.1f Gyr    $M_{today}$ = %.1f M$\odot$'
#                  %(Ages[i],M_today[j]),fontsize=15)
#        plt.xlabel('wavelength ($\AA$)',fontsize=12)
#        plt.ylabel('flux ($10^{-19} erg s^{-1} cm^{-2} \AA^{-1}$)',fontsize=12)
#        plt.xlim(0,25000)
##        #plt.savefig('SED-fitting/P2.5_M%dT%d.png'%(j+1,i+1))
#        plt.show()
        print i,j
        
SED_grid = SED_grid*u.solLum/u.AA

#==============================================================================
# 
#==============================================================================
from smpy.smpy import LoadEAZYFilters, FilterSet
eazy_library = LoadEAZYFilters('FILTER.RES.CANDELS')
filters = FilterSet()
filters.addEAZYFilter(eazy_library, range(len(eazy_library.filternames)))  
    # FUV, NUV, U, B, V, R, I, J, H, K

#==============================================================================
# 
#==============================================================================
class Observe_s(object):
    def __init__(self, SED, Filters, wave_u, redshift, units=u.uJy):
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
        units : '~astropy.units.Quantity'
            Desired output units, must be in spectral flux density equivalent
    
        Attributes
        ----------
        fluxes : array
            Apparent fluxes of CSP models observed through 'Filters' at 
            the desired redshifts
        AB : array
            Apparent AB magnitudes of CSP models observed through 'Filters' at
            the desired redshifts
        """
        self.F = Filters
        self.redshifts = np.array(redshift, ndmin=1)
        self.wave = wave_u
        
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
    
    def __getitem__(self, items):
        return self.fluxes[items]
    
    
synphot = Observe_s(SED_grid, filters, wl*u.AA, 0.001, units=u.uJy)
fluxes_conv = synphot.fluxes[0]

for i in range(Ages.size):
    for j in range(M_today.size):
        table_i  = asc.read('BEST_FITS/Aldo_SFH_P2.5_%d.input_res.fit'
                            %(i*M_today.size+j+1))
        wave, flux = table_i['col1'], table_i['col2']
        plt.figure(figsize=(8,4))
        plt.plot(wave,fluxes_conv[:,i,j],'ks',alpha=0.7)
        plt.plot(wave,flux/1e29,'rs',alpha=0.9)
        plt.title(r'T = %.1f Gyr    $M_{today}$ = %.1f M$\odot$'
                  %(Ages[i],M_today[j]),fontsize=15)
        plt.xlabel('wavelength ($\AA$)',fontsize=12)
        plt.ylabel('flux ($10^{-19} erg s^{-1} cm^{-2} \AA^{-1}$)',fontsize=12)
        plt.xlim(0,25000)
#        #plt.savefig('SED-fitting/P2.5_M%dT%d.png'%(j+1,i+1))
        plt.show()
        print i,j