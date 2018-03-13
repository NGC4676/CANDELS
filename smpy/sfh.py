import numpy as np
import astropy.units as u
from scipy.special import erfc

def exponential(t, tau):
    """ Exponential star formation history

    """
    sfh = np.exp(-1 * t / tau) / np.abs(tau)
    return sfh


def power(t, alpha):
    """ Power-law star formation history
    """
    sfh = np.power(t, alpha)
    return sfh


def delayed(t, tau):
    """ 'Delated' star formation history
    """
    sfh = t / (tau ** 2) * np.exp(-t / tau)
    return sfh

def dstb(t,tau,T1):
#    sfh = np.sin((t*2*np.pi/T1).value)**2 * np.exp(-t/tau) / tau
    sfh = np.sin((t*2*np.pi/T1).value)**2 / tau
    return sfh

def delayed_dstb(t, tau=1, k1=1., k2=0.05):
    """ 'Delated' star formation history
    """
    T1 = k1 * tau
    sfh = k2 * dstb(t, tau, T1) + (1-k2) * t * np.exp(-t/tau) / tau**2
    return sfh

def sinuodial(t, tau):
    sfh = (np.sin((t*2*np.pi/tau).value)+1)/tau
    return sfh

def truncated(t, tstop):
    """ Truncated star formation history

    Star-formation is continuous until some fraction, tstop, of the total
    time since onset of star-formation history np.max(t).

    """
    sfh = 1 / np.ones_like(t) 
    cut = np.argmin(np.abs(t - tstop*np.max(t))) # Nearest whole timestep.
    np.rollaxis(sfh, -1)[cut:] = 0.
    return sfh


def truncated_exp(t, tau, tstop):
    """ Truncated exponential star formation history

    Star-formation is exponential, tstop, of the total
    time since onset of star-formation history np.max(t).

    """
    sfh = np.exp(-1 * t / tau) / abs(tau)
    cut = np.argmin(np.abs(t - tstop*np.max(t))) # Nearest whole timestep.
    np.rollaxis(sfh, -1)[cut:] = 0.
    return sfh


	
def gaussian( x, params ):
    (c, mu, sigma) = params
    res =   c * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) )
    return res
def lorentzian( x, params ):
    (c, x0, gama) = params
    res =   c * gama / ((x-x0)**2+gama**2) / np.pi
    return res
def hermite( x, params):
    (h3, h4) = params
    res = (1 + h3*(8*x**3.-12*x)/np.sqrt(6*8.) + \
               h4*(16*x**4-48.*x**2.+12)/np.sqrt(24*16.))
    return res

def gaussian_hermite(x, params):
    (c, mu, sigma, h3, h4) = params
    res =   gaussian(x, (c, mu, sigma)) * hermite(x, (h3, h4))
    return res
def lorentzian_hermite(x, params):
    (c, x0, gama, h3, h4) = params
    res =   lorentzian(x, (c, x0, gama)) * hermite(x, (h3, h4))
    return res
	
def gauss_lorentz_hermite(t, params):
    t = t.to(u.Gyr).value
    c1, mu, sigma, h13, h14, c2, x0, gama, h23, h24 = params
    res = (gaussian_hermite(t, (c1, mu, sigma, h13, h14)) \
        + lorentzian_hermite(t, (c2, x0, gama, h23, h24)) ) / (u.Gyr)
    res[t<0.4] = 0. 
    return res 

def glh_prtb(t, params):
    t = t.to(u.Gyr).value
    c1, mu, sigma, h13, h14, \
    c2, x0, gama, h23, h24, \
    A, P = params
    phi = np.random.rand()*2*np.pi
    res = (gaussian_hermite(t, (c1, mu, sigma, h13, h14)) \
        + lorentzian_hermite(t, (c2, x0, gama, h23, h24)) ) \
        * (1 + A*np.sin((2*np.pi*t/P + phi)*u.rad)) / (u.Gyr)
    res[t<0.4] = 0. 
    return res 

def glh_absprtb(t, params):   #oscillation in dex
    t = t.to(u.Gyr).value
    c1, mu, sigma, h13, h14, \
    c2, x0, gama, h23, h24, \
    A, P = params
    phi = 3*np.pi/4.
    res = (gaussian_hermite(t, (c1, mu, sigma, h13, h14)) \
        + lorentzian_hermite(t, (c2, x0, gama, h23, h24)) ) \
        * 10**(-A*np.sin((2*np.pi*t/P + phi)*u.rad)) / (u.Gyr)
    res[t<0.4] = 0. 
    return res 

#def glh_ST(t, params):    #Cease the oscillation after a time
#    t = t.to(u.Gyr).value
#    c1, mu, sigma, h13, h14, \
#    c2, x0, gama, h23, h24, \
#    A, P = params
#    time_start = 4.
#    phi = 0.
#    res1 = (gaussian_hermite(t, (c1, mu, sigma, h13, h14)) \
#        + lorentzian_hermite(t, (c2, x0, gama, h23, h24)) ) 
#    res2 = (gaussian_hermite(t, (c1, mu, sigma, h13, h14)) \
#        + lorentzian_hermite(t, (c2, x0, gama, h23, h24)) ) \
#        * 10**(-A*np.sin((5*np.pi*np.log(t) + phi)*u.rad))
#    
#    if len(res2[t>time_start])==0:
#        res = res1 / (u.Gyr)
#    else:
#        res = [np.concatenate([(res1[t<time_start]).value, (res2[t>time_start]).value])] / (u.Gyr)
#    res[t<0.4] = 0. 
#    return res 

def glh_ST(t, params):
    t = t.to(u.Gyr).value
    c1, mu, sigma, h13, h14, \
    c2, x0, gama, h23, h24, \
    A, P = params
    phi = 3*np.pi/4.
    res = (gaussian_hermite(t, (c1, mu, sigma, h13, h14)) \
        + lorentzian_hermite(t, (c2, x0, gama, h23, h24)) ) \
        * 10**(-A*np.sin((5*np.pi*np.log(t) + phi)*u.rad) \
               +0.*np.random.normal(0.,0.3,t.size)) / (u.Gyr)
    res[t<0.4] = 0. 
    return res 

def RKPRG(t, params):
    t = t.to(u.Gyr).value
    m_seed = params
    A = 6e-3*np.exp(-np.log10(m_seed)/-0.84)
    mu = 47.39*np.exp(-np.log10(m_seed)/3.12)
    sigma = 17.08*np.exp(-np.log10(m_seed)/2.96)
    r_s = -0.56*np.log10(m_seed) + 7.03 
    res = A*(np.sqrt(np.pi)/2.) \
            *np.exp( (sigma/(r_s*2.))**2 - (t-mu)/r_s ) \
            *sigma*erfc((sigma/(r_s*2.) - (t-mu)/sigma).value)
    res[t<0.4] = 0. 
    return res
