import numpy as np
from matplotlib import pyplot as plt
import emcee
import corner
import sys
sys.path.insert(0,'../../')
from disk_model import *
from radmc.setup import *
from radmc.simulate import generate_simulation


class general_parameters:
    '''
    A class to store the parameters for individual kinds of grids.
    Details of individual parameters should refer to the functions that generate the grids.
    '''
    def __init__(self, **kwargs
                 ):
        for k, v in kwargs.items():
          # add parameters as attributes of this object
          setattr(self, k, v)

    def __del__(self):
      pass

    def add_attributes(self, **kwargs):
      '''
      Use this function to set the values of the attributs n1, n2, n3,
      which are number of pixels in the first, second, and third axes. 
      '''
      for k, v in kwargs.items():
        # add parameters as attributes of this object
        setattr(self, k, v)

###############################################################################
"""
CB68
Mass          : 0.08-0.30 Msun
Accretion rate: 4-7e-7    Msun/yr
Radius        : 20-40     au
Distance      : 140       pc
"""
###############################################################################
def sed_model(theta):
    amax, Mstar, Mdot, incl, r = theta
    
    model = radmc3d_setup(silent=False)
    model.get_mastercontrol(filename=None,
                            comment=None,
                            incl_dust=1,
                            incl_lines=1,
                            nphot=500000,
                            nphot_scat=5000000,
                            nphot_spec=500000, 
                            scattering_mode_max=2,
                            istar_sphere=1,
                            num_cpu=18)
    model.get_linecontrol(filename=None,
                        methanol='ch3oh leiden 0 0 0')
    model.get_continuumlambda(filename=None,
                            comment=None,
                            lambda_micron=None,
                            append=False)
    model.get_diskcontrol(  d_to_g_ratio = 0.01,
                            a_max=10**amax, 
                            Mass_of_star=0.1*Mstar, 
                            Accretion_rate=10**Mdot,
                            Radius_of_disk=r,
                            NR=200,
                            NTheta=200,
                            NPhi=10)
    model.get_vfieldcontrol(Kep=True,
                            vinfall=None,
                            Rcb=None,
                            outflow=None)
    model.get_heatcontrol(heat='accretion')
    model.get_gasdensitycontrol(abundance=1e-10,
                                snowline=None,
                                enhancement=1e5,
                                gas_inside_rcb=True)

    os.system(f'radmc3d spectrum incl {incl} lambdarange 1200. 1300. nlam 5 noline noscat')
    s = readSpectrum('spectrum.out')
    lam = s[:, 0]
    nu = (1e-2*cc)*1e-9/(1e-6*lam) # GHz
    fnu = s[:, 1]*1e26/(140**2) # mJy
    return nu[::-1], fnu[::-1]

# Synthetic data (Mock Observation)
observed_Freq = np.array([233.8, 233.8, 233.8, 246.7, 246.7, 246.7, 246.7, 246.7])
observed_Flux = np.array([   56,    55,    59,    62,    60,    60,    61,    66])
err           = np.array([0.054, 0.150, 0.029, 0.120, 0.300, 0.220, 0.120, 0.040])


# plt.errorbar(observed_Freq, observed_Flux, yerr=err, fmt=".k", capsize=0)
# plt.show()


# Log-prob, log-likelihood, and prior
def log_likelihood(theta, Freq, Flux, err):
    """
    y and r is the temperature and radius data point
    """ 
    freq_model, flux_model = sed_model(theta=theta)
    freq_index = np.searchsorted(freq_model, Freq)
    return -0.5 * np.sum((Flux - flux_model[freq_index]) ** 2 / err**2)

def log_prior(theta):
    a_max, Mstar, Mdot, incl, r = theta
    if -3 < a_max < 3 and 0.05 < Mstar < 0.5 and -9 < Mdot < -5 and 60 < incl < 80 and 10 < r < 60:
        return 0.0
    return -np.inf

def log_probability(theta, Freq, Flux, err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, Freq, Flux, err)


# MCMC Sampler
def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
    return sampler, pos, prob, state


def plotter(sampler, x=observed_Freq, y=observed_Flux, yerr=err):
    # plt.ion()
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0,label='CB68')
    samples = sampler.flatchain
    for theta in samples:
        freq_model, flux_model = sed_model(theta)
        plt.plot(freq_model, flux_model, "C1", alpha=0.1)
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    print(theta_max)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend()
    plt.title('MCMC Test')
    plt.ylabel('Flux density [mJy]')
    plt.xlabel('$\\nu [GHz]$')
    plt.ylim((0,100))
    plt.xlim((230,250))
    plt.xscale('log')
    plt.savefig('MCMC_test.pdf', transparent=True)
    # plt.show()

def posterior(sampler, label = [r'log($a_{max}$)',r'$M_{*}$', r'log($\dot{M}$)', r'$i^{\circ}$', 'Radius [au]']):
    samples = sampler.flatchain
    fig = corner.corner(
        samples, labels=label, 
        show_titles=True, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84]
        )
    fig.savefig('corner_test.pdf', transparent=True)

nwalkers, ndim = 150, 5  # Number of walkers and dimension of the parameter space
pos = [np.array([-2, 0.14, -7, 70, 30]) + [1, 5e-2, 0.5, 5, 5] * np.random.randn(ndim) for i in range(nwalkers)]
sampler, pos, prob, state = main(pos,nwalkers,50,ndim,log_probability, (observed_Freq, observed_Flux, err))
plotter(sampler)
posterior(sampler)



np.savez('result.npz',
         samples = sampler.flatchain,
         theta_max = sampler.flatchain[np.argmax(sampler.flatlnprobability)])
