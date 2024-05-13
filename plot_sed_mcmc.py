import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from disk_model import *
from problem_setup import problem_setup
from scipy.optimize import curve_fit
import emcee
import corner
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
    amax, Mstar, Mdot, incl= theta
    problem_setup(a_max=amax, Mass_of_star=Mstar*Msun, Accretion_rate=Mdot*1e-5*Msun/yr, Radius_of_disk=30*au, pancake=False, v_infall=0)
    os.system(f'radmc3d spectrum incl {incl*100} lambdarange 300. 3000. nlam 100 noline noscat')
    s = readSpectrum('spectrum.out')
    lam = s[:, 0]
    nu = (1e-2*cc)*1e-9/(1e-6*lam) # GHz
    fnu = s[:, 1]*1e26/(140**2) # mJy
    return nu, fnu

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
    a_max, Mstar, Mdot, incl = theta
    if 0.001 < a_max < 1 and 0.08 < Mstar < 0.50 and 0.02 < Mdot < 2 and .50 < incl < .90:
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
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0,label='training data')
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=100)]:
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
    plt.xlim((1e2,1e3))
    plt.xscale('log')
    plt.savefig('MCMC_test.pdf')
    # plt.show()

def posterior(sampler, label = ['$a_{max}$','$M_{*}$', '$\dot{M}$', '$i^{\circ}$']):
    samples = sampler.flatchain
    fig = corner.corner(
        samples, labels=label, 
        show_titles=True, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84]
        )
    fig.savefig('corner_test.pdf')

nwalkers, ndim = 100, 4  # Number of walkers and dimension of the parameter space
pos = [np.array([0.1, 0.14, 0.14, .70]) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
sampler, pos, prob, state = main(pos,nwalkers,100,ndim,log_probability, (observed_Freq, observed_Flux, err))
plotter(sampler)
posterior(sampler)
