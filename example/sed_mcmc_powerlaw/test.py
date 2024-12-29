import numpy as np
from matplotlib import pyplot as plt
import emcee
import corner
import sys
sys.path.insert(0,'../../')
from X22_model.disk_model import *
from radmc.setup import *
from multiprocessing import Pool
import time 
from datetime import datetime
import shutil
import h5py

n_processes = 20
nwalkers = 10  # Total number of walkers
ndim = 2        # Dimension of parameter space
niter = 100000     # Number of iterations

# # CB68's observation
observed_Freq = np.array([
                        # 95.0,
                        233.8,
                        246.8
                        ])
observed_Flux = np.array([
                        # 7.6,
                        59, 
                        66
                        ])
err           = np.array([
                        # 0.022,
                        0.029,
                        0.040
                        ])

        
def sed_model(theta):
    """
    Simple power-law model for SED
    Parameters:
        theta: list or array, [A, alpha]
            A: Amplitude (scaling factor)
            alpha: Spectral index
    Returns:
        nu: array
            Frequencies (same as observed_Freq for comparison)
        fnu: array
            Flux densities at corresponding frequencies
    """
    A, alpha = theta
    # nu = np.linspace(200, 300, 20) * 1e9
    nu = observed_Freq # Using the observed frequencies directly
    fnu = A * nu**alpha
    return nu, fnu


def log_prior(theta):
    A, alpha = theta
    if 0 < A  and -1 < alpha < 5:  # Example prior ranges
        return 0.0
    return -np.inf

def log_probability(theta, Freq, Flux, err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    nu, fnu = sed_model(theta)
    return lp - 0.5 * np.sum((Flux - fnu)**2 / err**2)

initial = [np.array([2.6e-4, 2.07]) + [1e-5, 1e-3] * np.random.randn(ndim) for i in range(nwalkers)]

with Pool(n_processes) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(observed_Freq, observed_Flux, err), pool=pool)
    sampler.run_mcmc(initial, niter, progress=True)

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["A", r"$\alpha$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
# plt.show()
plt.savefig('chain_step.pdf', transparent = True)
plt.close()


tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=2000, thin=50, flat=True)
fig = corner.corner(
    flat_samples, labels=labels,
    show_titles=True, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
fig.savefig('posterior.pdf', transparent=True)
plt.close()



inds = np.random.randint(len(flat_samples), size=1000)
for ind in inds:
    sample = flat_samples[ind]
    A, alpha = sample
    nu = np.linspace(230, 250, 100)
    plt.plot(nu, A*nu**alpha, "C1", alpha=0.005)
plt.errorbar(observed_Freq, observed_Flux, yerr=err, fmt=".k", capsize=0)
plt.xlabel('$\\nu [GHz]$')
plt.ylabel('Flux density [mJy]')
plt.savefig('plot.pdf', transparent=True)
plt.close()