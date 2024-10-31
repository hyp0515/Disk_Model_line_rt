import numpy as np
from matplotlib import pyplot as plt
import emcee
import corner
import sys
sys.path.insert(0,'../../')
from disk_model import *
from radmc.setup import *
from radmc.simulate import generate_simulation
from multiprocessing import Pool, current_process


n_processes = 20
nwalkers = 300  # Total number of walkers
ndim = 5        # Dimension of parameter space
niter = 50      # Number of iterations


work_dirs = [f"work_dir_{i}" for i in range(n_processes)]
for d in work_dirs:
    if not os.path.exists(d):
        os.makedirs(d)
        
        
def sed_model(theta, work_dir):
    amax, Mstar, Mdot, incl, r = theta
    if work_dir is not None:
        os.chdir(work_dir)  
        
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
                            num_cpu=1)
    model.get_linecontrol(filename=None,
                        methanol='ch3oh leiden 0 0 0')
    model.get_continuumlambda(filename=None,
                            comment=None,
                            lambda_micron=None,
                            append=False)
    model.get_diskcontrol(  d_to_g_ratio = 0.01,
                            a_max=10**amax, 
                            Mass_of_star=Mstar, 
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
    if work_dir is not None:
        os.chdir('..')
    return nu[::-1], fnu[::-1]


# Synthetic data (Mock Observation)
observed_Freq = np.array([233.8, 233.8, 233.8, 246.7, 246.7, 246.7, 246.7, 246.7])
observed_Flux = np.array([   56,    55,    59,    62,    60,    60,    61,    66])
err           = np.array([0.054, 0.150, 0.029, 0.120, 0.300, 0.220, 0.120, 0.040])




# Log-prob, log-likelihood, and prior
def log_likelihood(theta, Freq, Flux, err, work_dir):
    """
    y and r is the temperature and radius data point
    """ 
    freq_model, flux_model = sed_model(theta=theta, work_dir=work_dir)
    freq_index = np.searchsorted(freq_model, Freq)
    return -0.5 * np.sum((Flux - flux_model[freq_index]) ** 2 / err**2)

def log_prior(theta):
    a_max, Mstar, Mdot, incl, r = theta
    if -3 < a_max < 3 and 0.05 < Mstar < 0.5 and -9 < Mdot < -5 and 60 < incl < 80 and 10 < r < 60:
        return 0.0
    return -np.inf

def log_probability(theta, Freq, Flux, err, work_dir):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, Freq, Flux, err, work_dir)


def run_mcmc_in_dir(data):
    pos, Freq, Flux, err, work_dir = data
    sampler = emcee.EnsembleSampler(len(pos), ndim, log_probability, args=(Freq, Flux, err, work_dir))
    sampler.run_mcmc(pos, niter, progress=True)
    return sampler

pos = [np.array([-2, 0.14, -7, 70, 30]) + [1, 5e-2, 0.5, 5, 5] * np.random.randn(ndim) for i in range(nwalkers)]
walker_groups = np.array_split(pos, n_processes)
data = [(walker_groups[i], observed_Freq, observed_Flux, err, work_dirs[i]) for i in range(n_processes)]

# Run MCMC in parallel with each group assigned to a specific directory
with Pool(n_processes) as pool:
    samplers = pool.map(run_mcmc_in_dir, data)

try:
    np.savez('test.npz',
            all_samples = np.concatenate([sampler.get_chain(flat=True) for sampler in samplers], axis=0))
except:
    pass

def plotter(sampler=None,
            samplers_chain=None,
            fname='MCMC_test',
            x=observed_Freq, y=observed_Flux, yerr=err):
    plt.ion()
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0,label='CB68')
    if sampler is not None:
        if isinstance(sampler, list):
            samples = np.concatenate([sampler.get_chain(flat=True) for sampler in samplers], axis=0)
        else:
            samples = sampler.flatchain
    else:
        samples = samplers_chain
        
    for theta in samples:
        freq_model, flux_model = sed_model(theta, work_dir=None)
        plt.plot(freq_model, flux_model, "C1", alpha=0.1)
    # theta_max = samples[np.argmax(sampler.flatlnprobability)]
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend()
    plt.title('MCMC Test')
    plt.ylabel('Flux density [mJy]')
    plt.xlabel('$\\nu [GHz]$')
    plt.ylim((0,100))
    plt.xlim((230,250))
    plt.xscale('log')
    plt.ioff()
    plt.savefig(fname+'.pdf', transparent=True)
    # plt.show()

def posterior(sampler=None,
              samplers_chain=None,
              fname='corner_test',
              label = [r'log($a_{max}$)',r'$M_{*}$', r'log($\dot{M}$)', r'$i^{\circ}$', 'Radius [au]']):
    if sampler is not None:
        if isinstance(sampler, list):
            samples = np.concatenate([sampler.get_chain(flat=True) for sampler in samplers], axis=0)
        else:
            samples = sampler.flatchain
    else:
        samples = samplers_chain
    fig = corner.corner(
        samples, labels=label, 
        show_titles=True, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84]
        )
    fig.savefig(fname+'.pdf', transparent=True)
    

    
chain = np.load('test.npz')
chain = chain['all_samples']
posterior(samplers_chain=chain, fname=f'corner_all')
plotter(samplers_chain=chain, fname='plot_all')





