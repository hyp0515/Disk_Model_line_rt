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
from radmc3dPy import image
import os
sys.path.insert(0,'../')
from fit_with_GIdisk.find_center import find_center

n_processes = 6
nwalkers = 6  # Total number of walkers
ndim = 3        # Dimension of parameter space
niter = 100000     # Number of iterations


# # CB68's continuum
# cb68_conti = np.load('../../CB68/CB68_conti.npy')
# rms_noise = 30e-6
# background = np.sum(cb68_conti *(cb68_conti <(30*rms_noise))) / np.sum(cb68_conti <(30*rms_noise))
# cb68_conti -= background

# Initialize Observation data
cb68_alma_list = [
    '/run/media/hyp0515/fd14f880-ba6f-450f-b82d-98ba3710dc5f/backup/CB68/cb68_alma/cb68_setup1_all.rob2.I.image.tt0.pbcor.smooth.dropdeg.fits',
    '/run/media/hyp0515/fd14f880-ba6f-450f-b82d-98ba3710dc5f/backup/CB68/cb68_alma/cb68_setup2_all.rob2.I.image.tt0.pbcor.smooth.dropdeg.fits',
    '/run/media/hyp0515/fd14f880-ba6f-450f-b82d-98ba3710dc5f/backup/CB68/cb68_alma/cb68_setup3_all.rob2.I.image.tt0.pbcor.smooth.dropdeg.fits'
]

nu_list = [
    233.8,
    246.7,
    95.0
]

lam_list = cc * 1e4 / (np.array(nu_list) * 1e9)
print(lam_list)

sigma_list = [
    30e-6,
    40e-6,
    26e-6
]

observation_data = []
beam_pa = []
disk_posang = []
beam_axis = []
beam_area = []


for i, fname in enumerate(cb68_alma_list):
    ra_deg, dec_deg, disk_pa = find_center(fname)
    image_class = DiskImage(
        fname = fname,
        ra_deg = ra_deg,
        dec_deg = dec_deg,
        distance_pc = 140,
        rms_Jy = sigma_list[i], # convert to Jy/beam
        disk_pa = disk_pa,
        img_size_au = 100,
        remove_background=True
    )
    observation_data.append(image_class.img)
    beam_pa.append(image_class.beam_pa)
    beam_axis.append([image_class.beam_maj_au, image_class.beam_min_au])
    beam_area.append(pi/(4*np.log(2))*image_class.beam_maj_au*image_class.beam_min_au/\
                    (image_class.au_per_pix**2))
    disk_posang.append(disk_pa)
    
size_au  = image_class.img_size_au
au_per_pix  = image_class.au_per_pix
npix = image_class.img.shape[0]
pa = np.mean(disk_pa)


# model
def conti_model(theta):
    temp_dir_name = f'./temp/{datetime.now().strftime("%H%M%S")}_{datetime.now().microsecond}_{os.getpid()}'
    os.makedirs('./temp/', exist_ok=True)
    os.makedirs(temp_dir_name)
    os.chdir(temp_dir_name)
    
    with open('camera_wavelength_micron.inp', 'w+') as f:
        f.write('%d\n'%(len(lam_list)))
        for value in lam_list:
            f.write('%13.6e\n'%(value))
    
    amax, Mstar, Mdot = theta
 
    model = radmc3d_setup(silent=False)
    model.get_mastercontrol(filename=None,
                            comment=None,
                            incl_dust=1,
                            incl_lines=1,
                            nphot=500000,
                            nphot_scat=500000,
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
                            Radius_of_disk=30,
                            NR=200,
                            NTheta=200,
                            NPhi=10)
    model.get_heatcontrol(heat='accretion')

    
    os.system(f'radmc3d image npix {npix} sizeau {size_au} posang {pa} incl 73 loadlambda noline noscat')

    im = image.readImage()
    model_image_list = []
    for i in range(im.image.shape[2]):
        model_image = im.imageJyppix[:,:,i].T / (140**2)
        I = ndimage.rotate(model_image, -disk_pa+beam_pa[i], reshape=False)
        sigmas = np.array([beam_axis[i][0], beam_axis[i][1]])/au_per_pix/(2*np.sqrt(2*np.log(2)))
        I = ndimage.gaussian_filter(I, sigma=sigmas)
        # rotate to align with image
        I = ndimage.rotate(I, -beam_pa[i], reshape=False)
        # convert to flux density in Jy/beam
        model_image = I*beam_area[i]
        model_image_list.append(model_image)
    
    os.chdir("../..")
    shutil.rmtree(temp_dir_name)
    
    return model_image_list

def log_likelihood(theta, observation, err):
    model_image = conti_model(theta=theta)
    
    log_likelihood = []
    
    for i in range(len(observation)):
        ll = -0.5 * np.sum((observation[i][5:-5, 5:-5] - model_image[i][5:-5, 5:-5]) ** 2 / err[i]**2)
        log_likelihood.append(ll)

    return np.mean(log_likelihood)

def log_prior(theta):
    amax, Mstar, Mdot = theta
    if -3 < amax < 3 and 0.08 < Mstar < 0.3 and np.log10(1e-7) < Mdot < np.log10(10e-7):
        return 0.0
    return -np.inf

def log_probability(theta, observation, err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, observation, err)

pos = [np.array([0, 0.14, np.log10(5e-7)]) + [1e-1, 1e-3, 1e-2] * np.random.randn(ndim) for i in range(nwalkers)]

# File for saving progress
progress_file = "progress.h5"
backend = emcee.backends.HDFBackend(progress_file)
backend.reset(nwalkers, ndim)

# Initialize the sampler
with Pool(n_processes) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(observation_data, sigma_list), pool=pool, backend=backend, a=2)
    sampler.run_mcmc(pos, niter, progress=True)
