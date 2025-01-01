import numpy as np
from matplotlib import pyplot as plt
import emcee
import corner
from multiprocessing import Pool
import time 
from datetime import datetime
import shutil
import h5py
from radmc3dPy import image
import os

import sys
sys.path.insert(0,'../')
from fit_with_GIdisk.find_center import find_center
sys.path.insert(0,'../../')
from X22_model.disk_model import *
from radmc.setup import *
from CB68.data_dict import data_dict

n_processes = 6
nwalkers = 6  # Total number of walkers
ndim = 3        # Dimension of parameter space
niter = 100000     # Number of iterations

fit_data    = [data_dict["1.3_edisk"], data_dict["3.2_faust"]]
lam_list    = [fit_data[0]["wav"], fit_data[1]["wav"]]
sigma_list  = [fit_data[0]["sigma"], fit_data[1]["sigma"]]
desire_size = [60, 120]

observation_data = []
beam_pa = []
beam_axis = []
beam_area = []
disk_posang = []
size_au = []
au_per_pix = []
npix = []

for i, data in enumerate(fit_data):
    ra_deg, dec_deg, disk_pa = find_center(data["fname"])
    image_class = DiskImage(
        fname = data["fname"],
        ra_deg = ra_deg,
        dec_deg = dec_deg,
        distance_pc = 140,
        rms_Jy = data["sigma"], # convert to Jy/beam
        disk_pa = disk_pa,
        img_size_au = desire_size[i],
        remove_background=True
    )
    observation_data.append(image_class.img)
    beam_pa.append(image_class.beam_pa)
    beam_axis.append([image_class.beam_maj_au, image_class.beam_min_au])
    beam_area.append(pi/(4*np.log(2))*image_class.beam_maj_au*image_class.beam_min_au/\
                    (image_class.au_per_pix**2))
    disk_posang.append(disk_pa)
    
    size_au.append(image_class.img_size_au)
    au_per_pix.append(image_class.au_per_pix)
    npix.append(image_class.img.shape[0])

# model
def conti_model(theta):
    temp_dir_name = f'./temp/{datetime.now().strftime("%H%M%S")}_{datetime.now().microsecond}_{os.getpid()}'
    os.makedirs('./temp/', exist_ok=True)
    os.makedirs(temp_dir_name)
    os.chdir(temp_dir_name)
    
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

    # with open('camera_wavelength_micron.inp', 'w+') as f:
    #     f.write('%d\n'%(len(lam_list)))
    #     for value in lam_list:
    #         f.write('%13.6e\n'%(value*1e3))
    model_image_list = []

    for i, wav in enumerate(lam_list):
        os.system(f'radmc3d image npix {npix[i]} sizeau {size_au[i]} incl 73 {wav*1000} noline noscat')
        im = image.readImage()
        model_image = im.imageJyppix[:,:,0].T / (140**2)
        I = ndimage.rotate(model_image, -disk_posang[i]+beam_pa[i], reshape=False)
        sigmas = np.array([beam_axis[i][0], beam_axis[i][1]])/au_per_pix/(2*np.sqrt(2*np.log(2)))
        I = ndimage.gaussian_filter(I, sigma=sigmas)
        # rotate to align with image
        I = ndimage.rotate(I, -beam_pa[i], reshape=False)
        # convert to flux density in Jy/beam
        model_image = I*beam_area[i]
        model_image_list.append(model_image)


    # for i in range(im.image.shape[2]):
    #     model_image = im.imageJyppix[:,:,i].T / (140**2)
    #     I = ndimage.rotate(model_image, -disk_pa+beam_pa[i], reshape=False)
    #     sigmas = np.array([beam_axis[i][0], beam_axis[i][1]])/au_per_pix/(2*np.sqrt(2*np.log(2)))
    #     I = ndimage.gaussian_filter(I, sigma=sigmas)
    #     # rotate to align with image
    #     I = ndimage.rotate(I, -beam_pa[i], reshape=False)
    #     # convert to flux density in Jy/beam
    #     model_image = I*beam_area[i]
    #     model_image_list.append(model_image)
    
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
    if -3 < amax < 3 and 0.08 < Mstar < 0.3 and np.log10(1e-8) < Mdot < np.log10(1e-6):
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
