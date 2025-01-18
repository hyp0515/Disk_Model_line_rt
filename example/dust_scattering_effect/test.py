import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../../')
from radmc.setup import radmc3d_setup
from radmc.simulate import generate_simulation
from radmc.plot import generate_plot

# class general_parameters:
#     '''
#     A class to store the parameters for individual kinds of grids.
#     Details of individual parameters should refer to the functions that generate the grids.
#     '''
#     def __init__(self, **kwargs
#                  ):
#         for k, v in kwargs.items():
#           # add parameters as attributes of this object
#           setattr(self, k, v)

#     def __del__(self):
#       pass

#     def add_attributes(self, **kwargs):
#       '''
#       Use this function to set the values of the attributs n1, n2, n3,
#       which are number of pixels in the first, second, and third axes. 
#       '''
#       for k, v in kwargs.items():
#         # add parameters as attributes of this object
#         setattr(self, k, v)


# model = radmc3d_setup(silent=False)
# model.get_mastercontrol(filename=None,
#                         comment=None,
#                         incl_dust=1,
#                         incl_lines=1,
#                         nphot=100000,
#                         nphot_scat=100000,
#                         scattering_mode_max=2,
#                         istar_sphere=1,
#                         num_cpu=None)
# model.get_linecontrol(filename=None,
#                       methanol='ch3oh leiden 0 0 0')
# model.get_continuumlambda(filename=None,
#                           comment=None,
#                           lambda_micron=None,
#                           append=False)


# model.get_diskcontrol(  d_to_g_ratio = 0.01,
#                         a_max=0.1, 
#                         Mass_of_star=0.14, 
#                         Accretion_rate=5e-7,
#                         Radius_of_disk=50,
#                         NR=200,
#                         NTheta=200,
#                         NPhi=10)
# model.get_vfieldcontrol(Kep=True,
#                         vinfall=0.5,
#                         Rcb=None,
#                         outflow=None)
# model.get_heatcontrol(heat='accretion')
# model.get_gasdensitycontrol(abundance=1e-10,
#                             snowline=100,
#                             enhancement=1e5,
#                             gas_inside_rcb=True)

# ##############################################

# condition_parms = general_parameters(
#     nodust      = False,
#     scat        = True,
#     extract_gas = True,
# )

# channel_cube_parms = general_parameters(
#     incl=70, line=240,
#     v_width=10, nlam=7, vkms=0,
#     npix=500, sizeau=200,
#     dir='./test/', fname=f'channel_test',
#     read_cube=True
# )

# pv_cube_parms = general_parameters(
#     incl=70, line=240,
#     v_width=10, nlam=50, vkms=0,
#     npix=500, sizeau=200,
#     dir=f'./test/', fname=f'test',
#     read_cube=True
# )

# sed_parms = general_parameters(
#     incl = 70, scat=True,
#     freq_min=5e1, freq_max=5e2, nlam=10,
#     dir=f'./test/', fname=f'test',
#     read_sed=True
# )

# spectrum_parms = general_parameters(
#     incl=70, line=240,
#     v_width=10, nlam=10, vkms=0,
#     dir=f'./test/', fname=f'test',
#     read_spectrum=True
# )
# conti_parms = general_parameters(
#     incl=70, wav=1300,
#     npix=500, sizeau=200,
#     scat=True,
#     dir=f'./test/', fname=f'test',
#     read_conti=True
# )

# simulation_parms = general_parameters(
#     condition_parms    = condition_parms,
#     channel_cube_parms = channel_cube_parms,
#     pv_cube_parms      = pv_cube_parms,
#     conti_parms        = conti_parms,
#     sed_parms          = sed_parms,
#     spectrum_parms     = spectrum_parms,
#     save_out=True,
#     save_npz=True,
# )

# simulation = generate_simulation(
#     parms=simulation_parms,
#     channel       = False,
#     pv            = False,
#     conti         = False,
#     sed           = False,
#     line_spectrum = False
# )


# ##############################################
# plot_profile_parms = general_parameters(
#     dir=f'./test/', fname='profile'
# )

# plot_channel_parms = general_parameters(
#     cube_dir = './test/npzfile/',
#     cube_fname = [f'channel_channel_test_scat.npz',
#                     f'channel_channel_test_conti.npz'],
#     extracted = True,
#     conti = True,
#     convolve = True,
#     vkms = 5,
#     fwhm = 50,
#     d = 140,
#     title = f'test',
#     dir = f'./test/test/',
#     fname = 'channel'
# )

# plot_pv_parms = general_parameters(
#     cube_dir = './test/npzfile/',
#     cube_fname = [f'pv_test_scat.npz',
#                     f'pv_test_conti.npz'],
#     extracted = True,
#     conti = True,
#     convolve = True,
#     vkms = 5,
#     fwhm = 60,
#     d = 140,
#     CB68 = True,
#     title = f'test',
#     dir = f'./test/',
#     fname = 'pv'
# )

# plot_sed_parms = general_parameters(
#     sed_dir = './test/npzfile/',
#     sed_fname = f'sed_test_scat.npz',
#     d = 140,
#     CB68 = True,
#     title = f'test',
#     dir = f'./test/',
#     fname = 'sed'
# )

# plot_spectrum_parms = general_parameters(
#     spectra_dir = './test/npzfile/',
#     spectra_fname = [f'spectrum_test_scat.npz',
#                     f'spectrum_test_conti.npz'],
#     extracted = True,
#     vkms = 5,
#     d = 140,
#     title = f'test',
#     dir = f'./test/',
#     fname = 'spectrum'
# )

# plot_continuum_parms = general_parameters(
#     conti_dir = './test/npzfile/',
#     conti_fname = f'conti_test_scat.npz',
#     convolve = True,
#     fwhm = 60,
#     d = 140,
#     title = f'test',
#     dir = f'./test/',
#     fname = 'conti'
# )


# plot_parms = general_parameters(
#     profile_parms   = plot_profile_parms,
#     channel_parms   = plot_channel_parms,
#     pv_parms        = plot_pv_parms,
#     sed_parms       = plot_sed_parms,
#     spectra_parms   = plot_spectrum_parms,
#     continuum_parms = plot_continuum_parms
# )

# plot = generate_plot(
#     parms     = plot_parms, 
#     profile   = False,
#     channel   = True,
#     pv        = False,
#     continuum = False,
#     sed       = False,
#     spectrum  = False,
# )



cube = np.load('./test/npzfile/channel_channel_test_extracted.npz')
print(cube['freq'])

nu0 = 218.440063 * 1e9
c = 299792.458 # km/s

v = (cube['freq'] - nu0) / nu0 * c + 5

image_cube = cube['imageJyppix'] * 1e3
# def convolve(image, fwhm):
#             convolved_image = np.zeros(shape=image.shape)
#             sigma = fwhm * (npix/sizeau)/ (2*np.sqrt(2*np.log(2)))
#             convolved_image[:, :, 0] = gaussian_filter(image[:, :, 0], sigma=sigma)
#             return convolved_image*(np.pi/4*((fwhm*npix/sizeau)**2))


intensity_sum = np.sum(image_cube, axis=2)
weighted_velocity_sum = np.empty(image_cube.shape)
for i in range(image_cube.shape[2]):
    weighted_velocity_sum[:, :, i] = image_cube[:, :, i] * v[i]
weighted_velocity_sum = np.sum(weighted_velocity_sum, axis=2)

# Avoid division by zero
moment1 = np.zeros_like(intensity_sum)
nonzero_mask = intensity_sum > 0
moment1[nonzero_mask] = weighted_velocity_sum[nonzero_mask] / intensity_sum[nonzero_mask]

plt.figure(figsize=(8, 6))
plt.imshow(moment1, origin='lower', cmap='RdBu_r', extent=[0, image_cube.shape[1], 0, image_cube.shape[0]],
           vmin=-10, vmax=10)
plt.colorbar(label='Velocity (km/s)')
plt.title('Moment 1 Map')
plt.xlabel('Spatial X')
plt.ylabel('Spatial Y')
plt.show()
