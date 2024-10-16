import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../../')
from radmc.setup import radmc3d_setup
from radmc.simulate import generate_simulation
from radmc.plot import generate_plot

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


model = radmc3d_setup(silent=False)
model.get_mastercontrol(filename=None,
                        comment=None,
                        incl_dust=1,
                        incl_lines=1,
                        nphot=1000000,
                        nphot_scat=1000000,
                        scattering_mode_max=2,
                        istar_sphere=1,
                        num_cpu=None)
model.get_linecontrol(filename=None,
                      methanol='ch3oh leiden 0 0 0')
model.get_continuumlambda(filename=None,
                          comment=None,
                          lambda_micron=None,
                          append=False)
model.get_diskcontrol(  d_to_g_ratio = 0.01,
                        a_max=1, 
                        Mass_of_star=0.14, 
                        Accretion_rate=5e-7,
                        Radius_of_disk=30,
                        NR=200,
                        NTheta=200,
                        NPhi=10,
                        disk_boundary=1e-18)
model.get_vfieldcontrol(Kep=True,
                        vinfall=0.5,
                        Rcb=5,
                        outflow=0.5)
model.get_heatcontrol(heat='accretion')
model.get_gasdensitycontrol(abundance=1e-10,
                            snowline=None,
                            enhancement=1e5,
                            gas_inside_rcb=True)

##############################################
condition_parms = general_parameters(
    nodust      = False,
    scat        = True,
    extract_gas = True,
)

channel_cube_parms = general_parameters(
    incl=70, line=240,
    v_width=10, nlam=11, vkms=0,
    npix=200, sizeau=100,
    dir='./test/', fname='test',
    read_cube=True
)

pv_cube_parms = general_parameters(
    incl=70, line=240,
    v_width=10, nlam=50, vkms=0,
    npix=200, sizeau=100,
    dir='./test/', fname='test',
    read_cube=True
)

sed_parms = general_parameters(
    dir='./', fname='test',
    read_sed=True
)

spectrum_parms = general_parameters(
    dir='./', fname='test',
    read_spectrum=True
)

conti_parms = general_parameters(
    incl=70, wav=1300,
    npix=200, sizeau=100,
    scat=True,
    dir='./test/', fname='test',
    read_conti=True
)

simulation_parms = general_parameters(
    condition_parms    = condition_parms,
    channel_cube_parms = channel_cube_parms,
    pv_cube_parms      = pv_cube_parms,
    conti_parms        = conti_parms,
    sed_parms          = sed_parms,
    spectrum_parms     = spectrum_parms,
    save_out=True, save_npz=True,
)

simulation = generate_simulation(
    parms=simulation_parms,
    channel       = False,
    pv            = False,
    conti         = True,
    sed           = False,
    line_spectrum = False
)

##############################################
profile_parms = general_parameters(
    dir='./test/', fname='test_profile'
)

channel_parms = general_parameters(
    cube_dir = './test/npzfile/',
    cube_fname = ['channel_test_scat.npz', 'channel_test_conti.npz'],
    extracted = True,
    conti = True,
    convolve = True,
    vkms = 5,
    fwhm = 50,
    d = 140,
    title = 'test',
    dir = './test/',
    fname = 'test_channel'
)

pv_parms = general_parameters(
    cube_dir = './test/npzfile/',
    cube_fname = ['pv_test_scat.npz', 'pv_test_conti.npz'],
    extracted = True,
    conti = True,
    convolve = True,
    vkms = 5,
    fwhm = 50,
    d = 140,
    title = 'test',
    dir = './test/',
    fname = 'test_pv'
)

spectra_parms = general_parameters(
    spectra_dir = './test/npzfile/',
    spectra_fname = 'test_extracted.npz',
    extracted = True,
    conti = True,
    convolve = True,
    vkms = 5,
    fwhm = 50,
    d = 140,
    title = 'test',
    dir = './test/',
    fname = 'test_spectrum'
)

continuum_parms = general_parameters(
    conti_dir = './test/npzfile/',
    conti_fname = 'conti_test_scat.npz',
    convolve = True,
    fwhm = 50,
    d = 140,
    title = 'test',
    dir = './test/',
    fname = 'test_conti'
)


plot_parms = general_parameters(
    profile_parms   = profile_parms,
    channel_parms   = channel_parms,
    pv_parms        = pv_parms,
    spectra_parms   = spectra_parms,
    continuum_parms = continuum_parms
)

plot = generate_plot(
    parms     = plot_parms, 
    profile   = False,
    channel   = False,
    pv        = False,
    continuum = True,
    sed       = False,
    spectrum  = False,
)
