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
                        incl_dust=4,
                        incl_lines=1,
                        nphot=500000,
                        nphot_scat=1000000,
                        nphot_spec=500000, 
                        scattering_mode_max=2,
                        istar_sphere=1,
                        num_cpu=20)
model.get_linecontrol(filename=None,
                      methanol='ch3oh leiden 0 0 0')
model.get_continuumlambda(filename=None,
                          comment=None,
                          lambda_micron=None,
                          append=False)

for a in [10, 1, 0.1, 0.01]:
    for snow in [None, 100]:
        for vin in [None, 0.25, 0.5]:
            model.get_diskcontrol(  d_to_g_ratio = 0.01,
                                    a_max=a, 
                                    Mass_of_star=0.14, 
                                    Accretion_rate=5e-7,
                                    Radius_of_disk=50,
                                    NR=200,
                                    NTheta=200,
                                    NPhi=10)
            model.get_vfieldcontrol(Kep=True,
                                    vinfall=vin,
                                    Rcb=None,
                                    outflow=None)
            model.get_heatcontrol(heat='accretion')
            model.get_gasdensitycontrol(abundance=1e-10,
                                        snowline=snow,
                                        enhancement=1e5,
                                        gas_inside_rcb=True)

            #############################################

            condition_parms = general_parameters(
                nodust      = False,
                scat        = True,
                extract_gas = True,
            )

            channel_cube_parms = general_parameters(
                incl=70, line=240,
                v_width=10, nlam=11, vkms=0,
                npix=500, sizeau=200,
                dir='./test/a_vinfall_snowline/', fname=f'a_{a}_vinfall_{vin}_snowline_{snow}',
                read_cube=True
            )

            pv_cube_parms = general_parameters(
                incl=70, line=240,
                v_width=10, nlam=51, vkms=0,
                npix=500, sizeau=200,
                dir=f'./test/a_vinfall_snowline/', fname=f'a_{a}_vinfall_{vin}_snowline_{snow}',
                read_cube=True
            )

            sed_parms = general_parameters(
                incl = 70, scat=True,
                freq_min=5e1, freq_max=5e2, nlam=10,
                dir=f'./test/a_vinfall_snowline/', fname=f'a_{a}_vinfall_{vin}_snowline_{snow}',
                read_sed=True
            )

            spectrum_parms = general_parameters(
                incl=70, line=240,
                v_width=5, nlam=21, vkms=0,
                dir=f'./test/a_vinfall_snowline/', fname=f'a_{a}_vinfall_{vin}_snowline_{snow}',
                read_spectrum=True
            )

            conti_parms = general_parameters(
                incl=70, wav=1300,
                npix=500, sizeau=200,
                scat=True,
                dir=f'./test/a_vinfall_snowline/', fname=f'a_{a}_vinfall_{vin}_snowline_{snow}',
                read_conti=True
            )

            simulation_parms = general_parameters(
                condition_parms    = condition_parms,
                channel_cube_parms = channel_cube_parms,
                pv_cube_parms      = pv_cube_parms,
                conti_parms        = conti_parms,
                sed_parms          = sed_parms,
                spectrum_parms     = spectrum_parms,
                save_out=True,
                save_npz=True,
            )

            simulation = generate_simulation(
                parms=simulation_parms,
                channel       = True,
                pv            = True,
                conti         = False,
                sed           = False,
                line_spectrum = False
            )


            ##############################################
            plot_profile_parms = general_parameters(
                dir=f'./test/a_{a}_vinfall_{vin}_snowline_{snow}/', fname='profile'
            )

            plot_channel_parms = general_parameters(
                cube_dir = './test/a_vinfall_snowline/npzfile/',
                cube_fname = [f'channel_a_{a}_vinfall_{vin}_snowline_{snow}_scat.npz',
                                f'channel_a_{a}_vinfall_{vin}_snowline_{snow}_conti.npz'],
                extracted = True,
                conti = True,
                convolve = True,
                vkms = 5,
                fwhm = 60,
                d = 140,
                title = f'amax {a} vinfall {vin} snowline {snow}',
                dir = f'./test/a_{a}_vinfall_{vin}_snowline_{snow}/',
                fname = 'channel'
            )

            plot_pv_parms = general_parameters(
                cube_dir = './test/a_vinfall_snowline/npzfile/',
                cube_fname = [f'pv_a_{a}_vinfall_{vin}_snowline_{snow}_scat.npz',
                                f'pv_a_{a}_vinfall_{vin}_snowline_{snow}_conti.npz'],
                extracted = True,
                conti = True,
                convolve = True,
                vkms = 5,
                fwhm = 60,
                d = 140,
                CB68 = False,
                title = f'amax {a} vinfall {vin} snowline {snow}',
                dir = f'./test/a_{a}_vinfall_{vin}_snowline_{snow}/',
                fname = 'pv'
            )

            plot_sed_parms = general_parameters(
                sed_dir = './test/a_vinfall_snowline/npzfile/',
                sed_fname = f'sed_a_{a}_vinfall_{vin}_snowline_{snow}_scat.npz',
                d = 140,
                CB68 = True,
                title = f'amax {a} vinfall {vin} snowline {snow}',
                dir = f'./test/a_{a}_vinfall_{vin}_snowline_{snow}/',
                fname = 'sed'
            )
            
            plot_spectrum_parms = general_parameters(
                spectra_dir = './test/a_vinfall_snowline/npzfile/',
                spectra_fname = [f'spectrum_a_{a}_vinfall_{vin}_snowline_{snow}_scat.npz',
                                f'spectrum_a_{a}_vinfall_{vin}_snowline_{snow}_conti.npz'],
                extracted = True,
                vkms = 5,
                d = 140,
                title = f'amax {a} vinfall {vin} snowline {snow}',
                dir = f'./test/a_{a}_vinfall_{vin}_snowline_{snow}/',
                fname = 'spectrum'
            )

            plot_continuum_parms = general_parameters(
                conti_dir = './test/a_vinfall_snowline/npzfile/',
                conti_fname = f'conti_a_{a}_vinfall_{vin}_snowline_{snow}_scat.npz',
                convolve = True,
                fwhm = 60,
                d = 140,
                title = f'amax {a} vinfall {vin} snowline {snow}',
                dir = f'./test/a_{a}_vinfall_{vin}_snowline_{snow}/',
                fname = 'conti'
            )

            plot_parms = general_parameters(
                profile_parms   = plot_profile_parms,
                channel_parms   = plot_channel_parms,
                pv_parms        = plot_pv_parms,
                sed_parms       = plot_sed_parms,
                spectra_parms   = plot_spectrum_parms,
                continuum_parms = plot_continuum_parms
            )

            plot = generate_plot(
                parms     = plot_parms, 
                profile   = False,
                channel   = True,
                pv        = True,
                continuum = False,
                sed       = False,
                spectrum  = False,
            )