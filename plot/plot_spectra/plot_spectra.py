import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
import sys
sys.path.insert(0,'../../')
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup

###############################################################################
"""
CB68
Mass          : 0.08-0.30 Msun
Accretion rate: 4-7e-7    Msun/yr
Radius        : 20-40     au
Distance      : 140       pc
"""
###############################################################################
'''
Plot spectra
'''
def plot_spectra(incl=70, line=240, vkm=0, v_width=20, nlam=51,
                nodust=False, scat=True, extract_gas=False):
    if extract_gas is False:
        if nodust is True:
            prompt = ' noscat nodust'
        elif nodust is False:
            if scat is True:
                prompt = ' nphot_spec 100000'
            elif scat is False:
                prompt = ' noscat'

        os.system(f"radmc3d spectrum incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam}"+prompt)
        s = readSpectrum('spectrum.out')
        freq = (cc*1e-2) / (s[:, 0]*1e-6)
        v = cc / 1e5 * (freq[nlam//2] - freq) / freq[nlam//2]
        I = s[:,1]*1e26/(140*140) # mJy
        plt.plot(v, I)

        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Intensity (mJy/beam)')
        plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')  

    elif extract_gas is True:
        os.system(f"radmc3d spectrum incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam} nphot_spec 100000")
        os.system('mv spectrum.out spectrum_gas.out')
        s_gas = readSpectrum("spectrum_gas.out")
        freq = (cc*1e-2) / (s_gas[:, 0]*1e-6)
        v = cc / 1e5 * (freq[nlam//2] - freq) / freq[nlam//2]
        I_gas = s_gas[:, 1]*1e26/(140*140) # mJy

        os.system(f"radmc3d spectrum incl {incl} lambdarange {s_gas[0, 0]} {s_gas[-1, 0]} nlam {nlam} nphot_spec 100000 noline")
        os.system('mv spectrum.out spectrum_dust.out')
        s_dust = readSpectrum('spectrum_dust.out')
        I_dust = s_dust[:, 1]*1e26/(140*140) # mJy
        
        I_extracted_gas = I_gas-I_dust
        plt.plot(v, I_extracted_gas)

        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Intensity (mJy/beam)')
    return
###############################################################################
a_list = [1, 0.1, 0.01, 0.001]
incl_list = [0, 30, 60, 90]
for idx_mc, mcth in enumerate([True]):
    for idx_snow, snow in enumerate([True, False]):
        """
        Different a_max
        """    
        # nodust
        for idx_amax, amax in enumerate(a_list):

            problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
                      pancake=False, mctherm=mcth, snowline=snow, floor=True, kep=True, combine=True)
            
            plot_spectra(v_width=20, nlam=51, nodust=True, scat=False)

        label = [str(a*10)+' mm' for a in a_list]
        plt.legend(label)
        plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
        
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/different_amax/nodust_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/different_amax/nodust_snowline_{str(snow)}.png')
        plt.savefig(f'./figures/combine/different_amax/nodust_snowline_{str(snow)}.png')
        plt.close()
        
        # noscat
        # for idx_amax, amax in enumerate(a_list):

        #     problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
        #               pancake=False, mctherm=mcth, snowline=snow, floor=True, kep=True, combine=True)
            
        #     plot_spectra(v_width=20, nlam=51, nodust=False, scat=False)

        # label = [str(a*10)+' mm' for a in a_list]
        # plt.legend(label)
        # plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
        # # if mcth is True:
        # #     plt.savefig(f'./figures/mctherm/different_amax/noscat_snowline_{str(snow)}.png')
        # # elif mcth is False:
        # #     plt.savefig(f'./figures/x22/different_amax/noscat_snowline_{str(snow)}.png')
        # plt.savefig(f'./figures/combine/different_amax/noscat_snowline_{str(snow)}.png')
        # plt.close()

        # scat
        # for idx_amax, amax in enumerate(a_list):

        #     problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
        #               pancake=False, mctherm=mcth, snowline=snow, floor=True, kep=True, combine=True)
            
        #     plot_spectra(v_width=20, nlam=51, nodust=False, scat=True)

        # label = [str(a*10)+' mm' for a in a_list]
        # plt.legend(label)
        # plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
        # # if mcth is True:
        # #     plt.savefig(f'./figures/mctherm/different_amax/scat_snowline_{str(snow)}.png')
        # # elif mcth is False:
        # #     plt.savefig(f'./figures/x22/different_amax/scat_snowline_{str(snow)}.png')
        # plt.savefig(f'./figures/combine/different_amax/scat_snowline_{str(snow)}.png')
        # plt.close()

        # extracted_gas
        for idx_amax, amax in enumerate(a_list):

            problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
                      pancake=False, mctherm=mcth, snowline=snow, floor=True, kep=True, combine=True)
            
            plot_spectra(v_width=20, nlam=51, extract_gas=True)

        label = [str(a*10)+' mm' for a in a_list]
        plt.legend(label)
        plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/different_amax/extracted_gas_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/different_amax/extracted_gas_snowline_{str(snow)}.png')
        plt.savefig(f'./figures/combine/different_amax/extracted_gas_snowline_{str(snow)}.png')
        plt.close()



        """
        Different inclination
        """    
        problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
                      pancake=False, mctherm=mcth, snowline=snow, floor=True, kep=True, combine=True)
        # nodust
        for idx_incl, angle in enumerate(incl_list):
            plot_spectra(incl=angle, v_width=20, nlam=51, nodust=True, scat=False)
        
        label = [str(i)+r'$^\circ$' for i in incl_list]
        plt.legend(label)
        plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/different_incl/nodust_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/different_incl/nodust_snowline_{str(snow)}.png')
        plt.savefig(f'./figures/combine/different_incl/nodust_snowline_{str(snow)}.png')
        plt.close()

        # # noscat
        # for idx_incl, angle in enumerate(incl_list):
        #     plot_spectra(incl=angle, v_width=20, nlam=51, nodust=False, scat=False)
        
        # label = [str(i)+r'$^\circ$' for i in incl_list]
        # plt.legend(label)
        # plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/different_incl/noscat_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/different_incl/noscat_snowline_{str(snow)}.png')
        # plt.close()

        # # scat
        # for idx_incl, angle in enumerate(incl_list):
        #     plot_spectra(incl=angle, v_width=20, nlam=51, nodust=False, scat=True)
        
        # label = [str(i)+r'$^\circ$' for i in incl_list]
        # plt.legend(label)
        # plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/different_incl/scat_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/different_incl/scat_snowline_{str(snow)}.png')
        # plt.close()

        # extracted_gas
        for idx_incl, angle in enumerate(incl_list):
            plot_spectra(incl=angle, v_width=20, nlam=51, extract_gas=True)
        
        label = [str(i)+r'$^\circ$' for i in incl_list]
        plt.legend(label)
        plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/different_incl/extracted_gas_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/different_incl/extracted_gas_snowline_{str(snow)}.png')
        plt.savefig(f'./figures/combine/different_incl/extracted_gas_snowline_{str(snow)}.png')
        plt.close()

os.system('make cleanall')
