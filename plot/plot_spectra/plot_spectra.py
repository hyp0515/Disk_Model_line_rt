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
def plot_spectra(incl=70, line=240, vkm=0, v_width=20, nlam=50,
                nodust=False, scat=True, extract_gas=False, color='b',a=1):
    if extract_gas is False:
        if nodust is True:
            prompt = ' noscat nodust'
        elif nodust is False:
            if scat is True:
                prompt = ' nphot_spec 100000'
            elif scat is False:
                prompt = ' noscat'

        os.system(f"radmc3d spectrum incl {incl} iline {line} vkms 0 widthkms {v_width} linenlam {nlam}"+prompt)
        s = readSpectrum('spectrum.out')
        freq = (cc*1e-2) / (s[:, 0]*1e-6)
        freq0 = (freq[nlam//2] + freq[(nlam//2)-1])/2
        v = cc / 1e5 * (freq0 - freq) / freq0
        I = s[:,1]*1e26/(140*140) # mJy
        plt.plot(v+vkm, I, ':', color=color, label='w/o dust' +f'{a}cm')

        plt.xlabel('Velocity (km/s)',fontsize = 16)
        plt.ylabel('Intensity (mJy/beam)',fontsize = 16)
        plt.title('Spectra of $\mathregular{CH_3OH}$')  

    elif extract_gas is True:
        os.system(f"radmc3d spectrum incl {incl} iline {line} vkms 0 widthkms {v_width} linenlam {nlam} nphot_spec 100000")
        os.system('mv spectrum.out spectrum_gas.out')
        s_gas = readSpectrum("spectrum_gas.out")
        freq = (cc*1e-2) / (s_gas[:, 0]*1e-6)
        freq0 = (freq[nlam//2] + freq[(nlam//2)-1])/2
        v = cc / 1e5 * (freq0 - freq) / freq0
        I_gas = s_gas[:, 1]*1e26/(140*140) # mJy

        os.system(f"radmc3d spectrum incl {incl} lambdarange {s_gas[0, 0]} {s_gas[-1, 0]} nlam {nlam} nphot_spec 100000 noline")
        os.system('mv spectrum.out spectrum_dust.out')
        s_dust = readSpectrum('spectrum_dust.out')
        I_dust = s_dust[:, 1]*1e26/(140*140) # mJy
        
        I_extracted_gas = I_gas-I_dust
        plt.plot(v+vkm, I_extracted_gas, color=color, label='w/ dust, '+f'{a}cm')

        plt.xlabel('Velocity (km/s)',fontsize = 16)
        plt.ylabel('Intensity (mJy/beam)',fontsize = 16)
    return
###############################################################################
color_list = ['b', 'g', 'r', 'c']
a_list = [10, 1, 0.1, 0.01]

for idx_a, (a, color) in enumerate(zip(a_list, color_list)):

    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=None)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, nodust=True, color=color,a=a)
    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=None)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, extract_gas=True, color=color, a=a)
plt.legend()
plt.title('Dust Effect (irradiation + nosnowline)',fontsize = 16)
plt.savefig('./figures/mctherm_nosnowline')
os.system('make cleanall')
plt.close()

for idx_a, (a, color) in enumerate(zip(a_list, color_list)):

    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=False, snowline=False, floor=True, kep=True, Rcb=None)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, nodust=True, color=color,a=a)
    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=False, snowline=False, floor=True, kep=True, Rcb=None)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, extract_gas=True, color=color, a=a)
plt.legend()
plt.title('Dust Effect (accretion + nosnowline)',fontsize = 16)
plt.savefig('./figures/x22_nosnowline')
os.system('make cleanall')
plt.close()

for idx_a, (a, color) in enumerate(zip(a_list, color_list)):

    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=None)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, nodust=True, color=color,a=a)
    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=None)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, extract_gas=True, color=color, a=a)
plt.legend()
plt.title('Dust Effect (irradiation + snowline)',fontsize = 16)
plt.savefig('./figures/mctherm')
os.system('make cleanall')
plt.close()

for idx_a, (a, color) in enumerate(zip(a_list, color_list)):

    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=False, snowline=True, floor=True, kep=True, Rcb=None)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, nodust=True, color=color,a=a)
    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=False, snowline=True, floor=True, kep=True, Rcb=None)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, extract_gas=True, color=color, a=a)
plt.legend()
plt.title('Dust Effect (accretion + snowline)',fontsize = 16)
plt.savefig('./figures/x22')
os.system('make cleanall')
plt.close()
###############################################################################
for idx_a, (a, color) in enumerate(zip(a_list, color_list)):

    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=None, abundance_enhancement=1e-7)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, nodust=True, color=color,a=a)
    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=None, abundance_enhancement=1e-7)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, extract_gas=True, color=color, a=a)
plt.legend()
plt.title('Dust Effect (irradiation + nosnowline)',fontsize = 16)
plt.savefig('./figures/mctherm_nosnowline_abundance_7')
os.system('make cleanall')
plt.close()

for idx_a, (a, color) in enumerate(zip(a_list, color_list)):

    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=False, snowline=False, floor=True, kep=True, Rcb=None, abundance_enhancement=1e-7)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, nodust=True, color=color,a=a)
    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=False, snowline=False, floor=True, kep=True, Rcb=None, abundance_enhancement=1e-7)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, extract_gas=True, color=color, a=a)
plt.legend()
plt.title('Dust Effect (accretion + nosnowline)',fontsize = 16)
plt.savefig('./figures/x22_nosnowline_abundance_7')
os.system('make cleanall')
plt.close()

for idx_a, (a, color) in enumerate(zip(a_list, color_list)):

    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=None, abundance_enhancement=1e-7)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, nodust=True, color=color,a=a)
    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=None, abundance_enhancement=1e-7)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, extract_gas=True, color=color, a=a)
plt.legend()
plt.title('Dust Effect (irradiation + snowline)',fontsize = 16)
plt.savefig('./figures/mctherm_abundance_7')
os.system('make cleanall')
plt.close()

for idx_a, (a, color) in enumerate(zip(a_list, color_list)):

    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=False, snowline=True, floor=True, kep=True, Rcb=None, abundance_enhancement=1e-7)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, nodust=True, color=color,a=a)
    problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                pancake=False, mctherm=False, snowline=True, floor=True, kep=True, Rcb=None, abundance_enhancement=1e-7)
    plot_spectra(incl=70, vkm=5, v_width=5, nlam=40, extract_gas=True, color=color, a=a)
plt.legend()
plt.title('Dust Effect (accretion + nosnowline)',fontsize = 16)
plt.savefig('./figures/x22_abundance_7')
os.system('make cleanall')
plt.close()