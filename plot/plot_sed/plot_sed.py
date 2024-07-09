import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from scipy.optimize import curve_fit
import sys
sys.path.insert(0,'../../')
from disk_model import *
from problem_setup import problem_setup


###############################################################################
"""
CB68
Mass          : 0.08-0.30 Msun
Accretion rate: 4-7e-7    Msun/yr
Radius        : 20-40     au
Distance      : 140       pc
observed_Freq = [233.8, 233.8, 233.8, 246.7, 246.7, 246.7, 246.7, 246.7]
observed_Flux = [   56,    55,    59,    62,    60,    60,    61,    66]
"""
###############################################################################
# Plotter
def plot_sed(incl=70, scat=True, plot_nu=True, GHz=True, mjy=True, color='b', label=''):
    if scat is True:
        prompt = ' nphot_spec 100000 '
    elif scat is False:
        prompt = ' noscat '
    freq = np.array([5e1, 1e3]) # GHz
    wav = ((cc*1e-2)/(freq*1e9))*1e+6
    os.system(f"radmc3d spectrum incl {incl} lambdarange {wav[-1]} {wav[0]} nlam 200"+prompt+"noline")
    s = readSpectrum('spectrum.out')
    lam = s[:, 0]
    fnu = s[:, 1]/(140**2)
    if mjy is True:
        fnu = 1e26*fnu
        plt.ylabel('Flux Density [mJy]', fontsize=16)
        plt.ylim((1e-2, 1e4))
    else:
        fnu = 1e23*fnu
        plt.ylabel('Flux Density [Jy]', fontsize=16)
        plt.ylim((1e-5, 1e1))

    if plot_nu is True:
        nu = (1e-2*cc)/(1e-6*lam)
        if GHz is True:
            nu = 1e-9*nu
            plt.xlabel('$\\nu$'+'[GHz]', fontsize=16)
        else:
            plt.xlabel('$\\nu$'+'[GHz]', fontsize=16)
        if scat is True:
            fig = plt.plot(nu, fnu, color=color, label=label, linestyle=':')
        elif scat is False:
            fig = plt.plot(nu, fnu, color=color, label=label)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim((1e2, 1e3))
    else:
        fig = plt.plot(lam, fnu)
        plt.xlabel('$\\lambda$'+'[mm]', fontsize=16)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim((1e2, 1e6))
    return 
###############################################################################

color_list = ['b', 'g', 'r']
heat_list = ['Accretion', 'Irradiation', 'Combine']
for idx, (heat, color) in enumerate(zip(heat_list, color_list)):
    for idx_scat, scat in enumerate([True, False]):

        if heat == 'Accretion':
            problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=False, snowline=False, floor=True, kep=True)

        elif heat == 'Irradiation':
            problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=True, snowline=False, floor=True, kep=True, combine=False)
        elif heat == 'Combine':
            problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=True, snowline=False, floor=True, kep=True, combine=True)
        if scat is True:
            l = heat + ', scat'
        elif scat is False:
            l = heat + ', noscat'
        plot_sed(scat=scat, label=l, color=color, incl=70)
        observed_Freq = [233.8, 233.8, 233.8, 246.7, 246.7, 246.7, 246.7, 246.7]
        observed_Flux = [   56,    55,    59,    62,    60,    60,    61,    66]
        plt.scatter(observed_Freq, observed_Flux, color='black')
plt.legend()
plt.savefig('./figures/compare_heating.pdf', transparent=True)
plt.close()
os.system('make cleanall')


# color_list = ['b', 'g', 'r']
# a_list = [10, 0.1, 0.001]
# for idx, (a, color) in enumerate(zip(a_list, color_list)):
    
#     problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#                       pancake=False, mctherm=False, snowline=False, floor=True, kep=True, combine=False)

#     plot_sed(scat=False, label=f"{a}cm", color=color, incl=70)
#     observed_Freq = [233.8, 233.8, 233.8, 246.7, 246.7, 246.7, 246.7, 246.7]
#     observed_Flux = [   56,    55,    59,    62,    60,    60,    61,    66]
#     plt.scatter(observed_Freq, observed_Flux, color='black')
# plt.legend()
# plt.savefig('./figures/compare_amax.pdf', transparent=True)
# plt.close()
# os.system('make cleanall')