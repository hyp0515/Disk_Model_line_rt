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
"""
Plot Position-velocity (PV) diagrams
"""
def plot_pv(incl=70, line=240, vkm=0, v_width=20, nlam=51,
            nodust=False, scat=True, extract_gas=False, npix = 30):
    if extract_gas is False:
        if nodust is True:
            prompt = ' noscat nodust'
        elif nodust is False:
            if scat is True:
                prompt = ' nphot_scat 1000000'
            elif scat is False:
                prompt = ' noscat'
        
        os.system(f"radmc3d image npix {npix} sizeau 60 incl {incl} iline {line} vkms 0 widthkms {v_width} linenlam {nlam}"+prompt)
        im = readImage('image.out')

        freq0 = im.freq[nlam//2]
        v = cc / 1e5 * (freq0 - im.freq) / freq0
        center = int(len(im.y)//2)

        fig, ax = plt.subplots()
        c = ax.pcolormesh(im.x/au, v+vkm, (im.imageJyppix[:, center, :].T)*1e3/(140**2), shading="nearest", rasterized=True, cmap='jet')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('mJy/pixel')
        ax.set_xlabel("Offset [au]")
        ax.set_ylabel("Velocity [km/s]")
        ax.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax.plot([-30, 30], [vkm, vkm], 'w:')

    elif extract_gas is True:
        os.system(f"radmc3d image npix 30 sizeau 60 incl {incl} iline {line} vkms 0 widthkms {v_width} linenlam {nlam} nphot_scat 1000000")
        os.system('mv image.out image_gas.out')
        im_gas = readImage('image_gas.out')
        os.system(f"radmc3d image npix 30 sizeau 60 incl {incl} lambdarange {im_gas.wav[0]} {im_gas.wav[-1]} nlam {nlam} nphot_scat 1000000 noline")
        os.system('mv image.out image_dust.out')
        im_dust = readImage('image_dust.out')

        freq0 = im_gas.freq[nlam//2]
        v = cc / 1e5 * (freq0 - im_gas.freq) / freq0
        center = int(len(im_gas.y)//2)

        data_gas  = im_gas.imageJyppix
        data_dust = im_dust.imageJyppix
        extracted_gas = data_gas-data_dust

        fig, ax = plt.subplots()
        c = ax.pcolormesh(im_gas.x/au, v+vkm, (extracted_gas[:, center, :].T)*1e3/(140**2), shading="nearest", rasterized=True, cmap='jet')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('mJy/pixel')
        ax.set_xlabel("Offset [au]")
        ax.set_ylabel("Velocity [km/s]")
        ax.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax.plot([-30, 30], [vkm, vkm], 'w:')
    return

###############################################################################
# for idx_vin, vin in enumerate([1, 0]):

#     for idx_mc, mcth in enumerate([True, False]):
#         for idx_snow, snow in enumerate([True, False]):
#             """
#             Accretion/Mstar = 1e-5/yr
#             """
#             problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=vin, 
#                         pancake=False, mctherm=mcth, snowline=snow, floor=True, kep=True)
            
#             # extracted_gas
#             plot_pv(vkm=5, v_width=20, nlam=51, extract_gas=True)
#             if mcth is True:
#                 plt.savefig(f'./figures/mctherm/mock_CB68_snowline_{str(snow)}_mdot_5_vinfall_{str(vin)}_20.png')
#             elif mcth is False:
#                 plt.savefig(f'./figures/x22/mock_CB68_snowline_{str(snow)}_mdot_5_vinfall_{str(vin)}_20.png')
#             plt.close()

#             """
#             Accretion/Mstar = 1e-7/yr
#             """

#             problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-7*Msun/yr, Radius_of_disk=30*au, v_infall=vin, 
#                         pancake=False, mctherm=mcth, snowline=snow, floor=True, kep=True)

#             # extracted_gas
#             plot_pv(vkm=5, v_width=20, nlam=51, extract_gas=True)
#             if mcth is True:
#                 plt.savefig(f'./figures/mctherm/mock_CB68_snowline_{str(snow)}_mdot_7_vinfall_{str(vin)}_20.png')
#             elif mcth is False:
#                 plt.savefig(f'./figures/x22/mock_CB68_snowline_{str(snow)}_mdot_7_vinfall_{str(vin)}_20.png')
#             plt.close()

problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
                        pancake=False, mctherm=True, snowline=True, floor=True, kep=True)
plot_pv(vkm=5, v_width=10, nlam=21, extract_gas=True, npix=20)
plt.savefig(f'./figures/mock/mock_CB68_mctherm_nlam_21_npix_20_width_10.png')
plt.close()

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
#                         pancake=False, mctherm=False, snowline=True, floor=True, kep=True)
# plot_pv(vkm=5, v_width=5, nlam=21, extract_gas=True, npix=20)
# plt.savefig(f'./figures/mock/mock_CB68_x22_nlam_21_npix_20_width_5.png')
# plt.close()

problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
                        pancake=False, mctherm=True, snowline=True, floor=True, kep=True)
plot_pv(vkm=5, v_width=10, nlam=41, extract_gas=True, npix=40)
plt.savefig(f'./figures/mock/mock_CB68_mctherm_nlam_41_npix_40_width_10.png')
plt.close()

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
#                         pancake=False, mctherm=False, snowline=True, floor=True, kep=True)
# plot_pv(vkm=5, v_width=5, nlam=41, extract_gas=True, npix=40)
# plt.savefig(f'./figures/mock/mock_CB68_x22_nlam_41_npix_40_width_5.png')
plt.close()


os.system('make cleanall')