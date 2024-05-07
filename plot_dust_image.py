import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
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

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
#                   pancake=False,v_infall=0.5)
# os.system(f"radmc3d image npix 200 incl 70 lambda 1300 sizeau 50 nphot_scat 10000000 noline")
# os.system('mv image.out image_dust.out')
# im_dust = readImage('image_dust.out')
# data = np.transpose(im_dust.imageJyppix[:, ::-1, 0])
# plot = plt.imshow(data, cmap='hot',extent=[-50, 50, -50, 50])
# cbar = plt.colorbar(plot)
# plt.show()
incl = [0, 15, 30, 45, 60, 75, 90]
vkms = np.linspace(-5, 5, 11, endpoint=True)
problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                    pancake=False,v_infall=1)
for icl in incl:
    for v in vkms:

        os.system(f"radmc3d image npix 200 incl {icl} sizeau 60 iline 240 vkms {v} nphot_scat 10000000")
        os.system('mv image.out image_gas.out')
        im_gas = readImage('image_gas.out')
        # print(str(im_gas.wav[0]))
        # problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
        #                   pancake=False,v_infall=0.5)
        os.system(f"radmc3d image npix 200 incl {icl} lambda {im_gas.wav[0]} sizeau 60 nphot_scat 10000000 noline")
        os.system('mv image.out image_dust.out')
        im_dust = readImage('image_dust.out')

        data_gas  = np.transpose(im_gas.imageJyppix[:, ::-1, 0])
        data_dust = np.transpose(im_dust.imageJyppix[:, ::-1, 0])

        vmin = min(np.min(data_dust), np.min(data_gas))
        vmax = max(np.max(data_dust), np.max(data_gas))

        fig, ax = plt.subplots(1, 3, figsize=(15, 12), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
        im1 = ax[0].imshow(data_gas, cmap='hot', vmin=vmin, vmax=vmax, extent=[-30, 30, -30, 30])
        ax[0].set_title('Dust+Scattering+Gas')
        ax[0].set_ylabel('R[au]')
        ax[0].set_xlabel('R[au]')
        im2 = ax[1].imshow(data_dust, cmap='hot', vmin=vmin, vmax=vmax, extent=[-30, 30, -30, 30])
        ax[1].set_title('Dust+Scattering')
        ax[1].set_xlabel('R[au]')
        cbar_1 = fig.colorbar(im1, ax=ax[0:2], orientation='horizontal', shrink = 0.8)

        im3 = ax[2].imshow(data_gas-data_dust, cmap='hot', extent=[-30, 30, -30, 30])
        ax[2].set_title('Gas')
        ax[2].set_xlabel('R[au]')
        cbar_2 = fig.colorbar(im3, ax=ax[2], orientation='horizontal', shrink = .8)
        plt.savefig(f'./Figures/extracted_gas/incl_{icl}_vkms_{v}.png')
# plt.show()