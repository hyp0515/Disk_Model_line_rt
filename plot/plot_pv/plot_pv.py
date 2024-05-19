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
            nodust=False, scat=True, extract_gas=False):
    if extract_gas is False:
        if nodust is True:
            prompt = ' noscat nodust'
        elif nodust is False:
            if scat is True:
                prompt = ' nphot_scat 1000000'
            elif scat is False:
                prompt = ' noscat'
        
        os.system(f"radmc3d image npix 100 sizeau 60 incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam}"+prompt)
        im = readImage('image.out')
        freq0 = im.freq[nlam//2]
        v = cc / 1e5 * (freq0 - im.freq) / freq0
        center = int(len(im.y)//2)
        fig, ax = plt.subplots()
        c = ax.pcolormesh(im.x/au, v, (im.imageJyppix[:, center, :].T)*1e3/(140**2), shading="nearest", rasterized=True, cmap='jet')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('mJy/pixel')
        ax.set_xlabel("Offset [au]")
        ax.set_ylabel("Velocity [km/s]")
        ax.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax.plot([-30, 30], [vkm, vkm], 'w:')

    elif extract_gas is True:
        os.system(f"radmc3d image npix 100 sizeau 60 incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam} nphot_scat 1000000")
        os.system('mv image.out image_gas.out')
        im_gas = readImage('image_gas.out')
        os.system(f"radmc3d image npix 100 sizeau 60 incl {incl} lambdarange {im_gas.wav[0]} {im_gas.wav[-1]} nlam {nlam} nphot_scat 1000000 noline")
        os.system('mv image.out image_dust.out')
        im_dust = readImage('image_dust.out')

        freq0 = im_gas.freq[nlam//2]
        v = cc / 1e5 * (freq0 - im_gas.freq) / freq0
        center = int(len(im_gas.y)//2)

        data_gas  = im_gas.imageJyppix
        data_dust = im_dust.imageJyppix
        extracted_gas = data_gas-data_dust

        fig, ax = plt.subplots()
        c = ax.pcolormesh(im_gas.x/au, v, (extracted_gas[:, center, :].T)*1e3/(140**2), shading="nearest", rasterized=True, cmap='jet')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('mJy/pixel')
        ax.set_xlabel("Offset [au]")
        ax.set_ylabel("Velocity [km/s]")
        ax.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax.plot([-30, 30], [vkm, vkm], 'w:')
    return

###############################################################################
for idx_mc, mcth in enumerate([True, False]):
    for idx_snow, snow in enumerate([True, False]):
        """
        Accretion/Mstar = 1e-5/yr
        """
        problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
                      pancake=False, mctherm=mcth, snowline=snow, floor=True, kep=True)

        # nodust
        plot_pv(v_width=20, nlam=51, nodust=True, scat=False)
        if mcth is True:
            plt.savefig(f'./figures/mctherm/nodust_snowline_{str(snow)}_5.png')
        elif mcth is False:
            plt.savefig(f'./figures/x22/nodust_snowline_{str(snow)}_5.png')
        plt.close()
        
        # noscat
        plot_pv(v_width=20, nlam=51, nodust=False, scat=False)
        if mcth is True:
            plt.savefig(f'./figures/mctherm/noscat_snowline_{str(snow)}_5.png')
        elif mcth is False:
            plt.savefig(f'./figures/x22/noscat_snowline_{str(snow)}_5.png')
        plt.close()

        # scat
        plot_pv(v_width=20, nlam=51, nodust=False, scat=True)
        if mcth is True:
            plt.savefig(f'./figures/mctherm/scat_snowline_{str(snow)}_5.png')
        elif mcth is False:
            plt.savefig(f'./figures/x22/scat_snowline_{str(snow)}_5.png')
        plt.close()
        
        # extracted_gas
        plot_pv(v_width=20, nlam=51, extract_gas=True)
        if mcth is True:
            plt.savefig(f'./figures/mctherm/extracted_gas_snowline_{str(snow)}_5.png')
        elif mcth is False:
            plt.savefig(f'./figures/x22/extracted_gas_snowline_{str(snow)}_5.png')
        plt.close()

        """
        Accretion/Mstar = 1e-7/yr
        """

        problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-7*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
                      pancake=False, mctherm=mcth, snowline=snow, floor=True, kep=True)

        # nodust
        plot_pv(v_width=20, nlam=51, nodust=True, scat=False)
        if mcth is True:
            plt.savefig(f'./figures/mctherm/nodust_snowline_{str(snow)}_7.png')
        elif mcth is False:
            plt.savefig(f'./figures/x22/nodust_snowline_{str(snow)}_7.png')
        plt.close()
        
        # noscat
        plot_pv(v_width=20, nlam=51, nodust=False, scat=False)
        if mcth is True:
            plt.savefig(f'./figures/mctherm/noscat_snowline_{str(snow)}_7.png')
        elif mcth is False:
            plt.savefig(f'./figures/x22/noscat_snowline_{str(snow)}_7.png')
        plt.close()

        # scat
        plot_pv(v_width=20, nlam=51, nodust=False, scat=True)
        if mcth is True:
            plt.savefig(f'./figures/mctherm/scat_snowline_{str(snow)}_7.png')
        elif mcth is False:
            plt.savefig(f'./figures/x22/scat_snowline_{str(snow)}_7.png')
        plt.close()
        
        # extracted_gas
        plot_pv(v_width=20, nlam=51, extract_gas=True)
        if mcth is True:
            plt.savefig(f'./figures/mctherm/extracted_gas_snowline_{str(snow)}_7.png')
        elif mcth is False:
            plt.savefig(f'./figures/x22/extracted_gas_snowline_{str(snow)}_7.png')
        plt.close()

os.system('make cleanall')