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
vinfall = [0, 1]
incl    = [75, 90, 0]
amax = [1, 0.1, 0.01, 0.001]
mdot = [1e-5, 1e-6, 1e-7]
for idx_a, a in enumerate(amax):
    for idx_mdot, md in enumerate(mdot):

        for idx_v, vin in enumerate(vinfall):
            problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14*md*Msun/yr, Radius_of_disk=30*au, 
                                pancake=False, v_infall=vin)
            # os.system('radmc3d mctherm')
            for idx_incl, inc in enumerate(incl):

                os.system(f"radmc3d image npix 100 incl {inc} iline 240 vkms 0 widthkms 20 linenlam 51 nphot_scat 1000000 sizeau 60")
                os.system('mv image.out image_scat.out')
                im_scat = readImage('image_scat.out')
                freq0 = im_scat.freq[25]
                v = cc / 1e5 * (freq0 - im_scat.freq) / freq0
                jcenter = int(len(im_scat.y)//2)  # 3 au above the midplane
                fig, ax = plt.subplots()
                c = ax.pcolormesh(im_scat.x/au, v, (im_scat.imageJyppix[:, jcenter, :].T)*1e3/(140**2), shading="nearest", rasterized=True, cmap='jet')
                cbar = fig.colorbar(c, ax=ax)
                cbar.set_label('mJy/pixel')
                ax.set_xlabel("Offset [au]")
                ax.set_ylabel("Velocity [km/s]")
                ax.plot([0, 0], [-20, 20], 'w:')
                ax.plot([-30, 30], [0, 0], 'w:')
                plt.savefig(f'./Figures/pv/with_snowline/mctherm/scattering/incl_{inc}_vinfall_{vin}_amax_{a}_mdot_{idx_mdot+5}.png')
                plt.close()


                os.system(f"radmc3d image npix 100 incl {inc} lambdarange {im_scat.wav[0]} {im_scat.wav[-1]} nlam 51 sizeau 60 nphot_scat 1000000 noline")
                os.system('mv image.out image_noline.out')
                im_noline = readImage('image_noline.out')
                
                fig, ax = plt.subplots()
                c = ax.pcolormesh(im_noline.x/au, v, (im_noline.imageJyppix[:, jcenter, :].T)*1e3/(140**2), shading="nearest", rasterized=True, cmap='jet')
                cbar = fig.colorbar(c, ax=ax)
                cbar.set_label('mJy/pixel')
                ax.set_xlabel("Offset [au]")
                ax.set_ylabel("Velocity [km/s]")
                ax.plot([0, 0], [-20, 20], 'w:')
                ax.plot([-30, 30], [0, 0], 'w:')
                plt.savefig(f'./Figures/pv/with_snowline/mctherm/noline/incl_{inc}_vinfall_{vin}_amax_{a}_mdot_{idx_mdot+5}.png')
                plt.close()

                
                extracted = im_scat.imageJyppix[:, jcenter, :]-im_noline.imageJyppix[:, jcenter, :]
                fig, ax = plt.subplots()
                c = ax.pcolormesh(im_noline.x/au, v, (extracted.T)*1e3/(140**2), shading="nearest", rasterized=True, cmap='jet')
                cbar = fig.colorbar(c, ax=ax)
                cbar.set_label('mJy/pixel')
                ax.set_xlabel("Offset [au]")
                ax.set_ylabel("Velocity [km/s]")
                ax.plot([0, 0], [-20, 20], 'w:')
                ax.plot([-30, 30], [0, 0], 'w:')
                plt.savefig(f'./Figures/pv/with_snowline/mctherm/extracted_gas/incl_{inc}_vinfall_{vin}_amax_{a}_mdot_{idx_mdot+5}.png')
                plt.close()
os.system('make cleanmodel')
os.system('make cleanall')