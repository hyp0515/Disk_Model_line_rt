import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup

vinfall = np.linspace(0, 1, 6, endpoint=True)
incl = [0, 15, 30, 45, 60, 75, 90]
for idx_v, vin in enumerate(vinfall):
    problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                pancake=False, v_infall=vin)
    for idx_incl, inc in enumerate(incl):

        os.system(f"radmc3d image iline 240 incl {inc} vkms 0 widthkms 10 linenlam 101 npix 100 nodust")
        im = readImage()
        freq0 = im.freq[50]
        v = cc / 1e5 * (freq0 - im.freq) / freq0
        # print(im.wav)
        # print(im.freq)
        # print(v)
        jcenter = int(len(im.y)//2)
        plt.pcolormesh(im.x/au, v, im.image[:, jcenter, :].T, shading="nearest", rasterized=True, cmap='jet')
        plt.xlabel("Offset [au]")
        plt.ylabel("Velocity [km/s]")
        # plt.show()
        plt.savefig(f'./Figures/pv/width_10/no_dust/incl_{inc}_vinfall_{vin:.1f}.png')
        plt.close()

for idx_v, vin in enumerate(vinfall):
    problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                pancake=False, v_infall=vin)
    for idx_incl, inc in enumerate(incl):

        os.system(f"radmc3d image iline 240 incl {inc} vkms 0 widthkms 10 linenlam 101 npix 100")
        im = readImage()
        freq0 = im.freq[50]
        v = cc / 1e5 * (freq0 - im.freq) / freq0

        jcenter = int(len(im.y)//2)
        plt.pcolormesh(im.x/au, v, im.image[:, jcenter, :].T, shading="nearest", rasterized=True, cmap='jet')
        plt.xlabel("Offset [au]")
        plt.ylabel("Velocity [km/s]")
        # plt.show()
        plt.savefig(f'./Figures/pv/width_10/include_dust/incl_{inc}_vinfall_{vin:.1f}.png')
        plt.close()

# problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
#                 pancake=False, v_infall=1)
# os.system(f"radmc3d image iline 240 incl 45 vkms 0 widthkms 5 linenlam 101 npix 100 nodust")
# im = readImage()
# print(im.image.shape)