import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup

problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, pancake=False)
v_list = np.linspace(-5, 5, 11, endpoint=True)
# im = readImage('image.out')
# print(im.image.shape)

for v in v_list:
    os.system(f"radmc3d image incl 70 iline 240 vkms {v}")
    im = readImage('image.out')
    # plt.imshow(np.transpose(im.image[:,:,0]),cmap=plt.cm.gist_heat)
    # plt.show()
    plotImage(im, arcsec=True, dpc=140., cmap='hot')