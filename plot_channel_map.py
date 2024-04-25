import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup


###############################################################################
'''
Plot images one at a time and save by 'hand'
'''
# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, pancake=False)
# v_list = np.linspace(-5, 5, 11, endpoint=True)

# for v in v_list:
#     os.system(f"radmc3d image incl 70 iline 1 vkms {v}")
#     im = readImage('image.out')
#     plotImage(im, arcsec=True, dpc=140., cmap='hot', bunit='snu')
###############################################################################
'''
Plot images and save them into one plot
'''
def plot_channel_maps(incl=70, vlist=None, line=1, tworow=True):

    if vlist is None : vlist = np.linspace(-5, 5, 11, endpoint=True)
    
    if tworow is not True:
        fig, ax = plt.subplots(1, len(vlist), figsize=(15, 12), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
        for idx, v in enumerate(vlist):
            os.system(f"radmc3d image incl {incl} iline {line} vkms {v}")
            im = readImage('image.out')
            data = np.transpose(im.imageJyppix[:, ::-1, 0])
            ax[idx].imshow(data, cmap='hot')
            ax[idx].axis('off')
            ax[idx].set_title(f'{v} $km/s$')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=-1)
    else:
        fig, ax = plt.subplots(2, int((len(vlist)+1)/2), figsize=(15, 8), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
        for idx, v in enumerate(vlist):
            os.system(f"radmc3d image incl {incl} iline {line} vkms {v}")
            im = readImage('image.out')
            data = np.transpose(im.imageJyppix[:, ::-1, 0])
            if idx == (len(vlist)-1)/2:
                ax[0, idx].imshow(data, cmap='hot')
                ax[0, idx].axis('off')
                ax[0, idx].set_title(f'{v} $km/s$')
                ax[1, idx].imshow(data, cmap='hot')
                ax[1, idx].axis('off')
                ax[1, idx].set_title(f'{v} $km/s$')
            elif idx > (len(vlist)-1)/2:
                ax[1, len(vlist)-1-idx].imshow(data, cmap='hot')
                ax[1, len(vlist)-1-idx].axis('off')
                ax[1, len(vlist)-1-idx].set_title(f'{v} $km/s$')
            else:
                ax[0, idx].imshow(data, cmap='hot')
                ax[0, idx].axis('off')
                ax[0, idx].set_title(f'{v} $km/s$')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    return

###############################################################################
vinfall=[0, 1, 5, 10]
for vin in vinfall:
    incl_list = [0, 15, 30, 45, 60, 75, 90]
    problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                  pancake=False,v_infall=0.1*vin)
    for deg in incl_list:
        plot_channel_maps(incl = deg, line=240)
        plt.savefig(f'./Figures/channel_maps/iline_240/{vin}_infall_incl_{deg}.png')
        print('Finish...')



