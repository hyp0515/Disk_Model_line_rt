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
def plot_channel_maps(incl=70, vlist=None, line=1, tworow=True, dust=True):
    if dust is True:
        nodust = ''
    else:
        nodust = 'nodust'

    if vlist is None : vlist = np.linspace(-5, 5, 11, endpoint=True)
    
    if tworow is not True:
        fig, ax = plt.subplots(1, len(vlist), figsize=(15, 12), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
        for idx, v in enumerate(vlist):
            os.system(f"radmc3d image incl {incl} iline {line} vkms {v} {nodust}")
            im = readImage('image.out')
            data = np.transpose(im.imageJyppix[:, ::-1, 0])
            ax[idx].imshow(data, cmap='hot')
            ax[idx].axis('off')
            ax[idx].set_title(f'{v} $km/s$')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=-1)
    else:
        fig, ax = plt.subplots(2, int((len(vlist)+1)/2), figsize=(15, 8), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
        for idx, v in enumerate(vlist):
            os.system(f"radmc3d image incl {incl} iline {line} vkms {v} {nodust}")
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
# vinfall=[0, 1, 5, 10]
# for vin in vinfall:
#     problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
#                   pancake=False,v_infall=0.1*vin)
#     incl_list = [0, 15, 30, 45, 60, 75, 90]
#     for deg in incl_list:
#         plot_channel_maps(incl = deg, line=240,dust=True)
#         plt.savefig(f'./Figures/channel_maps/iline_240/include_dust/{vin}_infall_incl_{deg}.png')
#         print('Finish...')
#         plot_channel_maps(incl = deg, line=240,dust=False)
#         plt.savefig(f'./Figures/channel_maps/iline_240/nodust/{vin}_infall_incl_{deg}.png')
#         print('Finish...')
#         plot_channel_maps(incl = deg, line=1)
#         plt.savefig(f'./Figures/channel_maps/iline_1/pancake/{vin}_infall_incl_{deg}.png')
#         print('Finish...')

###############################################################################
'''
Plot extracted gas images and save them into one plot
'''
def plot_extracted_gas(incl=70, vlist=None, line=240, tworow=True):

    if vlist is None : vlist = np.linspace(-5, 5, 11, endpoint=True)
    
    if tworow is not True:
        fig, ax = plt.subplots(1, len(vlist), figsize=(15, 12), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
        for idx, v in enumerate(vlist):

            os.system(f"radmc3d image npix 200 incl {incl} sizeau 70 iline {line} vkms {v} nphot_scat 10000000")
            os.system('mv image.out image_gas.out')
            im_gas = readImage('image_gas.out')
            os.system(f"radmc3d image npix 200 incl {incl} lambda {im_gas.wav[0]} sizeau 70 nphot_scat 10000000 noline")
            os.system('mv image.out image_dust.out')
            im_dust = readImage('image_dust.out')
            data_gas  = np.transpose(im_gas.imageJyppix[:, ::-1, 0])/(140*140)
            data_dust = np.transpose(im_dust.imageJyppix[:, ::-1, 0])/(140*140)

            ax[idx].imshow(data_gas-data_dust, cmap='hot')
            ax[idx].axis('off')
            ax[idx].set_title(f'{v} $km/s$')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=-1)
    else:
        fig, ax = plt.subplots(2, int((len(vlist)+1)/2), figsize=(15, 8), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
        for idx, v in enumerate(vlist):
            os.system(f"radmc3d image npix 200 incl {incl} sizeau 70 iline {line} vkms {v} nphot_scat 1000000")
            os.system('mv image.out image_gas.out')
            im_gas = readImage('image_gas.out')
            os.system(f"radmc3d image npix 200 incl {incl} lambda {im_gas.wav[0]} sizeau 70 nphot_scat 1000000 noline")
            os.system('mv image.out image_dust.out')
            im_dust = readImage('image_dust.out')
            data_gas  = np.transpose(im_gas.imageJyppix[:, ::-1, 0])/(140**2)  # 140**2 is distance*2
            data_dust = np.transpose(im_dust.imageJyppix[:, ::-1, 0])/(140**2)
            
            if idx == (len(vlist)-1)/2:
                ax[0, idx].imshow(data_gas-data_dust, cmap='hot')
                ax[0, idx].axis('off')
                ax[0, idx].set_title(f'{v} $km/s$')
                ax[1, idx].imshow(data_gas-data_dust, cmap='hot')
                ax[1, idx].axis('off')
                ax[1, idx].set_title(f'{v} $km/s$')
            elif idx > (len(vlist)-1)/2:
                ax[1, len(vlist)-1-idx].imshow(data_gas-data_dust, cmap='hot')
                ax[1, len(vlist)-1-idx].axis('off')
                ax[1, len(vlist)-1-idx].set_title(f'{v} $km/s$')
            else:
                ax[0, idx].imshow(data_gas-data_dust, cmap='hot')
                ax[0, idx].axis('off')
                ax[0, idx].set_title(f'{v} $km/s$')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    return
###############################################################################
vinfall=[0, 0.1, 0.5, 1]
for vin in vinfall:
    problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                  pancake=False,v_infall=vin)
    incl_list = [0, 15, 30, 45, 60, 75, 90]
    for deg in incl_list:
        plot_extracted_gas(incl = deg, line=240)
        plt.savefig(f'./Figures/extracted_gas/with_snowline/channel_maps/{vin}_infall_incl_{deg}.png')
        print('Finish...')
###############################################################################
'''
Plot dust images
'''
# def plot_dust(incl=None):
#     incl_angle = incl
#     if incl is None:
#         incl_angle = [0, 15, 30, 45, 60, 75, 90]
#     os.system(f"radmc3d image npix 200 incl {str(incl_angle[0])} sizeau 70 iline 240 vkms 0 nphot_scat 1000")
#     os.system('mv image.out image_gas.out')
#     im_gas = readImage('image_gas.out')
#     fig, ax = plt.subplots(1, len(incl_angle), figsize=(40, 10), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
#     data_min = []
#     data_max = []
#     for idx, icl in enumerate(incl_angle):
#         os.system(f"radmc3d image npix 200 incl {icl} lambda {im_gas.wav[0]} sizeau 70 nphot_scat 10000000 noline")
#         os.system(f'mv image.out image_{idx}.out')
#         im_dust = readImage(f'image_{idx}.out')
#         data_dust = np.transpose(im_dust.imageJyppix[:, ::-1, 0])/(140*140)
#         data_min.append(np.min(data_dust))
#         data_max.append(np.max(data_dust))
#     for idx, icl in enumerate(incl_angle):
#         im_dust = readImage(f'image_{idx}.out')
#         data_dust = np.transpose(im_dust.imageJyppix[:, ::-1, 0])/(140*140)
#         ax[idx].imshow(data_dust, cmap='hot',extent=[-35,35,-35,35],vmin = min(data_min), vmax = max(data_max))
#         ax[idx].set_title(f'{icl}'+'$^{\circ}$')
#     return
###############################################################################
# for a in [0.1, 0.01, 0.001]:
#     for mstar in [0.1, 0.5, 1, 1, 5]:

#         problem_setup(a_max=a, Mass_of_star=mstar*Msun, Accretion_rate=mstar*1e-5*Msun/yr, Radius_of_disk=30*au, 
#                         pancake=False,v_infall=0)
#         plot_dust()
#         plt.savefig(f'./Figures/dust_maps/amax_{a}_mstar_{mstar}.png')
