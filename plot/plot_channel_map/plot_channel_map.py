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
Plot gas channel maps (with different assumption)
'''
def plot_gas_channel_maps(incl=70, line=240, vkm=0, v_width=5, nlam=11, tworow=True,
                          nodust=False, scat=True, extract_gas=False):
    if extract_gas is False:
        if nodust is True:
            prompt = ' noscat nodust'
        elif nodust is False:
            if scat is True:
                prompt = ' nphot_scat 1000000'
            elif scat is False:
                prompt = ' noscat'

        if tworow is not True:
            fig, ax = plt.subplots(1, nlam, figsize=(15, 12), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
            os.system(f"radmc3d image npix 100 sizeau 60 incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam}"+prompt)
            im = readImage('image.out')
            freq0 = im.freq[nlam//2]
            v = cc / 1e5 * (freq0 - im.freq) / freq0
            vmi = np.min(im.imageJyppix)/(140*140)
            vma = np.max(im.imageJyppix)/(140*140)
            for idx in range(nlam):
                data = np.transpose(im.imageJyppix[:, ::-1, idx])/(140*140)
                ax[idx].imshow(data, cmap='hot', vmin=vmi, vmax=vma)
                ax[idx].axis('off')
                ax[idx].set_title(f'{v[idx]:.1f} $km/s$')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=-1)
        else:
            fig, ax = plt.subplots(2, (nlam//2)+1, figsize=(15, 8), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
            os.system(f"radmc3d image npix 100 sizeau 60 incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam}"+prompt)
            im = readImage('image.out')
            freq0 = im.freq[nlam//2]
            v = cc / 1e5 * (freq0 - im.freq) / freq0
            
            vmi = np.min(im.imageJyppix/(140*140))
            vma = np.max(im.imageJyppix/(140*140))
            for idx in range(nlam):
                data = np.transpose(im.imageJyppix[:, ::-1, idx])/(140*140)
                if idx == nlam//2:
                    ax[0, idx].imshow(data, cmap='hot', vmin=vmi, vmax=vma)
                    ax[0, idx].axis('off')
                    ax[0, idx].set_title(f'{v[idx]:.1f} $km/s$')
                    ax[1, idx].imshow(data, cmap='hot', vmin=vmi, vmax=vma)
                    ax[1, idx].axis('off')
                    ax[1, idx].set_title(f'{v[idx]:.1f} $km/s$')
                elif idx > nlam//2:
                    ax[1, nlam-1-idx].imshow(data, cmap='hot', vmin=vmi, vmax=vma)
                    ax[1, nlam-1-idx].axis('off')
                    ax[1, nlam-1-idx].set_title(f'{v[idx]:.1f} $km/s$')
                else:
                    ax[0, idx].imshow(data, cmap='hot', vmin=vmi, vmax=vma)
                    ax[0, idx].axis('off')
                    ax[0, idx].set_title(f'{v[idx]:.1f} $km/s$')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    elif extract_gas is True:
        if tworow is not True:
            fig, ax = plt.subplots(1, nlam, figsize=(15, 12), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})

            os.system(f"radmc3d image npix 100 sizeau 60 incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam} nphot_scat 1000000")
            os.system('mv image.out image_gas.out')
            im_gas = readImage('image_gas.out')
            freq0 = im_gas.freq[nlam//2]
            v = cc / 1e5 * (freq0 - im_gas.freq) / freq0

            os.system(f"radmc3d image npix 100 sizeau 60 incl {incl} lambdarange {im_gas.wav[0]} {im_gas.wav[-1]} nlam {nlam} nphot_scat 1000000 noline")
            os.system('mv image.out image_dust.out')
            im_dust = readImage('image_dust.out')

            data_gas  = im_gas.imageJyppix/(140*140)
            data_dust = im_dust.imageJyppix/(140*140)
            extracted_gas = data_gas-data_dust

            vmi = np.min(extracted_gas)
            vmi = 0
            vma = np.max(extracted_gas)

            for idx in range(nlam):
                data = np.transpose(extracted_gas[:, ::-1, idx])
                ax[idx].imshow(data, cmap='hot', vmin=vmi, vmax=vma)
                ax[idx].axis('off')
                ax[idx].set_title(f'{v[idx]:.1f} $km/s$')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=-1)
        else:
            fig, ax = plt.subplots(2, (nlam//2)+1, figsize=(15, 8), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})

            os.system(f"radmc3d image npix 100 sizeau 60 incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam} nphot_scat 10000000")
            os.system('mv image.out image_gas.out')
            im_gas = readImage('image_gas.out')
            freq0 = im_gas.freq[nlam//2]
            v = cc / 1e5 * (freq0 - im_gas.freq) / freq0

            os.system(f"radmc3d image npix 100 sizeau 60 incl {incl} lambdarange {im_gas.wav[0]} {im_gas.wav[-1]} nlam {nlam} nphot_scat 10000000 noline")
            os.system('mv image.out image_dust.out')
            im_dust = readImage('image_dust.out')

            data_gas  = im_gas.imageJyppix/(140*140)
            data_dust = im_dust.imageJyppix/(140*140)
            extracted_gas = data_gas-data_dust

            vmi = np.min(extracted_gas)
            # vmi = 0
            vma = np.max(extracted_gas)
            
            for idx in range(nlam):
                data = np.transpose(extracted_gas[:, ::-1, idx])
                if idx == nlam//2:
                    ax[0, idx].imshow(data, cmap='hot', vmin=vmi, vmax=vma)
                    ax[0, idx].axis('off')
                    ax[0, idx].set_title(f'{v[idx]:.1f} $km/s$')
                    ax[1, idx].imshow(data, cmap='hot', vmin=vmi, vmax=vma)
                    ax[1, idx].axis('off')
                    ax[1, idx].set_title(f'{v[idx]:.1f} $km/s$')
                elif idx > nlam//2:
                    ax[1, nlam-1-idx].imshow(data, cmap='hot', vmin=vmi, vmax=vma)
                    ax[1, nlam-1-idx].axis('off')
                    ax[1, nlam-1-idx].set_title(f'{v[idx]:.1f} $km/s$')
                else:
                    ax[0, idx].imshow(data, cmap='hot', vmin=vmi, vmax=vma)
                    ax[0, idx].axis('off')
                    ax[0, idx].set_title(f'{v[idx]:.1f} $km/s$')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    return
###############################################################################
# for idx_snow, snow in enumerate([True, False]):
#         problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=1, 
#                       pancake=False, mctherm=True, snowline=snow, floor=True, kep=True, combine=True)

        # plot_gas_channel_maps(v_width=5, nlam=11, nodust=True, scat=False)
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/nodust_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/nodust_snowline_{str(snow)}.png')
        # plt.close()

        # plot_gas_channel_maps(v_width=5, nlam=11, nodust=False, scat=False)
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/noscat_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/noscat_snowline_{str(snow)}.png')
        # plt.close()

        # plot_gas_channel_maps(v_width=5, nlam=11, nodust=False, scat=True)
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/scat_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/scat_snowline_{str(snow)}.png')
        # plt.close()

        # plot_gas_channel_maps(incl=70, v_width=5, nlam=11, extract_gas=True)
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/extracted_gas_snowline_{str(snow)}_incl_0.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/extracted_gas_snowline_{str(snow)}_incl_0.png')
        # plt.savefig(f'./figures/combine/extracted_gas_snowline_{str(snow)}_incl_70.png')
        # plt.close()

        # problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, v_infall=0, 
        #               pancake=False, mctherm=True, snowline=snow, floor=True, kep=True)

        # plot_gas_channel_maps(v_width=5, nlam=11, nodust=True, scat=False)
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/nodust_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/nodust_snowline_{str(snow)}.png')
        # plt.close()

        # plot_gas_channel_maps(v_width=5, nlam=11, nodust=False, scat=False)
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/noscat_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/noscat_snowline_{str(snow)}.png')
        # plt.close()

        # plot_gas_channel_maps(v_width=5, nlam=11, nodust=False, scat=True)
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/scat_snowline_{str(snow)}.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/scat_snowline_{str(snow)}.png')
        # plt.close()

        # plot_gas_channel_maps(incl=70, v_width=5, nlam=11, extract_gas=True)
        # if mcth is True:
        #     plt.savefig(f'./figures/mctherm/extracted_gas_snowline_{str(snow)}_incl_0_noinfall.png')
        # elif mcth is False:
        #     plt.savefig(f'./figures/x22/extracted_gas_snowline_{str(snow)}_incl_0_noinfall.png')
        # plt.savefig(f'./figures/combine/extracted_gas_snowline_{str(snow)}_incl_70_noinfall.png')
        # plt.close()
    

# os.system('make cleanall')
###############################################################################
'''
Plot a dust image
'''
# def plot_dust_images(incl=70, wav=1.3, scat=True, tworow=True):
#     if scat is True:
#         scat_prompt = ''
#     elif scat is False:
#         scat_prompt = 'noscat'
#     if tworow is not True:
#         fig, ax = plt.subplots(1, 1, figsize=(15, 12), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
#         os.system(f"radmc3d image npix 100 sizeau 60 incl {incl} lambda {wav*1e3} {scat_prompt} noline")
#         im = readImage('image.out')
#         data = np.transpose(im.imageJyppix[:, ::-1, 0])
#         ax.imshow(data*1e3, cmap='hot')
        
#     return
###############################################################################
'''
Plot dust images
'''
def plot_dust(incl=None):
    incl_angle = incl
    if incl is None:
        incl_angle = [0, 15, 30, 45, 60, 75, 90]
    fig, ax = plt.subplots(1, len(incl_angle), figsize=(40, 10), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
    data_min = []
    data_max = []
    for idx, icl in enumerate(incl_angle):
        os.system(f"radmc3d image npix 25 incl {icl} lambda 1300 sizeau 70 nphot_scat 1000000 noline")
        os.system(f'mv image.out image_{idx}.out')
        im_dust = readImage(f'image_{idx}.out')
        data_dust = np.transpose(im_dust.imageJyppix[:, ::-1, 0])/(140*140)
        data_min.append(np.min(data_dust))
        data_max.append(np.max(data_dust))
    for idx, icl in enumerate(incl_angle):
        im_dust = readImage(f'image_{idx}.out')
        data_dust = np.transpose(im_dust.imageJyppix[:, ::-1, 0])/(140*140)
        c = ax[idx].imshow(data_dust, cmap='hot',extent=[-35, 35,-35,35],vmin = min(data_min), vmax = max(data_max))
        ax[idx].set_title(f'{icl}'+'$^{\circ}$')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Jy/pixel')
    return

problem_setup(a_max=.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14*1e-5*Msun/yr, Radius_of_disk=100*au, 
                        pancake=False,v_infall=0, mctherm=False, snowline=True, floor=True, kep=True)
plot_dust(incl=[70,80])
plt.show()
im_dust = readImage(f'image_0.out')
data_dust = np.transpose(im_dust.imageJyppix[:, ::-1, 0])/(140*140)
print(np.sum(data_dust))
os.system('make cleanall')
