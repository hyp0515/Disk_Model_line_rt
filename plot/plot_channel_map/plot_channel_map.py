import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
import sys
sys.path.insert(0,'../../')
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup
import matplotlib as mpl
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
def plot_gas_channel_maps(incl=70, line=240, vkm=0, v_width=5, nlam=11,
                          nodust=False, scat=True, extract_gas=False):
    if extract_gas is False:
        if nodust is True:
            prompt = ' noscat nodust'
        elif nodust is False:
            if scat is True:
                prompt = ' nphot_scat 1000000'
            elif scat is False:
                prompt = ' noscat'

        fig, ax = plt.subplots(2, (nlam//2)+1, figsize=(12, 6), sharex='all', sharey='all', gridspec_kw={'wspace': 0, 'hspace': 0})
        os.system(f"radmc3d image npix 100 sizeau 50 incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam}"+prompt)
        im = readImage('image.out')
        freq0 = im.freq[nlam//2]
        v = cc / 1e5 * (freq0 - im.freq) / freq0
            
        vmi = np.min(im.imageJyppix/(140*140))
        vma = np.max(im.imageJyppix/(140*140))
        cm = 'hot'
        tc = 'w'
        for idx in range(nlam):
            data = np.transpose(im.imageJyppix[:, ::-1, idx])/(140*140)
            if idx == nlam//2:
                ax[0, idx].imshow(data, cmap=cm, vmin=vmi, vmax=vma)
                ax[0, idx].text(90,10,f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                ax[1, idx].imshow(data, cmap=cm, vmin=vmi, vmax=vma)
                ax[1, idx].text(90,10,f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                ax[1, idx].set_xlabel('AU',fontsize=16)
                if idx == 0:
                    ax[1, idx].set_yticks([10, 50, 90])
                    ax[1, idx].set_yticklabels(['-20', '0', '20'], fontsize=14)
                    ax[1, idx].set_ylabel('AU',fontsize=16)
            elif idx > nlam//2:
                ax[1, nlam-1-idx].imshow(data, cmap=cm, vmin=vmi, vmax=vma)
                ax[1, nlam-1-idx].text(90,10,f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                ax[1, nlam-1-idx].set_xticks([10, 50, 90])
                ax[1, nlam-1-idx].set_xticklabels(['-20', '0', '20'], fontsize=14)
                ax[1, nlam-1-idx].set_xlabel('AU',fontsize=16)
                if nlam-1-idx == 0:
                    ax[1, nlam-1-idx].set_yticks([10, 50, 90])
                    ax[1, nlam-1-idx].set_yticklabels(['-20', '0', '20'], fontsize=14)
                    ax[1, nlam-1-idx].set_ylabel('AU',fontsize=16)
            else:
                ax[0, idx].imshow(data, cmap=cm, vmin=vmi, vmax=vma)
                ax[0, idx].text(90,10,f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                if idx == 0:
                    ax[0, idx].set_yticks([10, 50, 90])
                    ax[0, idx].set_yticklabels(['-20', '0', '20'], fontsize=14)
                    ax[0, idx].set_ylabel('AU',fontsize=16)
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.1, wspace=0, hspace=0)
    elif extract_gas is True:
        fig, ax = plt.subplots(2, (nlam//2)+1, figsize=(12, 6), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})

        os.system(f"radmc3d image npix 100 sizeau 50 incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam} nphot_scat 100000")
        os.system('mv image.out image_gas.out')
        im_gas = readImage('image_gas.out')
        freq0 = im_gas.freq[nlam//2]
        v = cc / 1e5 * (freq0 - im_gas.freq) / freq0

        os.system(f"radmc3d image npix 100 sizeau 50 incl {incl} lambdarange {im_gas.wav[0]} {im_gas.wav[-1]} nlam {nlam} nphot_scat 100000 noline")
        os.system('mv image.out image_dust.out')
        im_dust = readImage('image_dust.out')

        data_gas  = im_gas.imageJyppix/(140*140)
        data_dust = im_dust.imageJyppix/(140*140)
        extracted_gas = data_gas-data_dust

        # vmi = np.min(extracted_gas)
        vmi = 0
        vma = np.max(extracted_gas)
            
        cm = 'hot'
        tc = 'w'
        for idx in range(nlam):
            data = np.transpose(extracted_gas[:, ::-1, idx])
            if idx == nlam//2:
                ax[0, idx].imshow(data, cmap=cm, vmin=vmi, vmax=vma)
                ax[0, idx].text(90,10,f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                ax[1, idx].imshow(data, cmap=cm, vmin=vmi, vmax=vma)
                ax[1, idx].text(90,10,f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                ax[1, idx].set_xlabel('AU',fontsize=16)
                if idx == 0:
                    ax[1, idx].set_yticks([10, 50, 90])
                    ax[1, idx].set_yticklabels(['-20', '0', '20'], fontsize=14)
                    ax[1, idx].set_ylabel('AU',fontsize=16)
            elif idx > nlam//2:
                ax[1, nlam-1-idx].imshow(data, cmap=cm, vmin=vmi, vmax=vma)
                ax[1, nlam-1-idx].text(90,10,f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                ax[1, nlam-1-idx].set_xticks([10, 50, 90])
                ax[1, nlam-1-idx].set_xticklabels(['-20', '0', '20'], fontsize=14)
                ax[1, nlam-1-idx].set_xlabel('AU',fontsize=16)
                if nlam-1-idx == 0:
                    ax[1, nlam-1-idx].set_yticks([10, 50, 90])
                    ax[1, nlam-1-idx].set_yticklabels(['-20', '0', '20'], fontsize=14)
                    ax[1, nlam-1-idx].set_ylabel('AU',fontsize=16)
            else:
                ax[0, idx].imshow(data, cmap=cm, vmin=vmi, vmax=vma)
                ax[0, idx].text(90,10,f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                if idx == 0:
                    ax[0, idx].set_yticks([10, 50, 90])
                    ax[0, idx].set_yticklabels(['-20', '0', '20'], fontsize=14)
                    ax[0, idx].set_ylabel('AU',fontsize=16)
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.1, wspace=0, hspace=0)
    return
###############################################################################

problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
            pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=False, Rcb=None)
plot_gas_channel_maps(nodust=True)
plt.savefig('channel_map_mctherm_nodust.png')
plt.close()
os.system('make cleanall')

problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
            pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=False, Rcb=None)
plot_gas_channel_maps(extract_gas=True)
plt.savefig('channel_map_mctherm_with_dust.png')
plt.close()
os.system('make cleanall')

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=False, Rcb=None, abundance_enhancement=1e-7)
# plot_gas_channel_maps(nodust=True)
# plt.savefig('channel_map_mctherm_nodust.png_abundance_7')
# plt.close()
# os.system('make cleanall')

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=False, Rcb=None, abundance_enhancement=1e-7)
# plot_gas_channel_maps(extract_gas=True)
# plt.savefig('channel_map_mctherm_with_dust_abundance_7.png')
# plt.close()
# os.system('make cleanall')

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

