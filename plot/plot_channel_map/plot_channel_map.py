import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
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
                          nodust=False, scat=True, extract_gas=False, npix=100, sizeau=100,
                          convolve=True, fwhm=50):
    """
    Note: The unit of fwhm is the same with npix not sizeau.
    """
    if extract_gas is False:
        if nodust is True:
            prompt = ' noscat nodust'
        elif nodust is False:
            if scat is True:
                prompt = ' nphot_scat 1000000'
            elif scat is False:
                prompt = ' noscat'

        fig, ax = plt.subplots(2, (nlam // 2) + 1, figsize=(18, 6), constrained_layout=True, sharex=True, sharey=True,
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.1}, layout="constrained")
        os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam}"+prompt)
        im = readImage('image.out')
        
        freq0 = im.freq[nlam//2]
        v = cc / 1e5 * (freq0 - im.freq) / freq0
        
        os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} lambdarange {im.wav[0]} {im.wav[-1]} nlam {nlam} noscat noline")
        os.system('mv image.out image_dust.out')
        im_dust = readImage('image_dust.out')
        data = im.imageJyppix/(140*140)*1000  # unit : mJy
        dust_conti = im_dust.imageJyppix/(140*140)*1000  # unit : mJy

        if convolve is True:
            convolved_data = np.zeros(shape=data.shape)
            convolved_conti = np.zeros(shape=dust_conti.shape)
            for i in range(nlam):
                FWHM = fwhm
                sigma = FWHM / (2*np.sqrt(2*np.log(2)))
                convolved_data[:, :, i] = gaussian_filter(data[:, :, i], sigma=sigma)
                convolved_conti[:, :, i] = gaussian_filter(dust_conti[:, :, i], sigma=sigma)
            data = convolved_data
            dust_conti = convolved_conti
        
    elif extract_gas is True:
        fig, ax = plt.subplots(2, (nlam // 2) + 1, figsize=(18, 6), constrained_layout=True, sharex=True, sharey=True,
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.1}, layout="constrained")

        os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam} nphot_scat 1000000")
        os.system('mv image.out image_gas.out')
        im_gas = readImage('image_gas.out')
        freq0 = im_gas.freq[nlam//2]
        v = cc / 1e5 * (freq0 - im_gas.freq) / freq0

        os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} lambdarange {im_gas.wav[0]} {im_gas.wav[-1]} nlam {nlam} nphot_scat 1000000 noline")
        os.system('mv image.out image_dust.out')
        im_dust = readImage('image_dust.out')

        data_gas  = im_gas.imageJyppix/(140*140)*1000
        dust_conti = im_dust.imageJyppix/(140*140)*1000
        data = data_gas-dust_conti
        
        if convolve is True:
            convolved_data = np.zeros(shape=data.shape)
            convolved_conti = np.zeros(shape=dust_conti.shape)
            for i in range(nlam):
                FWHM = fwhm
                sigma = FWHM / (2*np.sqrt(2*np.log(2)))
                convolved_data[:, :, i] = gaussian_filter(data[:, :, i], sigma=sigma)
                convolved_conti[:, :, i] = gaussian_filter(dust_conti[:, :, i], sigma=sigma)
            data = convolved_data
            dust_conti = convolved_conti
        
    vmi = np.min(data)
    vma = np.max(data)
            
    cm = 'hot'
    tc = 'w'
    x = np.linspace(1, npix, npix, endpoint=True)
    y = np.linspace(1, npix, npix, endpoint=True)
    X, Y = np.meshgrid(x, y)
    contour_level = np.linspace(0, np.max(dust_conti), 5)
    for idx in range(nlam):
        d = np.transpose(data[:, ::-1, idx])
        if idx == nlam//2:
            image = ax[0, idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
            ax[0, idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
            # ax[0, idx].imshow(np.transpose(dust_conti[:, ::-1, idx]), cmap='nipy_spectral', vmin=np.min(dust_conti), vmax=np.max(dust_conti), alpha=0.5)
            ax[0, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
            
            ax[1, idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
            ax[1, idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
            # ax[1, idx].imshow(np.transpose(dust_conti[:, ::-1, idx]), cmap='nipy_spectral', vmin=np.min(dust_conti), vmax=np.max(dust_conti), alpha=0.5)
            ax[1, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
            
            ax[1, idx].set_xlabel('AU',fontsize=16)
            if idx == 0:
                ax[1, idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
                ax[1, idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
                ax[1, idx].set_ylabel('AU',fontsize=16)
        elif idx > nlam//2:
            ax[1, nlam-1-idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
            ax[1, nlam-1-idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
            # ax[1, nlam-1-idx].imshow(np.transpose(dust_conti[:, ::-1, idx]), cmap='nipy_spectral', vmin=np.min(dust_conti), vmax=np.max(dust_conti), alpha=0.5)
            ax[1, nlam-1-idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
            
            ax[1, nlam-1-idx].set_xticks([int(npix*0.1), npix//2, int(npix*0.9)])
            ax[1, nlam-1-idx].set_xticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
            ax[1, nlam-1-idx].set_xlabel('AU',fontsize=16)
            if nlam-1-idx == 0:
                ax[1, nlam-1-idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
                ax[1, nlam-1-idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
                ax[1, nlam-1-idx].set_ylabel('AU',fontsize=16)
        else:
            ax[0, idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
            ax[0, idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
            # ax[0, idx].imshow(np.transpose(dust_conti[:, ::-1, idx]), cmap='nipy_spectral', vmin=np.min(dust_conti), vmax=np.max(dust_conti), alpha=0.5)
            ax[0, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
            if idx == 0:
                ax[0, idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
                ax[0, idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
                ax[0, idx].set_ylabel('AU',fontsize=16)
    # fig.tight_layout(pad=1.0)
    cbar = fig.colorbar(image, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Intensity (mJy/pixel)')
    return

###############################################################################
# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=None)
# plot_gas_channel_maps(nodust=True, convolve=False)
# plt.savefig('./figures/combine/wo dust.pdf')
# plt.close()
# os.system('make cleanall')

problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
            pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=None)
plot_gas_channel_maps(nodust=True, convolve=True)
plt.savefig('./figures/combine/wo dust_convolved_fwhm_50.pdf')
plt.close()
os.system('make cleanall')

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=None)
# plot_gas_channel_maps(nodust=True, convolve=True, fwhm=10)
# plt.savefig('./figures/combine/wo dust_convolved_fwhm_10.pdf')
# plt.close()
# os.system('make cleanall')
###############################################################################
# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=None)
# plot_gas_channel_maps(extract_gas=True, convolve=False)
# plt.savefig('./figures/combine/w dust.pdf')
# plt.close()
# os.system('make cleanall')

problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
            pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=None)
plot_gas_channel_maps(extract_gas=True, convolve=True)
plt.savefig('./figures/combine/w dust_convolved_fwhm_50.pdf')
plt.close()
os.system('make cleanall')

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=None)
# plot_gas_channel_maps(extract_gas=True, convolve=True, fwhm=10)
# plt.savefig('./figures/combine/w dust_convolved_fwhm_10.pdf')
# plt.close()
# os.system('make cleanall')
###############################################################################
# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5)
# plot_gas_channel_maps(nodust=True)
# plt.savefig('./figures/combine/wo dust + rcb 5.pdf')
# plt.close()
# os.system('make cleanall')

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5)
# plot_gas_channel_maps(nodust=True, convolve=True)
# plt.savefig('./figures/combine/wo dust + rcb 5_convolved_fwhm_50.pdf')
# plt.close()
# os.system('make cleanall')

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5)
# plot_gas_channel_maps(nodust=True, convolve=True, fwhm=10)
# plt.savefig('./figures/combine/wo dust + rcb 5_convolved_fwhm_10.pdf')
# plt.close()
# os.system('make cleanall')
###############################################################################
# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5)
# plot_gas_channel_maps(extract_gas=True, convolve=False)
# plt.savefig('./figures/combine/w dust + rcb 5.pdf')
# plt.close()
# os.system('make cleanall')

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5)
# plot_gas_channel_maps(extract_gas=True,convolve=True)
# plt.savefig('./figures/combine/w dust + rcb 5_convolved_fwhm_50.pdf')
# plt.close()
# os.system('make cleanall')

# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5)
# plot_gas_channel_maps(extract_gas=True,convolve=True, fwhm=10)
# plt.savefig('./figures/combine/w dust + rcb 5_convolved_fwhm_10.pdf')
# plt.close()
# os.system('make cleanall')
###############################################################################
###############################################################################
'''
Plot dust images
'''
def plot_dust(incl=None):
    incl_angle = incl
    if incl is None:
        incl_angle = [0, 15, 30, 45, 60, 75,90]
    fig, ax = plt.subplots(1, len(incl_angle), figsize=(40, 10), sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
    data_min = []
    data_max = []
    integrated = []
    for idx, icl in enumerate(incl_angle):
        os.system(f"radmc3d image npix 50 incl {icl} lambda 1300 sizeau 70 nphot_scat 1000000 noline")
        os.system(f'mv image.out image_{idx}.out')
        im_dust = readImage(f'image_{idx}.out')
        data_dust = np.transpose(im_dust.imageJyppix[:, ::-1, 0])/(140*140)
        data_min.append(np.min(data_dust))
        data_max.append(np.max(data_dust))
        integrated.append(np.sum(data_dust))
    for idx, icl in enumerate(incl_angle):
        im_dust = readImage(f'image_{idx}.out')
        data_dust = np.transpose(im_dust.imageJyppix[:, ::-1, 0])/(140*140)
        c = ax[idx].imshow(data_dust, cmap='hot',extent=[-35, 35,-35,35],vmin = min(data_min), vmax = max(data_max))
        ax[idx].set_title(f'{icl}'+r'$^{\circ}$')
    print('peak intensity (Jy/pix):'+ str(data_max))
    print('integrated intensity (Jy/pix):'+ str(integrated))
    return


# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#               pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=False, Rcb=5)
# plot_dust()
# plt.savefig('dust_mctherm.png')
# os.system('make cleanall')
# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#               pancake=False, mctherm=False, snowline=True, floor=True, kep=True, combine=False, Rcb=5)
# plot_dust()
# plt.savefig('dust_x22.png')
# os.system('make cleanall')
# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
#               pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5)
# plot_dust()
# plt.savefig('dust_combine.png')
# os.system('make cleanall')