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
"""
Generate image cube
"""
def generate_cube(fname=None,
                  incl=70, line=240, v_width=10, nlam=10, npix=100, sizeau=100,
                  nodust=False, scat=True, extract_gas=True):
    """
    fname              : File name for saving image cube
    incl               : Inclination angle of the disk
    line               : Transistion level (see 'molecule_ch3oh.inp')
    v_width            : Range of velocity to simulate
    nlam               : Number of velocities
    npix               : Number of map's pixels
    nodust             : If False, dust effect is included
    scat               : If True and nodust=False, scattering is included. (Time-consuming)
    extracted_gas      : If True, spectral line is extracted (I_{dust+gas}-I{dust})
    sizeau             : Map's span
    convolve + fwhm    : Generate two images, with and without convolution
                         (The unit of fwhm is pixel)
    cube*              : File's name if images have been generated beforehand
        'cube' is for extract_gas=False
        'cube_gas' and 'cube_dust' is for extract_gas=True
    """
    if extract_gas is False:
        if nodust is True:
            prompt = ' noscat nodust'
            f = '_nodust'
        elif nodust is False:
            if scat is True:
                prompt = ' nphot_scat 1000000'
                f = ''
            elif scat is False:
                prompt = ' noscat'
                f = '_noscat'
            os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms 0 widthkms {v_width} linenlam {nlam}"+prompt)
            im = readImage('image.out')
            if fname is None:
                print('Be aware of repeating file\'s name')
            elif fname is not None:
                os.system(f'mv image.out '+fname+f+'.img')
        os.system('make cleanall')
        return im
    elif extract_gas is True:
        os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms 0 widthkms {v_width} linenlam {nlam} nphot_scat 1000000")
        im_gas = readImage('image.out')
        if fname is None:
            os.system('mv image.out image_gas.img')
            print('Be aware of repeating file\'s name')
        elif fname is not None:
            os.system('mv image.out image_gas_'+fname+'.img')
        os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} lambdarange {im_gas.wav[0]} {im_gas.wav[-1]} nlam {nlam} nphot_scat 1000000 noline")
        im_dust = readImage('image.out')
        if fname is None:
            os.system('mv image.out image_dust.img')
            print('Be aware of repeating file\'s name')
        elif fname is not None:
            os.system('mv image.out image_dust_'+fname+'.img')
        if fname is None:
            print('Be aware of repeating file\'s name')
        os.system('make cleanall')
        return im_gas, im_dust
###############################################################################
"""
Plot Position-velocity (PV) diagrams
"""
def plot_channel(dir=None, precomputed=False,
            cube=None, cube_gas=None, cube_dust=None,
            vkm=5,
            convolve=True, fwhm=50, fname=None, title=None):
    """
    vkm                : Systematic velocity
    convolve + fwhm    : Generate two images, with and without convolution
                         (The unit of fwhm is pixel)
    cube*              : File's name if images have been generated beforehand
                         'cube' is for extract_gas=False
                         'cube_gas' and 'cube_dust' is for extract_gas=True
    fname              : Plot's name
    title              : Title in the plot
    """
    if dir is not None:
        os.makedirs('./figures/'+dir, exist_ok=True)
        os.makedirs('./precomputed_data/'+dir, exist_ok=True)
        
    if cube is not None:
        if precomputed is True:
            cube = './precomputed_data/'+dir+'/'+cube
        im = readImage(cube)
        data = im.imageJyppix*1e3/(140**2) # mJy/pix
        if precomputed is False:
            os.system('mv '+cube+' ./precomputed_data/'+dir+'/'+cube)
        print('Under construction')
        
    elif (cube_dust is not None) and (cube_gas is not None):
        if precomputed is True:
            cube_gas = './precomputed_data/'+dir+'/'+cube_gas
            cube_dust = './precomputed_data/'+dir+'/'+cube_dust
        im = readImage(cube_gas)
        im_dust = readImage(cube_dust)
        if precomputed is False:
            os.system('mv '+cube_gas+' ./precomputed_data/'+dir+'/'+cube_gas)
            os.system('mv '+cube_dust+' ./precomputed_data/'+dir+'/'+cube_dust)
        data_gas  = im.imageJyppix
        data_dust = im_dust.imageJyppix
        data = (data_gas-data_dust)*1e3/(140**2) # mJy/pix
        absorption = np.where(data<0, data, 0)
        sizeau = int(round((im.x/au)[-1]))
        npix=im.nx
        nlam=len(im.wav)
        freq0 = (im.freq[nlam//2] + im.freq[(nlam//2)-1])/2
        v = cc / 1e5 * (freq0 - im.freq) / freq0
    
        fig, ax = plt.subplots(2, (nlam // 2) + 1, figsize=(18, 6), sharex=True, sharey=True,
                gridspec_kw={'wspace': 0.1, 'hspace': 0.1}, layout="constrained")
    
        # vmi = np.min(data)
        vmi = 0
        vma = np.max(data)
                
        cm = 'hot'
        abcm = 'viridis_r'
        tc = 'w'
        x = np.linspace(1, npix, npix, endpoint=True)
        y = np.linspace(1, npix, npix, endpoint=True)
        X, Y = np.meshgrid(x, y)
        contour_level = np.linspace(0, np.max(data_dust), 5)
        for idx in range(nlam):
            d = np.transpose(data[:, ::-1, idx])
            
            if idx == nlam//2:
                image = ax[0, idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
                ax[0, idx].contour(Y, X, data_dust[:, :, idx], levels=contour_level, colors='w', linewidths=1)
                ax[0, idx].imshow(absorption[:, :, idx], cmap=abcm, alpha=0.5)
                ax[0, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                
                ax[1, idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
                ax[1, idx].contour(Y, X, data_dust[:, :, idx], levels=contour_level, colors='w', linewidths=1)
                ax[1, idx].imshow(absorption[:, :, idx], cmap=abcm, alpha=0.5)
                ax[1, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                
                ax[1, idx].set_xlabel('AU',fontsize=16)
                if idx == 0:
                    ax[1, idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
                    ax[1, idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
                    ax[1, idx].set_ylabel('AU',fontsize=16)
            elif idx > nlam//2:
                ax[1, nlam-1-idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
                ax[1, nlam-1-idx].contour(Y, X, data_dust[:, :, idx], levels=contour_level, colors='w', linewidths=1)
                ax[1, nlam-1-idx].imshow(absorption[:, :, idx], cmap=abcm, alpha=0.5)
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
                ax[0, idx].contour(Y, X, data_dust[:, :, idx], levels=contour_level, colors='w', linewidths=1)
                ax[0, idx].imshow(absorption[:, :, idx], cmap=abcm, alpha=0.5)
                ax[0, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                if idx == 0:
                    ax[0, idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
                    ax[0, idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
                    ax[0, idx].set_ylabel('AU',fontsize=16)

        cbar = fig.colorbar(image, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Intensity (mJy/pixel)')
        if title is not None:
            ax.set_title(title, fontsize = 16) 
        fig.savefig('./figures/'+dir+'/'+fname+'.pdf', transparent=True)
        plt.close('all')
        return  
    else:
        print('No correct cube is given.')
        return
###############################################################################
problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=14e-7*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
              pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5, gas_inside_rcb=False)
generate_cube(nlam=9)
plot_channel()

# '''
# Plot gas channel maps ('problem_setup' is required before plot_gas_channel_maps)
# '''
# def plot_gas_channel_maps(incl=70, line=240, vkm=0, v_width=5, nlam=11,
#                           nodust=False, scat=True, extract_gas=False, npix=100, sizeau=100,
#                           convolve=True, fwhm=50,
#                           precomputed_data_gas=None, precomputed_data_dust=None, precomputed_data=None):
#     """
#     incl               : Inclination angle of the disk
#     line               : Transistion level (see 'molecule_ch3oh.inp')
#     vkm                : Systematic velocity
#     v_width            : Range of velocity to simulate
#     nlam               : Number of velocities
#     nodust             : If False, dust effect is included
#     scat               : If True and nodust=False, scattering is included. (Time-consuming)
#     extracted_gas      : If True, spectral line is extracted (I_{dust+gas}-I{dust})
#     npix               : Number of map's pixels
#     sizeau             : Map's span
#     convolve + fwhm    : Generate two images, with and without convolution
#                          (The unit of fwhm is pixel)
#     precomputed_data_* : File's name if images have been generated beforehand
#         'precomputed_data' is for extract_gas=False
#         'precomputed_data_gas' and 'precomputed_data_dust' is for extract_gas=True
#     """
#     if extract_gas is False:
#         if precomputed_data is None:
#             if nodust is True:
#                 prompt = ' noscat nodust'
#             elif nodust is False:
#                 if scat is True:
#                     prompt = ' nphot_scat 1000000'
#                 elif scat is False:
#                     prompt = ' noscat'
#             os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam}"+prompt)
#             im = readImage('image.out')
#             os.system('mv image.out image.img')
            
#         elif precomputed_data is not None:
#             im = readImage(precomputed_data)
#         freq0 = im.freq[nlam//2]
#         v = cc / 1e5 * (freq0 - im.freq) / freq0
        
#         os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} lambdarange {im.wav[0]} {im.wav[-1]} nlam {nlam} noscat noline")
#         os.system('mv image.out image_dust.out')
#         im_dust = readImage('image_dust.out')  # This is to plot the contour of dust continuum
#         data = im.imageJyppix/(140*140)*1000  # unit : mJy
#         dust_conti = im_dust.imageJyppix/(140*140)*1000  # unit : mJy

#         if convolve is True:
#             convolved_data = np.zeros(shape=data.shape)
#             convolved_conti = np.zeros(shape=dust_conti.shape)
#             for i in range(nlam):
#                 sigma = fwhm / (2*np.sqrt(2*np.log(2)))
#                 convolved_data[:, :, i] = gaussian_filter(data[:, :, i], sigma=sigma)
#                 convolved_conti[:, :, i] = gaussian_filter(dust_conti[:, :, i], sigma=sigma)
#             data = convolved_data
#             dust_conti = convolved_conti
        
#     elif extract_gas is True:
#         if (precomputed_data_gas is None) and (precomputed_data_dust is None):
#             os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms {vkm} widthkms {v_width} linenlam {nlam} nphot_scat 1000000")
#             os.system('mv image.out image_gas.img')
#             im_gas = readImage('image_gas.img')
#             os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} lambdarange {im_gas.wav[0]} {im_gas.wav[-1]} nlam {nlam} nphot_scat 1000000 noline")
#             os.system('mv image.out image_dust.img')
#             im_dust = readImage('image_dust.img')
            
#         if (precomputed_data_gas is not None) and (precomputed_data_dust is not None):
#             im_gas = readImage(precomputed_data_gas)
#             im_dust = readImage(precomputed_data_dust)
#         freq0 = im_gas.freq[nlam//2]
#         v = cc / 1e5 * (freq0 - im_gas.freq) / freq0
#         data_gas  = im_gas.imageJyppix/(140*140)*1000
#         dust_conti = im_dust.imageJyppix/(140*140)*1000
#         data = data_gas-dust_conti
#         absorption = np.where(data<0, data, 0)
        
#         if convolve is True:
#             convolved_data = np.zeros(shape=data.shape)
#             convolved_conti = np.zeros(shape=dust_conti.shape)
#             convolved_absorption = np.zeros(shape=absorption.shape)
#             for i in range(nlam):
#                 sigma = fwhm / (2*np.sqrt(2*np.log(2)))
#                 convolved_data[:, :, i] = gaussian_filter(data[:, :, i], sigma=sigma)
#                 convolved_conti[:, :, i] = gaussian_filter(dust_conti[:, :, i], sigma=sigma)
#                 convolved_absorption[:, :, i] = gaussian_filter(absorption[:, :, i], sigma=sigma)
#             data = convolved_data
#             dust_conti = convolved_conti
#             absorption = convolved_absorption
    
#     fig, ax = plt.subplots(2, (nlam // 2) + 1, figsize=(18, 6), sharex=True, sharey=True,
#                 gridspec_kw={'wspace': 0.1, 'hspace': 0.1}, layout="constrained")
    
#     # vmi = np.min(data)
#     vmi = 0
#     vma = np.max(data)
            
#     cm = 'hot'
#     abcm = 'viridis_r'
#     tc = 'w'
#     x = np.linspace(1, npix, npix, endpoint=True)
#     y = np.linspace(1, npix, npix, endpoint=True)
#     X, Y = np.meshgrid(x, y)
#     contour_level = np.linspace(0, np.max(dust_conti), 5)
#     for idx in range(nlam):
#         d = np.transpose(data[:, ::-1, idx])
        
#         if idx == nlam//2:
#             image = ax[0, idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
#             ax[0, idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
#             ax[0, idx].imshow(absorption[:, :, idx], cmap=abcm, alpha=0.5)
#             ax[0, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
            
#             ax[1, idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
#             ax[1, idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
#             ax[1, idx].imshow(absorption[:, :, idx], cmap=abcm, alpha=0.5)
#             ax[1, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
            
#             ax[1, idx].set_xlabel('AU',fontsize=16)
#             if idx == 0:
#                 ax[1, idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
#                 ax[1, idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
#                 ax[1, idx].set_ylabel('AU',fontsize=16)
#         elif idx > nlam//2:
#             ax[1, nlam-1-idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
#             ax[1, nlam-1-idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
#             ax[1, nlam-1-idx].imshow(absorption[:, :, idx], cmap=abcm, alpha=0.5)
#             ax[1, nlam-1-idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
            
#             ax[1, nlam-1-idx].set_xticks([int(npix*0.1), npix//2, int(npix*0.9)])
#             ax[1, nlam-1-idx].set_xticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
#             ax[1, nlam-1-idx].set_xlabel('AU',fontsize=16)
#             if nlam-1-idx == 0:
#                 ax[1, nlam-1-idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
#                 ax[1, nlam-1-idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
#                 ax[1, nlam-1-idx].set_ylabel('AU',fontsize=16)
#         else:
#             ax[0, idx].imshow(d, cmap=cm, vmin=vmi, vmax=vma)
#             ax[0, idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
#             ax[0, idx].imshow(absorption[:, :, idx], cmap=abcm, alpha=0.5)
#             ax[0, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
#             if idx == 0:
#                 ax[0, idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
#                 ax[0, idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
#                 ax[0, idx].set_ylabel('AU',fontsize=16)

#     cbar = fig.colorbar(image, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
#     cbar.set_label('Intensity (mJy/pixel)')
#     if title is not None:
#         ax.set_title(title, fontsize = 16)
#         ax_convolved.set_title(title, fontsize = 16)
        
#     fig.savefig('./figures/'+dir+'/'+fname+'.pdf', transparent=True)
#     fig_convolved.savefig('./figures/'+dir+'/'+fname+f'_convolved_fwhm_{fwhm}au.pdf', transparent=True)
#     plt.close('all')
#     return

###############################################################################
# Compare luminosity
# for idx_l, l in enumerate([0.1, 0.86, 1]):
#     # 0.86 L_sun is the luminosity of CB68
#     problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=14e-7*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#                 pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5, gas_inside_rcb=False, lum=l)
#     plot_gas_channel_maps(extract_gas=True, convolve=False)
#     plt.savefig(f'./figures/compare_luminosity/{l}_Lsun/high_mdot_high_infall.pdf', transparent=True)
#     os.system('mv image_gas.img image_gas_high.img')
#     os.system('mv image_dust.img image_dust_high.img')
#     plt.close()
#     plot_gas_channel_maps(extract_gas=True, convolve=True, fwhm=10, precomputed_data_dust='image_dust_high.img', precomputed_data_gas='image_gas_high.img')
#     plt.savefig(f'./figures/compare_luminosity/{l}_Lsun/high_mdot_high_infall_convolved_fwhm_10.pdf', transparent=True)
#     plt.close()

#     plot_gas_channel_maps(extract_gas=True, convolve=True, fwhm=50, precomputed_data_dust='image_dust_high.img', precomputed_data_gas='image_gas_high.img')
#     plt.savefig(f'./figures/compare_luminosity/{l}_Lsun/high_mdot_high_infall_convolved_fwhm_50.pdf', transparent=True)
#     plt.close()
#     os.system(f'mv image_gas_high.img ./precomputed_data/compare_luminosity/{l}_Lsun/image_gas_high.img')
#     os.system(f'mv image_dust_high.img ./precomputed_data/compare_luminosity/{l}_Lsun/image_dust_high.img')
    
#     problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=0.5, 
#                 pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5, gas_inside_rcb=False, lum=l)
#     plot_gas_channel_maps(extract_gas=True, convolve=False)
#     os.system('mv image_gas.img image_gas_low.img')
#     os.system('mv image_dust.img image_dust_low.img')
#     plt.savefig(f'./figures/compare_luminosity/{l}_Lsun/low_mdot_low_infall.pdf', transparent=True)
#     plt.close()

#     plot_gas_channel_maps(extract_gas=True, convolve=True, fwhm=10, precomputed_data_dust='image_dust_low.img', precomputed_data_gas='image_gas_low.img')
#     plt.savefig(f'./figures/compare_luminosity/{l}_Lsun/low_mdot_low_infall_convolved_fwhm_10.pdf', transparent=True)
#     plt.close()

#     plot_gas_channel_maps(extract_gas=True, convolve=True, fwhm=50, precomputed_data_dust='image_dust_low.img', precomputed_data_gas='image_gas_low.img')
#     plt.savefig(f'./figures/compare_luminosity/{l}_Lsun/low_mdot_low_infall_convolved_fwhm_50.pdf', transparent=True)
#     plt.close()
#     os.system(f'mv image_gas_low.img ./precomputed_data/compare_luminosity/{l}_Lsun/image_gas_low.img')
#     os.system(f'mv image_dust_low.img ./precomputed_data/compare_luminosity/{l}_Lsun/image_dust_low.img')   
# os.system('make cleanall') # image data won't be cleaned.
###############################################################################
# Compare different heating mechanism
# for idx_a, a in enumerate([1, 0.1, 0.01]):
#     for idx_h, h in enumerate(['Irradiation', 'Accretion', 'Combine']):
#         if h == 'Irradiation': 
#             problem_setup(a_max=a*0.1, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=0.5, 
#                         pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=False, Rcb=5, gas_inside_rcb=False)
#         elif h == 'Accretion':
#             problem_setup(a_max=a*0.1, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=0.5, 
#                         pancake=False, mctherm=False, snowline=True, floor=True, kep=True, combine=False, Rcb=5, gas_inside_rcb=False)
#         elif h == 'Combine':
#             problem_setup(a_max=a*0.1, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=0.5, 
#                         pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=True, Rcb=5, gas_inside_rcb=False)
#         plot_gas_channel_maps(extract_gas=True, convolve=False)
#         plt.savefig(f'./figures/compare_heating/amax_{a}/'+h+'.pdf', transparent=True)

#         plt.close()
#         plot_gas_channel_maps(extract_gas=True, convolve=True, fwhm=10, precomputed_data_dust='image_dust.img', precomputed_data_gas='image_gas.img')
#         plt.savefig(f'./figures/compare_heating/amax_{a}/'+h+'_convolved_fwhm_10.pdf', transparent=True)
#         plt.close()
#         plot_gas_channel_maps(extract_gas=True, convolve=True, fwhm=50, precomputed_data_dust='image_dust.img', precomputed_data_gas='image_gas.img')
#         plt.savefig(f'./figures/compare_heating/amax_{a}/'+h+'_convolved_fwhm_50.pdf', transparent=True)
#         plt.close()
#         os.system(f'mv image_gas.img ./precomputed_data/compare_heating/amax_{a}/image_gas_'+h+'.img')
#         os.system(f'mv image_dust.img ./precomputed_data/compare_heating/amax_{a}/image_dust_'+h+'.img')

# os.system('make cleanall') # image data won't be cleaned.
###############################################################################
# Compare infall velocity
# problem_setup(a_max=0.1*0.1, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#               pancake=False, mctherm=False, snowline=True, floor=True, kep=True, combine=False, Rcb=None, gas_inside_rcb=False)
# plot_gas_channel_maps(extract_gas=True, convolve=False, v_width=10, nlam=9)
# plt.savefig('vinfall_1_Accretion.pdf', transparent=True)
# plt.close()

# problem_setup(a_max=0.1*0.1, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=0.5, 
#               pancake=False, mctherm=False, snowline=True, floor=True, kep=True, combine=False, Rcb=None, gas_inside_rcb=False)
# plot_gas_channel_maps(extract_gas=True, convolve=False, v_width=10, nlam=9)
# plt.savefig('vinfall_0.5_Accretion.pdf', transparent=True)
# plt.close()

# problem_setup(a_max=0.1*0.1, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=0.1, 
#               pancake=False, mctherm=False, snowline=True, floor=True, kep=True, combine=False, Rcb=None, gas_inside_rcb=False)
# plot_gas_channel_maps(extract_gas=True, convolve=False, v_width=10, nlam=9)
# plt.savefig('vinfall_0.1_Accretion.pdf', transparent=True)
# plt.close()
###############################################################################

###############################################################################
'''
Plot dust images (under construction)
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