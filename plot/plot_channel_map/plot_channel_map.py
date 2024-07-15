import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from radmc3dPy.image import *
from radmc3dPy.analyze import *
import sys
sys.path.insert(0,'../../')
from disk_model import *
from vertical_profile_class import DiskModel_vertical
# from problem_setup import problem_setup
from radmc_setup import radmc3d_setup

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
                  incl=70, line=240, v_width=5, nlam=11, npix=100, sizeau=100,
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
    
    def pannels(dust_conti, data_to_plot, absorption_data=None, cm='hot', abcm='viridis_r', tc='w'):
        fig, ax = plt.subplots(2, (nlam // 2) + 1, figsize=(18, 6), sharex=True, sharey=True,
                gridspec_kw={'wspace': 0.1, 'hspace': 0.1}, layout="constrained")
        vmin = 0
        vmax = np.max(data_to_plot)
        
        x, y = np.linspace(1, npix, npix, endpoint=True), np.linspace(1, npix, npix, endpoint=True)
        X, Y = np.meshgrid(x, y)
        contour_level = np.linspace(0, np.max(dust_conti), 5)
        
        for idx in range(nlam):
            d = np.transpose(data_to_plot[:, ::-1, idx])
                
            if idx == nlam//2:
                image = ax[0, idx].imshow(d, cmap=cm, vmin=vmin, vmax=vmax)
                ax[0, idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
                if absorption_data is not None:
                    ax[0, idx].imshow(absorption_data[:, :, idx], cmap=abcm, alpha=0.5)
                ax[0, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                    
                ax[1, idx].imshow(d, cmap=cm, vmin=vmin, vmax=vmax)
                ax[1, idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
                if absorption_data is not None:
                    ax[1, idx].imshow(absorption_data[:, :, idx], cmap=abcm, alpha=0.5)
                ax[1, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                    
                ax[1, idx].set_xlabel('AU',fontsize=16)
                if idx == 0:
                    ax[1, idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
                    ax[1, idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
                    ax[1, idx].set_ylabel('AU',fontsize=16)
            elif idx > nlam//2:
                ax[1, nlam-1-idx].imshow(d, cmap=cm, vmin=vmin, vmax=vmax)
                ax[1, nlam-1-idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
                if absorption_data is not None:
                    ax[1, nlam-1-idx].imshow(absorption_data[:, :, idx], cmap=abcm, alpha=0.5)
                ax[1, nlam-1-idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                    
                ax[1, nlam-1-idx].set_xticks([int(npix*0.1), npix//2, int(npix*0.9)])
                ax[1, nlam-1-idx].set_xticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
                ax[1, nlam-1-idx].set_xlabel('AU',fontsize=16)
                if nlam-1-idx == 0:
                    ax[1, nlam-1-idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
                    ax[1, nlam-1-idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
                    ax[1, nlam-1-idx].set_ylabel('AU',fontsize=16)
            else:
                ax[0, idx].imshow(d, cmap=cm, vmin=vmin, vmax=vmax)
                ax[0, idx].contour(Y, X, dust_conti[:, :, idx], levels=contour_level, colors='w', linewidths=1)
                if absorption_data is not None:
                    ax[0, idx].imshow(absorption_data[:, :, idx], cmap=abcm, alpha=0.5)
                ax[0, idx].text(int(npix*0.9),int(npix*0.1),f'{v[idx]:.1f} $km/s$', ha='right', va='top', color=tc, fontsize=16)
                if idx == 0:
                    ax[0, idx].set_yticks([int(npix*0.1), npix//2, int(npix*0.9)])
                    ax[0, idx].set_yticklabels([f'-{int((sizeau//2)*0.8)}', '0', f'{int((sizeau//2)*0.8)}'], fontsize=14)
        return fig, ax, image
    
    
    if (cube is not None) and (cube_dust is not None):
        if precomputed is True:
            cube = './precomputed_data/'+dir+'/'+cube
            cube_dust = './precomputed_data/'+dir+'/'+cube_dust
        im = readImage(cube)
        im_dust = readImage(cube_dust)  # to plot dust contours
        if precomputed is False:
            os.system('mv '+cube+' ./precomputed_data/'+dir+'/'+cube)
        data = im.imageJyppix*1e3/(140**2) # mJy/pix
        data_dust = im_dust.imageJyppix*1e3/(140**2)
        
        sizeau = int(round((im.x/au)[-1]))
        npix=im.nx
        nlam=len(im.wav)
        freq0 = (im.freq[nlam//2] + im.freq[(nlam//2)-1])/2
        v = cc / 1e5 * (freq0 - im.freq) / freq0
        
        fig, ax, image = pannels(dust_conti=data_dust, data_to_plot=data, absorption_data=None)
        cbar = fig.colorbar(image, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Intensity (mJy/pixel)')
        
        if title is not None:
            ax.set_title(title, fontsize = 16) 
        fig.savefig('./figures/'+dir+'/'+fname+'.pdf', transparent=True)
        plt.close('all')
        
        if convolve is True:
            convolved_data = np.zeros(shape=data.shape)
            convolved_conti = np.zeros(shape=data_dust.shape)
            for i in range(nlam):
                sigma = fwhm / (2*np.sqrt(2*np.log(2)))
                convolved_data[:, :, i] = gaussian_filter(data[:, :, i], sigma=sigma)
                convolved_conti[:, :, i] = gaussian_filter(data_dust[:, :, i], sigma=sigma)
            fig, ax, image = pannels(dust_conti=convolved_conti, data_to_plot=convolved_data, absorption_data=None)
            cbar = fig.colorbar(image, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
            cbar.set_label('Intensity (mJy/pixel)')
            if title is not None:
                ax.set_title(title, fontsize = 16) 
            fig.savefig('./figures/'+dir+'/'+fname+f'_convolved_fwhm_{fwhm}.pdf', transparent=True)
            plt.close('all')    
        
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


        fig, ax, image = pannels(dust_conti=data_dust, data_to_plot=data, absorption_data=absorption)
        cbar = fig.colorbar(image, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Intensity (mJy/pixel)')
        if title is not None:
            ax.set_title(title, fontsize = 16) 
        fig.savefig('./figures/'+dir+'/'+fname+'.pdf', transparent=True)
        plt.close('all')
        if convolve is True:
            convolved_data = np.zeros(shape=data.shape)
            convolved_conti = np.zeros(shape=data_dust.shape)
            convolved_absorption = np.zeros(shape=absorption.shape)
            for i in range(nlam):
                sigma = fwhm / (2*np.sqrt(2*np.log(2)))
                convolved_data[:, :, i] = gaussian_filter(data[:, :, i], sigma=sigma)
                convolved_conti[:, :, i] = gaussian_filter(data_dust[:, :, i], sigma=sigma)
                convolved_absorption[:, :, i] = gaussian_filter(absorption[:, :, i], sigma=sigma)
            fig, ax, image = pannels(dust_conti=convolved_conti, data_to_plot=convolved_data, absorption_data=convolved_absorption)
            cbar = fig.colorbar(image, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
            cbar.set_label('Intensity (mJy/pixel)')
            if title is not None:
                ax.set_title(title, fontsize = 16) 
            fig.savefig('./figures/'+dir+'/'+fname+f'_convolved_fwhm_{fwhm}.pdf', transparent=True)
            plt.close('all')   
        return  
    else:
        print('No correct cube is given.')
        return
###############################################################################
model = radmc3d_setup(silent=False)
model.get_mastercontrol(filename=None,
                        comment=None,
                        incl_dust=1,
                        incl_lines=1,
                        nphot=1000000,
                        nphot_scat=1000000,
                        scattering_mode_max=2,
                        istar_sphere=1,
                        num_cpu=None)
model.get_linecontrol(filename=None,
                      methanol='ch3oh leiden 0 0 0')
model.get_continuumlambda(filename=None,
                          comment=None,
                          lambda_micron=None,
                          append=False)
model.get_diskcontrol(a_max=0.1, 
                        Mass_of_star=0.14, 
                        Accretion_rate=5e-7,
                        Radius_of_disk=50,
                        NR=200,
                        NTheta=200,
                        NPhi=10,
                        disk_boundary=1e-18)
model.get_vfieldcontrol(Kep=True,
                        vinfall=0.5,
                        Rcb=None)
model.get_heatcontrol(heat='accretion')
model.get_gasdensitycontrol(abundance=1e-10,
                            snowline=None,
                            enhancement=1e5,
                            gas_inside_rcb=True)
generate_cube(extract_gas=True, v_width=5, nlam=11, sizeau=50)
plot_channel(precomputed=False, cube_dust='image_dust.img', cube_gas='image_gas.img', dir='test', fname='test', vkm=0)
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
# model = radmc3d_setup(silent=False)
# model.get_mastercontrol(filename=None,
#                         comment=None,
#                         incl_dust=1,
#                         incl_lines=1,
#                         nphot=1000000,
#                         nphot_scat=1000000,
#                         scattering_mode_max=2,
#                         istar_sphere=1,
#                         num_cpu=1)
# model.get_linecontrol(filename='./figures/compare_rcb/compare_rcb_line.inp',
#                       methanol='ch3oh leiden 0 0 0')
# model.get_diskcontrol(a_max=0.1, 
#                         Mass_of_star=0.14, 
#                         Accretion_rate=5*1e-7,
#                         Radius_of_disk=50,
#                         NR=200,
#                         NTheta=200,
#                         NPhi=10,
#                         disk_boundary=1e-18)
# model.get_vfieldcontrol(Kep=True,
#                         vinfall=0.5,
#                         Rcb=None)
# model.get_heatcontrol(heat='accretion')
# model.get_gasdensitycontrol(abundance=1e-10,
#                             snowline=None,
#                             enhancement=1e5,
#                             gas_inside_rcb=True)

# plot_dust(incl=[45, 60])
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