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
###############################################################################
"""
CB68
Mass          : 0.08-0.30 Msun
Accretion rate: 4-7e-7    Msun/yr
Radius        : 20-40     au
Distance      : 140       pc
"""
###############################################################################
# from spectral_cube import SpectralCube
# from pvextractor import extract_pv_slice, Path
# cube = SpectralCube.read(
#     '/run/media/hyp0515/storage/CB68_setup1/CB68-Setup1-cube-products/CB68_218.440GHz_CH3OH_joint_0.5_clean.image.pbcor.common.fits'
#     )
# freq0 = 218.440063 * 1e9
# v = cc / 1e5 * (freq0 - cube.spectral_axis.value) / freq0
# path = Path([(793, 706), (710, 789)])
# pvdiagram = extract_pv_slice(cube=cube, path=path, spacing=1)
# v_axis = v[207:267]
# offset = np.linspace(-150, 150, 50, endpoint=True)
# O, V = np.meshgrid(offset, v_axis)
# contour_levels = np.linspace(0.01, pvdiagram.data[207:267, 36:86].max(), 4)
###############################################################################
CB68_PV = np.load('../../CB68_PV.npy')
'''
Shape of 'CB68_PV' is (60, 50)
The first dimension is velocity channel (0 - 10km/s)
The second dimension is offest (-150 - 150 AU)
'''
v_axis = np.linspace(0, 10, 60, endpoint=True)
offset = np.linspace(-150, 150, 50, endpoint=True)
O, V = np.meshgrid(offset, v_axis)
# contour_levels = np.linspace(0.01, CB68_PV.max(), 4)
###############################################################################
"""
Plot Position-velocity (PV) diagrams ('problem_setup' is required before plot_pv)
"""
def plot_pv(incl=70, line=240, vkm=0, v_width=20, nlam=50,
            nodust=False, scat=True, extract_gas=True, npix=30, sizeau=100,
            convolve=True, fwhm=50,
            precomputed_data_gas=None, precomputed_data_dust=None, precomputed_data=None):
    """
    incl               : Inclination angle of the disk
    line               : Transistion level (see 'molecule_ch3oh.inp')
    vkm                : Systematic velocity
    v_width            : Range of velocity to simulate
    nlam               : Number of velocities
    nodust             : If False, dust effect is included
    scat               : If True and nodust=False, scattering is included. (Time-consuming)
    extracted_gas      : If True, spectral line is extracted (I_{dust+gas}-I{dust})
    npix               : Number of map's pixels
    sizeau             : Map's span
    convolve + fwhm    : Generate two images, with and without convolution
                         (The unit of fwhm is pixel)
    precomputed_data_* : File's name if images have been generated beforehand
        'precomputed_data' is for extract_gas=False
        'precomputed_data_gas' and 'precomputed_data_dust' is for extract_gas=True
    """
    if extract_gas is False:
        if precomputed_data is None:
            if nodust is True:
                prompt = ' noscat nodust'
            elif nodust is False:
                if scat is True:
                    prompt = ' nphot_scat 1000000'
                elif scat is False:
                    prompt = ' noscat'
            
            os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms 0 widthkms {v_width} linenlam {nlam}"+prompt)
            im = readImage('image.out')
            os.system('mv image.out image.img')
        elif precomputed_data is not None:
            im = readImage(precomputed_data)
            
        freq0 = (im.freq[nlam//2] + im.freq[(nlam//2)-1])/2
        v = cc / 1e5 * (freq0 - im.freq) / freq0
        center = int(len(im.y)//2)
        data = im.imageJyppix*1e3/(140**2)
        
    elif extract_gas is True:
        if (precomputed_data_gas is None) and (precomputed_data_dust is None):
            os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms 0 widthkms {v_width} linenlam {nlam} nphot_scat 1000000")
            os.system('mv image.out image_gas.img')
            im = readImage('image_gas.img')
            os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} lambdarange {im.wav[0]} {im.wav[-1]} nlam {nlam} nphot_scat 1000000 noline")
            os.system('mv image.out image_dust.img')
            im_dust = readImage('image_dust.img')
            
        if (precomputed_data_gas is not None) and (precomputed_data_dust is not None):
            im = readImage(precomputed_data_gas)
            im_dust = readImage(precomputed_data_dust)
        freq0 = (im.freq[nlam//2] + im.freq[(nlam//2)-1])/2
        v = cc / 1e5 * (freq0 - im.freq) / freq0
        center = int(len(im.y)//2)

        data_gas  = im.imageJyppix
        data_dust = im_dust.imageJyppix
        data = (data_gas-data_dust)*1e3/(140**2)
        
    if convolve is not True: 
        fig, ax = plt.subplots()
        
        c = ax.pcolormesh(im.x/au, v+vkm, data[:, center, :].T, shading="nearest", rasterized=True, cmap='jet', vmin=0., vmax=0.4)
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('mJy/pixel',fontsize = 16)
        ax.set_xlabel("Offset [au]",fontsize = 16)
        ax.set_ylabel("Velocity [km/s]",fontsize = 16)
        ax.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax.plot([-(sizeau//2), (sizeau//2)], [vkm, vkm], 'w:')
        contour_levels = np.linspace(0.01, CB68_PV.max(), 4)
        contour = ax.contour(O, V, CB68_PV[:, ::-1], levels=contour_levels, colors='k', linewidths=1)
        return fig, ax
    elif convolve is True:
        """
        Save the images without convolution
        """
        fig, ax = plt.subplots()
        c = ax.pcolormesh(im.x/au, v+vkm, data[:, center, :].T, shading="nearest", rasterized=True, cmap='jet', vmin=0., vmax=0.4)
        
        absorption = np.where(data[:, center, :].T<0, data[:, center, :].T, 0)
        contour_levels = np.linspace(np.min(absorption), 0, 3)
        x = np.linspace(-sizeau//2, sizeau//2, npix, endpoint=True)
        y = np.linspace(-v_width+vkm, v_width+vkm, nlam, endpoint=True)
        X, Y = np.meshgrid(x, y)
        absorption_contours = ax.contour(X, Y, absorption, levels=contour_levels, colors='w', linewidths=1)
        
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('mJy/pixel',fontsize = 16)
        ax.set_xlabel("Offset [au]",fontsize = 16)
        ax.set_ylabel("Velocity [km/s]",fontsize = 16)
        ax.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax.plot([-(sizeau//2), (sizeau//2)], [vkm, vkm], 'w:')
        contour_levels = np.linspace(0.01, CB68_PV.max(), 4)
        contour = ax.contour(O, V, CB68_PV[:, ::-1], levels=contour_levels, colors='k', linewidths=1)
        """
        Save the images with convolution
        """
        fig_convolved, ax_convolved = plt.subplots()
        convolved_image = np.zeros(shape=data.shape)
        for i in range(nlam):
            FWHM = fwhm
            sigma = FWHM / (2*np.sqrt(2*np.log(2)))
            convolved_image[:, :, i] = gaussian_filter(data[:, :, i], sigma=sigma)
        convolved_model = (convolved_image[:, center, :].T)*1e3/(140**2)
        c = ax_convolved.pcolormesh(im.x/au, v+vkm, convolved_model, shading="nearest", rasterized=True, cmap='jet')
        cbar = fig_convolved.colorbar(c, ax=ax_convolved)
        cbar.set_label('mJy/pixel',fontsize = 16)
        ax_convolved.set_xlabel("Offset [au]",fontsize = 16)
        ax_convolved.set_ylabel("Velocity [km/s]",fontsize = 16)
        ax_convolved.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax_convolved.plot([-(sizeau//2), (sizeau//2)], [vkm, vkm], 'w:')
        contour_levels = np.linspace(0.01, CB68_PV.max(), 4)
        contour = ax.contour(O, V, CB68_PV[:, ::-1], levels=contour_levels, colors='k', linewidths=1)
        return fig, ax, fig_convolved, ax_convolved
        

###############################################################################
"""
This generates various plots if 'heat_list', 'snowline', and 'dust' are given. 
('problem_setup' is built-in)
"""
# This generates various plots if 'heat_list', 'snowline', and 'dust' are given.
def multiple_plots(amax, rcb, nlam, npix, sizeau, v0=0, vwidth=5, gas_inside=True, vinfall=1, convolve=True, fwhm=50, filename=None,
                   precomputed_data_gas=None, precomputed_data_dust=None, precomputed_data=None):
    """
    amax     : maximum grain size
    rcb      : centrifugal barrier
    v0       : vkm
    vinfall  : infall velocity in terms of Keplerian velocity
    filename : save plots
    """
    # The following codes are kind of redundant. Modification is required.
    for idx_h, heat in enumerate(heat_list):
        for idx_s, snow in enumerate(snowline):
            if heat == 'Accretion' and snow =='w/o snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
                            pancake=False, mctherm=False, snowline=False, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=gas_inside)
                t = 'Accretion + w/o snowline'
                f = 'Accretion + wo snowline'
            elif heat == 'Accretion' and snow =='w/ snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
                            pancake=False, mctherm=False, snowline=True, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=gas_inside)
                t = 'Accretion + w/ snowline'
                f = 'Accretion + w snowline'
            elif heat == 'Irradiation' and snow =='w/o snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
                            pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=gas_inside)
                t = 'Irradiation + w/o snowline'
                f = 'Irradiation + wo snowline'
            elif heat == 'Irradiation' and snow =='w/ snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
                            pancake=False, mctherm=True, snowline=True, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=gas_inside)
                t = 'Irradiation + w/ snowline'
                f = 'Irradiation + w snowline'
            elif heat == 'Combine' and snow =='w/o snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
                            pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=rcb, combine=True, gas_inside_rcb=gas_inside)
                t = 'Combine + w/o snowline'
                f = 'Combine + wo snowline'
            elif heat == 'Combine' and snow =='w/ snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
                            pancake=False, mctherm=True, snowline=True, floor=True, kep=True, Rcb=rcb, combine=True, gas_inside_rcb=gas_inside)
                t = 'Combine + w/ snowline'
                f = 'Combine + w snowline'        

            for idx_d, d in enumerate(dust):
                if convolve is not True:
                    if d == 'w/o dust':
                        fig, ax = plot_pv(
                            incl=70, vkm=v0, v_width=vwidth, nlam=nlam, nodust=True, npix=npix, sizeau=sizeau,
                            convolve=convolve, fwhm=fwhm,
                            precomputed_data=precomputed_data)
                        title = t + ' + w/o dust'
                        fname = f + ' + wo dust'
                    elif d == 'w/ dust':
                        fig, ax = plot_pv(
                            incl=70, vkm=v0, v_width=vwidth, nlam=nlam, extract_gas=True, npix=npix, sizeau=sizeau,
                            convolve=convolve, fwhm=fwhm,
                            precomputed_data_gas=precomputed_data_gas, precomputed_data_dust=precomputed_data_dust)
                        title = t + ' + w/ dust'
                        fname = f + ' + w dust'
                    ax.set_title(title, fontsize = 16)
                    if filename is None:
                        fig.savefig(f'./figures/high_resol/gas_inside_rcb_{gas_inside}/amax_{amax}/Rcb_{rcb}/'+fname+'.pdf', transparent=True)
                        plt.close('all')
                    elif filename is not None:
                        fig.savefig(filename+'.pdf', transparent=True)
                        plt.close('all')
                elif convolve is True:
                    if d == 'w/o dust':
                        fig, ax, fig_convolved, ax_convolved = plot_pv(
                            incl=70, vkm=v0, v_width=vwidth, nlam=nlam, nodust=True, npix=npix, sizeau=sizeau,
                            convolve=convolve, fwhm=fwhm,
                            precomputed_data=precomputed_data)
                        title = t + ' + w/o dust'
                        fname = f + ' + wo dust'
                    elif d == 'w/ dust':
                        fig, ax, fig_convolved, ax_convolved = plot_pv(
                            incl=70, vkm=v0, v_width=vwidth, nlam=nlam, extract_gas=True, npix=npix, sizeau=sizeau,
                            convolve=convolve, fwhm=fwhm,
                            precomputed_data_gas=precomputed_data_gas, precomputed_data_dust=precomputed_data_dust)
                        title = t + ' + w/ dust'
                        fname = f + ' + w dust'
                    ax.set_title(title, fontsize = 16)
                    ax_convolved.set_title(title, fontsize = 16)
                    if filename is None:
                        fig.savefig(f'./figures/high_resol/gas_inside_rcb_{gas_inside}/amax_{amax}/Rcb_{rcb}/'+fname+'.pdf', transparent=True)
                        fig_convolved.savefig(f'./figures/high_resol/gas_inside_rcb_{gas_inside}/amax_{amax}/Rcb_{rcb}/'+fname+f'_convolved_fwhm_{fwhm}au.pdf', transparent=True)
                        plt.close('all')
                    elif filename is not None:
                        fig.savefig(filename+fname+'.pdf', transparent=True)
                        fig_convolved.savefig(filename+fname+f'_convolved_fwhm_{fwhm}au.pdf', transparent=True)
                        plt.close('all')
        if (precomputed_data_gas is None) and (precomputed_data_dust is None):
            os.system(f'mv image_gas.img ./precomputed_data/amax_{a}/image_gas_{heat}.img')
            os.system(f'mv image_dust.img ./precomputed_data/amax_{a}/image_dust_{heat}.img')
            
    return
# ###############################################################################
# heat_list = ['Irradiation', 'Combine', 'Accretion']
# snowline = ['w/ snowline', 'w/o snowline']
# dust = ['w/o dust', 'w/ dust']
heat_list = ['Irradiation', 'Combine', 'Accretion']
snowline = ['w/o snowline']
dust = ['w/ dust']
for _, a in enumerate([0.01, 0.03, 0.05,0.1, 0.3, 0.5, 1, 3, 10]):
    multiple_plots(amax=a*0.1, rcb=5, nlam=100, npix=300, sizeau=300, v0=5, vwidth=10, gas_inside=False, convolve=True, fwhm=50, vinfall=1,
                filename=f"./figures/colorbar_rescaled/amax_{a}/",
                precomputed_data_gas=None, precomputed_data_dust=None)

# for _, a in enumerate([0.01, 0.03, 0.05,0.1, 0.3, 0.5, 1, 3, 10]):
#     heat_list = ['Accretion']
#     multiple_plots(amax=0.01*0.1, rcb=5, nlam=100, npix=300, sizeau=300, v0=5, vwidth=10, gas_inside=False, convolve=True, fwhm=50, vinfall=1,
#                 filename=f"./figures/colorbar_rescaled/amax_{a}/",
#                 precomputed_data_gas=f"./precomputed_data/amax_{a}/image_gas_Accretion.img",
#                 precomputed_data_dust=f"./precomputed_data/amax_{a}/image_dust_Accretion.img")
#     heat_list = ['Irradiation']
#     multiple_plots(amax=0.01*0.1, rcb=5, nlam=100, npix=300, sizeau=300, v0=5, vwidth=10, gas_inside=False, convolve=True, fwhm=50, vinfall=1,
#                 filename=f"./figures/colorbar_rescaled/amax_{a}/",
#                 precomputed_data_gas=f"./precomputed_data/amax_{a}/image_gas_Irradiation.img",
#                 precomputed_data_dust=f"./precomputed_data/amax_{a}/image_dust_Irradiation.img")
#     heat_list = ['Combine']
#     multiple_plots(amax=0.01*0.1, rcb=5, nlam=100, npix=300, sizeau=300, v0=5, vwidth=10, gas_inside=False, convolve=True, fwhm=50, vinfall=1,
#                 filename=f"./figures/colorbar_rescaled/amax_{a}/",
#                 precomputed_data_gas=f"./precomputed_data/amax_{a}/image_gas_Combine.img",
#                 precomputed_data_dust=f"./precomputed_data/amax_{a}/image_dust_Combine.img")

