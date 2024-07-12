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
# CB68_PV = np.load('../../CB68_PV.npy')
'''
Shape of 'CB68_PV' is (60, 50)
The first dimension is velocity channel (0 - 10km/s)
The second dimension is offest (-150 - 150 AU)
'''
# v_axis = np.linspace(0, 10, 60, endpoint=True)
# offset = np.linspace(-150, 150, 50, endpoint=True)
# O, V = np.meshgrid(offset, v_axis)
# contour_levels = np.linspace(0.01, CB68_PV.max(), 4)
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
        
        return im_gas, im_dust
###############################################################################
"""
Plot Position-velocity (PV) diagrams
"""
def plot_pv(dir=None, precomputed=False,
            cube=None, cube_gas=None, cube_dust=None,
            vkm=5,
            convolve=True, fwhm=50, fname=None, title=None,
            CB68=False):
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
        
    elif (cube_dust is not None) and (cube_gas is not None):
        if precomputed is True:
            cube_gas = './precomputed_data/'+dir+'/'+cube_gas
            cube_dust = './precomputed_data/'+dir+'/'+cube_dust
        im = readImage(cube_gas)
        im_dust = readImage(cube_dust)
        data_gas  = im.imageJyppix
        data_dust = im_dust.imageJyppix
        data = (data_gas-data_dust)*1e3/(140**2) # mJy/pix
        if precomputed is False:
            os.system('mv '+cube_gas+' ./precomputed_data/'+dir+'/'+cube_gas)
            os.system('mv '+cube_dust+' ./precomputed_data/'+dir+'/'+cube_dust)
    else:
        print('No correct cube is given.')
        return
    
    sizeau = int(round((im.x/au)[-1]))*2
    npix=im.nx
    nlam=len(im.wav)
    freq0 = (im.freq[nlam//2] + im.freq[(nlam//2)-1])/2
    v = cc / 1e5 * (freq0 - im.freq) / freq0
    v_width = round(v[-1]-v[0])//2
    center = int(im.ny//2)

    if convolve is not True: 
        fig, ax = plt.subplots()
        pv_slice = np.sum(data[:, center-fwhm//2:center+fwhm//2, :], axis=1)
        c = ax.pcolormesh(im.x/au, v+vkm, pv_slice.T, shading="nearest", rasterized=True, cmap='jet', vmin=0., vmax=3)
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('mJy/pixel',fontsize = 16)
        ax.set_xlabel("Offset [au]",fontsize = 16)
        ax.set_ylabel("Velocity [km/s]",fontsize = 16)
        ax.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax.plot([-(sizeau//2), (sizeau//2)], [vkm, vkm], 'w:')
        # contour_levels = np.linspace(0.01, CB68_PV.max(), 4)
        # contour = ax.contour(O, V, CB68_PV[::-1, :], levels=contour_levels, colors='w', linewidths=1)
        if title is not None:
            ax.set_title(title, fontsize = 16)
        fig.savefig('./figures/'+dir+'/'+fname+'.pdf', transparent=True)
        plt.close('all')
        return 
    elif convolve is True:
        """
        Save the images without convolution
        """
        fig, ax = plt.subplots()
        pv_slice = np.sum(data[:, center-fwhm//2:center+fwhm//2, :], axis=1)
        c = ax.pcolormesh(im.x/au, v+vkm, pv_slice.T, shading="nearest", rasterized=True, cmap='jet', vmin=0., vmax=3)
        
        absorption = np.where(pv_slice.T<0, pv_slice.T, 0)
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
        # contour_levels = np.linspace(0.01, CB68_PV.max(), 4)
        # contour = ax.contour(O, V, CB68_PV[::-1, :], levels=contour_levels, colors='w', linewidths=1)
                    
        """
        Save the images with convolution
        """
        fig_convolved, ax_convolved = plt.subplots()
        convolved_image = np.zeros(shape=data.shape)
        for i in range(nlam):
            sigma = fwhm / (2*np.sqrt(2*np.log(2)))
            convolved_image[:, :, i] = gaussian_filter(data[:, :, i], sigma=sigma)
        convolved_model = (convolved_image[:, center, :].T)*1e3/(140**2)
        c = ax_convolved.pcolormesh(im.x/au, v+vkm, convolved_model, shading="nearest", rasterized=True, cmap='jet')
        cbar = fig_convolved.colorbar(c, ax=ax_convolved)
        cbar.set_label('mJy/pixel',fontsize = 16)
        ax_convolved.set_xlabel("Offset [au]",fontsize = 16)
        ax_convolved.set_ylabel("Velocity [km/s]",fontsize = 16)
        ax_convolved.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax_convolved.plot([-(sizeau//2), (sizeau//2)], [vkm, vkm], 'w:')
        # contour_levels = np.linspace(0.01, CB68_PV.max(), 4)
        # contour = ax_convolved.contour(O, V, CB68_PV[::-1, :], levels=contour_levels, colors='w', linewidths=1)
        if title is not None:
            ax.set_title(title, fontsize = 16)
            ax_convolved.set_title(title, fontsize = 16)
        
        fig.savefig('./figures/'+dir+'/'+fname+'.pdf', transparent=True)
        fig_convolved.savefig('./figures/'+dir+'/'+fname+f'_convolved_fwhm_{fwhm}au.pdf', transparent=True)
        plt.close('all')
        return 
###############################################################################
d = 'compare_rcb'
os.makedirs('./figures/'+d, exist_ok=True)
model = radmc3d_setup(silent=False)
model.get_mastercontrol(filename='./figures/compare_rcb/compare_rcb.inp',
                        comment='compare_rcb',
                        incl_dust=1,
                        incl_lines=1,
                        nphot=1000000,
                        nphot_scat=1000000,
                        scattering_mode_max=2,
                        istar_sphere=1,
                        num_cpu=None)
model.get_linecontrol(filename='./figures/compare_rcb/compare_rcb_line.inp',
                      methanol='ch3oh leiden 0 0 0')
# model.get_diskcontrol(a_max=0.1, 
#                       Mass_of_star=0.14, 
#                       Accretion_rate=5e-7,
#                       Radius_of_disk=50,
#                       NR=200,
#                       NTheta=200,
#                       NPhi=10,
#                       disk_boundary=1e-18)
# model.get_vfieldcontrol(Kep=True,
#                         vinfall=0.5,
#                         Rcb=5)
# model.get_heatcontrol(L_star=0.86,
#                       heat = 'accretion')
# model.get_gasdensitycontrol(abundance=1e-10,
#                             snowline=100,
#                             enhancement=1e5,
#                             gas_inside_rcb=True)

for idx_a, a in enumerate([0.01, 0.03, 0.05,0.1, 0.3, 0.5, 1, 3, 10]):
    for idx_r, r in enumerate([None, 5, 10, 20, 30]):
        for idx_mdot, mdot in enumerate([5, 10, 15]):
            model.get_diskcontrol(a_max=a, 
                        Mass_of_star=0.14, 
                        Accretion_rate=mdot*1e-7,
                        Radius_of_disk=50,
                        NR=200,
                        NTheta=200,
                        NPhi=10,
                        disk_boundary=1e-18)
            model.get_vfieldcontrol(Kep=True,
                                vinfall=0.5,
                                    Rcb=r)
            for idx_s, snow in enumerate([100, None]):
                if r == None:
                    for idx_h, mechanism in enumerate(['Accretion', 'Irradiation', 'Combine']):
                        model.get_heatcontrol(heat=mechanism)
                        model.get_gasdensitycontrol(abundance=1e-10,
                                        snowline=snow,
                                        enhancement=1e5,
                                        gas_inside_rcb=True)
                        if snow is not None:
                            generate_cube(fname=mechanism+f'_snowline_True', v_width=10, nlam=50, npix=200, sizeau=200)
                            plot_pv(cube_dust='image_dust_'+mechanism+f'_snowline_True'+'.img',
                                    cube_gas='image_gas_'+mechanism+f'_snowline_True'+'.img',
                                    dir=d+f'/rcb_{r}/mdot_{mdot}/amax_{a}',
                                    fname=mechanism+f'_snowline_True',
                                    title=mechanism+"+ w/ snowline + w/ dust")
                        elif snow is None:
                            generate_cube(fname=mechanism+f'_snowline_{snow}', v_width=10, nlam=50, npix=200, sizeau=200)
                            plot_pv(cube_dust='image_dust_'+mechanism+f'_snowline_{snow}'+'.img',
                                    cube_gas='image_gas_'+mechanism+f'_snowline_{snow}'+'.img',
                                    dir=d+f'/rcb_{r}/mdot_{mdot}/amax_{a}',
                                    fname=mechanism+f'_snowline_{snow}',
                                    title=mechanism+"+ w/o snowline + w/ dust")
                else:
                    for _, gas_rcb in enumerate([True, False]):
                        for idx_h, mechanism in enumerate(['Accretion', 'Irradiation', 'Combine']):
                            model.get_heatcontrol(heat=mechanism)
                            model.get_gasdensitycontrol(abundance=1e-10,
                                        snowline=snow,
                                        enhancement=1e5,
                                        gas_inside_rcb=gas_rcb)
                            if snow is not None:
                                generate_cube(fname=mechanism+f'_gas_inside_{gas_rcb}_snowline_True', v_width=10, nlam=50, npix=200, sizeau=200)
                                plot_pv(cube_dust='image_dust_'+mechanism+f'_gas_inside_{gas_rcb}_snowline_True'+'.img',
                                        cube_gas='image_gas_'+mechanism+f'_gas_inside_{gas_rcb}_snowline_True'+'.img',
                                        dir=d+f'/rcb_{r}/mdot_{mdot}/amax_{a}',
                                        fname=mechanism+f'_gas_inside_{gas_rcb}_snowline_True',
                                        title=mechanism+"+ w/ snowline + w/ dust")
                            if snow is None:
                                generate_cube(fname=mechanism+f'_gas_inside_{gas_rcb}_snowline_{snow}', v_width=10, nlam=50, npix=200, sizeau=200)
                                plot_pv(cube_dust='image_dust_'+mechanism+f'_gas_inside_{gas_rcb}_snowline_{snow}'+'.img',
                                        cube_gas='image_gas_'+mechanism+f'_gas_inside_{gas_rcb}_snowline_{snow}'+'.img',
                                        dir=d+f'/rcb_{r}/mdot_{mdot}/amax_{a}',
                                        fname=mechanism+f'_gas_inside_{gas_rcb}_snowline_{snow}',
                                        title=mechanism+"+ w/o snowline + w/ dust")

###############################################################################
"""
This generates various plots if 'heat_list', 'snowline', and 'dust' are given. 
('problem_setup' is built-in)
"""
# This generates various plots if 'heat_list', 'snowline', and 'dust' are given.
# def multiple_plots(amax, rcb, nlam, npix, sizeau, v0=0, vwidth=5, gas_inside=True, vinfall=1, convolve=True, fwhm=50, filename=None,
#                    precomputed_data_gas=None, precomputed_data_dust=None, precomputed_data=None):
#     """
#     amax     : maximum grain size
#     rcb      : centrifugal barrier
#     v0       : vkm
#     vinfall  : infall velocity in terms of Keplerian velocity
#     filename : save plots
#     """
#     # The following codes are kind of redundant. Modification is required.
#     for idx_h, heat in enumerate(heat_list):
#         for idx_s, snow in enumerate(snowline):
#             if heat == 'Accretion' and snow =='w/o snowline':
#                 problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
#                             pancake=False, mctherm=False, snowline=False, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=gas_inside)
#                 t = 'Accretion + w/o snowline'
#                 f = 'Accretion + wo snowline'
#             elif heat == 'Accretion' and snow =='w/ snowline':
#                 problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
#                             pancake=False, mctherm=False, snowline=True, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=gas_inside)
#                 t = 'Accretion + w/ snowline'
#                 f = 'Accretion + w snowline'
#             elif heat == 'Irradiation' and snow =='w/o snowline':
#                 problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
#                             pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=gas_inside)
#                 t = 'Irradiation + w/o snowline'
#                 f = 'Irradiation + wo snowline'
#             elif heat == 'Irradiation' and snow =='w/ snowline':
#                 problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
#                             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=gas_inside)
#                 t = 'Irradiation + w/ snowline'
#                 f = 'Irradiation + w snowline'
#             elif heat == 'Combine' and snow =='w/o snowline':
#                 problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
#                             pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=rcb, combine=True, gas_inside_rcb=gas_inside)
#                 t = 'Combine + w/o snowline'
#                 f = 'Combine + wo snowline'
#             elif heat == 'Combine' and snow =='w/ snowline':
#                 problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=5e-7*Msun/yr, Radius_of_disk=100*au, v_infall=vinfall, 
#                             pancake=False, mctherm=True, snowline=True, floor=True, kep=True, Rcb=rcb, combine=True, gas_inside_rcb=gas_inside)
#                 t = 'Combine + w/ snowline'
#                 f = 'Combine + w snowline'        

#             for idx_d, d in enumerate(dust):
#                 if convolve is not True:
#                     if d == 'w/o dust':
#                         fig, ax = plot_pv(
#                             incl=70, vkm=v0, v_width=vwidth, nlam=nlam, nodust=True, npix=npix, sizeau=sizeau,
#                             convolve=convolve, fwhm=fwhm,
#                             precomputed_data=precomputed_data)
#                         title = t + ' + w/o dust'
#                         fname = f + ' + wo dust'
#                     elif d == 'w/ dust':
#                         fig, ax = plot_pv(
#                             incl=70, vkm=v0, v_width=vwidth, nlam=nlam, extract_gas=True, npix=npix, sizeau=sizeau,
#                             convolve=convolve, fwhm=fwhm,
#                             precomputed_data_gas=precomputed_data_gas, precomputed_data_dust=precomputed_data_dust)
#                         title = t + ' + w/ dust'
#                         fname = f + ' + w dust'
#                     ax.set_title(title, fontsize = 16)
#                     if filename is None:
#                         fig.savefig(f'./figures/high_resol/gas_inside_rcb_{gas_inside}/amax_{amax}/Rcb_{rcb}/'+fname+'.pdf', transparent=True)
#                         plt.close('all')
#                     elif filename is not None:
#                         fig.savefig(filename+'.pdf', transparent=True)
#                         plt.close('all')
#                 elif convolve is True:
#                     if d == 'w/o dust':
#                         fig, ax, fig_convolved, ax_convolved = plot_pv(
#                             incl=70, vkm=v0, v_width=vwidth, nlam=nlam, nodust=True, npix=npix, sizeau=sizeau,
#                             convolve=convolve, fwhm=fwhm,
#                             precomputed_data=precomputed_data)
#                         title = t + ' + w/o dust'
#                         fname = f + ' + wo dust'
#                     elif d == 'w/ dust':
#                         fig, ax, fig_convolved, ax_convolved = plot_pv(
#                             incl=70, vkm=v0, v_width=vwidth, nlam=nlam, extract_gas=True, npix=npix, sizeau=sizeau,
#                             convolve=convolve, fwhm=fwhm,
#                             precomputed_data_gas=precomputed_data_gas, precomputed_data_dust=precomputed_data_dust)
#                         title = t + ' + w/ dust'
#                         fname = f + ' + w dust'
#                     ax.set_title(title, fontsize = 16)
#                     ax_convolved.set_title(title, fontsize = 16)
#                     if filename is None:
#                         fig.savefig(f'./figures/high_resol/gas_inside_rcb_{gas_inside}/amax_{amax}/Rcb_{rcb}/'+fname+'.pdf', transparent=True)
#                         fig_convolved.savefig(f'./figures/high_resol/gas_inside_rcb_{gas_inside}/amax_{amax}/Rcb_{rcb}/'+fname+f'_convolved_fwhm_{fwhm}au.pdf', transparent=True)
#                         plt.close('all')
#                     elif filename is not None:
#                         fig.savefig(filename+fname+'.pdf', transparent=True)
#                         fig_convolved.savefig(filename+fname+f'_convolved_fwhm_{fwhm}au.pdf', transparent=True)
#                         plt.close('all')
#                 if (precomputed_data_gas is None) and (precomputed_data_dust is None):
#                     os.system(f'mv image_gas.img ./precomputed_data/amax_{a}/vinfall_0.5/image_gas_{fname}.img')
#                     os.system(f'mv image_dust.img ./precomputed_data/amax_{a}/vinfall_0.5/image_dust_{fname}.img')

            
#     return
# ###############################################################################
# heat_list = ['Irradiation', 'Combine', 'Accretion']
# snowline = ['w/ snowline', 'w/o snowline']
# dust = ['w/o dust', 'w/ dust']
# heat_list = ['Irradiation', 'Combine', 'Accretion']
# snowline = ['w/ snowline', 'w/o snowline']
# dust = ['w/ dust', 'w/o dust']



# for _, a in enumerate([0.01, 0.03, 0.05,0.1, 0.3, 0.5, 1, 3, 10]):
#     multiple_plots(amax=a*0.1, rcb=5, nlam=100, npix=300, sizeau=300, v0=5, vwidth=10, gas_inside=False, convolve=True, fwhm=50, vinfall=0.5,
#                 filename=f"./figures/colorbar_rescaled/amax_{a}/vinfall_0.5/",
#                 precomputed_data_gas=None, precomputed_data_dust=None)

# for _, a in enumerate([0.01, 0.03, 0.05, 0.1]):
#     heat_list = ['Accretion']
#     multiple_plots(amax=a*0.1, rcb=5, nlam=100, npix=300, sizeau=300, v0=5, vwidth=10, gas_inside=False, convolve=True, fwhm=50, vinfall=1,
#                 filename=f"./figures/colorbar_rescaled/amax_{a}/vinfall_0.5/",
#                 precomputed_data_gas=f"./precomputed_data/amax_{a}/vinfall_0.5/image_gas_Accretion.img",
#                 precomputed_data_dust=f"./precomputed_data/amax_{a}/vinfall_0.5/image_dust_Accretion.img")
#     heat_list = ['Irradiation']
#     multiple_plots(amax=a*0.1, rcb=5, nlam=100, npix=300, sizeau=300, v0=5, vwidth=10, gas_inside=False, convolve=True, fwhm=50, vinfall=1,
#                 filename=f"./figures/colorbar_rescaled/amax_{a}/vinfall_0.5/",
#                 precomputed_data_gas=f"./precomputed_data/amax_{a}/vinfall_0.5/image_gas_Irradiation.img",
#                 precomputed_data_dust=f"./precomputed_data/amax_{a}/vinfall_0.5/image_dust_Irradiation.img")
#     heat_list = ['Combine']
#     multiple_plots(amax=a*0.1, rcb=5, nlam=100, npix=300, sizeau=300, v0=5, vwidth=10, gas_inside=False, convolve=True, fwhm=50, vinfall=1,
#                 filename=f"./figures/colorbar_rescaled/amax_{a}/vinfall_0.5/",
#                 precomputed_data_gas=f"./precomputed_data/amax_{a}/vinfall_0.5/image_gas_Combine.img",
#                 precomputed_data_dust=f"./precomputed_data/amax_{a}/vinfall_0.5/image_dust_Combine.img")

