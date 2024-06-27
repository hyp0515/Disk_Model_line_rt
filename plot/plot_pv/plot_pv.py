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
from spectral_cube import SpectralCube
from pvextractor import extract_pv_slice, Path
cube = SpectralCube.read(
    '/mnt/storage/CB68_setup1/CB68-Setup1-cube-products/CB68_218.440GHz_CH3OH_joint_0.5_clean.image.pbcor.common.fits'
    )
freq0 = 218.440063 * 1e9
v = cc / 1e5 * (freq0 - cube.spectral_axis.value) / freq0
path = Path([(793, 706), (710, 789)])
pvdiagram = extract_pv_slice(cube=cube, path=path, spacing=1)
v_axis = v[207:267]
offset = np.linspace(-150, 150, 50, endpoint=True)
O, V = np.meshgrid(offset, v_axis)
# contour_levels = np.linspace(0.01, pvdiagram.data[207:267, 36:86].max(), 4)
###############################################################################
"""
Plot Position-velocity (PV) diagrams
"""
def plot_pv(incl=70, line=240, vkm=0, v_width=20, nlam=50,
            nodust=False, scat=True, extract_gas=False, npix = 30, sizeau=100):
    if extract_gas is False:
        if nodust is True:
            prompt = ' noscat nodust'
        elif nodust is False:
            if scat is True:
                prompt = ' nphot_scat 1000000'
            elif scat is False:
                prompt = ' noscat'
        
        os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms 0 widthkms {v_width} linenlam {nlam}"+prompt)
        im = readImage('image.out')

        freq0 = (im.freq[nlam//2] + im.freq[(nlam//2)-1])/2
        v = cc / 1e5 * (freq0 - im.freq) / freq0
        center = int(len(im.y)//2)

        fig, ax = plt.subplots()
        c = ax.pcolormesh(im.x/au, v+vkm, (im.imageJyppix[:, center, :].T)*1e3/(140**2), shading="nearest", rasterized=True, cmap='jet')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('mJy/pixel',fontsize = 16)
        ax.set_xlabel("Offset [au]",fontsize = 16)
        ax.set_ylabel("Velocity [km/s]",fontsize = 16)
        ax.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax.plot([-(sizeau//2), (sizeau//2)], [vkm, vkm], 'w:')

        contour_levels = np.linspace(0.01, pvdiagram.data[207:267, 36:86].max(), 4)
        contour = ax.contour(O, V, pvdiagram.data[207:267, 36:86][:, ::-1], levels=contour_levels, colors='k', linewidths=0.5)

    elif extract_gas is True:
        os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms 0 widthkms {v_width} linenlam {nlam} nphot_scat 1000000")
        os.system('mv image.out image_gas.out')
        im_gas = readImage('image_gas.out')
        os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} lambdarange {im_gas.wav[0]} {im_gas.wav[-1]} nlam {nlam} nphot_scat 1000000 noline")
        os.system('mv image.out image_dust.out')
        im_dust = readImage('image_dust.out')

        freq0 = (im_gas.freq[nlam//2] + im_gas.freq[(nlam//2)-1])/2
        v = cc / 1e5 * (freq0 - im_gas.freq) / freq0
        center = int(len(im_gas.y)//2)

        data_gas  = im_gas.imageJyppix
        data_dust = im_dust.imageJyppix
        extracted_gas = data_gas-data_dust

        fig, ax = plt.subplots()
        c = ax.pcolormesh(im_gas.x/au, v+vkm, (extracted_gas[:, center, :].T)*1e3/(140**2), shading="nearest", rasterized=True, cmap='jet')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('mJy/pixel',fontsize = 16)
        ax.set_xlabel("Offset [au]",fontsize = 16)
        ax.set_ylabel("Velocity [km/s]",fontsize = 16)
        ax.plot([0, 0], [-v_width+vkm, v_width+vkm], 'w:')
        ax.plot([-(sizeau//2), (sizeau//2)], [vkm, vkm], 'w:')
        contour_levels = np.linspace(0.01, pvdiagram.data[207:267, 36:86].max(), 4)
        contour = ax.contour(O, V, pvdiagram.data[207:267, 36:86][:, ::-1], levels=contour_levels, colors='k', linewidths=1)

    return ax

###############################################################################
heat_list = ['Accretion', 'Irradiation', 'Combine']
# snowline = ['w/o snowline', 'w/ snowline']
snowline = ['w/ snowline']
# dust = ['w/o dust', 'w/ dust']
dust = ['w/ dust']
def multiple_plots(amax, rcb, nlam, npix, sizeau, v0=0, vwidth=5):
    for idx_h, heat in enumerate(heat_list):
        for idx_s, snow in enumerate(snowline):
            if heat == 'Accretion' and snow =='w/o snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=False, snowline=False, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=False)
                t = 'Accretion + w/o snowline'
                f = 'Accretion + wo snowline'
            elif heat == 'Accretion' and snow =='w/ snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=False, snowline=True, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=False)
                t = 'Accretion + w/ snowline'
                f = 'Accretion + w snowline'
            elif heat == 'Irradiation' and snow =='w/o snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=False)
                t = 'Irradiation + w/o snowline'
                f = 'Irradiation + wo snowline'
            elif heat == 'Irradiation' and snow =='w/ snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=True, snowline=True, floor=True, kep=True, Rcb=rcb, gas_inside_rcb=False)
                t = 'Irradiation + w/ snowline'
                f = 'Irradiation + w snowline'
            elif heat == 'Combine' and snow =='w/o snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=rcb, combine=True, gas_inside_rcb=False)
                t = 'Combine + w/o snowline'
                f = 'Combine + wo snowline'
            elif heat == 'Combine' and snow =='w/ snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=True, snowline=True, floor=True, kep=True, Rcb=rcb, combine=True, gas_inside_rcb=False)
                t = 'Combine + w/ snowline'
                f = 'Combine + w snowline'        

            for idx_d, d in enumerate(dust):
                if d == 'w/o dust':
                    p = plot_pv(incl=70, vkm=v0, v_width=vwidth, nlam=nlam, nodust=True, npix=npix, sizeau=sizeau)
                    title = t + ' + w/o dust'
                    fname = f + ' + wo dust'
                elif d == 'w/ dust':
                    p = plot_pv(incl=70, vkm=v0, v_width=vwidth, nlam=nlam, extract_gas=True, npix=npix, sizeau=sizeau)
                    title = t + ' + w/ dust'
                    fname = f + ' + w dust'
                p.set_title(title, fontsize = 16)
                # if sizeau == 200:
                #     plt.savefig(f'./figures/v_width_{vwidth}/mid_scale(200au)/amax_{amax}/Rcb_{rcb}/nlam_{nlam}_npix_{npix}/'+fname+'.png')
                # elif sizeau == 100:
                #     plt.savefig(f'./figures/v_width_{vwidth}/small_scale(100au)/amax_{amax}/Rcb_{rcb}/nlam_{nlam}_npix_{npix}/'+fname+'.png')
                # elif sizeau == 300:
                #     plt.savefig(f'./figures/v_width_{vwidth}/large_scale(300au)/amax_{amax}/Rcb_{rcb}/nlam_{nlam}_npix_{npix}/'+fname+'.png')
                plt.savefig(f'./figures/high_resol/amax_{amax}/Rcb_{rcb}/'+fname+'.pdf')
                plt.close()
    return
# ###############################################################################
a_list = [0.1, 0.001, 10]
r_list = [None, 5, 30]

for idx_r, r in enumerate(r_list):
    for idx_a, a in enumerate(a_list):
        multiple_plots(amax=a, rcb=r, nlam=100, npix=300, sizeau=300, v0=5, vwidth=10)
        os.system('make cleanall')
# # Kep
# for idx_mc, mctherm in enumerate([True, False]):
#     for idx_snow, snow in enumerate([True, False]):

#         problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=0, 
#                       pancake=False, mctherm=mctherm, snowline=snow, floor=True, kep=True, Rcb=None)
#         plot_pv(incl=70, vkm=0, v_width=5, nlam=50, extract_gas=True, npix=40, sizeau=300)
#         plt.savefig(f'./rcb_30/mctherm_{mctherm}_snowline_{snow}_kep.png')

#         plot_pv(incl=70, vkm=0, v_width=5, nlam=50, nodust=True, npix=40, sizeau=300)
#         plt.savefig(f'./rcb_30/mctherm_{mctherm}_snowline_{snow}_nodust_kep.png')
# os.system('make cleanall')

# # Infalling
# for idx_mc, mctherm in enumerate([True, False]):
#     for idx_snow, snow in enumerate([True, False]):

#         problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#                       pancake=False, mctherm=mctherm, snowline=snow, floor=True, kep=True, Rcb=None)
#         plot_pv(incl=70, vkm=0, v_width=5, nlam=50, extract_gas=True, npix=40, sizeau=300)
#         plt.savefig(f'./rcb_30/mctherm_{mctherm}_snowline_{snow}_infalling.png')

#         plot_pv(incl=70, vkm=0, v_width=5, nlam=50, nodust=True, npix=40, sizeau=300)
#         plt.savefig(f'./rcb_30/mctherm_{mctherm}_snowline_{snow}_nodust_infalling.png')
# os.system('make cleanall')

# # Gas inside Rcb
# for idx_mc, mctherm in enumerate([True, False]):
#     for idx_snow, snow in enumerate([True, False]):

#         problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#                       pancake=False, mctherm=mctherm, snowline=snow, floor=True, kep=True, Rcb=30, gas_inside_rcb=True)
#         plot_pv(incl=70, vkm=0, v_width=5, nlam=50, extract_gas=True, npix=40, sizeau=300)
#         plt.savefig(f'./rcb_30/mctherm_{mctherm}_snowline_{snow}_gas_inside.png')

#         plot_pv(incl=70, vkm=0, v_width=5, nlam=50, nodust=True, npix=40, sizeau=300)
#         plt.savefig(f'./rcb_30/mctherm_{mctherm}_snowline_{snow}_nodust_gas_inside.png')
# os.system('make cleanall')

# Oya
# for idx_h, heat in enumerate(['Accretion', 'Irradiation', 'Combine']):
#     if heat == 'Accretion':
#         problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#                         pancake=False, mctherm=False, snowline=True, floor=True, kep=True, Rcb=30, gas_inside_rcb=False)
#     if heat == 'Irradiation':
#         problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#                         pancake=False, mctherm=True, snowline=True, floor=True, kep=True, Rcb=30, gas_inside_rcb=False)
#     if heat == 'Combine':
#         problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#                         pancake=False, mctherm=True, snowline=True, floor=True, kep=True, Rcb=30, gas_inside_rcb=False, combine=True)
    
#     p = plot_pv(incl=70, vkm=5, v_width=10, nlam=100, extract_gas=True, npix=60, sizeau=300)
    

#     plt.savefig(f'./rcb_30/heat_{heat}_oya.pdf')

# os.system('make cleanall')
# for idx_a, a in enumerate([1, 0.1, 0.01, 0.001]):
#     for idx_h, heat in enumerate(['Accretion', 'Irradiation', 'Combine']):
#         for idx_s, snow in enumerate([True, False]):
            

#             if heat == 'Accretion':
#                 problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#                                 pancake=False, mctherm=False, snowline=snow, floor=True, kep=True, Rcb=5, gas_inside_rcb=False)
#             if heat == 'Irradiation':
#                 problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#                                 pancake=False, mctherm=True, snowline=snow, floor=True, kep=True, Rcb=5, gas_inside_rcb=False)
#             if heat == 'Combine':
#                 problem_setup(a_max=a, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#                                 pancake=False, mctherm=True, snowline=snow, floor=True, kep=True, Rcb=5, gas_inside_rcb=False, combine=True)
#             for idx_d, dust in enumerate([True, False]):
#                 if dust is True:
#                     p = plot_pv(incl=70, vkm=5, v_width=10, nlam=50, extract_gas=True, npix=300, sizeau=300)
#                 elif dust is False:
#                     p = plot_pv(incl=70, vkm=5, v_width=10, nlam=50, nodust=True, npix=300, sizeau=300)
    
# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=100*au, v_infall=1, 
#                                 pancake=False, mctherm=False, snowline=True, floor=True, kep=True, Rcb=5, gas_inside_rcb=False)
# p = plot_pv(incl=70, vkm=5, v_width=10, nlam=50, nodust=True, npix=300, sizeau=300)
# plt.savefig('test.pdf')
os.system('make cleanall')