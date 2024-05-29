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

    return ax
###############################################################################
heat_list = ['Accretion', 'Irradiation']
snowline = ['w/o snowline', 'w/ snowline']
dust = ['w/o dust', 'w/ dust']
def multiple_plots(amax, rcb, nlam, npix, sizeau, v0=0, vwidth=5):
    for idx_h, heat in enumerate(heat_list):
        for idx_s, snow in enumerate(snowline):
            if heat == 'Accretion' and snow =='w/o snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=False, snowline=False, floor=True, kep=True, Rcb=rcb)
                t = 'Accretion + w/o snowline'
                f = 'Accretion + wo snowline'
            elif heat == 'Accretion' and snow =='w/ snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=False, snowline=True, floor=True, kep=True, Rcb=rcb)
                t = 'Accretion + w/ snowline'
                f = 'Accretion + w snowline'
            elif heat == 'Irradiation' and snow =='w/o snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=True, snowline=False, floor=True, kep=True, Rcb=rcb)
                t = 'Irradiation + w/o snowline'
                f = 'Irradiation + wo snowline'
            elif heat == 'Irradiation' and snow =='w/ snowline':
                problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=True, snowline=True, floor=True, kep=True, Rcb=rcb)
                t = 'Irradiation + w/ snowline'
                f = 'Irradiation + w snowline'
            
            for idx_d, d in enumerate(dust):
                if d == 'w/o dust':
                    p = plot_pv(incl=70,vkm=v0, v_width=vwidth, nlam=nlam, nodust=True, npix=npix, sizeau=sizeau)
                    title = t + ' + w/o dust'
                    fname = f + ' + wo dust'
                elif d == 'w/ dust':
                    p = plot_pv(incl=70,vkm=v0, v_width=vwidth, nlam=nlam, extract_gas=True, npix=npix, sizeau=sizeau)
                    title = t + ' + w/ dust'
                    fname = f + ' + w dust'
                p.set_title(title, fontsize = 16)
                if sizeau == 200:
                    plt.savefig(f'./figures/mid_scale(200au)/amax_{amax}/Rcb_{rcb}/nlam_{nlam}_npix_{npix}/'+fname+'.png')
                elif sizeau == 100:
                    plt.savefig(f'./figures/small_scale(100au)/amax_{amax}/Rcb_{rcb}/nlam_{nlam}_npix_{npix}/'+fname+'.png')
                elif sizeau == 300:
                    plt.savefig(f'./figures/large_scale(300au)/amax_{amax}/Rcb_{rcb}/nlam_{nlam}_npix_{npix}/'+fname+'.png')
                plt.close()
    return
###############################################################################
for _, size in enumerate([300]):
    for _, a in enumerate([10, 0.1, 0.001]):
        for _, r in enumerate([5, 10, None]):
            for _, n in enumerate([20, 40]):
                multiple_plots(amax=a, rcb=r, nlam=n, npix=n, sizeau=size)

os.system('make cleanall')
