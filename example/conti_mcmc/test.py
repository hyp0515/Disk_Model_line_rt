import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
top = mpl.colormaps['Reds_r'].resampled(128)
bottom = mpl.colormaps['Blues'].resampled(128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
residual_cmp = ListedColormap(newcolors, name='RedsBlue')
from matplotlib.patches import Ellipse
import os
from scipy import ndimage
from astropy.coordinates import SkyCoord

import sys
sys.path.insert(0,'../../')
from disk_model import *
from radmc.setup import *
from radmc3dPy import *

sys.path.insert(0,'../')
from fit_with_GIdisk.find_center import find_center


filename = '/run/media/hyp0515/fd14f880-ba6f-450f-b82d-98ba3710dc5f/backup/CB68/cb68_alma/cb68_setup1_all.rob2.I.image.tt0.pbcor.smooth.dropdeg.fits'
ra_deg, dec_deg, disk_pa = find_center(filename)


cb68_image = DiskImage(
    fname=filename,
    ra_deg = ra_deg,
    dec_deg = dec_deg,
    distance_pc = 140,
    rms_Jy = 30e-6, # convert to Jy/beam
    disk_pa = disk_pa,
    img_size_au = 100,
    remove_background=True
)

cb68_conti  = cb68_image.img
beam_pa     = cb68_image.beam_pa
beam_maj_au = cb68_image.beam_maj_au
beam_min_au = cb68_image.beam_min_au
beam_area   = cb68_image.beam_area
au_per_pix  = cb68_image.au_per_pix
disk_pa     = cb68_image.disk_pa
size_au     = cb68_image.img_size_au


model = radmc3d_setup(silent=False)
model.get_mastercontrol(filename=None,
                        comment=None,
                        incl_dust=1,
                        incl_lines=1,
                        nphot=500000,
                        nphot_scat=1000000,
                        scattering_mode_max=2,
                        istar_sphere=1,
                        num_cpu=1,
                        modified_random_walk=1
                        )
model.get_linecontrol(filename=None,
                      methanol='ch3oh leiden 0 0 0')
model.get_continuumlambda(filename=None,
                          comment=None,
                          lambda_micron=None,
                          append=False)
model.get_diskcontrol(  d_to_g_ratio=0.01,
                        a_max=.001, 
                        Mass_of_star=0.14, 
                        Accretion_rate=5e-7,
                        Radius_of_disk=30,
                        NR=200,
                        NTheta=200,
                        NPhi=10)
model.get_vfieldcontrol(Kep=True,
                        vinfall=0.5,
                        Rcb=None,
                        outflow=None)
model.get_heatcontrol(heat='accretion')
model.get_gasdensitycontrol(abundance=1e-10,
                            snowline=100,
                            enhancement=1e5,
                            gas_inside_rcb=True)
# os.system('radmc3d mctherm')
os.system(f'radmc3d image npix {cb68_conti.shape[0]} sizeau {size_au} posang {disk_pa} incl 73 lambda 1300 noline')

im = image.readImage()
model_image = im.image[:,:,0].T

I = ndimage.rotate(model_image, -disk_pa+beam_pa,reshape=False)
sigmas = np.array([beam_maj_au, beam_min_au])/au_per_pix/(2*np.sqrt(2*np.log(2)))
I = ndimage.gaussian_filter(I, sigma=sigmas)
# rotate to align with image
I = ndimage.rotate(I, -beam_pa,reshape=False)
# convert to flux density in Jy/beam
model_image = I*1e23*beam_area

# sigma_levels = [10, 20, 40, 80, 160, 320, 640]
# contour_levels = [level * 30e-6 for level in sigma_levels]


fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
fig.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.1, wspace=0.1, hspace=0.05)

cb68 = ax[0].imshow(cb68_conti, cmap='Oranges', origin='lower', vmin=0.00, vmax=0.06)
# ax[0].plot([])
ax[0].set_title('CB68')
ax[0].set_yticks([0, 49//2, 48])
ax[0].set_yticklabels([-50, 0, 50])
ax[0].set_ylabel('AU')
ax[0].set_xticks([0, 49//2, 48])
ax[0].set_xticklabels([-50, 0, 50])
ax[0].set_xlabel('AU')
colorbar = fig.colorbar(cb68, ax=ax[0], pad=0.005, aspect=50, shrink=0.85)
beam = Ellipse((12, 12), width=cb68_image.beam_min_au/cb68_image.au_per_pix, height=cb68_image.beam_maj_au/cb68_image.au_per_pix,
               angle=-90, edgecolor='black', facecolor=None, lw=1.5, fill=False)
ax[0].add_patch(beam)
# contours = ax[0].contour(cb68_conti, levels=contour_levels, colors='black', linewidths=0.8)
# colorbar.set_label('Intensity (Jy/beam)')

model = ax[1].imshow(model_image, cmap='Oranges', origin='lower', vmin=0.00, vmax=0.06)
ax[1].set_title('Synthetic Observation')
ax[1].set_xticks([0, 49//2, 48])
ax[1].set_xticklabels([-50, 0, 50])
ax[1].set_xlabel('AU')
colorbar = fig.colorbar(model, ax=ax[1], pad=0.005, aspect=50, shrink=0.85)
beam = Ellipse((12, 12), width=cb68_image.beam_min_au/cb68_image.au_per_pix, height=cb68_image.beam_maj_au/cb68_image.au_per_pix,
               angle=-90, edgecolor='black', facecolor=None, lw=1.5, fill=False)
ax[1].add_patch(beam)
# colorbar.set_label('Intensity (Jy/beam)')

residual = ax[2].imshow(cb68_conti-model_image, cmap=residual_cmp, origin='lower', vmin=-0.06, vmax=0.06)
ax[2].set_title('Residual')
ax[2].set_xticks([0, 49//2, 48])
ax[2].set_xticklabels([-50, 0, 50])
ax[2].set_xlabel('AU')
colorbar = fig.colorbar(residual, ax=ax[2], pad=0.005, aspect=50, shrink=0.85)
colorbar.set_label('Intensity (Jy/beam)')

plt.savefig('residual.pdf', transparent=True)

# print(np.unravel_index(np.argmax(convolved_image.T), convolved_image.shape))
# print(np.unravel_index(np.argmax(cb68_conti), cb68_conti.shape))
